"""CLI to merge THINKING datasets into unified sharded Parquet, push to HF.

Fixes:
- Iterative shard upload with retry (each shard pushed immediately after writing)
- Concurrent upload via ThreadPoolExecutor — each thread gets its OWN HfApi instance
  to avoid shared-state thread-safety issues with the HF client
- dataset_prefix passed to write_shards so filenames are globally unique
- HF_TOKEN read from env var, Kaggle secrets, and Colab userdata
- Shard-level progress bar via tqdm
- --dry-run flag: streams up to N examples per dataset and prints counts without
  writing any shards — useful for validating pipeline on Colab/Kaggle before a full run
"""

from __future__ import annotations

import argparse
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, login
from tqdm import tqdm

from src.config import read_yaml, write_json
from src.adapters import DatasetSpec, iter_dataset
from src.writer import write_shards

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ZenyxMerger")


# ─────────────────────────────────────────────────────────────────────────────
# Token resolution — works on Colab, Kaggle, and plain env vars
# ─────────────────────────────────────────────────────────────────────────────

def _get_hf_token() -> str | None:
    """Read HF token from environment or platform secret stores."""
    # 1. Plain env var (works everywhere)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok

    # 2. Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            log.info("HF token loaded from Kaggle secrets.")
            return tok
    except Exception:
        pass

    # 3. Google Colab userdata
    try:
        from google.colab import userdata  # type: ignore
        tok = userdata.get("HF_TOKEN")
        if tok:
            log.info("HF token loaded from Colab userdata.")
            return tok
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Upload helpers
# ─────────────────────────────────────────────────────────────────────────────

def upload_with_retry(
    local_path: str,
    repo_path: str,
    repo_id: str,
    token: str,
    retries: int = 5,
    backoff: float = 10.0,
) -> None:
    """
    Upload a single file to HF Hub with exponential backoff retry.

    IMPORTANT: Creates its own HfApi instance so this function is safe to call
    from multiple threads concurrently (HfApi holds internal HTTP session state
    that is not thread-safe when shared across threads).
    """
    api = HfApi(token=token)  # one client per call — thread-safe
    for attempt in range(1, retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            log.info(f"  ✓ Uploaded: {repo_path}")
            return
        except Exception as e:
            if attempt == retries:
                log.error(f"  ✗ Upload failed after {retries} attempts: {repo_path} — {e}")
                raise
            wait = backoff * (1.5 ** (attempt - 1))
            log.warning(f"  Upload attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)


def upload_shards_concurrent(
    shard_paths: list[str],
    repo_id: str,
    token: str,
    max_workers: int = 4,
) -> None:
    """
    Upload a list of shard files concurrently with a progress bar.

    Each worker thread calls upload_with_retry which creates its own HfApi
    instance, so there is no shared mutable state between threads.
    """
    if not shard_paths:
        return

    def _upload_one(sp: str) -> str:
        upload_with_retry(
            local_path=sp,
            repo_path=f"data/{Path(sp).name}",
            repo_id=repo_id,
            token=token,
        )
        return sp

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_one, sp): sp for sp in shard_paths}
        with tqdm(total=len(futures), desc="Uploading shards", unit="shard") as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    log.error(f"  ✗ Shard upload failed: {futures[fut]} — {e}")
                finally:
                    pbar.update(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge HuggingFace thinking/CoT datasets into unified sharded Parquet."
    )
    ap.add_argument("--config", default="config_thinking.yaml",
                    help="Path to YAML config file.")
    ap.add_argument("--upload-workers", type=int, default=4,
                    help="Concurrent HF upload threads (default: 4).")
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Stream up to --dry-run-examples examples per dataset and print "
            "extraction counts WITHOUT writing any shards or uploading anything. "
            "Use this to validate the pipeline cheaply before a full run."
        ),
    )
    ap.add_argument("--dry-run-examples", type=int, default=200,
                    help="Examples to sample per dataset in dry-run mode (default: 200).")
    args = ap.parse_args()

    cfg   = read_yaml(args.config)
    token = _get_hf_token()

    out_dir    = Path(cfg["output"]["local_dir"])
    push_to_hf = bool(cfg["output"].get("push_to_hf", False))
    repo_id    = cfg["output"].get("repo_id", "")
    private    = bool(cfg["output"].get("private", True))

    # In dry-run mode we never write or upload anything
    if args.dry_run:
        log.info("[DRY-RUN] No shards will be written and nothing will be uploaded.")
        push_to_hf = False
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    api = None
    if push_to_hf:
        if not token:
            raise RuntimeError(
                "HF token not found. Set HF_TOKEN as an env var, "
                "Kaggle secret, or Colab userdata secret."
            )
        login(token=token, add_to_git_credential=False)
        # We only need one HfApi here for repo creation; uploads use per-thread instances
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        log.info(f"✓ HF repo ready: {repo_id}")

    specs = [
        DatasetSpec(
            id=d["id"],
            subset=d.get("subset"),
            split=d.get("split", "train"),
        )
        for d in cfg["datasets"]
    ]

    streaming      = bool(cfg["runtime"].get("streaming", True))
    shard_size     = int(cfg["runtime"].get("shard_size", 5000))
    max_per_ds     = cfg["runtime"].get("max_examples_per_dataset")
    candidates     = cfg["schema"]["text_key_candidates"]
    upload_workers = args.upload_workers

    # In dry-run mode, cap each dataset at dry_run_examples
    if args.dry_run:
        max_per_ds = args.dry_run_examples

    all_shards  = []
    total_start = time.time()

    for spec in specs:
        log.info(f"\n{'='*60}")
        log.info(f"Loading: {spec.id}  (subset={spec.subset}, split={spec.split})")

        # Human-readable prefix for shard filenames: last component of dataset id
        ds_label = spec.id.split("/")[-1]

        def gen_one(s=spec):
            n = 0
            for ex in iter_dataset(
                s,
                streaming=streaming,
                token=token,
                text_key_candidates=candidates,
            ):
                yield ex
                n += 1
                if max_per_ds is not None and n >= int(max_per_ds):
                    break

        ds_start = time.time()

        if args.dry_run:
            # Just consume and count — no disk writes
            count = sum(1 for _ in gen_one())
            log.info(
                f"[DRY-RUN] {spec.id}: {count} examples extracted "
                f"in {time.time()-ds_start:.1f}s"
            )
        else:
            shard_paths = write_shards(
                gen_one(), out_dir, shard_size, dataset_prefix=ds_label
            )
            all_shards.extend(shard_paths)
            log.info(
                f"✓ {spec.id}: {len(shard_paths)} shard(s) written in "
                f"{time.time()-ds_start:.1f}s"
            )

            # Push this dataset's shards immediately (concurrent, thread-safe)
            if push_to_hf and shard_paths:
                upload_shards_concurrent(
                    shard_paths=shard_paths,
                    repo_id=repo_id,
                    token=token,
                    max_workers=upload_workers,
                )

    if args.dry_run:
        log.info(f"\n{'='*60}")
        log.info(f"[DRY-RUN] DONE | {time.time()-total_start:.1f}s total — no data written.")
        log.info(f"{'='*60}")
        return

    # ── Manifest — written and uploaded AFTER all shards are confirmed done ────
    manifest = {
        "created_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_shards":  len(all_shards),
        "shards":      [Path(p).name for p in all_shards],
        "streaming":   streaming,
        "shard_size":  shard_size,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    log.info(f"\n✓ Manifest written: {manifest_path}")

    if push_to_hf:
        upload_with_retry(
            local_path=str(manifest_path),
            repo_path="manifest.json",
            repo_id=repo_id,
            token=token,
        )

    log.info(f"\n{'='*60}")
    log.info(f"✓ DONE | {len(all_shards)} shards | {time.time()-total_start:.1f}s total")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
