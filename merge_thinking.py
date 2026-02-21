"""CLI to merge THINKING datasets into unified sharded Parquet, push to HF.

Usage — single config:
    python merge_thinking.py --config configs/reasoning_math.yaml

Usage — multiple configs in one run:
    python merge_thinking.py --config configs/reasoning_math.yaml configs/reasoning_code.yaml

Usage — all configs at once (glob):
    python merge_thinking.py --config configs/*.yaml

Fixes:
- Multi-config support: --config accepts one or more YAML files
- Shards land in category subfolders: out_thinking/reasoning/math/, out_thinking/reasoning/code/, etc.
- HF upload mirrors the same subfolder structure: data/reasoning/math/*.parquet
- Per-thread HfApi instances for thread-safe concurrent uploads
- --dry-run flag: validates pipeline without writing anything
"""

from __future__ import annotations

import argparse
import glob
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
# Token resolution
# ─────────────────────────────────────────────────────────────────────────────

def _get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            log.info("HF token loaded from Kaggle secrets.")
            return tok
    except Exception:
        pass
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
    """Upload a single file with exponential backoff. Thread-safe (own HfApi instance)."""
    api = HfApi(token=token)
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
    category: str,
    max_workers: int = 4,
) -> None:
    """
    Upload shards concurrently into data/<category>/ subfolder on HF.
    Each worker thread creates its own HfApi instance — thread-safe.
    """
    if not shard_paths:
        return

    # HF repo path: data/reasoning/math/shard.parquet
    hf_prefix = f"data/{category}" if category else "data"

    def _upload_one(sp: str) -> str:
        upload_with_retry(
            local_path=sp,
            repo_path=f"{hf_prefix}/{Path(sp).name}",
            repo_id=repo_id,
            token=token,
        )
        return sp

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_one, sp): sp for sp in shard_paths}
        with tqdm(total=len(futures), desc=f"Uploading [{category}]", unit="shard") as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    log.error(f"  ✗ Shard upload failed: {futures[fut]} — {e}")
                finally:
                    pbar.update(1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-config runner
# ─────────────────────────────────────────────────────────────────────────────

def run_config(
    config_path: str,
    token: str | None,
    push_to_hf: bool,
    upload_workers: int,
    dry_run: bool,
    dry_run_examples: int,
) -> dict:
    """Process one config file. Returns a summary dict."""
    cfg      = read_yaml(config_path)
    category = cfg["output"].get("category", "")
    repo_id  = cfg["output"].get("repo_id", "")
    private  = bool(cfg["output"].get("private", True))
    base_out = Path(cfg["output"]["local_dir"])

    # Shards for this category go into out_dir/reasoning/math/ etc.
    out_dir = base_out / category if category else base_out

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    if push_to_hf and not dry_run:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        log.info(f"✓ HF repo ready: {repo_id}")

    streaming  = bool(cfg["runtime"].get("streaming", True))
    shard_size = int(cfg["runtime"].get("shard_size", 5000))
    max_per_ds = cfg["runtime"].get("max_examples_per_dataset")
    candidates = cfg["schema"]["text_key_candidates"]

    if dry_run:
        max_per_ds = dry_run_examples

    specs = [
        DatasetSpec(
            id=d["id"],
            subset=d.get("subset"),
            split=d.get("split", "train"),
            category=category,
        )
        for d in cfg.get("datasets") or []
    ]

    if not specs:
        log.info(f"[{config_path}] No datasets configured — skipping.")
        return {"category": category, "shards": [], "skipped": True}

    all_shards: list[str] = []
    cat_start = time.time()

    log.info(f"\n{'='*60}")
    log.info(f"Category: {category or '(none)'}  |  Config: {config_path}")
    log.info(f"{'='*60}")

    for spec in specs:
        log.info(f"  Loading: {spec.id}  (subset={spec.subset}, split={spec.split})")
        ds_label = spec.id.split("/")[-1]
        if spec.subset:
            ds_label += f"-{spec.subset}"

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

        if dry_run:
            count = sum(1 for _ in gen_one())
            log.info(f"  [DRY-RUN] {spec.id} ({spec.subset}): {count} examples in {time.time()-ds_start:.1f}s")
        else:
            shard_paths = write_shards(gen_one(), out_dir, shard_size, dataset_prefix=ds_label)
            all_shards.extend(shard_paths)
            log.info(f"  ✓ {spec.id}: {len(shard_paths)} shard(s) in {time.time()-ds_start:.1f}s")

            if push_to_hf and shard_paths:
                upload_shards_concurrent(
                    shard_paths=shard_paths,
                    repo_id=repo_id,
                    token=token,
                    category=category,
                    max_workers=upload_workers,
                )

    if dry_run:
        log.info(f"  [DRY-RUN] {category}: done in {time.time()-cat_start:.1f}s")
        return {"category": category, "shards": [], "skipped": False}

    return {"category": category, "shards": all_shards, "out_dir": str(out_dir)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge HuggingFace datasets into categorised sharded Parquet."
    )
    ap.add_argument(
        "--config", nargs="+", default=["configs/reasoning_math.yaml"],
        help="One or more YAML config files (supports shell glob expansion).",
    )
    ap.add_argument("--push", action="store_true",
                    help="Push shards to HuggingFace Hub after writing.")
    ap.add_argument("--upload-workers", type=int, default=4)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Stream examples and count without writing any files.",
    )
    ap.add_argument("--dry-run-examples", type=int, default=200)
    args = ap.parse_args()

    # Expand globs (shell may not expand *.yaml on all platforms)
    config_files: list[str] = []
    for pat in args.config:
        expanded = glob.glob(pat)
        config_files.extend(sorted(expanded) if expanded else [pat])
    config_files = list(dict.fromkeys(config_files))  # deduplicate, preserve order

    if not config_files:
        log.error("No config files found. Check --config path.")
        return

    token      = _get_hf_token()
    push_to_hf = args.push

    if push_to_hf and not token:
        raise RuntimeError(
            "HF token not found. Set HF_TOKEN as env var, Kaggle secret, or Colab userdata."
        )
    if push_to_hf and token:
        login(token=token, add_to_git_credential=False)

    if args.dry_run:
        log.info("[DRY-RUN] No shards will be written and nothing will be uploaded.")

    total_start = time.time()
    all_results: list[dict] = []

    for cfg_path in config_files:
        result = run_config(
            config_path=cfg_path,
            token=token,
            push_to_hf=push_to_hf,
            upload_workers=args.upload_workers,
            dry_run=args.dry_run,
            dry_run_examples=args.dry_run_examples,
        )
        all_results.append(result)

    if args.dry_run:
        log.info(f"\n{'='*60}")
        log.info(f"[DRY-RUN] ALL DONE | {time.time()-total_start:.1f}s total")
        log.info(f"{'='*60}")
        return

    # ── Write global manifest after ALL configs are done ───────────────────────
    base_out = Path(read_yaml(config_files[0])["output"]["local_dir"])
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "categories": [
            {
                "category":   r["category"],
                "num_shards": len(r.get("shards", [])),
                "shards":     [Path(p).name for p in r.get("shards", [])],
                "out_dir":    r.get("out_dir", ""),
            }
            for r in all_results if not r.get("skipped")
        ],
    }
    manifest_path = base_out / "manifest.json"
    write_json(manifest_path, manifest)
    log.info(f"\n✓ Manifest written: {manifest_path}")

    if push_to_hf and token:
        # Upload manifest to the first config's repo
        first_cfg = read_yaml(config_files[0])
        repo_id   = first_cfg["output"].get("repo_id", "")
        if repo_id:
            upload_with_retry(
                local_path=str(manifest_path),
                repo_path="manifest.json",
                repo_id=repo_id,
                token=token,
            )

    total_shards = sum(len(r.get("shards", [])) for r in all_results)
    log.info(f"\n{'='*60}")
    log.info(f"✓ ALL DONE | {total_shards} total shards | {time.time()-total_start:.1f}s")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
