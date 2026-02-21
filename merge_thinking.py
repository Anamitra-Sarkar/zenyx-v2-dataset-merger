"""CLI to merge THINKING datasets into unified sharded Parquet, push to HF iteratively.

Fixes:
- Iterative shard upload: each shard is pushed to HF immediately after writing
  so a network failure mid-upload doesn't lose hours of work
- Retry wrapper on upload
- Progress logging per dataset
"""

from __future__ import annotations

import argparse
import os
import time
import logging
from pathlib import Path

from huggingface_hub import HfApi, login

from src.config import read_yaml, write_json
from src.adapters import DatasetSpec, iter_dataset
from src.writer import write_shards

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ZenyxMerger")


def upload_with_retry(
    api: HfApi,
    local_path: str,
    repo_path: str,
    repo_id: str,
    token: str,
    retries: int = 5,
    backoff: float = 10.0,
) -> None:
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
            log.warning(f"  Upload attempt {attempt} failed: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_thinking.yaml")
    args = ap.parse_args()

    cfg   = read_yaml(args.config)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    out_dir    = Path(cfg["output"]["local_dir"])
    push_to_hf = bool(cfg["output"].get("push_to_hf", False))
    repo_id    = cfg["output"].get("repo_id", "")
    private    = bool(cfg["output"].get("private", True))

    out_dir.mkdir(parents=True, exist_ok=True)

    api = None
    if push_to_hf:
        if not token:
            raise RuntimeError("Set HF_TOKEN env var to push to HF.")
        login(token=token, add_to_git_credential=False)
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

    streaming  = bool(cfg["runtime"].get("streaming", True))
    shard_size = int(cfg["runtime"].get("shard_size", 5000))
    max_per_ds = cfg["runtime"].get("max_examples_per_dataset")
    candidates = cfg["schema"]["text_key_candidates"]

    all_shards   = []
    total_start  = time.time()

    for spec in specs:
        log.info(f"\n{'='*60}")
        log.info(f"Loading: {spec.id}  (subset={spec.subset}, split={spec.split})")

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

        ds_start    = time.time()
        shard_paths = write_shards(gen_one(), out_dir, shard_size)
        all_shards.extend(shard_paths)
        log.info(f"✓ {spec.id}: {len(shard_paths)} shards in {time.time()-ds_start:.1f}s")

        # push each shard immediately after writing
        if push_to_hf and api:
            for sp in shard_paths:
                upload_with_retry(
                    api=api,
                    local_path=sp,
                    repo_path=f"data/{Path(sp).name}",
                    repo_id=repo_id,
                    token=token,
                )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_shards": len(all_shards),
        "shards": [Path(p).name for p in all_shards],
        "streaming": streaming,
        "shard_size": shard_size,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    log.info(f"\n✓ Manifest written: {manifest_path}")

    if push_to_hf and api:
        upload_with_retry(
            api=api,
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
