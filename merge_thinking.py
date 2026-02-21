"""CLI to merge THINKING datasets into a unified dataset on disk, and optionally push to HF."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, login

from src.config import read_yaml, write_json
from src.adapters import DatasetSpec, iter_dataset
from src.writer import write_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_thinking.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    out_dir = Path(cfg["output"]["local_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec(id=d["id"], subset=d.get("subset"), split=d.get("split", "train"))
        for d in cfg["datasets"]
    ]

    streaming = bool(cfg["runtime"].get("streaming", True))
    shard_size = int(cfg["runtime"].get("shard_size", 100000))
    max_per_ds = cfg["runtime"].get("max_examples_per_dataset")

    candidates = cfg["schema"]["text_key_candidates"]

    def gen_all():
        for spec in specs:
            n = 0
            for ex in iter_dataset(spec, streaming=streaming, token=token, text_key_candidates=candidates):
                yield ex
                n += 1
                if max_per_ds is not None and n >= int(max_per_ds):
                    break

    t0 = time.time()
    shard_paths = write_shards(gen_all(), out_dir, shard_size)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg,
        "num_shards": len(shard_paths),
        "shards": [Path(p).name for p in shard_paths],
        "streaming": streaming,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"Wrote {len(shard_paths)} shards to {out_dir} in {time.time() - t0:.1f}s")

    if cfg["output"].get("push_to_hf", False):
        if not token:
            raise RuntimeError("Set HF_TOKEN env var to push.")
        login(token=token, add_to_git_credential=False)
        api = HfApi(token=token)
        repo_id = cfg["output"]["repo_id"]
        private = bool(cfg["output"].get("private", True))
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Pushed to HF: {repo_id}")


if __name__ == "__main__":
    main()
