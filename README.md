# zenyx-v2-dataset-merger

Dataset merger pipeline for Zenyx-V2.

## What it does
- Downloads multiple Hugging Face datasets (streaming where possible).
- Normalizes to a unified schema:
  - `text`: full prompt/response training text (may contain `<think>` blocks)
  - `source`: dataset id
  - `subset`: optional config/subset
  - `split`: split name
  - `license`: best-effort (optional)
  - `meta`: optional JSON metadata
- Writes sharded Parquet + a lightweight manifest.
- Optionally pushes the merged dataset to the Hugging Face Hub.

## Quickstart

```bash
python -m pip install -r requirements.txt
export HF_TOKEN=...   # or edit config.yaml
python merge_thinking.py --config config_thinking.yaml
```

## Notes
- Some datasets have varying schemas; the merger uses heuristics + per-dataset adapters.
- For very large datasets, prefer `--streaming` and use `--max_examples` for dry runs.
