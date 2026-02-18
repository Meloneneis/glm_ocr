"""
Tokenize all texts in finetuning/output/labels.json with the GLM-OCR tokenizer
and report max (and optional) token counts per page.

Uses the same tokenizer as train_unsloth.py (no special tokens on the label text)
so counts match what the model sees for the response.

Run from project root:
  python finetuning/train/tokenize_labels.py
  python finetuning/train/tokenize_labels.py --labels path/to/labels.json
"""
import argparse
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output"
MODEL_ID = "unsloth/GLM-OCR"


def main():
    parser = argparse.ArgumentParser(description="Tokenize labels.json and report max tokens per page.")
    parser.add_argument("--labels", type=Path, default=OUTPUT_DIR / "labels.json", help="Path to labels.json")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID for tokenizer (default: unsloth/GLM-OCR)")
    parser.add_argument("--percentiles", action="store_true", help="Print 50th, 90th, 95th, 99th percentiles")
    args = parser.parse_args()

    if not args.labels.is_file():
        print(f"File not found: {args.labels}", flush=True)
        return 1

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if not labels:
        print("No entries in labels.json.", flush=True)
        return 0

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)

    lengths = []
    max_len = 0
    max_key = None
    for key, text in labels.items():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = []
        elif isinstance(ids[0], list):
            ids = ids[0]
        n = len(ids)
        lengths.append(n)
        if n > max_len:
            max_len = n
            max_key = key

    lengths.sort()
    n_pages = len(lengths)
    print(f"Labels: {n_pages} pages")
    print(f"Max tokens per page: {max_len}")
    if max_key:
        print(f"  (longest: {max_key})")
    print(f"Min tokens per page: {min(lengths)}")
    print(f"Mean tokens per page: {sum(lengths) / n_pages:.1f}")

    if args.percentiles:
        for p in (50, 90, 95, 99):
            idx = max(0, int(n_pages * p / 100) - 1)
            print(f"  {p}th percentile: {lengths[idx]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
