"""
Tokenize all texts in finetuning/output/labels.json with the GLM-OCR tokenizer
and report max (and optional) token counts per page. Optionally report input
(image + prompt) token counts for a sample of images.

Uses the same tokenizer as train_unsloth.py (no special tokens on the label text)
so counts match what the model sees for the response.

Run from project root:
  python finetuning/train/tokenize_labels.py
  python finetuning/train/tokenize_labels.py --labels path/to/labels.json --percentiles
  python finetuning/train/tokenize_labels.py --input-tokens --max-samples 100
"""
import argparse
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output"
MODEL_ID = "unsloth/GLM-OCR"
PROMPT = "Text Recognition:"


def main():
    parser = argparse.ArgumentParser(description="Tokenize labels.json and report max tokens per page.")
    parser.add_argument("--labels", type=Path, default=OUTPUT_DIR / "labels.json", help="Path to labels.json")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID for tokenizer (default: unsloth/GLM-OCR)")
    parser.add_argument("--percentiles", action="store_true", help="Print 50th, 90th, 95th, 99th percentiles for label lengths")
    parser.add_argument("--input-tokens", action="store_true", help="Also compute input (image+prompt) token count for a sample of images")
    parser.add_argument("--max-samples", type=int, default=50, help="Max images to use for input-tokens (default 50)")
    args = parser.parse_args()

    if not args.labels.is_file():
        print(f"File not found: {args.labels}", flush=True)
        return 1

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if not labels:
        print("No entries in labels.json.", flush=True)
        return 0

    from transformers import AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Label token counts
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
    print("=== Label tokens (response text) ===")
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

    # Input (image + prompt) token counts for a sample of images
    if args.input_tokens:
        image_dir = args.labels.parent
        if not image_dir.is_dir():
            print(f"\nImage dir not found: {image_dir} (skip --input-tokens)", flush=True)
        else:
            keys = list(labels.keys())[: args.max_samples]
            input_lengths = []
            for key in keys:
                path = image_dir / key
                if not path.is_file():
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                except Exception:
                    continue
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": PROMPT},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                n = inputs["input_ids"].shape[1]
                input_lengths.append(n)
            if input_lengths:
                input_lengths.sort()
                n_samp = len(input_lengths)
                print("\n=== Input tokens (image + prompt) per page ===")
                print(f"Sample: {n_samp} images from {image_dir}")
                print(f"Min input tokens: {min(input_lengths)}")
                print(f"Max input tokens: {max(input_lengths)}")
                print(f"Mean input tokens: {sum(input_lengths) / n_samp:.1f}")
                if args.percentiles:
                    for p in (50, 90, 95, 99):
                        idx = max(0, int(n_samp * p / 100) - 1)
                        print(f"  {p}th percentile: {input_lengths[idx]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
