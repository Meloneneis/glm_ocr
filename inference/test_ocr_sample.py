"""
Run OCR on a small sample of images and print formatted output to the terminal.
Uses the same model and logic as run_ocr.py.

Run from project root:
  python inference/test_ocr_sample.py
  python inference/test_ocr_sample.py --images-dir inference/21jhd/images --num 5
"""
import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_ocr

_PROJECT_ROOT = _SCRIPT_DIR.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR on a small sample of images and print formatted output.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=_PROJECT_ROOT / "inference" / "21jhd" / "images",
        help="Directory containing PNG images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meloneneneis/glm_ocr_21jhd",
        help="HuggingFace adapter id or local path to adapter.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of images to run (default: 5).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=run_ocr.DEFAULT_MAX_NEW_TOKENS,
        help="Max new tokens per image (safety cap; generation stops at EOS).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per batch (default: 8).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single image file to run on (overrides images-dir/num).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw output: no stripping of special tokens, decode with special tokens visible.",
    )
    parser.add_argument(
        "--with-probs",
        action="store_true",
        help="Compute and show per-token probabilities (for confidence/QC).",
    )
    args = parser.parse_args()

    if args.image is not None:
        path = args.image.resolve()
        if not path.is_file():
            print(f"Image file not found: {path}", file=sys.stderr)
            sys.exit(1)
        image_paths = [path]
        images_dir = path.parent
    else:
        images_dir = args.images_dir
        if not images_dir.is_dir():
            print(f"Images directory not found: {images_dir}", file=sys.stderr)
            sys.exit(1)
        image_paths = sorted(images_dir.rglob("*.png"))[: args.num]
    if not image_paths:
        print("No PNG images found in images-dir.", file=sys.stderr)
        sys.exit(1)

    batch_size = max(1, args.batch_size)
    print("Inference params:")
    print(f"  model:           {args.model}")
    print(f"  images_dir:      {images_dir}")
    if args.image is not None:
        print(f"  image:           {args.image}")
    else:
        print(f"  num:             {args.num}")
    print(f"  batch_size:      {batch_size}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  raw:             {args.raw}")
    print(f"  with_probs:     {args.with_probs}")
    print(f"  images to run:   {len(image_paths)}")
    print()

    print(f"Loading model and processor (model: {args.model})...")
    model, processor = run_ocr._load_model_and_processor(args.model)
    print(f"Running OCR on {len(image_paths)} image(s) (batch_size={batch_size})...\n")

    t0 = time.perf_counter()
    texts = []
    token_probs_list = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_texts, _, batch_probs = run_ocr._run_ocr_batch(
            processor, model, batch_paths, args.max_new_tokens, raw=args.raw, with_probs=args.with_probs
        )
        texts.extend(batch_texts)
        if batch_probs:
            for i in range(len(batch_paths)):
                token_probs_list.append(batch_probs[i] if batch_probs[i] is not None else [])
    elapsed = time.perf_counter() - t0
    n = len(image_paths)
    print(f"OCR run time: {elapsed:.2f} s  |  Time per image: {elapsed / n:.2f} s\n")

    tokenizer = getattr(processor, "tokenizer", processor)
    for img_idx, (path, text) in enumerate(zip(image_paths, texts)):
        rel = path.relative_to(images_dir) if path.is_relative_to(images_dir) else path.name
        abs_path = path.resolve()
        file_url = abs_path.as_uri()
        ids = tokenizer.encode(text or "", add_special_tokens=False)
        num_tokens = len(ids) if not ids or not isinstance(ids[0], list) else sum(len(x) for x in ids)
        print("=" * 60)
        print(f"  {rel}")
        print(f"  path:   {abs_path}")
        print(f"  open:   {file_url}")
        print(f"  tokens: {num_tokens}")
        print("=" * 60)
        if args.raw:
            print(repr(text) if text else "(no text)")
        else:
            content = text or "(no text)"
            for line_no, line in enumerate(content.splitlines(), start=1):
                print(f"  {line_no:4d}| {line}")
        if args.with_probs and img_idx < len(token_probs_list) and token_probs_list[img_idx]:
            probs = token_probs_list[img_idx]
            # Top 10 lowest with 5-token context
            lowest = sorted(enumerate(probs), key=lambda x: x[1][1])[:10]
            show_n = len(lowest)
            print(f"\n  Lowest-prob tokens (top {show_n} of {len(probs)}), 5 tokens context:")
            RED, RESET = "\033[31m", "\033[0m"
            for idx, (tok, p) in lowest:
                left = probs[max(0, idx - 5) : idx]
                right = probs[idx + 1 : min(len(probs), idx + 6)]
                left_str = "".join(t for t, _ in left)
                right_str = "".join(t for t, _ in right)
                print(f"    …{left_str}{RED}{tok}{RESET}{right_str}…  (p={p:.4f})")
            # Full list sorted lowest to highest
            print(f"\n  Full list ({len(probs)} tokens, low to high):")
            for tok, p in sorted(probs, key=lambda x: x[1]):
                print(f"    {repr(tok):20s}  {p:.4f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
