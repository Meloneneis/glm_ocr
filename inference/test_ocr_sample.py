"""
Run OCR on a small sample of images and print formatted output to the terminal.
Uses the same model and logic as run_ocr.py.

Run from project root:
  python inference/test_ocr_sample.py
  python inference/test_ocr_sample.py --images-dir inference/21jhd/images --num 5
"""
import argparse
import sys
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
    print(f"  images to run:   {len(image_paths)}")
    print()

    print(f"Loading model and processor (model: {args.model})...")
    model, processor = run_ocr._load_model_and_processor(args.model)
    print(f"Running OCR on {len(image_paths)} image(s) (batch_size={batch_size})...\n")

    texts = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        texts.extend(run_ocr._run_ocr_batch(processor, model, batch_paths, args.max_new_tokens, raw=args.raw))

    for path, text in zip(image_paths, texts):
        rel = path.relative_to(images_dir) if path.is_relative_to(images_dir) else path.name
        abs_path = path.resolve()
        file_url = abs_path.as_uri()
        print("=" * 60)
        print(f"  {rel}")
        print(f"  path:   {abs_path}")
        print(f"  open:   {file_url}")
        print("=" * 60)
        if args.raw:
            print(repr(text) if text else "(no text)")
        else:
            print(text or "(no text)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
