"""
Split images in finetuning/output into train (1000) and test (100).

Writes train.txt and test.txt (one image filename per line) into the output dir.
Uses a fixed seed for reproducible splits.

Run from project root: python finetuning/data_prep/split_train_test.py
"""
import argparse
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Split images into train (1000) and test (100).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "finetuning" / "output",
        help="Directory containing the PNG images (default: finetuning/output/).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=8500,
        help="Number of images for training (default 1000).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=510,
        help="Number of images for test (default 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default 42).",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Output directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    images = sorted(f.name for f in args.output_dir.glob("*.png"))
    if not images:
        print("No PNG images found in output dir.", file=sys.stderr)
        sys.exit(1)

    total = args.train_size + args.test_size
    if len(images) < total:
        print(f"Not enough images: have {len(images)}, need {total}. Reduce --train-size/--test-size.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    shuffled = images.copy()
    random.shuffle(shuffled)

    train_list = shuffled[: args.train_size]
    test_list = shuffled[args.train_size : args.train_size + args.test_size]

    train_file = args.output_dir / "train.txt"
    test_file = args.output_dir / "test.txt"
    train_file.write_text("\n".join(train_list) + "\n")
    test_file.write_text("\n".join(test_list) + "\n")

    print(f"Wrote {len(train_list)} lines to {train_file}")
    print(f"Wrote {len(test_list)} lines to {test_file}")


if __name__ == "__main__":
    main()
