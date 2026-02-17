"""
Data prep: sample PDFs from a data source, render pages to images, write to output.

Hyperparameters: num_pdfs (how many PDFs to convert), scale_factor (render scale),
seed (for reproducible sampling). Output images are named like the PDF with page number
(e.g. mydoc.pdf -> mydoc_page_0001.png, mydoc_page_0002.png, ...).

Run from project root: python finetuning/data_prep/pdf_to_images.py --num-pdfs 5 --seed 42
"""
import argparse
import random
import sys
from pathlib import Path

import pypdfium2 as pdfium

# Project root (parent of finetuning/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent


def pdf_to_images(pdf_path: Path, scale: float) -> list:
    """Render each PDF page to a PIL Image. scale: pixels per point (e.g. 2 = ~144 DPI)."""
    images = []
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            images.append(pil_image)
    finally:
        pdf.close()
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Sample PDFs, render pages to images; output named like PDF + page number."
    )
    parser.add_argument(
        "--num-pdfs",
        type=int,
        default=1100,
        metavar="N",
        help="Number of PDFs to convert to images.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.5,
        metavar="S",
        help="PDF render scale in pixels per point (default 2.0 ≈ 144 DPI).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for reproducible PDF sampling (default 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/21Jhd",
        help="Directory to search for PDFs (default: project data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "finetuning" / "output",
        help="Directory to write rendered images (default: finetuning/output/).",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(args.data_dir.rglob("*.pdf"))
    if not pdfs:
        print("No PDFs found under data-dir.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    n = min(args.num_pdfs, len(pdfs))
    chosen = random.sample(pdfs, n)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in chosen:
        stem = pdf_path.stem
        images = pdf_to_images(pdf_path, scale=args.scale_factor)
        for page_idx, pil_image in enumerate(images):
            # Name: like the PDF but with page number (1-based for readability)
            page_num = page_idx + 1
            out_name = f"{stem}_page_{page_num:04d}.png"
            out_path = args.output_dir / out_name
            pil_image.save(out_path)
        print(f"Saved {len(images)} images from {pdf_path.name} -> {args.output_dir}/ ({stem}_page_*.png)")

    print(f"Done: {n} PDFs, images in {args.output_dir}")


if __name__ == "__main__":
    main()
