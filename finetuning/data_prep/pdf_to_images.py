"""
Data prep: render PDF pages to images up to a target page count. Incremental: only
converts additional PDFs until total PNG count in output-dir reaches --num-pages.
PDFs that already have any image in output-dir are skipped so existing images are
never re-generated (deterministic order via --seed).

Output images: {pdf_stem}_page_0001.png, _page_0002.png, ...

Run from project root:
  python finetuning/data_prep/pdf_to_images.py --num-pages 10000 --seed 42
  # With 5k PNGs already present, converts only enough new PDFs to add 5k more pages.
"""
import argparse
import random
import sys
from pathlib import Path

import pypdfium2 as pdfium

# Project root (parent of finetuning/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent


def _pdf_stem_from_png_path(png_path: Path) -> str:
    """From output filename like mydoc_page_0001.png return PDF stem 'mydoc'."""
    s = png_path.stem  # no extension
    if "_page_" in s:
        return s.rsplit("_page_", 1)[0]
    return s


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
        description="Render PDF pages to images until output-dir has --num-pages PNGs (incremental)."
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=11000,
        metavar="N",
        help="Target total number of page images in output-dir (default 10000).",
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
        help="Random seed for deterministic PDF order (default 42).",
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
        help="Directory to write rendered images (default: finetuning/output_20Jhd).",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing PNGs and which PDF stems already have images (do not re-convert those)
    existing_pngs = list(args.output_dir.glob("*.png"))
    existing_count = len(existing_pngs)
    stems_done = {_pdf_stem_from_png_path(p) for p in existing_pngs}

    pdfs_all = sorted(args.data_dir.rglob("*.pdf"))
    if not pdfs_all:
        print("No PDFs found under data-dir.", file=sys.stderr)
        sys.exit(1)

    # One PDF per stem so output filenames are unique (no overwrites, no duplicate pages)
    seen_stems = set()
    pdfs_unique = []
    for p in pdfs_all:
        if p.stem in seen_stems:
            continue
        seen_stems.add(p.stem)
        pdfs_unique.append(p)

    pdfs_todo = [p for p in pdfs_unique if p.stem not in stems_done]
    random.seed(args.seed)
    random.shuffle(pdfs_todo)

    pages_needed = max(0, args.num_pages - existing_count)
    if pages_needed == 0:
        print(f"Already have {existing_count} page images (>= {args.num_pages}). Nothing to do.")
        return

    print(f"Existing: {existing_count} PNGs. Target: {args.num_pages} pages. Will add up to {pages_needed} more.")

    added_pages = 0
    converted_pdfs = 0
    for pdf_path in pdfs_todo:
        if added_pages >= pages_needed:
            break
        stem = pdf_path.stem
        images = pdf_to_images(pdf_path, scale=args.scale_factor)
        for page_idx, pil_image in enumerate(images):
            page_num = page_idx + 1
            out_name = f"{stem}_page_{page_num:04d}.png"
            out_path = args.output_dir / out_name
            pil_image.save(out_path)
            added_pages += 1
        converted_pdfs += 1
        print(f"Saved {len(images)} images from {pdf_path.name} -> {args.output_dir}/ ({stem}_page_*.png)")

    total_now = existing_count + added_pages
    print(f"Done: added {added_pages} pages from {converted_pdfs} PDFs. Total in {args.output_dir}: {total_now} PNGs.")


if __name__ == "__main__":
    main()
