"""
Render PDF pages to PNG images. Single script for both data prep and inference.

- If --num-pages is set: add pages until output-dir has that many PNGs (incremental; --seed for PDF order).
- If --num-pages is not set: convert all PDFs in pdf-path (incremental: skip PDFs that already have any image).

Output filenames: {pdf_stem}_page_0001.png, _page_0002.png, ...

Run from project root:
  python scripts/pdf_to_images.py
  python scripts/pdf_to_images.py --pdf-path data/21Jhd/single_doc.pdf --output-dir inference/21jhd/images
  python scripts/pdf_to_images.py --pdf-dir data/21Jhd --output-dir inference/21jhd/images --workers 8
  python scripts/pdf_to_images.py --num-pages 10000 --output-dir finetuning/output --seed 42
"""
import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def _pdf_stem_from_png_path(png_path: Path) -> str:
    """From output filename like mydoc_page_0001.png return PDF stem 'mydoc'."""
    s = png_path.stem
    if "_page_" in s:
        return s.rsplit("_page_", 1)[0]
    return s


def pdf_to_images(pdf_path: Path, scale: float):
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


def _convert_one_pdf(args_tuple):
    """Worker: convert one PDF to PNGs in output_dir. Top-level for ProcessPoolExecutor."""
    pdf_path_str, output_dir_str, scale = args_tuple
    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)
    stem = pdf_path.stem
    images = pdf_to_images(pdf_path, scale)
    for page_idx, pil_image in enumerate(images):
        page_num = page_idx + 1
        out_name = f"{stem}_page_{page_num:04d}.png"
        out_path = output_dir / out_name
        pil_image.save(out_path)
    return stem, len(images)


def main():
    parser = argparse.ArgumentParser(
        description="Render PDF pages to PNGs. Full folder or single file; use --num-pages to cap (e.g. for data prep).",
    )
    parser.add_argument(
        "--pdf-path", "--pdf-dir",
        dest="pdf_path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "21Jhd",
        help="Directory to search for PDFs, or path to a single PDF (default: data/21Jhd).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "inference" / "21jhd" / "images",
        help="Directory to write PNGs (default: inference/21jhd/images).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.5,
        metavar="S",
        help="PDF render scale in pixels per point (default: 1.5).",
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=None,
        metavar="N",
        help="If set, add pages until output-dir has N PNGs (incremental). If not set, convert all PDFs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for PDF order when using --num-pages (default: 42).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of parallel workers when converting full folder (default: 8).",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"PDF path not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_pngs = list(args.output_dir.glob("*.png"))
    existing_count = len(existing_pngs)
    stems_done = {_pdf_stem_from_png_path(p) for p in existing_pngs}

    # Handle single file vs directory
    if args.pdf_path.is_file():
        if args.pdf_path.suffix.lower() != ".pdf":
            print(f"Provided file is not a PDF: {args.pdf_path}", file=sys.stderr)
            sys.exit(1)
        pdfs_all = [args.pdf_path]
    else:
        pdfs_all = sorted(args.pdf_path.rglob("*.pdf"))

    if not pdfs_all:
        print("No PDFs found at the specified path.", file=sys.stderr)
        sys.exit(1)

    seen_stems = set()
    pdfs_unique = []
    for p in pdfs_all:
        if p.stem in seen_stems:
            continue
        seen_stems.add(p.stem)
        pdfs_unique.append(p)

    pdfs_todo = [p for p in pdfs_unique if p.stem not in stems_done]
    if not pdfs_todo:
        if args.num_pages is not None and existing_count >= args.num_pages:
            print(f"Already have {existing_count} page images (>= {args.num_pages}). Nothing to do.")
        else:
            print(f"All {len(pdfs_unique)} PDFs already have images in {args.output_dir}. Nothing to do.")
        return

    if args.num_pages is not None:
        # Cap mode: add until we have num_pages total; process PDFs sequentially (seed order)
        pages_needed = max(0, args.num_pages - existing_count)
        if pages_needed == 0:
            print(f"Already have {existing_count} page images (>= {args.num_pages}). Nothing to do.")
            return
        random.seed(args.seed)
        random.shuffle(pdfs_todo)
        added_pages = 0
        converted_pdfs = 0
        for pdf_path in pdfs_todo:
            if added_pages >= pages_needed:
                break
            stem = pdf_path.stem
            images = pdf_to_images(pdf_path, args.scale_factor)
            for page_idx, pil_image in enumerate(images):
                page_num = page_idx + 1
                out_name = f"{stem}_page_{page_num:04d}.png"
                out_path = args.output_dir / out_name
                pil_image.save(out_path)
                added_pages += 1
            converted_pdfs += 1
        total_now = existing_count + added_pages
        print(f"Done: added {added_pages} pages from {converted_pdfs} PDFs. Total in {args.output_dir}: {total_now} PNGs.")
    else:
        # Full folder/file: convert all todo PDFs in parallel
        workers = max(1, args.workers)
        print(f"Converting {len(pdfs_todo)} PDFs (workers={workers}) -> {args.output_dir}")
        task_tuples = [
            (str(p.resolve()), str(args.output_dir.resolve()), args.scale_factor)
            for p in pdfs_todo
        ]
        total_added = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_convert_one_pdf, t): t for t in task_tuples}
            with tqdm(total=len(pdfs_todo), desc="PDFs", unit="pdf") as pbar:
                for future in as_completed(futures):
                    try:
                        stem, num_pages = future.result()
                        total_added += num_pages
                    except Exception as e:
                        t = futures[future]
                        pdf_name = Path(t[0]).name
                        print(f"\nError on {pdf_name}: {e}", file=sys.stderr)
                    pbar.update(1)
        total_now = existing_count + total_added
        print(f"Done: added {total_added} pages from {len(pdfs_todo)} PDFs. Total in {args.output_dir}: {total_now} PNGs.")


if __name__ == "__main__":
    main()