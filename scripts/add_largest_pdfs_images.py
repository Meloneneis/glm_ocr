"""
Add images from the N largest PDFs (by file size) to an output folder.
Use case: enrich 20Jhd training data with noisier scans (larger PDFs = noisier).

- Selects top N PDFs by size from --data-dir (flat or recursive).
- Renders all pages to PNG and writes to --output-dir (same naming as pdf_to_images.py).
- If output-dir exists: appends; existing same-named PNGs are overwritten.
- Asks for confirmation before converting (shows N PDFs and total image count X).

Run from project root:
  python scripts/add_largest_pdfs_images.py --data-dir data/20Jhd --output-dir finetuning/output_20Jhd --top-n 50
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.pdf_to_images import pdf_to_images


def get_page_count(pdf_path: Path) -> int:
    """Return number of pages in PDF without rendering."""
    doc = pdfium.PdfDocument(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def _convert_one_pdf(args_tuple):
    """Worker: convert one PDF to PNGs in output_dir. For ProcessPoolExecutor."""
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
        description="Add images from the N largest PDFs (by file size) to an output folder. Asks for confirmation.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "20Jhd",
        help="Directory to search for PDFs (default: data/20Jhd).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "finetuning" / "output_20Jhd",
        help="Directory to write PNGs; created if missing; existing same-named files overwritten (default: finetuning/output_20Jhd).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        metavar="N",
        help="Number of largest PDFs by file size to convert (default: 50).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=2,
        metavar="S",
        help="PDF render scale in pixels per point (default: 1.5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers for conversion (default: 4).",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    pdfs_all = sorted(args.data_dir.rglob("*.pdf"))
    if not pdfs_all:
        print("No PDFs found under data-dir.", file=sys.stderr)
        sys.exit(1)

    # Dedupe by stem (keep path with largest size per stem); then sort by size desc, take top N
    by_stem = {}
    for p in pdfs_all:
        size = p.stat().st_size
        if p.stem not in by_stem or by_stem[p.stem][1] < size:
            by_stem[p.stem] = (p, size)
    pdfs_unique = [t[0] for t in by_stem.values()]
    pdfs_sorted = sorted(pdfs_unique, key=lambda p: p.stat().st_size, reverse=True)
    top_pdfs = pdfs_sorted[: args.top_n]

    if len(top_pdfs) < args.top_n:
        print(f"Only {len(top_pdfs)} PDFs available; using all of them (requested --top-n {args.top_n}).")

    # Compute total page count for confirmation
    total_pages = 0
    for pdf_path in top_pdfs:
        try:
            total_pages += get_page_count(pdf_path)
        except Exception as e:
            print(f"Warning: could not get page count for {pdf_path.name}: {e}", file=sys.stderr)

    print(f"Converting the {len(top_pdfs)} PDFs to {total_pages} images.")
    response = input("Do you wish to continue? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("Aborted.")
        sys.exit(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    task_tuples = [
        (str(p.resolve()), str(args.output_dir.resolve()), args.scale_factor)
        for p in top_pdfs
    ]
    total_added = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_convert_one_pdf, t): t for t in task_tuples}
        with tqdm(total=len(top_pdfs), desc="PDFs", unit="pdf") as pbar:
            for future in as_completed(futures):
                try:
                    stem, num_pages = future.result()
                    total_added += num_pages
                except Exception as e:
                    t = futures[future]
                    print(f"\nError on {Path(t[0]).name}: {e}", file=sys.stderr)
                pbar.update(1)

    print(f"Done: wrote {total_added} images from {len(top_pdfs)} PDFs to {args.output_dir}.")


if __name__ == "__main__":
    main()
