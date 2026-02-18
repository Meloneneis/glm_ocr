"""
Label all ~1100 images using Vertex AI Gemini (ADC). Optional parallel workers.
On error: pause 1 minute and retry, up to 5 attempts per image.
Saves to finetuning/output/labels.json (and checkpoint every 100 images).

If train.txt and test.txt exist in output-dir, uses those (1000 + 100). Otherwise
uses all PNGs. Already-labeled images (present in labels.json) are skipped so you
can resume after a stop.

Run from project root:
  conda activate glm_ocr
  python finetuning/labels/label_all_vertex.py
  python finetuning/labels/label_all_vertex.py --workers 5
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output"

# Same prompt as test_gemini_ocr
from test_gemini_ocr import OCR_PROMPT

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "project-7f39f9fe-005d-48f0-bf1"))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

MAX_RETRIES = 5
RETRY_PAUSE_SECONDS = 60
CHECKPOINT_EVERY = 100


def _process_one(name, output_dir, client, prompt, delay_sec):
    """Run OCR on one image; retry up to MAX_RETRIES on error. Returns (name, text)."""
    from google.genai.types import Part

    path = output_dir / name
    if not path.is_file():
        return (name, "")
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    text = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    Part.from_bytes(data=image_bytes, mime_type=mime),
                    prompt,
                ],
            )
            text = (response.text or "").strip()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_PAUSE_SECONDS)
            else:
                # Leave text empty; caller can log if desired
                pass
    if delay_sec > 0:
        time.sleep(delay_sec)
    return (name, text)


def main():
    parser = argparse.ArgumentParser(description="Label all images with Vertex AI Gemini (retry on error).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing PNGs and where labels.json is written (default: finetuning/output/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <output-dir>/labels.json).",
    )
    parser.add_argument(
        "--list",
        type=Path,
        action="append",
        metavar="FILE",
        help="Use only images listed in FILE(s), e.g. train.txt and test.txt (one filename per line).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many images (for testing).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Seconds to pause between each API call (default 0.5). Use 0 for no delay.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default 1). Use e.g. 5 to speed up.",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Output directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    # Image list: from --list files, or train.txt + test.txt if present, or all PNGs
    if args.list:
        filenames = []
        for list_path in args.list:
            p = list_path if list_path.is_file() else args.output_dir / list_path.name
            if not p.is_file():
                print(f"List file not found: {list_path}", file=sys.stderr)
                sys.exit(1)
            filenames.extend([line.strip() for line in p.read_text().splitlines() if line.strip()])
        filenames = list(dict.fromkeys(filenames))
    else:
        train_txt = args.output_dir / "train.txt"
        test_txt = args.output_dir / "test.txt"
        if train_txt.is_file() and test_txt.is_file():
            filenames = [line.strip() for line in train_txt.read_text().splitlines() if line.strip()]
            filenames += [line.strip() for line in test_txt.read_text().splitlines() if line.strip()]
            filenames = list(dict.fromkeys(filenames))
        else:
            filenames = sorted(f.name for f in args.output_dir.glob("*.png"))

    if not filenames:
        print("No images to process.", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        filenames = filenames[: args.limit]

    out_path = args.output or (args.output_dir / "labels.json")
    # Load existing labels if resuming (e.g. after crash)
    if out_path.is_file():
        try:
            labels = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            labels = {}
    else:
        labels = {}

    try:
        from google import genai
        from google.genai.types import HttpOptions, Part
        from tqdm import tqdm
    except ImportError as e:
        print(f"Missing dependency: {e}. Install: pip install google-genai tqdm", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    todo = [n for n in filenames if n not in labels]
    num_done_before = len(labels)

    def _checkpoint():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.workers <= 1:
        # Sequential
        for name in tqdm(todo, desc="Labeling", unit="img"):
            _, text = _process_one(name, args.output_dir, client, OCR_PROMPT, args.delay)
            labels[name] = text
            if (len(labels) - num_done_before) % CHECKPOINT_EVERY == 0:
                _checkpoint()
    else:
        # Parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_one, name, args.output_dir, client, OCR_PROMPT, args.delay
                ): name
                for name in todo
            }
            with tqdm(total=len(todo), desc="Labeling", unit="img") as pbar:
                for future in as_completed(futures):
                    name, text = future.result()
                    labels[name] = text
                    completed += 1
                    pbar.update(1)
                    if completed % CHECKPOINT_EVERY == 0:
                        _checkpoint()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(labels)} labels to {out_path}")


if __name__ == "__main__":
    main()
