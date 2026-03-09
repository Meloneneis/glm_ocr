"""
Label images using Vertex AI Gemini (ADC). Optional parallel workers.
On error: all workers pause for RETRY_PAUSE_SECONDS, then retry. No image
is ever skipped.

Normal mode:  process images not yet present in labels.json.
Repair mode (--repair):  re-process entries whose value is an empty string
  (one attempt each; accepts empty results as valid for blank pages).

Run from project root:
  conda activate glm_ocr
  python finetuning/labels/label_all_vertex.py
  python finetuning/labels/label_all_vertex.py --workers 5
  python finetuning/labels/label_all_vertex.py --repair
"""
import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output_20Jhd"

# Same prompts as test_gemini_ocr
from test_gemini_ocr import OCR_PROMPT, OCR_PROMPT_OLD

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "onyx-zodiac-349715"))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

RETRY_PAUSE_SECONDS = 60
CHECKPOINT_EVERY = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _process_one(name, output_dir, client, prompt, model, delay_sec, shared_state, accept_empty=False):
    """Run OCR on one image. Retries indefinitely on any error or (in normal
    mode) on empty responses.  When *accept_empty* is True the first
    successful response is returned even if it is empty."""
    from google.genai.types import Part

    path = output_dir / name
    if not path.is_file():
        log.warning("File not found, skipping: %s", name)
        return (name, None)
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    while True:
        with shared_state["lock"]:
            until = shared_state["pause_until"]
        if until > time.time():
            time.sleep(until - time.time())
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    Part.from_bytes(data=image_bytes, mime_type=mime),
                    prompt,
                ],
            )
            text = (response.text or "").strip()
            text = re.sub(r"\[REDACTED\]", "[redacted]", text, flags=re.IGNORECASE)
            if text or accept_empty:
                if delay_sec > 0:
                    time.sleep(delay_sec)
                return (name, text)
            log.warning("Empty response for %s — pausing %ds and retrying",
                        name, RETRY_PAUSE_SECONDS)
            with shared_state["lock"]:
                shared_state["pause_until"] = time.time() + RETRY_PAUSE_SECONDS
            time.sleep(RETRY_PAUSE_SECONDS)
        except Exception as exc:
            log.error("Error processing %s: %s — pausing %ds and retrying",
                      name, exc, RETRY_PAUSE_SECONDS)
            with shared_state["lock"]:
                shared_state["pause_until"] = time.time() + RETRY_PAUSE_SECONDS
            time.sleep(RETRY_PAUSE_SECONDS)


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
        "--all-pngs",
        action="store_true",
        help="Use all PNGs in output-dir (ignore train.txt/test.txt).",
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
        default=1,
        metavar="SEC",
        help="Seconds to pause between each API call (default 0.5). Use 0 for no delay.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        metavar="N",
        help="Number of parallel workers (default 64).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Vertex AI Gemini model (default: gemini-2.5-pro for better German/OCR).",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-process only entries with empty strings in labels.json (one attempt per entry, accepts empty results as valid for blank pages).",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Output directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or (args.output_dir / "labels.json")
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

    output_dir_str = str(args.output_dir)
    prompt_for_all = OCR_PROMPT_OLD if "20Jhd" in output_dir_str else OCR_PROMPT
    model = args.model

    if args.repair:
        # Scan labels.json directly for empty-string values — no dependency on filenames.
        todo = sorted(k for k, v in labels.items() if (v or "").strip() == "")
        accept_empty = True
        if not todo:
            log.info("No empty entries found in labels.json — nothing to repair.")
            return
        log.info("Repair mode: re-processing %d entries with empty labels:", len(todo))
        for name in todo:
            full_path = (args.output_dir / name).resolve()
            log.info("  will repair: %s  ->  file://%s", name, full_path)
    else:
        # Build image list: --all-pngs, or --list files, or train/test.txt, or all PNGs
        if args.all_pngs:
            filenames = sorted(f.name for f in args.output_dir.glob("*.png"))
        elif args.list:
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

        todo = [n for n in filenames if n not in labels]
        accept_empty = False
        if not todo:
            log.info("All %d images already labeled.", len(labels))
            return
        log.info("To label: %d new images.", len(todo))

    shared_state = {"pause_until": 0.0, "lock": threading.Lock()}

    def _checkpoint():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.workers <= 1:
        done = 0
        for name in tqdm(todo, desc="Labeling", unit="img"):
            _, text = _process_one(name, args.output_dir, client, prompt_for_all, model, args.delay, shared_state, accept_empty=accept_empty)
            if text is not None:
                labels[name] = text
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                _checkpoint()
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_one, name, args.output_dir, client, prompt_for_all, model, args.delay, shared_state, accept_empty=accept_empty
                ): name
                for name in todo
            }
            with tqdm(total=len(todo), desc="Labeling", unit="img") as pbar:
                for future in as_completed(futures):
                    name, text = future.result()
                    if text is not None:
                        labels[name] = text
                    completed += 1
                    pbar.update(1)
                    if completed % CHECKPOINT_EVERY == 0:
                        _checkpoint()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %d labels to %s", len(labels), out_path)


if __name__ == "__main__":
    main()
