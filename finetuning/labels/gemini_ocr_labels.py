"""
Generate labels for OCR images using the Gemini API.

Reads images from finetuning/output (or a list file like train.txt), sends each
image + a configurable prompt to Gemini, and saves the model output as the label.
Uses GEMINI_API_KEY from the environment (do not hardcode the key).

Run from project root:
  export GEMINI_API_KEY=your_key
  python finetuning/labels/gemini_ocr_labels.py --list train.txt
  python finetuning/labels/gemini_ocr_labels.py --list train.txt --list test.txt --output labels.json
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

DEFAULT_PROMPT = "Extract all text from this document image exactly as it appears. Preserve layout and line breaks. Output only the extracted text, no explanation."


def get_gemini_client():
    try:
        import google.generativeai as genai
    except ImportError:
        print("Install the Gemini SDK: pip install google-generativeai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def ocr_image(model, image_path: Path, prompt: str) -> str:
    """Send image and prompt to Gemini; return the model's text response."""
    data = image_path.read_bytes()
    image_part = {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(data).decode("utf-8"),
        }
    }
    response = model.generate_content([image_part, prompt])
    if not response.text:
        return ""
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate OCR labels using Gemini (set GEMINI_API_KEY)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "finetuning" / "output",
        help="Directory containing PNG images (default: finetuning/output/).",
    )
    parser.add_argument(
        "--list",
        type=Path,
        action="append",
        metavar="FILE",
        help="Text file with one image filename per line (e.g. train.txt). Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for labels (default: output-dir/labels.json).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="OCR prompt sent to Gemini (default: extract all text).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N images (for testing).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (default 0.5).",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Output directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect image filenames from list file(s) or all PNGs
    if args.list:
        filenames = []
        for list_path in args.list:
            if not list_path.is_file():
                # Try relative to output_dir
                list_path = args.output_dir / list_path.name
            if not list_path.is_file():
                print(f"List file not found: {list_path}", file=sys.stderr)
                sys.exit(1)
            filenames.extend([line.strip() for line in list_path.read_text().splitlines() if line.strip()])
        filenames = list(dict.fromkeys(filenames))  # keep order, remove dupes
    else:
        filenames = sorted(f.name for f in args.output_dir.glob("*.png"))

    if not filenames:
        print("No images to process.", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        filenames = filenames[: args.limit]

    model = get_gemini_client()
    labels = {}

    for i, name in enumerate(filenames):
        path = args.output_dir / name
        if not path.is_file():
            print(f"Skip (not found): {name}", file=sys.stderr)
            continue
        try:
            text = ocr_image(model, path, args.prompt)
            labels[name] = text
            print(f"[{i+1}/{len(filenames)}] {name} -> {len(text)} chars")
        except Exception as e:
            print(f"Error for {name}: {e}", file=sys.stderr)
            labels[name] = ""
        if args.delay and i < len(filenames) - 1:
            time.sleep(args.delay)

    out_path = args.output or (args.output_dir / "labels.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(labels)} labels to {out_path}")


if __name__ == "__main__":
    main()
