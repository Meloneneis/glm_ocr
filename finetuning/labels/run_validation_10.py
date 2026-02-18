"""
Run OCR on 10 images and save results to validation/ for manual check.
Uses same Vertex AI Gemini setup as test_gemini_ocr.py.
Run from project root: python finetuning/labels/run_validation_10.py
"""
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output"
VALIDATION_DIR = _SCRIPT_DIR / "validation"

# Import prompt and run logic from test script
sys.path.insert(0, str(_SCRIPT_DIR))
from test_gemini_ocr import OCR_PROMPT

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "project-7f39f9fe-005d-48f0-bf1"))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


def main():
    from google import genai
    from google.genai.types import HttpOptions, Part

    pngs = sorted(OUTPUT_DIR.glob("*.png"))
    if len(pngs) < 10:
        print(f"Need at least 10 PNGs in {OUTPUT_DIR}, found {len(pngs)}", file=sys.stderr)
        sys.exit(1)

    # Sample 10: mix from different documents (every Nth to get variety)
    step = max(1, len(pngs) // 10)
    chosen = [pngs[i] for i in range(0, min(10 * step, len(pngs)), step)][:10]

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    for i, img_path in enumerate(chosen, start=1):
        out_name = img_path.stem + ".txt"
        out_path = VALIDATION_DIR / out_name
        print(f"[{i}/10] {img_path.name} -> {out_path.name}")
        image_bytes = img_path.read_bytes()
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Part.from_bytes(data=image_bytes, mime_type=mime),
                OCR_PROMPT,
            ],
        )
        text = response.text or "(no text)"
        lines_with_nums = "\n".join(f"{j:4d}  {line}" for j, line in enumerate(text.splitlines(), start=1))
        out_path.write_text(lines_with_nums, encoding="utf-8")

    print(f"Done. Results in {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
