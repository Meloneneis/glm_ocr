"""
Test Gemini OCR with one image using your Google Cloud project (no API key).

Uses Vertex AI Gemini + Application Default Credentials. Run once:
  gcloud auth application-default login

Then run (from project root):
  python finetuning/labels/test_gemini_ocr.py --image path/to/image.png
  python finetuning/labels/test_gemini_ocr.py --image finetuning/output/$(head -1 finetuning/output/train.txt)
"""
import argparse
import os
import sys
from pathlib import Path

# Use your GCP project ID (from the console). Override with env GOOGLE_CLOUD_PROJECT if needed.
DEFAULT_PROJECT_ID = "project-7f39f9fe-005d-48f0-bf1"
DEFAULT_LOCATION = "us-central1"

OCR_PROMPT = """Extract all text from this document image and output it according to these rules. Output only the extracted text, no explanation.

1) Page numbers: Remove centered page numbers like "- 2 -" or "- 3 -" from each page. Do not include them in the output.

2) Left margin numbers: Keep all numbers in the left margin (e.g. line numbers, paragraph numbers) as they appear. Do not remove them.

3) Paragraphs: Write exactly one line per paragraph. Do not insert line breaks within a paragraph; each paragraph is a single output line.

4) De-hyphenation: If a word was split across two lines with a hyphen (e.g. "docu-" on one line and "ment" on the next), join it into one word and remove the hyphen (e.g. "document"). Do this for all line-break hyphenations.

5) Separate text blocks: On the first page (and anywhere else) there are often text blocks on the right side of the page (e.g. marginalia, side notes, or a second column). Treat these as separate text blocks from the main left-side body. Do not merge text that appears on the same visual line but belongs to different blocks (left vs right). Output each block separately: each line within a block on its own line. So if one visual line has left-side text and right-side text, output the left-side line and the right-side line as two distinct lines, in logical order (e.g. main block first, then right-side block), rather than concatenating them into one line."""


def main():
    parser = argparse.ArgumentParser(description="Test Gemini OCR on one image (Vertex AI, ADC).")
    parser.add_argument("--image", type=Path, required=True, help="Path to a PNG or JPEG image.")
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT_ID),
        help="GCP project ID (default from env or built-in).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
        help="Vertex AI location (default us-central1).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=OCR_PROMPT,
        help="OCR prompt to send to Gemini.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Ensure Vertex AI env vars for google-genai
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", args.project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", args.location)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

    try:
        from google import genai
        from google.genai.types import HttpOptions, Part
    except ImportError:
        print("Install the SDK: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    # ADC required for Vertex AI (no API key). Run once: gcloud auth application-default login
    image_bytes = args.image.read_bytes()
    mime = "image/png" if args.image.suffix.lower() == ".png" else "image/jpeg"

    print("Calling Vertex AI Gemini (ADC)...")
    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Part.from_bytes(data=image_bytes, mime_type=mime),
                args.prompt,
            ],
        )
    except Exception as e:
        try:
            from google.auth.exceptions import DefaultCredentialsError
            if isinstance(e, DefaultCredentialsError):
                print("Credentials not set. Run once: gcloud auth application-default login", file=sys.stderr)
        except ImportError:
            pass
        raise

    print("--- OCR result ---")
    text = response.text or "(no text)"
    # Show line numbers so you can tell real line breaks from output formatting
    for i, line in enumerate(text.splitlines(), start=1):
        print(f"{i:4d}  {line}")
    print("--- done ---")


if __name__ == "__main__":
    main()
