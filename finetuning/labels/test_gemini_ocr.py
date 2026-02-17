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
OCR_PROMPT = "Extract all text from this document image exactly as it appears. Preserve layout and line breaks. Output only the extracted text, no explanation."


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
    print(response.text or "(no text)")
    print("--- done ---")


if __name__ == "__main__":
    main()
