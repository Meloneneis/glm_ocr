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
import re
import sys
from pathlib import Path

# Use your GCP project ID (from the console). Override with env GOOGLE_CLOUD_PROJECT if needed.
DEFAULT_PROJECT_ID = "project-7f39f9fe-005d-48f0-bf1"
DEFAULT_LOCATION = "us-central1"

OCR_PROMPT = """Extract all text from this document image and output it according to these rules. Output only the extracted text, no explanation.

1) Page numbers: Remove centered page numbers like "- 2 -" or "- 3 -" from each page. Do not include them in the output.

2) Randnummern (left-margin paragraph numbers): Numbers in the left margin that label a paragraph (e.g. "13", "41") are Randnummern. Never output a Randnummer on its own line. Always put it at the beginning of the same line as the paragraph it refers to, with a space after it (e.g. "13 1. Die Kostenentscheidung beruht auf ..." as one line, not "13" on one line and "1. Die Kostenentscheidung..." on the next).

3) Paragraphs: Write exactly one line per paragraph. Do not insert line breaks within a paragraph; each paragraph is a single output line. If there is a Randnummer for a paragraph, it is the first token on that line (see rule 2).

4) Judge names: When judge names appear in a staggered or multi-line layout (e.g. several names on one row, more names indented on the next row), output all of them as a single line, separated by spaces (e.g. "Meier-Beck Schmidt-Räntsch Kirchhoff Roloff Tolkmitt").

5) De-hyphenation: If a word was split across two lines with a hyphen (e.g. "docu-" on one line and "ment" on the next), join it into one word and remove the hyphen (e.g. "document"). Do this for all line-break hyphenations.

6) Separate text blocks (right-side only): Only treat as separate blocks content that is clearly in a different column or area—e.g. marginalia or a second column on the right side of the page. Left-margin numbers (Randnummern) are part of the main body and must stay on the same line as their paragraph (rule 2). Judge names in a staggered layout are one block—output as one line (rule 4). For other right-side blocks: do not merge text that belongs to different blocks; output each block separately, each line within a block on its own line.

7) Any handwritten notes, or numbers without context that are clearly not part of the document can be safely removed. However do not remove legit Metadata such as the Aktenzeichen or ECLI.

8) Blank or empty pages: If the page contains no meaningful text after applying all rules above (e.g. only a page number, only whitespace, or a completely blank page), output nothing — produce an empty response. Do NOT output the page number, do NOT write phrases like "blank page" or "no text found".

9) Strict output format: Your entire output must consist exclusively of extracted document text. Never output JSON, code blocks, markdown formatting, commentary, or meta-text. If the image is corrupted or unreadable to the point where you cannot extract any text at all, output exactly the single word ERROR and nothing else."""

# Dedicated prompt for older court decisions (1950s/1960s): simpler and self-contained.
OCR_PROMPT_OLD = """Extract all text from this document image. Output only the extracted text, no explanation.

Layout:
- One output line per paragraph. Do not split a paragraph across multiple lines.
- The block that lists judges and roles (e.g. Senatspräsident Richter, Bundesrichter Dr. Peetz, ... als Urkundsbeamter) is one paragraph: output it as a single line with names and roles separated by spaces or commas, not each person on a new line.
- If a margin number (Randnummer) labels a paragraph, put it at the start of that paragraph line, then the text (e.g. "13 1. Die Kostenentscheidung...").

Cleanup:
- Omit centered page numbers like "- 2 -".
- Omit handwritten numbers or notes in corners (e.g. "35").
- Join hyphenated line-breaks into one word (e.g. "docu-" + "ment" → "document").

Redactions:
- Where solid black rectangles cover text, output [redacted] (exactly that, lowercase in brackets). One [redacted] per black rectangle, in place. Example: "Karl [redacted] aus H [redacted] geboren dort am [redacted] 1902".

Unclear print:
- If a letter is hard to read, choose the spelling that makes sense in German. Do not invent text.

Blank or empty pages:
- If the page contains no meaningful text after applying all rules above (e.g. only a page number, only whitespace, or a completely blank page), output nothing — produce an empty response. Do NOT output the page number, do NOT write phrases like "blank page" or "no text found".

Strict output format:
- Your entire output must consist exclusively of extracted document text. Never output JSON, code blocks, markdown formatting, commentary, or meta-text. If the image is corrupted or unreadable to the point where you cannot extract any text at all, output exactly the single word ERROR and nothing else."""


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
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Vertex AI Gemini model (default: gemini-2.5-pro for better German/OCR).",
    )
    parser.add_argument(
        "--write-labels",
        type=Path,
        default=None,
        metavar="DIR",
        help="After OCR, write the result to DIR/labels.json under the image filename. Use e.g. finetuning/output.",
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
            model=args.model,
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
    # Normalize redaction placeholder to [redacted] (model sometimes outputs [REDACTED])
    text = re.sub(r"\[REDACTED\]", "[redacted]", text, flags=re.IGNORECASE)
    # Show line numbers so you can tell real line breaks from output formatting
    for i, line in enumerate(text.splitlines(), start=1):
        print(f"{i:4d}  {line}")
    print("--- done ---")

    if args.write_labels is not None:
        import json
        labels_path = args.write_labels / "labels.json"
        labels = {}
        if labels_path.is_file():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
        name = args.image.name
        labels[name] = text if text != "(no text)" else ""
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Updated {labels_path} with label for {name}")


if __name__ == "__main__":
    main()
