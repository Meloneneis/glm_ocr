"""
Run GLM-OCR on a single PDF: convert to images, OCR each page with timing, output text.

Note: GLM-OCR does not accept PDF natively. The model (HF Transformers and vLLM) only
consumes images (pixel_values). The official SDK (glmocr.parse) accepts PDF paths but
converts them to images internally via pypdfium2. So we must convert PDF→images first;
this script does that explicitly with pypdfium2, then runs the HF model on each page.

Refs: https://huggingface.co/docs/transformers/model_doc/glm_ocr
      https://github.com/zai-org/GLM-OCR (SDK PageLoader uses pdf_to_images_pil)
"""
import argparse
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
import torch
from tqdm import tqdm
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

model_id = "zai-org/GLM-OCR"


def pdf_to_images(pdf_path: Path, scale: float = 2.0) -> list:
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


def run_ocr_on_image(processor, model, image, prompt: str = "Text Recognition:", max_new_tokens: int = 512, return_n_tokens: bool = False):
    """Run GLM-OCR on a single PIL Image; return decoded text (and optionally num generated tokens)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    n_new_tokens = output.shape[1] - input_length
    text = processor.decode(output[0], skip_special_tokens=True)
    if prompt in text:
        text = text.split(prompt)[-1]
    text = text.replace("<|image|>", "").strip()
    if return_n_tokens:
        return (text or "", n_new_tokens)
    return text or ""


def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR on a single PDF.")
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=None,
        help="Path to PDF (default: first PDF under data/)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="PDF render scale (default 2.0 ≈ 144 DPI)",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Print generated token count per page (to verify time vs. output length)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        metavar="N",
        help="Max new tokens per page (default 1024; increase if dense pages get cut off)",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention 2 (use default attention; try this if flash-attn fails)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    if args.pdf is None:
        pdfs = list(data_dir.rglob("*.pdf"))
        if not pdfs:
            print("No PDF found under data/. Pass a path to a PDF.", file=sys.stderr)
            sys.exit(1)
        pdf_path = pdfs[0]
        print(f"No PDF given; using first under data/: {pdf_path}")
    else:
        pdf_path = args.pdf
    if not pdf_path.is_file():
        print(f"Not a file: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "kernels-community/flash-attn2"
        print("Using Flash Attention 2 (kernels-community/flash-attn2).")
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        if not args.no_flash_attn and "flash" in str(e).lower():
            print(f"Flash Attention load failed: {e}", file=sys.stderr)
            print("Retry with --no-flash-attn to use default attention.", file=sys.stderr)
            sys.exit(1)
        raise

    print(f"Converting PDF to images (scale={args.scale}): {pdf_path}")
    images = pdf_to_images(pdf_path, scale=args.scale)
    n_pages = len(images)

    page_times: list[float] = []
    page_texts: list[str] = []

    max_tokens = args.max_tokens
    for i, image in enumerate(tqdm(images, desc="OCR pages", unit="page")):
        t0 = time.perf_counter()
        if args.show_tokens:
            text, n_tokens = run_ocr_on_image(processor, model, image, max_new_tokens=max_tokens, return_n_tokens=True)
            page_texts.append((text, n_tokens))
        else:
            text = run_ocr_on_image(processor, model, image, max_new_tokens=max_tokens)
            page_texts.append((text, None))
        elapsed = time.perf_counter() - t0
        page_times.append(elapsed)

    # --- Output: times and text ---
    print("\n" + "=" * 60)
    print("TIMING (seconds per page)")
    print("=" * 60)
    for i, t in enumerate(page_times, start=1):
        tok_info = f"  ({page_texts[i-1][1]} tok)" if page_texts[i-1][1] is not None else ""
        print(f"  Page {i}: {t:.2f}s{tok_info}")
    total_time = sum(page_times)
    print(f"  Total: {total_time:.2f}s  (avg {total_time / n_pages:.2f}s/page)")
    if args.show_tokens and any(p[1] is not None for p in page_texts):
        total_tok = sum(p[1] for p in page_texts if p[1] is not None)
        print(f"  Total generated tokens: {total_tok}  (avg {total_tok / n_pages:.0f}/page)")
    print()

    print("=" * 60)
    print("TEXT (per page)")
    print("=" * 60)
    for i, (text, _) in enumerate(page_texts, start=1):
        print(f"\n--- Page {i} ---")
        print(text)
    print()


if __name__ == "__main__":
    main()
