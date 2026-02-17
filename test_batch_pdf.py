"""
Run GLM-OCR on a PDF using batch inference: convert to images, process pages in
batches for better GPU utilization.

Refs: https://huggingface.co/docs/transformers/model_doc/glm_ocr (Batch inference)
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
PROMPT = "Text Recognition:"


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


def _clean_decoded(text: str) -> str:
    if PROMPT in text:
        text = text.split(PROMPT)[-1]
    return text.replace("<|image|>", "").strip() or ""


def run_ocr_batch(processor, model, images: list, max_new_tokens: int = 1024, return_batch_tokens: bool = False):
    """Run GLM-OCR on a batch of PIL Images; return list of decoded texts (and optionally total new tokens in batch)."""
    if not images:
        return ([], 0) if return_batch_tokens else []
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        for img in images
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    n_new_per_seq = output.shape[1] - input_len
    total_new_tokens = len(images) * n_new_per_seq
    decoded = processor.batch_decode(output, skip_special_tokens=True)
    texts = [_clean_decoded(s) for s in decoded]
    if return_batch_tokens:
        return (texts, total_new_tokens)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR on a PDF with batch inference.")
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=None,
        help="Path to PDF (default: first PDF under data/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        metavar="N",
        help="Number of pages per batch (default 2)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="PDF render scale (default 2.0 ≈ 144 DPI)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        metavar="N",
        help="Max new tokens per page (default 1024)",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention 2",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Print generated token count per batch and total (avg per page)",
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

    batch_size = max(1, args.batch_size)

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
    n_batches = (n_pages + batch_size - 1) // batch_size

    print(f"OCR in batches (batch_size={batch_size}, {n_batches} batches for {n_pages} pages)...")
    page_texts: list[str] = []
    batch_times: list[float] = []
    batch_tokens: list[int] = []  # total new tokens per batch (when --show-tokens)

    for start in tqdm(range(0, n_pages, batch_size), desc="OCR batches", unit="batch"):
        batch = images[start : start + batch_size]
        t0 = time.perf_counter()
        if args.show_tokens:
            texts, tok = run_ocr_batch(
                processor, model, batch, max_new_tokens=args.max_tokens, return_batch_tokens=True
            )
            batch_tokens.append(tok)
        else:
            texts = run_ocr_batch(processor, model, batch, max_new_tokens=args.max_tokens)
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)
        page_texts.extend(texts)

    total_time = sum(batch_times)

    # --- Timing summary (considering batch size) ---
    print("\n" + "=" * 60)
    print("TIMING (batch inference)")
    print("=" * 60)
    print(f"  Pages: {n_pages}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches: {n_batches}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg seconds per batch: {total_time / n_batches:.2f}s")
    # Avg seconds per page: total time / number of pages (batch size is already considered)
    avg_sec_per_page = total_time / n_pages
    print(f"  Avg seconds per page: {avg_sec_per_page:.2f}s  (total_time / n_pages)")
    print(f"  Pages per second: {n_pages / total_time:.2f}")
    if args.show_tokens and batch_tokens:
        for i, tok in enumerate(batch_tokens, start=1):
            print(f"  Batch {i}: {batch_times[i-1]:.2f}s  ({tok} tok)")
        total_tok = sum(batch_tokens)
        print(f"  Total generated tokens: {total_tok}  (avg {total_tok / n_pages:.0f}/page)")
    print()

    print("=" * 60)
    print("TEXT (per page)")
    print("=" * 60)
    for i, text in enumerate(page_texts, start=1):
        print(f"\n--- Page {i} ---")
        print(text)
    print()


if __name__ == "__main__":
    main()
