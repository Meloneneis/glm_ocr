"""
Run finetuned GLM-OCR on all images in a directory; save results to JSON.
Supports batch inference. Keys = image path relative to images dir; value = generated text.
Resume: skips images already in output JSON; checkpoints every 100 images.

Run from project root:
  python inference/run_ocr.py
  python inference/run_ocr.py --images-dir inference/21jhd/images --batch-size 8
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

# Unsloth must be imported before transformers/peft for optimizations
from unsloth import FastVisionModel

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

PROMPT = "Text Recognition:"
CHECKPOINT_EVERY = 100
# Base model (Unsloth); adapter is loaded on top via PEFT (Hub id or local path).
BASE_MODEL_ID = "unsloth/GLM-OCR"
# Safety cap only: generation stops at EOS when the model finishes. Use same as training max_seq_length so no truncation.
DEFAULT_MAX_NEW_TOKENS = 4096


def _is_merged_model_path(path: Path) -> bool:
    """True if path is a local directory with a full model (config.json) and no PEFT adapter_config.json."""
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and not (path / "adapter_config.json").exists()
    )


def _load_model_and_processor(adapter_id_or_path: str):
    """Load model and processor. Accepts (1) merged model path (local dir with no adapter_config.json) or (2) PEFT adapter (Hub id or local path)."""
    path = Path(adapter_id_or_path).resolve()
    if _is_merged_model_path(path):
        # Full merged model: load single model and processor from that dir
        model_path = str(path)
        model, _ = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=8192,
            load_in_4bit=False,
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    else:
        # Base + PEFT adapter
        model, _ = FastVisionModel.from_pretrained(
            BASE_MODEL_ID,
            max_seq_length=8192,
        )
        model = PeftModel.from_pretrained(model, adapter_id_or_path, is_trainable=False)
        processor = AutoProcessor.from_pretrained(adapter_id_or_path, use_fast=False)
    model.to(model.device)
    FastVisionModel.for_inference(model)
    return model, processor


def _clean_decoded(text: str) -> str:
    """Drop prompt prefix and strip GLM-OCR special tokens after decode."""
    if PROMPT in text:
        text = text.split(PROMPT)[-1]
    text = text.replace("<think>", "").replace("</think>", "").replace("<|image|>", "")
    return text.strip()


def _run_ocr_batch(processor, model, image_paths: List[Path], max_new_tokens: int, raw: bool = False) -> List[str]:
    """Run OCR on a batch of images; return list of texts (same order as image_paths). If raw=True, no cleaning and skip_special_tokens=False."""
    if not image_paths:
        return []
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            images.append(None)
    valid_indices = [i for i, im in enumerate(images) if im is not None]
    if not valid_indices:
        return [""] * len(image_paths)
    valid_images = [images[i] for i in valid_indices]
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
        for img in valid_images
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(model.config, "pad_token_id", None)
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = processor.batch_decode(out, skip_special_tokens=not raw)
    if raw:
        texts = list(decoded)
    else:
        texts = [_clean_decoded(s) for s in decoded]
    result = [""] * len(image_paths)
    for idx, i in enumerate(valid_indices):
        result[i] = texts[idx]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run finetuned GLM-OCR on all images; save results to JSON (resume and checkpoint supported)."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=_PROJECT_ROOT / "inference" / "21jhd" / "images",
        help="Directory containing PNG images (default: inference/21jhd/images).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meloneneneis/glm_ocr_21jhd",
        help="HuggingFace adapter id (e.g. meloneneneis/glm_ocr_21jhd) or local path to adapter dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <images-dir>/../results.json).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Safety cap per image; generation stops at EOS so no truncation (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images per batch (default: 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many images (e.g. --limit 100 to test; useful for 100k+ image runs).",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        print(f"Images directory not found: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or (args.images_dir.parent / "results.json")
    if out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        results = {}

    image_paths = sorted(args.images_dir.rglob("*.png"))
    if not image_paths:
        print("No PNG images found in images-dir.", file=sys.stderr)
        sys.exit(1)

    def rel_key(p: Path) -> str:
        return str(p.relative_to(args.images_dir)).replace("\\", "/")

    todo = [p for p in image_paths if rel_key(p) not in results]
    if args.limit is not None:
        todo = todo[: args.limit]
    if not todo:
        print(f"All {len(image_paths)} images already in {out_path}. Nothing to do.")
        return

    if len(todo) > 100_000:
        print(f"Note: {len(todo)} images is a large run. Use --limit N to test, or lower --max-new-tokens for speed if docs are short.")

    batch_size = max(1, args.batch_size)
    print("Inference params:")
    print(f"  model:           {args.model}")
    print(f"  images_dir:      {args.images_dir}")
    print(f"  output:          {out_path}")
    print(f"  batch_size:      {batch_size}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    if args.limit is not None:
        print(f"  limit:           {args.limit}")
    print(f"  images to run:   {len(todo)}")
    print()

    print(f"Loading model and processor (model: {args.model})...")
    model, processor = _load_model_and_processor(args.model)
    print(f"Processing {len(todo)} images -> {out_path}")

    # One tiny warmup to trigger CUDA/kernel compilation so first real batch isn't 5–15 min
    print("Warmup run (triggers CUDA compilation, ~30–60 s)...")
    warmup_paths = todo[:1]
    _run_ocr_batch(processor, model, warmup_paths, max_new_tokens=2)
    print("Warmup done. Starting OCR (first batch will still be slower than the rest).")

    def checkpoint():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    processed = 0
    with tqdm(total=len(todo), desc="OCR", unit="img", mininterval=2.0) as pbar:
        for start in range(0, len(todo), batch_size):
            batch_paths = todo[start : start + batch_size]
            try:
                texts = _run_ocr_batch(processor, model, batch_paths, args.max_new_tokens)
                for path, text in zip(batch_paths, texts):
                    results[rel_key(path)] = text
            except Exception as e:
                print(f"\nError on batch at {start}: {e}", file=sys.stderr)
                for path in batch_paths:
                    results[rel_key(path)] = ""
            processed += len(batch_paths)
            pbar.update(len(batch_paths))
            if processed % CHECKPOINT_EVERY == 0:
                checkpoint()

    checkpoint()
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
