"""
Run finetuned GLM-OCR on all images in a directory; save results to JSONL (one line per image).
Each line: {"path": "<rel_path>", "text": "...", "truncated": bool, "token_probs": [...] (if --with-probs)}.
Resume: skips images already in output JSONL (streams file to build done set). Appends after every batch.

Run from project root:
  python inference/run_ocr.py
  python inference/run_ocr.py --images-dir inference/21jhd/images --batch-size 8
"""
import argparse
import gc
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Unsloth must be imported before transformers/peft for optimizations
from unsloth import FastVisionModel

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.generation_with_probs import _scores_to_token_probs, _token_probs_to_ordered_pairs

PROMPT = "Text Recognition:"
# Base model (Unsloth); adapter is loaded on top via PEFT (Hub id or local path).
BASE_MODEL_ID = "unsloth/GLM-OCR"
# Safety cap only: generation stops at EOS when the model finishes. Use same as training max_seq_length so no truncation.
DEFAULT_MAX_NEW_TOKENS = 1024


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


def _run_ocr_batch(
    processor,
    model,
    image_paths: List[Path],
    max_new_tokens: int,
    raw: bool = False,
    with_probs: bool = False,
) -> Tuple[List[str], List[bool], Optional[List[Optional[List[List]]]]]:
    """Run OCR on a batch of images; return (texts, truncated_mask, token_probs_or_none). truncated_mask[i]=True if no EOS. If with_probs, token_probs_or_none[i] is list of [token_str, prob] for JSON."""
    if not image_paths:
        return [], [], None
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            images.append(None)
    valid_indices = [i for i, im in enumerate(images) if im is not None]
    if not valid_indices:
        return [""] * len(image_paths), [False] * len(image_paths), None
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
    # Release PIL images immediately after processor has copied to tensors (PIL is known to retain memory in long-running loops)
    for im in images:
        if im is not None:
            try:
                im.close()
            except Exception:
                pass
    del images, messages, valid_images
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
    if with_probs:
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    sequences = out.sequences if hasattr(out, "sequences") else out
    scores = getattr(out, "scores", None) or ()
    input_len = inputs["input_ids"].shape[1]
    # Release input tensors so GPU memory can be reused (inputs no longer needed)
    del inputs
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else (list(eos_token_id) if eos_token_id else [])
    truncated_mask = [False] * len(image_paths)
    for idx, i in enumerate(valid_indices):
        gen_tokens = sequences[idx, input_len:].tolist()
        if eos_ids and not any(eid in gen_tokens for eid in eos_ids):
            truncated_mask[i] = True
    decoded = processor.batch_decode(sequences, skip_special_tokens=not raw)
    if raw:
        texts = list(decoded)
    else:
        texts = [_clean_decoded(s) for s in decoded]
    result = [""] * len(image_paths)
    for idx, i in enumerate(valid_indices):
        result[i] = texts[idx]

    token_probs_result = None
    if with_probs and scores:
        eos_for_stop = eos_ids[0] if eos_ids else None
        token_probs_per_batch = _scores_to_token_probs(sequences, scores, input_len, eos_for_stop)
        token_probs_result = [None] * len(image_paths)
        for idx, i in enumerate(valid_indices):
            pairs = _token_probs_to_ordered_pairs(
                sequences[idx, input_len:], token_probs_per_batch[idx], tokenizer
            )
            token_probs_result[i] = [[s, round(p, 6)] for s, p in pairs]
        # Release large GPU tensors so memory can be freed (avoid per-batch gc/empty_cache here; done periodically in main loop)
        del out, scores, sequences
    else:
        del out, sequences
    return result, truncated_mask, token_probs_result


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
        help="Output JSONL path (default: <images-dir>/../results.jsonl or results_with_probs.jsonl if --with-probs).",
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
    parser.add_argument(
        "--with-probs",
        action="store_true",
        help="Store per-token probabilities in JSONL line (fused: each line has token_probs); use for confidence/QC.",
    )
    parser.add_argument(
        "--cleanup-every",
        type=int,
        default=5,
        metavar="N",
        help="Run gc + CUDA empty_cache every N batches to limit memory growth (default: 5). Use 0 to disable, 1 every batch (slower).",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        print(f"Images directory not found: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    default_stem = "results_with_probs" if args.with_probs else "results"
    out_path = args.output or (args.images_dir.parent / f"{default_stem}.jsonl")
    done_paths = set()
    truncated_paths = []
    if out_path.is_file():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    path_key = obj.get("path")
                    if path_key is not None:
                        done_paths.add(path_key)
                        if obj.get("truncated"):
                            truncated_paths.append(path_key)
                except json.JSONDecodeError:
                    pass

    image_paths = sorted(args.images_dir.rglob("*.png"))
    if not image_paths:
        print("No PNG images found in images-dir.", file=sys.stderr)
        sys.exit(1)

    def rel_key(p: Path) -> str:
        return str(p.relative_to(args.images_dir)).replace("\\", "/")

    todo = [p for p in image_paths if rel_key(p) not in done_paths]
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
    print(f"  with_probs:      {args.with_probs}")
    print(f"  cleanup_every:   every {args.cleanup_every} batches" if args.cleanup_every else "  cleanup_every:   disabled")
    print(f"  images to run:   {len(todo)}")
    print()

    print(f"Loading model and processor (model: {args.model})...")
    model, processor = _load_model_and_processor(args.model)
    print(f"Processing {len(todo)} images -> {out_path}")

    # One tiny warmup to trigger CUDA/kernel compilation so first real batch isn't 5–15 min
    print("Warmup run (triggers CUDA compilation, ~30–60 s)...")
    warmup_paths = todo[:1]
    _run_ocr_batch(processor, model, warmup_paths, max_new_tokens=2, with_probs=args.with_probs)
    print("Warmup done. Starting OCR (first batch will still be slower than the rest).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = len(done_paths)
    cleanup_every = max(0, args.cleanup_every)
    with open(out_path, "a", encoding="utf-8") as out_file:
        with tqdm(total=len(todo), desc="OCR", unit="img", mininterval=2.0) as pbar:
            for batch_idx, start in enumerate(range(0, len(todo), batch_size)):
                batch_paths = todo[start : start + batch_size]
                try:
                    texts, batch_truncated, batch_probs = _run_ocr_batch(
                        processor, model, batch_paths, args.max_new_tokens, with_probs=args.with_probs
                    )
                    for i, (path, text, truncated) in enumerate(zip(batch_paths, texts, batch_truncated)):
                        key = rel_key(path)
                        if truncated:
                            truncated_paths.append(key)
                        line_obj = {"path": key, "text": text, "truncated": truncated}
                        if args.with_probs and batch_probs and batch_probs[i] is not None:
                            line_obj["token_probs"] = batch_probs[i]
                        out_file.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                        total_written += 1
                    # Drop references to batch results so GC can free large token_probs lists
                    del texts, batch_truncated, batch_probs
                except Exception as e:
                    print(f"\nError on batch at {start}: {e}", file=sys.stderr)
                    for path in batch_paths:
                        key = rel_key(path)
                        line_obj = {"path": key, "text": "", "truncated": False}
                        if args.with_probs:
                            line_obj["token_probs"] = []
                        out_file.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                        total_written += 1
                pbar.update(len(batch_paths))
                out_file.flush()
                if cleanup_every and (batch_idx + 1) % cleanup_every == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if truncated_paths:
        truncated_path = out_path.parent / (out_path.stem + "_truncated.json")
        truncated_path.write_text(json.dumps(truncated_paths, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Saved {total_written} results to {out_path}")


if __name__ == "__main__":
    main()
