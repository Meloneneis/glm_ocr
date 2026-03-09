"""
Merge a PEFT adapter into the base model and save a single full model.
Result: one directory with the same param count as the base (no separate adapter).
Inference is identical; deployment is simpler and can be slightly faster.

Run from project root:
  python scripts/merge_adapter.py --adapter meloneneneis/glm_ocr_21jhd --output inference/merged_21jhd
  python scripts/merge_adapter.py --adapter ./finetuning/output/checkpoint-100 --output ./merged_model
"""
import argparse
import sys
from pathlib import Path

# Unsloth must be imported before transformers/peft
from unsloth import FastVisionModel

from peft import PeftModel
from transformers import AutoProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

BASE_MODEL_ID = "unsloth/GLM-OCR"


def main():
    parser = argparse.ArgumentParser(
        description="Merge PEFT adapter into base model and save a single full model.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="meloneneneis/glm_ocr_21jhd",
        help="HuggingFace adapter id or local path to adapter directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the merged model (and processor).",
    )
    args = parser.parse_args()

    print("Loading base model (full precision, no 4-bit)...")
    model, _ = FastVisionModel.from_pretrained(
        BASE_MODEL_ID,
        max_seq_length=8192,
        load_in_4bit=False,
    )
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    print("Merging adapter into base...")
    model = model.merge_and_unload()
    print("Loading processor from adapter...")
    processor = AutoProcessor.from_pretrained(args.adapter, use_fast=False)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model and processor to {args.output}...")
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print("Done. Use the output path as a single model for inference (e.g. load with AutoModel + AutoProcessor).")


if __name__ == "__main__":
    main()
