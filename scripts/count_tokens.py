"""
Count average tokens per training sample for GLM-OCR fine-tuning.

Reports image tokens (from image dimensions), label tokens (from labels.json),
prompt/template overhead, and the total sequence length per sample.

Image tokens are computed analytically from image dimensions without loading the
full processor, so this runs fast even on thousands of images.

Run from project root:
  conda run -n glm_ocr python scripts/count_tokens.py finetuning/output
  conda run -n glm_ocr python scripts/count_tokens.py finetuning/output_20Jhd --labels finetuning/output/labels.json
  conda run -n glm_ocr python scripts/count_tokens.py finetuning/output --percentiles
"""
import argparse
import json
import math
import statistics
from pathlib import Path
from PIL import Image

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28
MIN_PIXELS = 12544       # 112 * 112
MAX_PIXELS = 9_633_792   # 14*14*2*2*2*6144
TEMPLATE_OVERHEAD = 12   # fixed tokens from chat template + "Text Recognition:" prompt

MODEL_ID = "unsloth/GLM-OCR"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Reproduce the GLM-46V / Qwen2-VL smart_resize for a single image (t=1)."""
    if height < FACTOR or width < FACTOR:
        scale = max(FACTOR / height, FACTOR / width)
        height = int(height * scale)
        width = int(width * scale)

    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR

    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR

    return h_bar, w_bar


def image_token_count(height: int, width: int) -> int:
    h_bar, w_bar = smart_resize(height, width)
    return (h_bar // FACTOR) * (w_bar // FACTOR)


def print_stats(name: str, values: list[int]):
    if not values:
        print(f"  {name}: (no data)")
        return
    print(f"  {name}:  min={min(values)}  max={max(values)}  "
          f"mean={statistics.mean(values):.1f}  median={statistics.median(values):.1f}")


def print_percentiles(name: str, values: list[int]):
    if not values:
        return
    s = sorted(values)
    n = len(s)
    for p in (50, 75, 90, 95, 99):
        idx = min(int(n * p / 100), n - 1)
        print(f"    {name} p{p}: {s[idx]}")


def main():
    parser = argparse.ArgumentParser(
        description="Count avg tokens per GLM-OCR training sample (image + label).")
    parser.add_argument("input_dir", type=Path,
                        help="Folder with images (and optionally labels.json).")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Path to labels.json (if not inside input_dir).")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help="HF model ID for tokenizer (default: unsloth/GLM-OCR).")
    parser.add_argument("--percentiles", action="store_true",
                        help="Print percentile breakdown.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        return 1

    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {input_dir}")
        return 1

    labels_path = args.labels or (input_dir / "labels.json")
    labels = {}
    if labels_path.is_file():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        print(f"Labels file: {labels_path}  ({len(labels)} entries)")
    else:
        print(f"No labels.json found — reporting image tokens only.")

    tokenizer = None
    if labels:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = getattr(processor, "tokenizer", processor)

    img_tokens_list = []
    label_tokens_list = []
    total_tokens_list = []
    missing_labels = 0

    for img_path in images:
        w, h = Image.open(img_path).size
        n_img = image_token_count(h, w)
        img_tokens_list.append(n_img)

        n_label = 0
        if labels:
            text = labels.get(img_path.name)
            if text is not None:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                n_label = len(ids)
                label_tokens_list.append(n_label)
            else:
                missing_labels += 1

        total_tokens_list.append(TEMPLATE_OVERHEAD + n_img + n_label)

    print(f"\nImages processed: {len(images)}")
    if missing_labels:
        print(f"Images without labels: {missing_labels}")
    if label_tokens_list:
        print(f"Images with labels:    {len(label_tokens_list)}")

    print(f"\nTemplate overhead per sample: {TEMPLATE_OVERHEAD} tokens")
    print()
    print_stats("Image tokens", img_tokens_list)
    if label_tokens_list:
        print_stats("Label tokens", label_tokens_list)
    print_stats("Total tokens", total_tokens_list)

    if args.percentiles:
        print()
        print_percentiles("Image", img_tokens_list)
        if label_tokens_list:
            print_percentiles("Label", label_tokens_list)
        print_percentiles("Total", total_tokens_list)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
