# GLM-OCR with Hugging Face Transformers

This project runs [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (Z.ai’s multimodal OCR model) via **Hugging Face Transformers** for document images and PDFs. It converts PDFs to page images, then runs the model to extract text (and optionally uses Flash Attention 2 and batch inference).

**References:** [Hugging Face — GLM-OCR](https://huggingface.co/docs/transformers/model_doc/glm_ocr), [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR).

---

## What’s in this repo

| Item | Description |
|------|-------------|
| **`src/`** | Shared code: `generation_with_probs.py` (token-probability helpers used by PDF scripts when `--show-probs` is set). |
| **`scripts/`** | Exploratory/test scripts: `test_single_pdf.py`, `test_batch_pdf.py`, `test_single_image.py`. Run from project root (e.g. `python scripts/test_single_pdf.py`). |
| **`finetuning/`** | Pipeline for fine-tuning: `data_prep/`, `labels/`, `train/`, `output/` (for rendered images and gold labels). |
| **`inference/`** | Mass inference (to be added after fine-tuning). |
| **`requirements.txt`** | Python dependencies; includes notes for optional Flash Attention and `kernels`. |
| **`data/`** | Directory for PDFs (e.g. drop files here; scripts can pick the first PDF under `data/` if no path is given). |

The model does **not** accept PDF bytes directly; it expects **images** (`pixel_values`). PDFs are turned into page images with **pypdfium2**, then each image (or batch of images) is sent to the model.

**Project structure:** Shared code lives in `src/`; exploratory scripts in `scripts/`. The `finetuning/` and `inference/` trees are set up for the fine-tune-then-inference pipeline (data prep → labels → train → mass inference). Empty directories are kept in Git via `.gitkeep` files so the folder layout is preserved when cloning.

**Finetuning subfolders:**

| Folder | Purpose |
|--------|--------|
| **`finetuning/data_prep/`** | Scripts to sample PDFs from a data source, render pages to images, and write them (plus a manifest) into `output/`. |
| **`finetuning/labels/`** | Scripts to produce or attach gold text per image (e.g. LLM-generated or manual labels), written into the manifest or sidecar files in `output/`. |
| **`finetuning/train/`** | Scripts and config to load (image, label) pairs from `output/` and run fine-tuning of GLM-OCR. |
| **`finetuning/output/`** | Generated data only: rendered images, manifest, and labels produced by the steps above; consumed by `train/`. Typically gitignored. |

---

## Setup

- **Python:** 3.11 recommended.
- **GPU:** CUDA-capable GPU (model uses `device_map="auto"` and bfloat16).
- **Conda (recommended):**

  ```bash
  conda create -n glm_ocr python=3.11
  conda activate glm_ocr
  pip install -r requirements.txt
  ```

- **Optional — Flash Attention 2 (faster inference):**  
  For **PyTorch 2.10 + CUDA 12.8 + Python 3.11** you can install a prebuilt wheel (no build from source):

  ```bash
  pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl
  ```

  Other combos: see [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).  
  If you don’t install Flash Attention, run the PDF scripts with `--no-flash-attn`. Run scripts from the project root (e.g. `python scripts/test_single_pdf.py`).

- **Optional — `kernels` (for Flash Attention with Transformers):**  
  If you use the `kernels-community/flash-attn2` attention implementation:

  ```bash
  pip install -U kernels
  ```

---

## Usage

### Single PDF (one page per forward pass)

Run from project root:

```bash
# First PDF under data/
python scripts/test_single_pdf.py

# Specific PDF
python scripts/test_single_pdf.py path/to/file.pdf

# Options: scale, max tokens, token stats, token probabilities, disable Flash Attention
python scripts/test_single_pdf.py path/to/file.pdf --scale 2.0 --max-tokens 1024 --show-tokens
python scripts/test_single_pdf.py path/to/file.pdf --show-probs
python scripts/test_single_pdf.py path/to/file.pdf --no-flash-attn
```

- **Output:** Timing per page, total time, and extracted text per page (with line numbers). With `--show-tokens`, generated token counts per page and total. With `--show-probs`, per-page and global token-probability stats plus the 50 lowest-probability tokens with surrounding context (token in bold red).

### Batch PDF (multiple pages per forward pass)

```bash
# Batch size 8 (default)
python scripts/test_batch_pdf.py path/to/file.pdf

# Larger batches
python scripts/test_batch_pdf.py path/to/file.pdf --batch-size 4

# Token stats or token probabilities
python scripts/test_batch_pdf.py path/to/file.pdf --show-tokens
python scripts/test_batch_pdf.py path/to/file.pdf --show-probs --batch-size 4
```

- **Output:** Timing summary (pages, batch size, batches, total time, avg sec/page, pages/sec) and text per page (with line numbers). With `--show-tokens`, token counts per batch and total/avg per page. With `--show-probs`, per-batch and global token-probability stats plus the 50 lowest-probability tokens with 10-token context (token in bold red).

### Single image

```bash
# Edit scripts/test_single_image.py to set the image path, then (from project root):
python scripts/test_single_image.py
```

---

## Main options (PDF scripts)

| Option | Scripts | Default | Description |
|--------|---------|--------|-------------|
| `--scale` | single, batch | 2.0 | PDF render scale (pixels per point; 2 ≈ 144 DPI). |
| `--max-tokens` | single, batch | 1024 / 2048 | Max new tokens per page (single: 1024; batch: 2048). |
| `--show-tokens` | single, batch | off | Print generated token counts (per page or per batch + total). |
| `--show-probs` | single, batch | off | Compute token probabilities (softmax over logits), show global avg/min, and worst 50 tokens with context (bold red in terminal). |
| `--no-flash-attn` | single, batch | off | Disable Flash Attention 2 (use default attention). |
| `--batch-size` | batch only | 8 | Number of pages per batch in `test_batch_pdf.py`. |

---

## How it works (for users / agents)

1. **PDF → images:** A PDF is opened with **pypdfium2**; each page is rendered to a PIL image at the given `--scale`.
2. **Model:** **zai-org/GLM-OCR** is loaded via `AutoProcessor` and `GlmOcrForConditionalGeneration` (bfloat16, `device_map="auto"`). If available and not disabled, Flash Attention 2 is used (`attn_implementation="kernels-community/flash-attn2"`).
3. **Single-page mode:** For each page image, a chat message (image + prompt `"Text Recognition:"`) is built, passed through the processor, then the model generates text (up to `--max-tokens`). Output is decoded and the prompt/artifacts are stripped.
4. **Batch mode:** Pages are processed in chunks of `--batch-size`. Each chunk is sent as a batch of messages with `padding=True`; the model returns one sequence per page; `batch_decode` and the same cleanup yield one text per page in order.
5. **Token probabilities (`--show-probs`):** When enabled, the scripts call `run_ocr_batch_with_probs()` from **src/generation_with_probs.py**, which runs `model.generate(..., output_scores=True)`. Logits per generated token are turned into probabilities via softmax; the probability of the chosen token is recorded. Special tokens (e.g. end-of-text) are excluded. Each page gets an ordered list of `(token_str, probability)` pairs. The scripts then print global avg/min, the 50 tokens with lowest probability with 10 tokens of context left and right (target token in bold red), and the full OCR text with line numbers (1-based).
6. **Timing:** Wall-clock time is recorded (per page in single-PDF, per batch in batch-PDF). Avg seconds per page is `total_time / n_pages`; batch size is already reflected in that ratio.

---

## Files not tracked in git

- **`data/`** — PDFs and other inputs (often in `.gitignore` so they stay local).
- **Model weights** — Downloaded from Hugging Face Hub on first run (e.g. `~/.cache/huggingface/`).
