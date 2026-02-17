# GLM-OCR with Hugging Face Transformers

This project runs [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (Z.ai’s multimodal OCR model) via **Hugging Face Transformers** for document images and PDFs. It converts PDFs to page images, then runs the model to extract text (and optionally uses Flash Attention 2 and batch inference).

**References:** [Hugging Face — GLM-OCR](https://huggingface.co/docs/transformers/model_doc/glm_ocr), [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR).

---

## What’s in this repo

| Item | Description |
|------|-------------|
| **`test_single_pdf.py`** | Run OCR on one PDF: render all pages to images, process **one page at a time**, print timing and text per page. |
| **`test_batch_pdf.py`** | Same PDF workflow but process pages **in batches** (e.g. 2 pages per forward pass) for better GPU use. |
| **`test_single_image.py`** | Run OCR on a **single image** (no PDF). |
| **`requirements.txt`** | Python dependencies; includes notes for optional Flash Attention and `kernels`. |
| **`data/`** | Directory for PDFs (e.g. drop files here; scripts can pick the first PDF under `data/` if no path is given). |

The model does **not** accept PDF bytes directly; it expects **images** (`pixel_values`). PDFs are turned into page images with **pypdfium2**, then each image (or batch of images) is sent to the model.

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
  If you don’t install Flash Attention, run the PDF scripts with `--no-flash-attn`.

- **Optional — `kernels` (for Flash Attention with Transformers):**  
  If you use the `kernels-community/flash-attn2` attention implementation:

  ```bash
  pip install -U kernels
  ```

---

## Usage

### Single PDF (one page per forward pass)

```bash
# First PDF under data/
python test_single_pdf.py

# Specific PDF
python test_single_pdf.py path/to/file.pdf

# Options: scale, max tokens, token stats, disable Flash Attention
python test_single_pdf.py path/to/file.pdf --scale 2.0 --max-tokens 1024 --show-tokens
python test_single_pdf.py path/to/file.pdf --no-flash-attn
```

- **Output:** Timing per page, total time, and extracted text per page. With `--show-tokens`, generated token counts per page and total.

### Batch PDF (multiple pages per forward pass)

```bash
# Batch size 2 (default)
python test_batch_pdf.py path/to/file.pdf

# Larger batches
python test_batch_pdf.py path/to/file.pdf --batch-size 4

# Token stats
python test_batch_pdf.py path/to/file.pdf --show-tokens
```

- **Output:** Timing summary (pages, batch size, batches, total time, avg sec/page, pages/sec) and text per page. With `--show-tokens`, token counts per batch and total/avg per page.

### Single image

```bash
# Edit test_single_image.py to set the image path, then:
python test_single_image.py
```

---

## Main options (PDF scripts)

| Option | Scripts | Default | Description |
|--------|---------|--------|-------------|
| `--scale` | single, batch | 2.0 | PDF render scale (pixels per point; 2 ≈ 144 DPI). |
| `--max-tokens` | single, batch | 1024 | Max new tokens per page (increase if long pages are cut off). |
| `--show-tokens` | single, batch | off | Print generated token counts (per page or per batch + total). |
| `--no-flash-attn` | single, batch | off | Disable Flash Attention 2 (use default attention). |
| `--batch-size` | batch only | 2 | Number of pages per batch in `test_batch_pdf.py`. |

---

## How it works (for users / agents)

1. **PDF → images:** A PDF is opened with **pypdfium2**; each page is rendered to a PIL image at the given `--scale`.
2. **Model:** **zai-org/GLM-OCR** is loaded via `AutoProcessor` and `GlmOcrForConditionalGeneration` (bfloat16, `device_map="auto"`). If available and not disabled, Flash Attention 2 is used (`attn_implementation="kernels-community/flash-attn2"`).
3. **Single-page mode:** For each page image, a chat message (image + prompt `"Text Recognition:"`) is built, passed through the processor, then the model generates text (up to `--max-tokens`). Output is decoded and the prompt/artifacts are stripped.
4. **Batch mode:** Pages are processed in chunks of `--batch-size`. Each chunk is sent as a batch of messages with `padding=True`; the model returns one sequence per page; `batch_decode` and the same cleanup yield one text per page in order.
5. **Timing:** Wall-clock time is recorded (per page in single-PDF, per batch in batch-PDF). Avg seconds per page is `total_time / n_pages`; batch size is already reflected in that ratio.

---

## Files not tracked in git

- **`data/`** — PDFs and other inputs (often in `.gitignore` so they stay local).
- **Model weights** — Downloaded from Hugging Face Hub on first run (e.g. `~/.cache/huggingface/`).
