# GLM-OCR Unsloth training on Google Colab

Copy the cells below into a new Colab notebook. Set **Runtime → Change runtime type → GPU** (T4 or better) before running.

---

## Cell 1: Clone repo and go to project root

```python
# Set your repo URL (use HTTPS for public repos; for private, use a token or upload the repo manually)
REPO_URL = "https://github.com/Meloneneis/glm_ocr.git"  # or your fork

!git clone --depth 1 $REPO_URL glm_ocr
%cd glm_ocr
```

---

## Cell 2: Install dependencies

Colab already has PyTorch and CUDA. We install the rest; Unsloth may pull a specific transformers version—if GLM-OCR fails with “model type not found”, upgrade transformers in the next cell.

```python
!pip install -q transformers>=5.2.0 accelerate tqdm pypdfium2 kernels
!pip install -q unsloth trl
!pip install -q google-genai   # only needed for the labels step (Vertex Gemini)
```

---

## Cell 3 (optional): Upgrade transformers for GLM-OCR

If you see `ValueError: model type 'glm_ocr' is not recognized`, run:

```python
!pip install -q --upgrade "transformers>=5.2.0"
```

Then **Runtime → Restart runtime**, and run the install cell again (or skip it) and continue from Cell 4.

---

## Cell 4: Check GPU and list data

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA: {torch.version.cuda}")

# Ensure we're in project root
import os
assert os.path.isfile("requirements.txt"), "Run from glm_ocr repo root (e.g. after %cd glm_ocr)"
!ls -la finetuning/output/ 2>/dev/null || echo "Upload or create finetuning/output/ with train.txt, test.txt, labels.json and images."
```

---

## Cell 5: Run Unsloth training

Upload your data to `finetuning/output/` (or mount Drive and symlink) so that `train.txt`, `test.txt`, `labels.json`, and the image files are there. Then run:

```python
# Default: uses all samples in train.txt / test.txt. Reduce with --train-size / --test-size if OOM.
!python finetuning/train/train_unsloth.py

# Or with options, e.g. 4-bit and fewer samples:
# !python finetuning/train/train_unsloth.py --epochs 3 --load-in-4bit --train-size 2000 --test-size 200
```

---

## Cell 6 (optional): Mount Google Drive for data or saving checkpoints

```python
from google.colab import drive
drive.mount("/content/drive")

# Example: link your data from Drive into the repo
# !ln -s /content/drive/MyDrive/glm_ocr_output finetuning/output

# Example: copy checkpoint back to Drive after training
# !cp -r finetuning/train/out_unsloth /content/drive/MyDrive/
```

---

## Quick one-shot (paste as one cell)

If you prefer a single cell (clone + install + check), use this. Replace `REPO_URL` if needed.

```python
REPO_URL = "https://github.com/Meloneneis/glm_ocr.git"
!git clone --depth 1 $REPO_URL glm_ocr
%cd glm_ocr
!pip install -q "transformers>=5.2.0" accelerate tqdm pypdfium2 kernels unsloth trl google-genai
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
# Then run: !python finetuning/train/train_unsloth.py
```

After that, run training in a new cell: `!python finetuning/train/train_unsloth.py`
