# Colab setup

## 1. Upload data to Google Drive (one-time, free)

From your machine (in the repo, with `finetuning/output` filled):

```bash
cd finetuning/output
zip -r glm_ocr_output.zip labels.json train.txt test.txt *.png
```

Upload `glm_ocr_output.zip` to **Google Drive** → **My Drive** (root is fine).

Or create a folder **glm_ocr_output** in My Drive and upload `labels.json`, `train.txt`, `test.txt`, and all `*.png` into it.

## 2. Open the notebook in Colab

- In Colab: **File → Open notebook → GitHub** and enter:
  `https://github.com/Meloneneis/glm_ocr`
- Open `scripts/colab_train.ipynb`.

Or clone and upload the notebook; or run the same steps in a new Colab notebook by pasting the cells from `colab_train.ipynb`.

## 3. Run cells

1. **Cell 1**: Install dependencies (run once).
2. **Cell 2**: Clone repo, mount Drive, copy data into `finetuning/output`, `cd finetuning/train`. Authorize Drive when prompted.
3. **Cell 3**: Run training:
   ```bash
   !python train_unsloth.py --batch-size 16 --gradient-accumulation-steps 2
   ```

That’s it.
