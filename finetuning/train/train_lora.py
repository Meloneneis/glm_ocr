"""
Fine-tune GLM-OCR with LoRA on (image, label) pairs from finetuning/output.

Quick test: 100 train samples, 10 test samples. Uses train.txt, test.txt, labels.json.
Run from project root:
  python finetuning/train/train_lora.py
  python finetuning/train/train_lora.py --train-size 500 --test-size 50
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output"
MODEL_ID = "zai-org/GLM-OCR"
PROMPT = "Text Recognition:"


def get_data(output_dir: Path, labels_path: Path, train_txt: Path, test_txt: Path, train_size: int, test_size: int):
    """Load labels and train/test lists; return (train_list, test_list) of (image_name, label_text)."""
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    train_names = [line.strip() for line in train_txt.read_text().splitlines() if line.strip()][:train_size]
    test_names = [line.strip() for line in test_txt.read_text().splitlines() if line.strip()][:test_size]
    train = [(n, labels.get(n, "")) for n in train_names if n in labels]
    test = [(n, labels.get(n, "")) for n in test_names if n in labels]
    return train, test


class OcrDataset(torch.utils.data.Dataset):
    """Dataset of (image path, label text) for GLM-OCR. Returns dict with input_ids, attention_mask, pixel_values, labels."""

    def __init__(self, data_dir: Path, pairs: list, processor, max_length: int = 2048):
        self.data_dir = data_dir
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length
        self.eos_id = getattr(processor.tokenizer, "eos_token_id") or processor.tokenizer.pad_token_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name, label = self.pairs[idx]
        path = self.data_dir / name
        image = Image.open(path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors=None,
        )
        input_ids = inputs["input_ids"]
        if isinstance(input_ids, list) and len(input_ids) and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if hasattr(input_ids, "shape") and len(input_ids.shape) > 1:
            input_ids = input_ids[0].tolist()
        elif not isinstance(input_ids, list):
            input_ids = input_ids.tolist()
        input_ids = list(input_ids)
        prompt_len = len(input_ids)
        response_ids = self.processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        if isinstance(response_ids, list) and len(response_ids) and isinstance(response_ids[0], list):
            response_ids = response_ids[0]
        if self.eos_id is not None:
            response_ids = list(response_ids) + [self.eos_id]
        else:
            response_ids = list(response_ids)
        # Truncate only the response so we never cut into prompt/image tokens (model needs token count to match image features).
        max_response_len = max(0, self.max_length - prompt_len)
        response_ids = response_ids[:max_response_len]
        full_ids = list(input_ids) + response_ids
        labels = [-100] * prompt_len + response_ids
        if len(labels) < len(full_ids):
            labels = labels + [-100] * (len(full_ids) - len(labels))
        full_ids = list(full_ids)
        labels = list(labels)

        out = {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if hasattr(pv, "numpy"):
                pv = torch.from_numpy(pv) if not isinstance(pv, torch.Tensor) else pv
            else:
                pv = torch.tensor(pv, dtype=torch.float32)
            if pv.dim() == 4:
                pv = pv.squeeze(0)
            out["pixel_values"] = pv
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"]
            if hasattr(thw, "numpy"):
                thw = torch.from_numpy(thw) if not isinstance(thw, torch.Tensor) else thw
            else:
                thw = torch.tensor(thw, dtype=torch.long)
            if thw.dim() > 1:
                thw = thw.squeeze(0)
            out["image_grid_thw"] = thw
        return out


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GLM-OCR with LoRA.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory with images, train.txt, test.txt, labels.json.")
    parser.add_argument("--train-size", type=int, default=100, help="Max train samples (default 100).")
    parser.add_argument("--test-size", type=int, default=10, help="Max test samples (default 10).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default 1 to reduce OOM).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-dir", type=Path, default=None, help="Where to save adapter (default: finetuning/train/out_lora).")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention.")
    args = parser.parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT))

    train_txt = args.output_dir / "train.txt"
    test_txt = args.output_dir / "test.txt"
    labels_path = args.output_dir / "labels.json"
    for f in (train_txt, test_txt, labels_path):
        if not f.is_file():
            print(f"Missing: {f}", file=sys.stderr)
            sys.exit(1)

    train_list, test_list = get_data(args.output_dir, labels_path, train_txt, test_txt, args.train_size, args.test_size)
    if not train_list or not test_list:
        print("No train or test samples after filtering.", file=sys.stderr)
        sys.exit(1)
    print(f"Train samples: {len(train_list)}, test samples: {len(test_list)}")

    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not args.no_flash_attn:
        try:
            model_kwargs["attn_implementation"] = "kernels-community/flash-attn2"
        except Exception:
            pass
    model = GlmOcrForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)

    # LoRA on decoder attention (GLM-OCR text decoder uses q_proj, k_proj, v_proj, o_proj in layers)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "base_model"):
        model.base_model.model.gradient_checkpointing_enable()

    train_ds = OcrDataset(args.output_dir, train_list, processor, max_length=args.max_length)
    eval_ds = OcrDataset(args.output_dir, test_list, processor, max_length=args.max_length)

    def _collate(examples):
        # Pad input_ids, attention_mask, labels; stack pixel_values
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [e["input_ids"] for e in examples], batch_first=True, padding_value=processor.tokenizer.pad_token_id or 0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [e["attention_mask"] for e in examples], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [e["labels"] for e in examples], batch_first=True, padding_value=-100
        )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in examples[0]:
            batch["pixel_values"] = torch.stack([e["pixel_values"] for e in examples])
        if "image_grid_thw" in examples[0]:
            batch["image_grid_thw"] = torch.stack([e["image_grid_thw"] for e in examples])
        return batch

    from transformers import TrainingArguments, Trainer

    save_dir = args.save_dir or (_PROJECT_ROOT / "finetuning" / "train" / "out_lora")
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
    )
    trainer.train()
    trainer.save_model(str(save_dir))
    processor.save_pretrained(save_dir)
    print(f"Saved adapter and processor to {save_dir}")


if __name__ == "__main__":
    main()
