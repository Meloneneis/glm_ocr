"""
Fine-tune GLM-OCR with Unsloth (FastVisionModel + LoRA + UnslothVisionDataCollator).

Uses finetuning/output (train.txt, test.txt, labels.json).
Default: 100 train / 10 test, 1 epoch. Saves to finetuning/train/out_unsloth.
CER (character error rate) is computed after each evaluation and uses extra GPU memory;
if OOM occurs after the first epoch, lower --cer-samples and/or --cer-max-new-tokens.

Run from project root:
  python finetuning/train/train_unsloth.py
  python finetuning/train/train_unsloth.py --train-size 500 --epochs 2 --load-in-4bit
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "finetuning" / "output"
MODEL_ID = "unsloth/GLM-OCR"
PROMPT = "Text Recognition:"


def get_data(output_dir: Path, labels_path: Path, train_txt: Path, test_txt: Path, train_size: int, test_size: int):
    """Load labels and train/test lists; return (train_list, test_list) of (image_name, label_text)."""
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    train_names = [line.strip() for line in train_txt.read_text().splitlines() if line.strip()][:train_size]
    test_names = [line.strip() for line in test_txt.read_text().splitlines() if line.strip()][:test_size]
    train = [(n, labels.get(n, "")) for n in train_names if n in labels]
    test = [(n, labels.get(n, "")) for n in test_names if n in labels]
    return train, test


class LazyMessagesDataset:
    """
    Dataset that loads images on demand in __getitem__ to avoid loading
    thousands of images into RAM at once (which causes OOM kill with large train_size).
    Returns Unsloth vision format: {"messages": [user with image+text, assistant with text]}.
    """

    def __init__(self, data_dir: Path, pairs: list):
        self.data_dir = data_dir
        self.pairs = pairs

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
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]
        return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GLM-OCR with Unsloth.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=99999)
    parser.add_argument("--test-size", type=int, default=99999)
    # Defaults aligned with Unsloth DeepSeek-OCR 2 / DeepSeek-OCR (3B) notebook.
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (notebook uses max_steps=60 for quick run).")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size (notebook: 2).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (notebook: 2e-4).")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4bit to reduce VRAM.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (notebook: 16).")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (notebook: 16).")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout (notebook: 0).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps (notebook: 4; effective batch = batch_size * this).")
    parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps.")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay (notebook: 0.001).")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed (notebook: 3407).")
    parser.add_argument("--eval-steps", type=int, default=None, help="Evaluate every N steps. If not set, evaluate once per epoch.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-device batch size for evaluation (default 1 to avoid OOM; set to match --batch-size if you have enough VRAM).")
    parser.add_argument("--cer-samples", type=int, default=10, help="Number of test samples used for CER after each eval. Lower (e.g. 5 or 3) if OOM occurs after evaluation.")
    parser.add_argument("--cer-max-new-tokens", type=int, default=512, help="Max new tokens for CER generation. Lower (e.g. 256 or 128) to reduce VRAM during CER.")
    args = parser.parse_args()

    train_txt = args.output_dir / "train.txt"
    test_txt = args.output_dir / "test.txt"
    labels_path = args.output_dir / "labels.json"
    for f in (train_txt, test_txt, labels_path):
        if not f.is_file():
            print(f"Missing: {f}", file=sys.stderr)
            sys.exit(1)

    train_list, test_list = get_data(
        args.output_dir, labels_path, train_txt, test_txt, args.train_size, args.test_size
    )
    if not train_list or not test_list:
        print("No train or test samples after filtering.", file=sys.stderr)
        sys.exit(1)
    print(f"Train samples: {len(train_list)}, test samples: {len(test_list)}")

    import torch
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_length,
    )
    tokenizer = getattr(processor, "tokenizer", processor)

    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
        use_rslora=False,
        target_modules="all-linear",
    )
    FastVisionModel.for_training(model)

    # Lazy datasets: load images in __getitem__ to avoid OOM with large train_size (e.g. 8500).
    train_data = LazyMessagesDataset(args.output_dir, train_list)
    eval_data = LazyMessagesDataset(args.output_dir, test_list)

    # GLM-OCR chat template: user yields "[gMASK]\n" + prompt + "\n", then add_generation_prompt adds "\n"; assistant starts with "\n".
    instruction_part = "[gMASK]\n" + PROMPT + "\n\n"
    response_part = "\n"

    data_collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=args.max_length,
        train_on_responses_only=True,
        instruction_part=instruction_part,
        response_part=response_part,
        completion_only_loss=True,
        resize="min",
    )

    save_dir = args.save_dir or (_PROJECT_ROOT / "finetuning" / "train" / "out_unsloth")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validation on test set: after each epoch (default), or every --eval-steps if set.
    eval_strategy = "steps" if args.eval_steps is not None else "epoch"
    eval_steps = args.eval_steps if args.eval_steps is not None else 1  # ignored when strategy is "epoch"
    save_strategy = "steps" if args.eval_steps is not None else "epoch"
    save_steps = args.eval_steps if args.eval_steps is not None else 1  # save checkpoint when we eval
    if eval_strategy == "epoch":
        print("Validation will run on the test set at the end of each epoch.")
    else:
        print(f"Validation will run on the test set every {eval_steps} steps.")

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else 1
    if eval_batch_size > args.batch_size:
        eval_batch_size = args.batch_size
    print(f"Eval batch size: {eval_batch_size} (train batch size: {args.batch_size}).")
    print(f"CER: {args.cer_samples} samples, max_new_tokens={args.cer_max_new_tokens} (use --cer-samples / --cer-max-new-tokens to reduce if OOM after eval).")

    training_args = SFTConfig(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        seed=args.seed,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=1,
        max_seq_length=args.max_length,
    )

    class ClearCacheEvalCallback(TrainerCallback):
        """Clear CUDA cache before evaluation to reduce OOM (eval often uses more memory)."""

        def on_evaluate(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return control

    def _cap_label_to_tokens(processor, label: str, max_tokens: int) -> str:
        """Truncate label to first max_tokens tokens and decode back, for fair CER vs capped generation."""
        tokenizer = getattr(processor, "tokenizer", processor)
        ids = tokenizer.encode(label, add_special_tokens=False)
        ids_capped = ids[:max_tokens]
        return tokenizer.decode(ids_capped, skip_special_tokens=True)

    def _char_error_rate(prediction: str, reference: str) -> float:
        """Compute Character Error Rate via Levenshtein distance."""
        n = len(reference)
        m = len(prediction)
        if n == 0:
            return 0.0 if m == 0 else 1.0
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if reference[i - 1] == prediction[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[m] / n

    class CERCallback(TrainerCallback):
        """After each evaluation, compute CER across the full eval set and show one sample."""

        def __init__(self, processor, output_dir: Path, test_list: list, max_new_tokens: int = 512, cer_samples: int = 10):
            import random
            self.processor = processor
            self.output_dir = output_dir
            self.test_list = test_list
            self.cer_subset = random.sample(test_list, min(cer_samples, len(test_list)))
            self.max_new_tokens = max_new_tokens

        def _generate_one(self, model, name):
            image = Image.open(self.output_dir / name).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            else:
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids = out[0].cpu().clone()
            del inputs, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            generated = self.processor.decode(generated_ids, skip_special_tokens=True)
            if PROMPT in generated:
                generated = generated.split(PROMPT)[-1]
            generated = generated.replace("<think>", "").replace("</think>", "").replace("<|image|>", "").strip()
            return generated

        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            step = state.global_step if state else 0
            print("\n" + "=" * 60)
            print(f"Evaluation results (step {step})")
            print("=" * 60)
            if metrics:
                for k, v in sorted(metrics.items()):
                    print(f"  {k}: {v}")
            if model is None or not self.test_list:
                print("=" * 60 + "\n")
                return control

            model.eval()
            cer_scores = []
            first_generated = None
            first_label = None
            first_name = None
            samples = self.cer_subset
            total = len(samples)
            for i, (name, label) in enumerate(samples):
                print(f"  CER generation: {i + 1}/{total}", end="\r", flush=True)
                generated = self._generate_one(model, name)
                label_capped = _cap_label_to_tokens(self.processor, label, self.max_new_tokens)
                cer = _char_error_rate(generated, label_capped)
                cer_scores.append(cer)
                if first_generated is None:
                    first_generated, first_label, first_name = generated, label_capped, name
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_cer = sum(cer_scores) / len(cer_scores)
            print("-" * 60)
            print(f"  Eval CER (avg over {total} samples): {avg_cer:.4f}")
            print(f"  CER min: {min(cer_scores):.4f}  max: {max(cer_scores):.4f}")
            print("-" * 60)
            print(f"Sample ({first_name}):")
            print("  Label:     ", (first_label[:500] + "..." if len(first_label) > 500 else first_label) or "(empty)")
            print("  Generated: ", (first_generated[:500] + "..." if len(first_generated) > 500 else first_generated) or "(empty)")
            print("=" * 60 + "\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return control

    callbacks = [
        ClearCacheEvalCallback(),
        CERCallback(
            processor,
            args.output_dir,
            test_list,
            max_new_tokens=args.cer_max_new_tokens,
            cer_samples=args.cer_samples,
        ),
    ]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    # Evaluation before training (baseline)
    print("Running baseline evaluation before training...")
    baseline_metrics = trainer.evaluate()
    print("Baseline metrics:", baseline_metrics)
    trainer.train()
    trainer.save_model(str(save_dir))
    processor.save_pretrained(save_dir)
    print(f"Saved adapter and processor to {save_dir}")


if __name__ == "__main__":
    main()
