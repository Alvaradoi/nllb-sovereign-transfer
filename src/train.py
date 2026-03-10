"""
src/train.py

LoRA fine-tuning of NLLB-200-distilled-600M on Kalispel Salish ↔ English pairs.

Usage:
    python src/train.py \
        --data-dir data/processed \
        --model-dir models/nllb-200-600M \
        --output-dir outputs/checkpoints \
        [--direction salish_to_english | english_to_salish | both] \
        [--epochs 10] [--batch-size 4] [--grad-accum 8] [--lr 3e-4]

All inference is local. No network calls at runtime. MLflow tracking is local.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLLB language codes
# ---------------------------------------------------------------------------
KALISPEL_LANG = "kal_Latn"   # Kalispel Salish — not an official NLLB code;
                               # use a placeholder BCP-47 tag. NLLB will treat
                               # it as an unknown tag and rely on LoRA transfer.
ENGLISH_LANG = "eng_Latn"

# ---------------------------------------------------------------------------
# LoRA config 
# ---------------------------------------------------------------------------
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

MAX_INPUT_LEN = 128    # tokens — generous for short Salish utterances
MAX_TARGET_LEN = 128

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_dataset(
    records: list[dict],
    direction: str,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
) -> Dataset:
    """Convert JSONL records into a tokenized HuggingFace Dataset.

    direction: "salish_to_english" | "english_to_salish"
    src_lang / tgt_lang: NLLB BCP-47 language codes already resolved by caller.
    """
    inputs, targets = [], []
    for r in records:
        salish = r["salish"].strip()
        english = r["english"].strip()
        if not salish or not english:
            continue
        if direction == "salish_to_english":
            inputs.append(salish)
            targets.append(english)
        else:
            inputs.append(english)
            targets.append(salish)

    tokenizer.src_lang = src_lang

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input"],
            text_target=batch["target"],
            max_length=MAX_INPUT_LEN,
            max_target_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        return model_inputs

    raw = Dataset.from_dict({"input": inputs, "target": targets})
    tokenized = raw.map(tokenize, batched=True, remove_columns=["input", "target"])
    return tokenized


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer):
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 (ignored positions) with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = chrf.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": round(bleu_result["score"], 4),
            "chrf": round(chrf_result["score"], 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    direction: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    log.info("Loading base model from %s", model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    log.info("Applying LoRA (r=%d, alpha=%d)", LORA_CONFIG.r, LORA_CONFIG.lora_alpha)
    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Resolve language codes and forced_bos for generation
    # NLLB uses language tokens like "eng_Latn" in its vocabulary.
    # kal_Latn is not official — we fall back to a closely related code
    # for the BOS token (used only to steer generation direction).
    # LoRA adapts the representations regardless of the BOS token.
    # ------------------------------------------------------------------
    def resolve_lang_code(lang: str) -> str:
        tok_id = tokenizer.convert_tokens_to_ids(lang)
        if tok_id == tokenizer.unk_token_id:
            log.warning(
                "Language code %r not in tokenizer vocab (id=%d). "
                "Salish→English direction will use eng_Latn BOS. "
                "English→Salish will use zza_Latn (Zazaki, placeholder).",
                lang, tok_id,
            )
            # Return a real NLLB code as fallback; LoRA will override representations
            return "zza_Latn"
        return lang

    # direction → (src_lang, tgt_lang, forced_bos_id)
    def lang_pair(d: str) -> tuple[str, str, int]:
        if d == "salish_to_english":
            src = KALISPEL_LANG
            tgt = ENGLISH_LANG
        else:
            src = ENGLISH_LANG
            tgt = KALISPEL_LANG
        resolved_tgt = resolve_lang_code(tgt)
        forced_bos = tokenizer.convert_tokens_to_ids(resolved_tgt)
        return src, resolved_tgt, forced_bos

    # For single-direction runs the forced_bos is fixed; set on generation config.
    # For "both", the two directions are concatenated — forced_bos is set per-direction
    # before the respective build_hf_dataset call.
    directions = (
        [direction] if direction != "both"
        else ["salish_to_english", "english_to_salish"]
    )

    train_records: list[dict] = []
    dev_records: list[dict] = []
    for split_file, store in [
        (data_dir / "camp2007_train.jsonl", train_records),
        (data_dir / "camp2007_dev.jsonl",   dev_records),
    ]:
        if split_file.exists():
            store.extend(load_jsonl(split_file))
        else:
            log.warning("Missing %s", split_file)

    log.info("Train records: %d | Dev records: %d", len(train_records), len(dev_records))

    all_train: list[Dataset] = []
    all_dev: list[Dataset] = []
    primary_forced_bos: Optional[int] = None
    for d in directions:
        src, tgt, forced_bos = lang_pair(d)
        if primary_forced_bos is None:
            primary_forced_bos = forced_bos
        all_train.append(build_hf_dataset(train_records, d, tokenizer, src, tgt))
        all_dev.append(build_hf_dataset(dev_records,   d, tokenizer, src, tgt))

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets(all_train) if len(all_train) > 1 else all_train[0]
    dev_dataset   = concatenate_datasets(all_dev)   if len(all_dev)   > 1 else all_dev[0]

    log.info("Tokenized train: %d | dev: %d", len(train_dataset), len(dev_dataset))

    # Set forced_bos on the model generation config so beam search decodes
    # into the correct target language during evaluation.
    if primary_forced_bos is not None:
        model.generation_config.forced_bos_token_id = primary_forced_bos
        log.info("forced_bos_token_id = %d", primary_forced_bos)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    run_name = f"nllb-kalispel-{direction}-r{LORA_CONFIG.r}"
    checkpoint_dir = output_dir / run_name

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=torch.cuda.is_available(),
        learning_rate=lr,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_steps=10,
        save_total_limit=3,
        seed=seed,
        report_to=[],    # disable all external reporters
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    compute_metrics = make_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ------------------------------------------------------------------
    # MLflow local run
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nllb-kalispel")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "direction": direction,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "lr": lr,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "train_pairs": len(train_dataset),
            "dev_pairs": len(dev_dataset),
        })

        log.info("Starting training — run: %s", run_name)
        trainer.train()

        log.info("Saving LoRA adapter to %s", checkpoint_dir / "final")
        model.save_pretrained(str(checkpoint_dir / "final"))
        tokenizer.save_pretrained(str(checkpoint_dir / "final"))

        # Log final eval metrics
        eval_results = trainer.evaluate()
        log.info("Final eval: %s", eval_results)
        mlflow.log_metrics({
            "final_bleu": eval_results.get("eval_bleu", 0.0),
            "final_chrf": eval_results.get("eval_chrf", 0.0),
        })

    log.info("Training complete. Adapter saved to %s/final", checkpoint_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune NLLB-200 on Kalispel Salish")
    p.add_argument("--data-dir",   type=Path, default=Path("data/processed"))
    p.add_argument("--model-dir",  type=Path, default=Path("models/nllb-200-600M"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"))
    p.add_argument(
        "--direction",
        choices=["salish_to_english", "english_to_salish", "both"],
        default="salish_to_english",
        help="Translation direction(s) to train",
    )
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    if not args.data_dir.exists():
        log.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)
    if not args.model_dir.exists():
        log.error("Model directory not found: %s  — run setup first", args.model_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        direction=args.direction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
