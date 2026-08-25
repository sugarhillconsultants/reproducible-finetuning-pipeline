"""
training/finetune_lora.py

Fine-tunes a tiny transformer classifier with PEFT/LoRA on the log
event dataset. Deliberately uses a tiny model (prajjwal1/bert-tiny,
~4M params) so this is CPU-feasible in a GitHub Actions runner in a
few minutes — a real production fine-tune of a larger model would use
QLoRA on a GPU (ZeroGPU or dedicated), as covered in this project's
"honest limitations" section.

Install once:
  pip install transformers peft datasets torch accelerate evaluate scikit-learn

Usage:
  python finetune_lora.py --dataset-repo your-username/log-events --dataset-version v2 \
      --output-dir ./adapter-output
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

BASE_MODEL = "prajjwal1/bert-tiny"  # ~4M params, CPU-feasible for CI demo


def load_and_tokenize(dataset_repo: str, dataset_version: str, tokenizer):
    dataset = load_dataset(dataset_repo, revision=dataset_version)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=64)

    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
    }


def main(dataset_repo: str, dataset_version: str, output_dir: str, epochs: int):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],  # BERT-family attention projections
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = load_and_tokenize(dataset_repo, dataset_version, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    print(f"Final eval metrics: {final_metrics}")

    adapter_path = f"{output_dir}/final-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter to {adapter_path}")

    # Write metrics to a file evaluate.py can read without re-running training
    import json
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump({
            "eval_f1": final_metrics["eval_f1"],
            "eval_accuracy": final_metrics["eval_accuracy"],
        }, f)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", default="./adapter-output")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.dataset_repo, args.dataset_version, args.output_dir, args.epochs)
