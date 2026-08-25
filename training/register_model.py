"""
training/register_model.py

Merges the trained LoRA adapter into a full-precision copy of the base
model (never merge directly from quantized weights — see this project's
README for why), then pushes the merged model to the Hugging Face Hub
with a version tag, only ever called after evaluate.py has approved it.

Install once:
  pip install transformers peft huggingface_hub

Usage:
  python register_model.py --adapter-path ./adapter-output/final-adapter \
      --repo-id your-username/log-classifier-tiny --version v1.0.0
"""

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from huggingface_hub import create_tag

BASE_MODEL = "prajjwal1/bert-tiny"


def merge_and_push(adapter_path: str, repo_id: str, version: str):
    print(f"Loading base model {BASE_MODEL} in full precision for merging...")
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print(f"Loading adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Pushing merged model to {repo_id}...")
    merged_model.push_to_hub(repo_id, commit_message=f"Register model {version}")
    tokenizer.push_to_hub(repo_id, commit_message=f"Register model {version}")

    create_tag(repo_id=repo_id, tag=version, repo_type="model", exist_ok=True)

    print(f"Registered and tagged as {version}. Load later with:")
    print(f'  AutoModelForSequenceClassification.from_pretrained("{repo_id}", revision="{version}")')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    merge_and_push(args.adapter_path, args.repo_id, args.version)
