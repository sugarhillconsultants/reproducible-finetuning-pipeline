"""
export/verify_onnx_inference.py

Loads BOTH the original PyTorch model and the exported ONNX model, runs
the same inputs through each, and confirms they agree — proving the
export didn't silently change model behavior. This is the step most
ONNX-export tutorials skip; skipping it means you never actually know
the exported file works correctly, just that a file got created.

Install once:
  pip install optimum[onnxruntime] transformers

Usage:
  python verify_onnx_inference.py --repo-id your-username/log-classifier-tiny \
      --onnx-dir ./onnx-export
"""

import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

TEST_TEXTS = [
    "User alice logged in successfully from 10.0.0.5",
    "Failed password for invalid user root from 203.0.113.9 port 51000 ssh2",
    "Unauthorized access attempt: privilege escalation detected for user svc-app",
    "DescribeInstances called by role svc-app",
]


def verify(repo_id: str, onnx_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print("Loading original PyTorch model...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    pt_model.eval()

    print(f"Loading ONNX model from {onnx_dir}...")
    onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)

    mismatches = 0
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        import torch
        with torch.no_grad():
            pt_logits = pt_model(**inputs).logits.numpy()
        pt_pred = int(np.argmax(pt_logits, axis=-1)[0])

        onnx_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        onnx_logits = onnx_model(**onnx_inputs).logits.detach().numpy()
        onnx_pred = int(np.argmax(onnx_logits, axis=-1)[0])

        agree = pt_pred == onnx_pred
        logit_diff = float(np.abs(pt_logits - onnx_logits).max())

        status = "MATCH" if agree else "MISMATCH"
        print(f"[{status}] pt={pt_pred} onnx={onnx_pred} max_logit_diff={logit_diff:.6f}  '{text[:50]}'")

        if not agree:
            mismatches += 1

    if mismatches > 0:
        print(f"\n{mismatches}/{len(TEST_TEXTS)} predictions disagree between PyTorch and ONNX. "
              f"Do not trust the exported model without investigating.")
        raise SystemExit(1)

    print(f"\nAll {len(TEST_TEXTS)} predictions match. ONNX export verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--onnx-dir", default="./onnx-export")
    args = parser.parse_args()
    verify(args.repo_id, args.onnx_dir)
