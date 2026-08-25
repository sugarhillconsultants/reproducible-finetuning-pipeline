"""
training/gate.py

The gate: reads metrics.json produced by finetune_lora.py and exits
non-zero if the model doesn't clear the F1 threshold — mirroring the
same evaluate-before-register pattern from this author's other MLOps
project (Multi-Cloud MLOps Showcase's evaluate_and_register.py), just
targeting the Hugging Face Hub instead of Azure ML.

Note: this file is deliberately NOT named evaluate.py, even though
that's the more obvious name for what it does — Python adds a script's
own directory to sys.path when run directly, so a local evaluate.py
sitting next to finetune_lora.py would shadow the installed `evaluate`
pip package's `import evaluate` inside that file. Found this the hard
way; renaming was simpler and more robust than reordering imports.

Usage:
  python gate.py --metrics-file ./adapter-output/metrics.json --threshold 0.75
"""

import argparse
import json
import sys


def evaluate_metrics(metrics_file: str, f1_threshold: float) -> bool:
    with open(metrics_file) as f:
        metrics = json.load(f)

    f1 = metrics.get("eval_f1", 0.0)
    accuracy = metrics.get("eval_accuracy", 0.0)

    print(f"eval_f1={f1:.4f}  eval_accuracy={accuracy:.4f}  (threshold: f1 >= {f1_threshold})")

    if f1 < f1_threshold:
        print(f"REJECTED: F1 {f1:.4f} is below the {f1_threshold} threshold — not registering.")
        return False

    print("APPROVED: model clears the threshold, proceeding to registration.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    passed = evaluate_metrics(args.metrics_file, args.threshold)
    sys.exit(0 if passed else 1)
