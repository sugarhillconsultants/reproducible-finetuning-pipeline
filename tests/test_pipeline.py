"""
tests/test_pipeline.py

Fast, dependency-light tests for the pipeline's core logic — the parts
that don't require transformers/torch/peft to be installed just to
verify correctness. Gates CI before any fine-tuning job runs.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))


def test_dataset_generation_is_balanced_enough():
    from build_dataset import build

    dataset = build(n_samples=200, seed=42)
    assert len(dataset["train"]) + len(dataset["test"]) == 200

    train_labels = dataset["train"]["label"]
    anomaly_rate = sum(train_labels) / len(train_labels)
    # Should be imbalanced (minority anomaly class) but not degenerate
    assert 0.15 < anomaly_rate < 0.55


def test_dataset_generation_is_deterministic():
    from build_dataset import build

    d1 = build(n_samples=50, seed=1)
    d2 = build(n_samples=50, seed=1)
    assert d1["train"]["text"] == d2["train"]["text"]
    assert d1["train"]["label"] == d2["train"]["label"]


def test_evaluate_gate_rejects_below_threshold(tmp_path):
    from gate import evaluate_metrics

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"eval_f1": 0.60, "eval_accuracy": 0.80}))

    assert evaluate_metrics(str(metrics_file), f1_threshold=0.75) is False


def test_evaluate_gate_approves_above_threshold(tmp_path):
    from gate import evaluate_metrics

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"eval_f1": 0.88, "eval_accuracy": 0.91}))

    assert evaluate_metrics(str(metrics_file), f1_threshold=0.75) is True


def test_evaluate_gate_boundary_is_inclusive(tmp_path):
    from gate import evaluate_metrics

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"eval_f1": 0.75, "eval_accuracy": 0.80}))

    # Exactly at threshold should pass (>= not >)
    assert evaluate_metrics(str(metrics_file), f1_threshold=0.75) is True
