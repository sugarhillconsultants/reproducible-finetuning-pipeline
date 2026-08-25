"""
data/build_dataset.py

Builds a labeled log-classification dataset (normal vs. security_anomaly),
pushes it to the Hugging Face Hub as a versioned dataset with a real
dataset card — not just raw files with no documentation.

Install once:
  pip install datasets huggingface_hub

Usage:
  python build_dataset.py --version v1 --n-samples 200 --repo-id your-username/log-events
  python build_dataset.py --version v2 --n-samples 800 --repo-id your-username/log-events
"""

import argparse
import random
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from huggingface_hub import create_tag

LABELS = ["normal", "security_anomaly"]

NORMAL_TEMPLATES = [
    "User {user} logged in successfully from {ip}",
    "DescribeInstances called by role {user}",
    "GetObject from bucket {bucket} by {user}",
    "Scheduled backup job completed successfully for {bucket}",
    "AssumeRole succeeded for CI pipeline service account {user}",
    "CreateAccessKey called for user {user} during onboarding",
    "AttachUserPolicy granted ReadOnlyAccess to user {user}",
]

ANOMALY_TEMPLATES = [
    "Failed password for invalid user root from {ip} port {port} ssh2",
    "Unauthorized access attempt: privilege escalation detected for user {user}",
    "CreateAccessKey called for user root outside business hours",
    "AttachUserPolicy granted AdministratorAccess to new user {user}",
    "DeleteTrail called disabling CloudTrail logging by {user}",
    "CreateAccessKey called for user {user} during onboarding",
    "AttachUserPolicy granted PowerUserAccess to user {user}",
]

LABEL_NOISE_RATE = 0.12  # deliberately imperfect labels — real security data is rarely clean

USERS = ["alice", "bob", "svc-app", "svc-ci", "root", "carol"]
BUCKETS = ["app-logs", "config-store", "backups", "reports"]
IPS = ["10.0.0.5", "203.0.113.9", "198.51.100.23", "192.0.2.44"]


def generate_record(rng: random.Random) -> dict:
    is_anomaly = rng.random() < 0.35  # deliberately imbalanced, like real security data
    template = rng.choice(ANOMALY_TEMPLATES if is_anomaly else NORMAL_TEMPLATES)
    text = template.format(
        user=rng.choice(USERS), ip=rng.choice(IPS),
        bucket=rng.choice(BUCKETS), port=rng.randint(40000, 60000),
    )
    label = 1 if is_anomaly else 0
    if rng.random() < LABEL_NOISE_RATE:  # realistic imperfect labeling
        label = 1 - label
    return {"text": text, "label": label}


def build(n_samples: int, seed: int = 42) -> DatasetDict:
    rng = random.Random(seed)
    records = [generate_record(rng) for _ in range(n_samples)]

    features = Features({"text": Value("string"), "label": ClassLabel(names=LABELS)})
    full = Dataset.from_list(records, features=features)

    split = full.train_test_split(test_size=0.2, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def write_dataset_card(repo_id: str, version: str, dataset: DatasetDict, local_path: str):
    train_labels = dataset["train"]["label"]
    anomaly_rate = sum(train_labels) / len(train_labels)

    card = f"""---
license: mit
task_categories:
  - text-classification
tags:
  - cybersecurity
  - log-analysis
  - synthetic
---

# Log Event Classification Dataset — {version}

Synthetic, templated log lines labeled `normal` (0) or `security_anomaly` (1).
Built for demonstrating a reproducible fine-tuning pipeline, not for
production security use — see the parent repo's honest caveats.

## Version: {version}

- Train rows: {len(dataset['train'])}
- Test rows: {len(dataset['test'])}
- Anomaly rate (train): {anomaly_rate:.1%}

## Intended use

Small enough to fine-tune a tiny transformer classifier on CPU in CI,
in minutes, while still exhibiting realistic class imbalance (anomalies
are deliberately the minority class, as in real security log data).

## Known limitations

- Synthetic/templated, not real log data — sentence structure is far
  more uniform than genuine logs.
- Small vocabulary; a model trained on this will not generalize to
  real-world log formats without further fine-tuning on real data.
"""
    with open(f"{local_path}/README.md", "w") as f:
        f.write(card)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. v1, v2")
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    dataset = build(args.n_samples)
    local_path = f"./dataset-versions/{args.version}"
    dataset.save_to_disk(local_path)
    write_dataset_card(args.repo_id, args.version, dataset, local_path)

    print(f"Built dataset {args.version}: {dataset}")

    if args.push:
        dataset.push_to_hub(args.repo_id, commit_message=f"Build dataset {args.version}")
        create_tag(repo_id=args.repo_id, tag=args.version, repo_type="dataset", exist_ok=True)
        print(f"Pushed and tagged as {args.version}. Load later with:")
        print(f'  load_dataset("{args.repo_id}", revision="{args.version}")')
