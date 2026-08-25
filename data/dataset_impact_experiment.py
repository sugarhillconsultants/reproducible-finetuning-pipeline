"""
data/dataset_impact_experiment.py

Answers "does more labeled data actually improve the model" empirically
rather than assuming yes. Trains a fast TF-IDF + Logistic Regression
baseline (not the full LoRA fine-tune — this needs to run in seconds,
many times, to be a useful comparison tool) on both dataset versions
and compares metrics directly.

Install once:
  pip install scikit-learn datasets pandas

Usage:
  python dataset_impact_experiment.py --repo-id your-username/log-events \
      --versions v1 v2
"""

import argparse
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate_version(repo_id: str, version: str) -> dict:
    dataset = load_dataset(repo_id, revision=version)

    vectorizer = TfidfVectorizer(max_features=500)
    X_train = vectorizer.fit_transform(dataset["train"]["text"])
    X_test = vectorizer.transform(dataset["test"]["text"])
    y_train = dataset["train"]["label"]
    y_test = dataset["test"]["label"]

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "version": version,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


def run_comparison(repo_id: str, versions: list[str]):
    results = [evaluate_version(repo_id, v) for v in versions]

    print(f"\n{'Version':<10}{'Train rows':<12}{'Test rows':<12}{'Accuracy':<12}F1")
    print("-" * 58)
    for r in results:
        print(f"{r['version']:<10}{r['n_train']:<12}{r['n_test']:<12}"
              f"{r['accuracy']:<12.4f}{r['f1']:.4f}")

    if len(results) >= 2:
        delta_acc = results[-1]["accuracy"] - results[0]["accuracy"]
        delta_f1 = results[-1]["f1"] - results[0]["f1"]
        print(f"\n{results[0]['version']} -> {results[-1]['version']}: "
              f"accuracy {delta_acc:+.4f}, F1 {delta_f1:+.4f}")
        if delta_f1 < 0:
            print("Note: more data did NOT improve F1 here — worth investigating "
                  "why before assuming the larger dataset version is strictly better.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--versions", nargs="+", required=True)
    args = parser.parse_args()
    run_comparison(args.repo_id, args.versions)
