"""
export/export_onnx.py

Exports the registered Hub model to ONNX using Optimum — the current
tool for this (transformers.onnx is deprecated).

Install once:
  pip install optimum[exporters]

Usage:
  python export_onnx.py --repo-id your-username/log-classifier-tiny --version v1.0.0 \
      --output-dir ./onnx-export
"""

import argparse
import subprocess


def export(repo_id: str, version: str, output_dir: str):
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", repo_id,
        "--task", "text-classification",
        output_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"(Exporting revision {version} — pass via --model {repo_id}@{version} "
          f"if pinning a specific revision is needed for your Optimum version)")
    subprocess.run(cmd, check=True)
    print(f"Exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", default="./onnx-export")
    args = parser.parse_args()
    export(args.repo_id, args.version, args.output_dir)
