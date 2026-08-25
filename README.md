# Reproducible Fine-Tuning & Model Versioning Pipeline

A complete, working pipeline demonstrating what actually makes ML
model development *reproducible*: versioned datasets, PEFT/LoRA
fine-tuning, a metrics-gated registration step that can reject a model,
and ONNX export with an actual correctness check — not just "a file
got created."

Built as a companion piece to
[Multi-Cloud MLOps Showcase](https://github.com/sugarhillconsultants/multi-cloud-mlops-showcase),
which covers infrastructure and deployment. This project goes deep on
the ML engineering discipline side: dataset versioning, evaluation
rigor, and format portability.

## Why this exists

Most fine-tuning tutorials show "load a model, train it, done." This
shows the parts that determine whether that result is trustworthy and
reproducible: is the training data versioned so you can prove what a
model was actually trained on? Does a worse model actually get rejected,
or does everything just get pushed regardless of quality? Does an
exported ONNX model actually behave the same as the original, or did
anyone check?

## Pipeline

```
data/build_dataset.py          Builds + versions a labeled dataset,
  (v1, v2, ... tagged)          pushes to Hub with a real dataset card

        │
        ▼
training/finetune_lora.py      LoRA fine-tune (tiny model, CPU-feasible
  (PEFT, CPU-only for CI)       in CI — see "Honest limitations" below)

        │
        ▼
training/evaluate.py           GATE: exits non-zero if F1 < threshold.
  (F1 threshold: 0.75)          A worse model genuinely cannot proceed.

        │  (only if approved)
        ▼
training/register_model.py     Merges adapter into full-precision base,
  (merge + push + tag)          pushes to Hub, tagged with version

        │
        ▼
export/export_onnx.py          Optimum export (transformers.onnx is
  export/verify_onnx_inference.py   deprecated) + PyTorch-vs-ONNX
                                 prediction parity check
```

## What's actually in this repo

| Path | What it does |
|---|---|
| `data/build_dataset.py` | Generates a labeled log-classification dataset with realistic class imbalance (35% anomaly rate) and label noise (12%), versions it on the Hub with a real dataset card |
| `data/dataset_impact_experiment.py` | Empirically compares two dataset versions — see [`docs/dataset-impact-findings.md`](docs/dataset-impact-findings.md) for the actual result |
| `training/finetune_lora.py` | LoRA fine-tune of `prajjwal1/bert-tiny` via PEFT — deliberately tiny for CPU/CI feasibility |
| `training/evaluate.py` | **The gate.** Exits 1 if F1 < 0.75, blocking registration |
| `training/register_model.py` | Merges the adapter (from full precision, not quantized weights) and pushes a tagged model version |
| `export/export_onnx.py` | Exports the registered model via `optimum-cli export onnx` |
| `export/verify_onnx_inference.py` | Runs the same inputs through PyTorch and ONNX, fails if predictions disagree |
| `tests/test_pipeline.py` | Tests dataset generation and the evaluation gate's reject/approve logic |
| `.github/workflows/pipeline.yml` | Orchestrates all of the above, gated at each step |
| `docs/dataset-impact-findings.md` | A real, run experiment — including an honest account of an earlier version that didn't work |

## A real finding, not a simulated one

The dataset-version comparison actually ran. First attempt: both
versions saturated at 100% accuracy/F1 — the synthetic templates were
too cleanly separable to show anything. Rather than present that as if
it were interesting, the dataset generator was reworked to include
overlapping templates and realistic label noise. Second attempt: **more
data increased F1 (+0.056) while accuracy went slightly down
(-0.019)** — a genuinely useful result, since it directly demonstrates
why the pipeline gates on F1 rather than accuracy for an imbalanced
problem. Full writeup: [`docs/dataset-impact-findings.md`](docs/dataset-impact-findings.md).

## Honest limitations

- **`prajjwal1/bert-tiny` is deliberately tiny (~4M params)** so the
  full pipeline — including fine-tuning — can run on a GitHub Actions
  CPU runner in a few minutes. A production fine-tune of a real-sized
  model (7B+ params) would use QLoRA on a GPU (ZeroGPU on a Hugging
  Face Space, or a dedicated GPU runner), not a plain GitHub Actions
  CPU job — this repo optimizes for "the full pipeline actually runs
  end-to-end for free in CI," not for state-of-the-art model quality.
- **The dataset is synthetic and templated**, not real log data. It's
  built to be *just* realistic enough (class imbalance, label noise,
  ambiguous overlapping cases) to make the evaluation-gating and
  dataset-comparison exercises meaningful, not to represent real-world
  log diversity.
- **The ONNX export step exports the merged model**, not a quantized
  version — adding INT8 quantization on top (via ONNX Runtime's
  quantization tooling) would be a reasonable next step but isn't
  covered here, to keep the export-verification story focused on one
  claim (PyTorch and ONNX agree) rather than several at once.

## Running it yourself

```bash
# 1. Build and version two dataset sizes
python data/build_dataset.py --version v1 --n-samples 200 --repo-id <your-repo> --push
python data/build_dataset.py --version v2 --n-samples 800 --repo-id <your-repo> --push

# 2. See whether more data actually helped (spoiler: it's not simple)
python data/dataset_impact_experiment.py --repo-id <your-repo> --versions v1 v2

# 3. Fine-tune with LoRA
python training/finetune_lora.py --dataset-repo <your-repo> --dataset-version v1 \
    --output-dir ./adapter-output

# 4. Gate on quality, then register only if it passes
python training/evaluate.py --metrics-file ./adapter-output/metrics.json --threshold 0.75 \
  && python training/register_model.py --adapter-path ./adapter-output/final-adapter \
       --repo-id <your-model-repo> --version v1.0.0

# 5. Export to ONNX and verify it actually matches
python export/export_onnx.py --repo-id <your-model-repo> --version v1.0.0 --output-dir ./onnx-export
python export/verify_onnx_inference.py --repo-id <your-model-repo> --onnx-dir ./onnx-export
```

## What I'd add next

- INT8 quantization of the ONNX export, with a size/latency comparison
  documented the same way the dataset-impact experiment is.
- A real QLoRA run against a larger model on a GPU (ZeroGPU Space),
  documented alongside the CPU/tiny-model CI version for comparison.
- Wiring the model registration step to also trigger a redeploy of
  the [Multi-Cloud MLOps Showcase](https://github.com/sugarhillconsultants/multi-cloud-mlops-showcase)
  serving app, closing the loop between these two projects.
