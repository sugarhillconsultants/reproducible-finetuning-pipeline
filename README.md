# Reproducible Fine-Tuning & Model Versioning Pipeline

**Status: fully verified end-to-end against live infrastructure** — every
stage below has actually run successfully on GitHub Actions against a
real Hugging Face account, not just locally or in theory. Getting here
took six distinct real bugs, each found from an actual failure log and
fixed, not anticipated in advance — the full account is in
[`docs/incidents.md`](docs/incidents.md).

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
training/finetune_lora.py      LoRA fine-tune (distilbert-base-uncased,
  (PEFT, CPU-only for CI)       CPU-feasible in CI — see limitations below)

        │
        ▼
training/gate.py           GATE: exits non-zero if F1 < threshold.
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
| `training/finetune_lora.py` | LoRA fine-tune of `distilbert-base-uncased` via PEFT — small enough for CPU/CI feasibility |
| `training/gate.py` | **The gate.** Exits 1 if F1 < 0.75, blocking registration (deliberately not named `evaluate.py` — see [`docs/incidents.md`](docs/incidents.md) #4 for why) |
| `training/register_model.py` | Merges the adapter (from full precision, not quantized weights) and pushes a tagged model version |
| `export/export_onnx.py` | Exports the registered model via `optimum-cli export onnx` |
| `export/verify_onnx_inference.py` | Runs the same inputs through PyTorch and ONNX, fails if predictions disagree |
| `tests/test_pipeline.py` | Tests dataset generation and the evaluation gate's reject/approve logic |
| `.github/workflows/pipeline.yml` | Orchestrates all of the above, gated at each step |
| `docs/dataset-impact-findings.md` | A real, run experiment — including an honest account of an earlier version that didn't work |
| `docs/incidents.md` | Six real bugs found running this against live infrastructure, each with the actual error and actual fix |

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

## Verified results

The full pipeline — build dataset → LoRA fine-tune → F1 gate → merge
and register → ONNX export → parity check — has actually run
successfully end to end on GitHub Actions. The final verification step
produced:

```
[MATCH] pt=0 onnx=0 max_logit_diff=0.000001  'User alice logged in successfully...'
[MATCH] pt=1 onnx=1 max_logit_diff=0.000000  'Failed password for invalid user root...'
[MATCH] pt=1 onnx=1 max_logit_diff=0.000000  'Unauthorized access attempt...'
[MATCH] pt=0 onnx=0 max_logit_diff=0.000000  'DescribeInstances called by role...'
All 4 predictions match. ONNX export verified.
```

Every prediction is correct given the actual content of each test
sentence (two normal, two anomalous), and every PyTorch/ONNX logit
difference is at floating-point noise level — genuine proof the export
preserved model behavior, not just confirmation a file got created.
Both the versioned dataset and the registered, tagged model exist for
real on the Hugging Face Hub, not just as local artifacts.

## Honest limitations

- **`distilbert-base-uncased` (~66M params), not a frontier-scale
  model.** Large enough to need a real base-model swap partway through
  development (see [`docs/incidents.md`](docs/incidents.md) #3), small
  enough that the full pipeline — including fine-tuning — completes on
  a GitHub Actions CPU runner in a few minutes. A production fine-tune
  of a real-sized model (7B+ params) would use QLoRA on a GPU (ZeroGPU
  on a Hugging Face Space, or a dedicated GPU runner), not a plain
  GitHub Actions CPU job — this repo optimizes for "the full pipeline
  actually runs end-to-end, for free, and has been proven to work," not
  for state-of-the-art model quality.
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
- **Getting here required six real fixes**, documented in full in
  [`docs/incidents.md`](docs/incidents.md) — a secret/variable mix-up,
  two rounds of token-scope gaps, a base-model swap forced by a
  tokenizer incompatibility, a Python import collision, a pip resolver
  bug, and an architecture-specific tokenizer/model mismatch. None of
  these were hypothetical "things that could go wrong" written in
  advance — each is a real failure this pipeline actually hit and
  actually recovered from.

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
python training/gate.py --metrics-file ./adapter-output/metrics.json --threshold 0.75 \
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
