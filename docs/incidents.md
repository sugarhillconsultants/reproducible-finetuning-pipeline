# Real Incidents Encountered Running This Pipeline

Same rationale as the companion project
([Multi-Cloud MLOps Showcase](https://github.com/sugarhillconsultants/multi-cloud-mlops-showcase)):
a pipeline that "just worked" on the first run would be less honest,
and less useful to anyone else hitting the same issues, than an
accurate account of what actually broke and how it got fixed. Every
issue below was diagnosed from real GitHub Actions logs, not guessed —
and several took multiple attempts to actually resolve, which is left
in below rather than cleaned up into a tidier-looking narrative.

## 1. `HF_TOKEN` saved as a GitHub Variable instead of a Secret

The very first `build-dataset` run failed with `401 Unauthorized`.
The workflow's `env:` block showed `HF_TOKEN:` with nothing after it —
GitHub always masks a real secret's value as `***` in logs, so a
genuinely blank line meant the reference wasn't resolving to anything
at all. Root cause: `HF_TOKEN` had been added under the repo's
**Variables** tab, not **Secrets** — an easy mix-up since they sit on
adjacent tabs in GitHub's UI, and `${{ secrets.HF_TOKEN }}` will
silently resolve to an empty string rather than error if no matching
secret exists. Fixed by deleting it from Variables and adding it fresh
under Secrets. **Rule of thumb going forward: anything that's a
credential goes in Secrets; anything that's just configuration (repo
names, IDs) goes in Variables.**

## 2. Fine-grained token scope missing the dataset repo, then the model repo

Once the token was wired correctly, the same push failed differently:
`403 Forbidden` on Hugging Face's `xet-write-token` endpoint — the
token was authenticating fine but lacked write permission for this
specific repo. The token had been scoped to a Space, but not to the
dataset repo (`datasets/oromeop/log-events`). Fixed by editing the
token's permissions to add read/write on the dataset repo — then the
exact same class of failure recurred one stage later when the pipeline
reached model registration, because the token also hadn't been scoped
to the model repo (`oromeop/log-classifier-tiny`). Both had to be added
before the full pipeline could run without stopping.

## 3. `sentencepiece` missing, then found to be the wrong diagnosis entirely

The `finetune` job failed loading the tokenizer for `prajjwal1/bert-tiny`:
`ValueError: ... You need to have sentencepiece or tiktoken installed
to convert a slow tokenizer to a fast one.` Added `sentencepiece` to the
job's `pip install` line. The exact same error recurred on the next run
— and adding `use_fast=False` to the `AutoTokenizer.from_pretrained()`
call *also* didn't help, which was the real signal that this wasn't
actually a missing-dependency problem. `prajjwal1/bert-tiny` is a
community model that predates the modern fast-tokenizer-file convention
on the Hub — it likely only ships a legacy `vocab.txt`, and newer
`transformers` versions apparently no longer let `use_fast=False`
bypass the conversion attempt the way older versions did. The actual
fix was switching the base model entirely, to `distilbert-base-uncased`
— a well-established model guaranteed to ship a proper `tokenizer.json`
— which also required updating the LoRA `target_modules` from BERT's
attention-layer names (`query`, `value`) to DistilBERT's (`q_lin`,
`v_lin`), since the two architectures name their layers differently.

## 4. A local file named `evaluate.py` shadowed the installed `evaluate` package

After fix #3, `finetune` got genuinely further — the model loaded,
LoRA applied correctly, and training actually ran, with loss dropping
across real steps. It then failed with
`AttributeError: module 'evaluate' has no attribute 'load'` inside
`compute_metrics()`, which calls `evaluate.load("accuracy")` from the
Hugging Face `evaluate` metrics library. The real cause: this repo also
has a file at `training/evaluate.py` (the F1-gating script), and Python
prepends a script's own directory to `sys.path` when run directly with
`python training/finetune_lora.py`. Since both files live in
`training/`, `import evaluate` inside `finetune_lora.py` resolved to
the *local* `training/evaluate.py` file instead of the installed pip
package — and that local file, naturally, has no `.load()` function.
Fixed by renaming the local gate script to `training/gate.py`,
updating every reference to it (`pipeline.yml`, `tests/test_pipeline.py`,
this repo's README). This is arguably the most instructive bug in this
list: a completely valid, well-named local file broke a totally
unrelated import purely through a Python path-resolution quirk that
has nothing to do with either file's actual logic.

## 5. pip's resolver backtracked into broken, years-old package versions

The final job, `export-and-verify-onnx`, failed installing dependencies:
`pip install optimum[exporters] optimum[onnxruntime] transformers torch
sentencepiece` printed
`WARNING: optimum 2.3.0 does not provide the extra 'exporters'`, then
proceeded to backtrack through `optimum-1.0.0`, `0.1.3`, `0.1.2`,
`0.1.1`, and finally tried to build `optimum-0.1.0` from source, which
failed outright (`FileNotFoundError: optimum/version.py` — that release's
own packaging was broken). Current Optimum has apparently split ONNX
export support into a separate `optimum-onnx` package, and passing two
separate `optimum[...]` install targets in one pip command created an
ambiguous requirement that sent the resolver hunting through years of
old releases trying to satisfy both extras at once. Fixed by combining
the extras into a single spec and adding the new package name
explicitly: `pip install "optimum[exporters,onnxruntime]" optimum-onnx
transformers torch sentencepiece`.

## 6. DistilBERT's tokenizer emits a field its own model's forward() rejects

With the install fixed, ONNX export itself succeeded — but the
verification step failed: `TypeError: DistilBertForSequenceClassification
.forward() got an unexpected keyword argument 'token_type_ids'`.
DistilBERT's tokenizer is `BertTokenizerFast` under the hood (DistilBERT
never shipped its own tokenizer class) and dutifully produces
`token_type_ids`, just like BERT's tokenizer does — but DistilBERT's
architecture doesn't use segment/token-type embeddings at all, so its
`forward()` method never accepts that argument. Passing the tokenizer's
full output dict straight into the model via `**inputs` throws. Fixed
by explicitly popping `token_type_ids` from the tokenized inputs before
calling either the PyTorch or ONNX model.

## After all six fixes: a genuine, verified pass

```
[MATCH] pt=0 onnx=0 max_logit_diff=0.000001  'User alice logged in successfully...'
[MATCH] pt=1 onnx=1 max_logit_diff=0.000000  'Failed password for invalid user root...'
[MATCH] pt=1 onnx=1 max_logit_diff=0.000000  'Unauthorized access attempt...'
[MATCH] pt=0 onnx=0 max_logit_diff=0.000000  'DescribeInstances called by role...'
All 4 predictions match. ONNX export verified.
```

Every prediction is correct given the actual content of each test
sentence, and every logit difference between PyTorch and ONNX is at
floating-point noise level — genuine proof the export preserved model
behavior, not just confirmation that a file got created.

## The throughline

Unlike the companion Azure project, where most incidents were
infrastructure/IAM-shaped (missing role assignments, OIDC subject
mismatches), every incident here was a **Python/ML-tooling-shaped**
problem: a namespace collision, a tokenizer-conversion edge case, a
dependency resolver going sideways, an architecture-specific API
mismatch. None of these are things a code reviewer would catch by
reading the code — they only surface by actually running the pipeline
against real infrastructure, which is the entire argument for why this
project exists: a fine-tuning script that has never been executed
end-to-end isn't reproducible, it's just unverified.
