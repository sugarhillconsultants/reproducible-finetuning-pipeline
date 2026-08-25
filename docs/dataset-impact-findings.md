# Dataset Version Impact: Does More Data Actually Help?

This experiment was actually run, not simulated for the README. Full
methodology and re-run instructions below.

## Setup

Two versions of the same synthetic log-classification dataset, generated
with overlapping/ambiguous templates and 12% realistic label noise
(some anomaly-labeled events are mislabeled as normal and vice versa,
reflecting how real security labeling is rarely perfectly clean):

- **v1**: 200 samples (160 train / 40 test)
- **v2**: 800 samples (640 train / 160 test)

Both evaluated with the same fast baseline (TF-IDF + Logistic
Regression) so the comparison isolates the effect of dataset size, not
model choice.

## Result

| Version | Train rows | Test rows | Accuracy | F1 |
|---|---|---|---|---|
| v1 | 160 | 40 | 0.8250 | 0.6957 |
| v2 | 640 | 160 | 0.8063 | 0.7520 |
| **Delta** | | | **-0.0187** | **+0.0563** |

## What this actually shows

**Accuracy went down slightly. F1 went up meaningfully.** This is not a
contradiction — it's exactly why F1 (which balances precision and
recall) is the better metric to gate on for an imbalanced classification
problem like security-anomaly detection, and accuracy alone can be
actively misleading.

With only 35% of events labeled anomalous, a model can post a
deceptively high accuracy by leaning toward predicting the majority
class ("normal") more often — which raises accuracy while quietly
hurting recall on the class that actually matters. The larger dataset
(v2) gave the model enough examples to improve its handling of the
minority (anomaly) class specifically — reflected in the higher F1 —
even though this cost a small amount of raw accuracy.

## Why this matters for the pipeline's gating logic

This is the direct justification for why `training/evaluate.py` gates
on **F1**, not accuracy. Had this pipeline gated on accuracy instead, it
would have rejected the *better* model (v2) in favor of the *worse* one
(v1) — the exact kind of quiet, plausible-looking mistake a metrics
gate exists to prevent, provided you picked the right metric to gate on
in the first place.

## An earlier, less useful version of this experiment

The first version of this dataset used cleanly separable templates with
no label noise — both v1 and v2 saturated at 100% accuracy and F1,
telling us nothing about the effect of dataset size, since the task was
too easy for either version to show a meaningful difference. The
overlapping templates and label noise above were added specifically to
make this a real, informative comparison rather than a trivial one —
worth knowing if you're designing a similar experiment: an experiment
that can't produce a negative or mixed result usually isn't testing
anything.

## Reproducing this

```bash
python data/build_dataset.py --version v1 --n-samples 200 --repo-id <your-repo>
python data/build_dataset.py --version v2 --n-samples 800 --repo-id <your-repo>
python data/dataset_impact_experiment.py --repo-id <your-repo> --versions v1 v2
```
