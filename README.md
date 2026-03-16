# autoresearch-gdpa1

Autonomous model development for predicting antibody developability attributes. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted from LLM pretraining to multi-task regression on the GDPa1 dataset.

## The idea

Give an AI agent a dataset of 246 therapeutic antibodies with experimentally measured developability properties, a baseline model, and let it experiment autonomously overnight on an A100. It modifies the code, trains, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a much better model.

The agent modifies `train.py` — everything else is fixed.

## What we're optimizing

**Mean Spearman rank correlation across 5 antibody developability targets** (higher is better).

Spearman measures how well the model *ranks* antibodies by each property, not how close the absolute predictions are. This is the right metric because in practice you use the model to rank candidates — "which antibodies are likely least sticky?" or "which will express best?" — not to predict exact assay values.

### The 5 targets

| Target | What it measures | Why it matters | N valid |
|---|---|---|---|
| **HIC** | Hydrophobicity (chromatographic retention) | Sticky antibodies aggregate, have poor PK | 242 |
| **Tm2** | Thermal stability (second melting point) | Low Tm = unstable, short shelf life | 193 |
| **PR_CHO** | Polyreactivity to CHO cell extract | High polyreactivity = off-target binding, fast clearance | 197 |
| **AC-SINS pH 7.4** | Self-interaction propensity | High self-interaction = viscosity, aggregation | 242 |
| **Titer** | Expression yield in CHO cells | Low titer = expensive/infeasible manufacturing | 239 |

These properties determine whether a therapeutic antibody candidate can actually be manufactured and dosed as a drug. A model that accurately ranks antibodies on these would let you filter candidates computationally before running expensive wet lab assays.

### Cross-validation

Folds are split by **sequence similarity clustering** (`hierarchical_cluster_fold`), not randomly. The validation set contains antibodies from sequence families the model has never seen during training. This is harder than random CV but reflects the real use case: predicting developability for *new* antibody sequences.

## How it works

### The loop

An AI coding agent runs in a terminal with access to this repo, following `program.md`:

1. **Edit `train.py`** — change model, features, hyperparameters, anything
2. **`git commit`**
3. **`modal run modal_run.py > run.log 2>&1`** — runs `train.py` on an A100 via Modal
4. **Extract result** — `grep "^mean_spearman:" run.log`
5. **If improved**: keep the commit, advance the branch
6. **If not**: `git reset --hard`, discard
7. **Log to `results.tsv`** either way
8. **Repeat forever** until manually stopped

The agent never modifies `prepare.py` (data loading + evaluation) or `modal_run.py` (GPU execution). This separation ensures the metric can't be gamed.

### What `train.py` does

Each run performs 5-fold cross-validation: train on ~196 samples, predict on ~50, collect all out-of-fold predictions, pass to `evaluate()`. The agent can change anything inside: swap the MLP for XGBoost, add physicochemical features, fine-tune ESM-2, try ensembling, etc.

### The branch structure

The agent creates a branch like `autoresearch/mar15`. Successful experiments advance the branch tip. Failed experiments get reverted. So `git log` shows the chain of improvements, and `results.tsv` has the full history including discards and crashes.

### Available packages in the Modal image

torch, numpy, pandas, scipy, scikit-learn, transformers, fair-esm, xgboost, lightgbm

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), [Modal](https://modal.com/) account (for A100 GPU execution).

```bash
# 1. Install dependencies
uv sync

# 2. Authenticate with Modal (one-time)
modal setup

# 3. Verify data and encoding (one-time sanity check)
uv run prepare.py

# 4. Run a single training experiment on A100
modal run modal_run.py
```

## Running the agent

Spin up Claude Code (or similar) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will establish a baseline, then autonomously iterate on `train.py` — trying different architectures, features, hyperparameters, etc. — logging results to `results.tsv`.

## Project structure

```
prepare.py      — constants, data loading, encoding, evaluation (do not modify)
train.py        — model + training loop (agent modifies this)
modal_run.py    — runs train.py on Modal A100 GPU (do not modify)
program.md      — agent instructions
data/GDPa1.csv  — dataset (246 antibodies, 5 targets)
pyproject.toml  — dependencies
```

## License

MIT
