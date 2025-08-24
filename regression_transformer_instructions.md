Below is a turn‑key implementation workflow to adapt IBM’s Regression Transformer (RT) to your own continuous property (or properties). It’s opinionated, reproducible, and includes ready‑to‑run code and file scaffolding. Another capable AI (or engineer) can copy these files into a fresh repo and execute the pipeline end‑to‑end.

⸻

0) Why this works (one‑line rationale)

RT represents numbers as tokens and trains a single LM that does both regression and property‑conditioned generation—controlled by where you place masks/primers. You just add your property token(s), format your data, (fine‑)tune, and prompt.  ￼ ￼

⸻

1) Repo + environment

Clone IBM RT (dev code) and use their environment as base.  ￼

git clone https://github.com/IBM/regression-transformer.git
cd regression-transformer

# create env (uses XLNet; expect CUDA)
conda env create -f conda.yml
conda activate terminator

# install package in editable mode
pip install -e .

Why IBM RT repo? It already ships:
	•	scripts/run_language_modeling.py / scripts/eval_language_modeling.py (training & eval)
	•	scripts/create_vocabulary.py and terminator.tokenization.ExpressionBertTokenizer (number tokenization + chemical/protein vocabs)
	•	Working training configs (incl. “alternating CC” schedule) and examples.  ￼

License: MIT (adapting/commercial use permitted).  ￼

⸻

2) Project scaffold (drop these next to the cloned repo)

regression-transformer/
├─ adapters/
│  ├─ config.myprop.yaml
│  ├─ prepare_rt_data.py
│  ├─ train_rt.sh
│  ├─ eval_rt.sh
│  ├─ generate_with_gt4sd.py
│  ├─ Makefile

2.1 adapters/config.myprop.yaml

A single source of truth for your property schema.

# adapters/config.myprop.yaml
name: myprop                   # display name
token: "<myprop>"              # the literal primer token you want in the vocab
decimals: 3                    # fixed decimals to emit in data (e.g., 12.345)
scale: "zscore"                # one of: none | zscore | minmax
scale_args: {min: null, max: null, mean: null, std: null}  # optional preset
negative_ok: true              # property can be negative pre-scaling
units: "kcal/mol"              # for your notes; not used programmatically
domain:
  kind: "molecule"             # molecule | protein | reaction | generic
  sequence_column: "sequence"  # input sequence column name in your CSV
  representation: "SMILES"     # SMILES | SELFIES | AA | rxnSMILES | custom
data:
  train_csv: "data/train.csv"  # CSV with columns: sequence, myprop
  valid_csv: "data/valid.csv"
  out_dir: "runs/myprop"       # all artifacts land here
training:
  base_config: "configs/rt_small.json"   # model size from repo
  regime: "training_configs/qed_alternated_cc.json"  # alternating objective
  epochs: 8
  lr: 1.0e-4
  batch: 16
  block_size: 510
  eval_every_steps: 200
  save_steps: 1000
  seed: 13
generation:
  use_gt4sd: true              # recommended for easier generation APIs
  num_samples: 16
  fraction_to_mask: 0.2
  temperature: 1.5


⸻

3) Prepare data in RT line format

RT expects one sample per line:
<myprop>12.700 | <SEQUENCE> (multi‑property = multiple primers before |). The repo shows the same pattern and includes a helper to generate examples for QED; we’re generalizing it here.  ￼

3.1 adapters/prepare_rt_data.py

# adapters/prepare_rt_data.py
import argparse, csv, math, os, json, pathlib
from statistics import mean, pstdev

def load_yaml(path):
    import yaml
    with open(path) as f: return yaml.safe_load(f)

def _load_values(path, col_sequence, col_prop):
    seq, y = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            s = (row[col_sequence] or "").strip()
            v = row[col_prop]
            if s == "" or v is None or v == "":
                continue
            try:
                y.append(float(v))
                seq.append(s)
            except ValueError:
                continue
    return seq, y

def _scaler(y, mode, args):
    if mode == "none":
        return lambda v: v, {"mode":"none"}
    if mode == "zscore":
        mu = mean(y) if args.get("mean") is None else float(args["mean"])
        sd = pstdev(y) if args.get("std") is None else float(args["std"])
        sd = sd if sd > 0 else 1.0
        return lambda v: (v - mu) / sd, {"mode":"zscore","mean":mu,"std":sd}
    if mode == "minmax":
        lo = min(y) if args.get("min") is None else float(args["min"])
        hi = max(y) if args.get("max") is None else float(args["max"])
        span = (hi - lo) if hi > lo else 1.0
        return lambda v: (v - lo) / span, {"mode":"minmax","min":lo,"max":hi}
    raise ValueError(f"Unknown scale: {mode}")

def _format_value(v, decimals):
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(v)

def make_lines(csv_path, seq_col, prop_col, token, decimals, scale, scale_args):
    seq, y = _load_values(csv_path, seq_col, prop_col)
    fx, meta = _scaler(y, scale, scale_args or {})
    lines = []
    for s, v in zip(seq, y):
        val = _format_value(fx(v), decimals)
        # Spaces are fine; IBM scripts split on '|' in examples
        lines.append(f"{token}{val} | {s}")
    return lines, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    out = pathlib.Path(cfg["data"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # TRAIN
    lines, meta = make_lines(
        cfg["data"]["train_csv"],
        cfg["domain"]["sequence_column"],
        cfg["name"],  # property column name equals 'name'
        cfg["token"],
        cfg["decimals"],
        cfg["scale"],
        cfg.get("scale_args", {}),
    )
    (out / "train.txt").write_text("\n".join(lines) + "\n")

    # VALID
    vlines, _ = make_lines(
        cfg["data"]["valid_csv"],
        cfg["domain"]["sequence_column"],
        cfg["name"],
        cfg["token"],
        cfg["decimals"],
        cfg["scale"],
        cfg.get("scale_args", {}),
    )
    (out / "valid.txt").write_text("\n".join(vlines) + "\n")

    # Save scaling metadata for invertibility
    (out / "scaling.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(lines)} train / {len(vlines)} valid lines to {out}")

if __name__ == "__main__":
    main()

Run:

python adapters/prepare_rt_data.py --config adapters/config.myprop.yaml
# => runs/myprop/train.txt and runs/myprop/valid.txt


⸻

4) Build/extend the vocabulary

Use IBM’s helper to build a tokenizer vocabulary that includes your property token and all symbols needed for your sequences (SMILES/SELFIES/AA/etc.). The README shows the exact command and demonstrates how numbers become digit‑wise tokens with exponents (e.g., 0.3936 → ['_0_0_','_._','_3_-1_',…]).  ￼

# create a vocab from your TRAIN file (it auto-adds special tokens)
python scripts/create_vocabulary.py runs/myprop/train.txt runs/myprop/vocab.txt

(Optional) Sanity‑check tokenization (as in the README):  ￼

python - <<'PY'
from terminator.tokenization import ExpressionBertTokenizer
tok = ExpressionBertTokenizer.from_pretrained('runs/myprop')  # folder containing vocab.txt
x = "<myprop>0.3936|CCO"
print(tok.tokenize(x))
PY


⸻

5) Train / Fine‑tune (alternating objective)

The repo provides a working training invocation (we simply point it at our train/valid and our vocab). The “alternating CC” regime is what enables the RT’s dichotomy (numbers vs. text).  ￼

5.1 adapters/train_rt.sh

#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-adapters/config.myprop.yaml}

# Read a few fields from YAML with python (no external deps)
read_yaml () { python - "$1" "$2" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for key in sys.argv[2:]:
    for k in key.split('.'): cfg = cfg[k]
print(cfg)
PY
}
OUTDIR=$(read_yaml "$CFG" data.out_dir)
VOCAB="$OUTDIR/vocab.txt"
TRAIN="$OUTDIR/train.txt"
VALID="$OUTDIR/valid.txt"
BASE_CONFIG=$(read_yaml "$CFG" training.base_config)
REGIME=$(read_yaml "$CFG" training.regime)
EPOCHS=$(read_yaml "$CFG" training.epochs)
LR=$(read_yaml "$CFG" training.lr)
BATCH=$(read_yaml "$CFG" training.batch)
BLOCK=$(read_yaml "$CFG" training.block_size)
SEED=$(read_yaml "$CFG" training.seed)
EVAL_EVERY=$(read_yaml "$CFG" training.eval_every_steps)
SAVE_STEPS=$(read_yaml "$CFG" training.save_steps)

python scripts/run_language_modeling.py \
  --output_dir "$OUTDIR/model" \
  --config_name "$BASE_CONFIG" \
  --tokenizer_name "$(dirname "$VOCAB")" \
  --do_train --do_eval \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --save_total_limit 2 \
  --save_steps "$SAVE_STEPS" \
  --per_gpu_train_batch_size "$BATCH" \
  --evaluate_during_training \
  --eval_steps "$EVAL_EVERY" \
  --eval_data_file "$VALID" \
  --train_data_file "$TRAIN" \
  --line_by_line --block_size "$BLOCK" \
  --seed "$SEED" --logging_steps 100 --eval_accumulation_steps 2 \
  --training_config_path "$REGIME"

Run:

bash adapters/train_rt.sh adapters/config.myprop.yaml

The exact CLI mirrors the README’s “Training a model” section; only the paths change to your artifacts.  ￼

⸻

6) Evaluate (regression)

The repo also ships an evaluation script; here’s a thin wrapper.  ￼

6.1 adapters/eval_rt.sh

#!/usr/bin/env bash
set -euo pipefail
OUTDIR=${1:-runs/myprop}
python scripts/eval_language_modeling.py \
  --output_dir "$OUTDIR/model" \
  --eval_file "$OUTDIR/valid.txt" \
  --eval_accumulation_steps 2 \
  --param_path configs/qed_eval.json

Run:

bash adapters/eval_rt.sh runs/myprop

Notes:
	•	configs/qed_eval.json is a ready example in the repo—sufficient to exercise the evaluation pipeline.  ￼
	•	For domain‑specific metrics (e.g., RDKit validity, SA score), compute on the generated strings (see §7).

⸻

7) Generate sequences at a target property

You have two options:

Option A (recommended): GT4SD wrapper

IBM explicitly recommends using the GT4SD implementation for pretrained models and (lightweight) fine‑tuning/deployment. Their README shows code for property‑conditioned generation; we mirror it and make it configurable.  ￼ ￼

7.1 adapters/generate_with_gt4sd.py

import argparse, json
from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer, RegressionTransformerMolecules
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_sequence", required=True)  # e.g., SMILES/SELFIES
    ap.add_argument("--prop_token", required=True)     # e.g., <myprop>
    ap.add_argument("--target_value", type=float, required=True)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--fraction_to_mask", type=float, default=0.2)
    ap.add_argument("--alg_version", default="custom") # your fine-tuned tag
    args = ap.parse_args()

    cfg = RegressionTransformerMolecules(
        algorithm_version=args.alg_version,
        search="sample",
        temperature=args.temperature,
        sampling_wrapper={
            'property_goal': {args.prop_token: args.target_value},
            'fraction_to_mask': args.fraction_to_mask
        }
    )
    gen = RegressionTransformer(configuration=cfg, target=args.seed_sequence)
    items = list(gen.sample(args.num))
    print(json.dumps(items, indent=2))

if __name__ == "__main__":
    main()

Run (example):

python adapters/generate_with_gt4sd.py \
  --seed_sequence "CC(C#C)N(C)C(=O)NC1=CC=C(Cl)C=C1" \
  --prop_token "<myprop>" --target_value 1.25 --num 8

The exact pattern above is the same as the README’s GT4SD snippet (they show <esol>). If you fine‑tune via GT4SD, set --alg_version to your registered artifact.  ￼

Option B: Raw repo (advanced)

Prompt the HF model directly by constructing strings like
<myprop>1.25 | <SEED_SEQUENCE_WITH_MASKS> and using the RT tokenizer to feed XLNet; sampling wrappers mirror the training regime. Prefer Option A unless you need custom decoding.

⸻

8) Makefile (one‑command UX)

8.1 adapters/Makefile

CFG=adapters/config.myprop.yaml

.PHONY: prep vocab train eval gen

prep:
	python adapters/prepare_rt_data.py --config $(CFG)

vocab:
	python scripts/create_vocabulary.py runs/myprop/train.txt runs/myprop/vocab.txt

train:
	bash adapters/train_rt.sh $(CFG)

eval:
	bash adapters/eval_rt.sh runs/myprop

gen:
	python adapters/generate_with_gt4sd.py \
	  --seed_sequence "CCO" \
	  --prop_token "<myprop>" \
	  --target_value 0.0 \
	  --num 8

End‑to‑end:

make prep vocab train eval gen


⸻

9) Practical guidance (bakes in paper best‑practices)
	•	Scaling: Use zscore or minmax; RT learns numeric proximity from digit tokens, and scaling stabilizes training (especially across wide magnitudes).  ￼
	•	Decimals: Cap precision (e.g., 3–4 decimals). Excess decimals inflate sequence length with marginal value.  ￼
	•	Token choice: Pick a short, unique property token (e.g., <ΔG> → escape to <dG> if non‑ASCII). The vocab tool will add it; tokenize with ExpressionBertTokenizer.  ￼
	•	Objective: Use the alternating regime from training_configs for multitask behavior (regression vs. conditional generation).  ￼
	•	Speed: XLNet pretraining is slow; starting from a pretrained RT via GT4SD and then fine‑tuning is explicitly recommended by the authors.  ￼
	•	Domains already supported: molecules (SMILES/SELFIES), proteins (AA), reactions (rxnSMILES), polymers—pretrained variants exist in GT4SD.  ￼

⸻

10) Minimal acceptance tests (so an AI can self‑verify)
	1.	Tokenizer test: Given "<myprop>0.3936|CBr", token list includes number tokens like ['_0_0_', '_._', '_3_-1_', ...] + symbols; this matches the README demo.  ￼
	2.	Dry‑run training: With tiny train.txt/valid.txt (100–500 lines), run_language_modeling.py runs, saves checkpoints in runs/myprop/model/ and logs eval steps without error.  ￼
	3.	Eval: eval_language_model.py completes on valid.txt using configs/qed_eval.json.  ￼
	4.	Generation (GT4SD): Calling generate_with_gt4sd.py returns a list of strings (sequences) of length --num.  ￼

⸻

11) What to change for multi‑property use
	•	Put multiple primers before |:
"<p1>0.73 <p2>-3.5 | <SEQUENCE>"
	•	Set multiple columns in your CSV and emit them in fixed order in prepare_rt_data.py (duplicate the property block or generalize it to a list).
	•	Use the same training procedure; RT is multitask by design.  ￼

⸻

12) References you’ll rely on while implementing
	•	IBM RT README (training CLI, vocab creation, tokenizer demo).  ￼
	•	Nature Machine Intelligence / arXiv paper (method: numbers‑as‑tokens; regression ↔ generation via masking/priming).  ￼ ￼
	•	GT4SD docs & PyPI (RT inference/training wrappers; model hub).  ￼

⸻

TL;DR (execution order)

# 0) env
conda activate terminator && pip install -e .

# 1) data → RT format (+ scaling)
python adapters/prepare_rt_data.py --config adapters/config.myprop.yaml

# 2) vocab
python scripts/create_vocabulary.py runs/myprop/train.txt runs/myprop/vocab.txt

# 3) train (alternating regime)
bash adapters/train_rt.sh adapters/config.myprop.yaml

# 4) eval
bash adapters/eval_rt.sh runs/myprop

# 5) generate (GT4SD)
python adapters/generate_with_gt4sd.py --seed_sequence "CCO" --prop_token "<myprop>" --target_value 0.0 --num 8

That’s it. Swap in your property name/column, sequences, and scaling; the rest is boilerplate.