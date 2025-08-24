#!/usr/bin/env python
"""
Small HPO for LoRA on RT v4 adapter:
- Tries attention targets {attn_qv, attn_qvo} and ffn as baseline.
- r in {4,8,16}, alpha in {8,16,32}, layers in {1,2}, dropout in {0.0,0.1} (small grid).
- For each config, updates toggles in rt_adapter_train_v4.py via env vars and runs one training.
- Parses final metrics from stack.json and records best by rate-space R2.
"""

import json
import os
import subprocess
from pathlib import Path
from itertools import product

ROOT = Path("/home/passos/ml_measurable_hte_rates")
TRAIN = ROOT / "gt4sd_hte_integration" / "rt_adapter_train_v4.py"
ART = ROOT / "gt4sd_hte_integration" / "artifacts" / "rt_adapter_v4"


def run_one(target: str, r: int, alpha: int, layers: int, dropout: float) -> float:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["USE_LORA"] = "1"
    env["LORA_TARGET"] = target
    env["LORA_R"] = str(r)
    env["LORA_ALPHA"] = str(alpha)
    env["LORA_NUM_LAST_LAYERS"] = str(layers)
    env["LORA_DROPOUT"] = str(dropout)
    env["WRAP_DATA_PARALLEL"] = "1"
    log = ART / f"hpo_{target}_r{r}_a{alpha}_L{layers}_d{dropout}.log"
    cmd = ["/home/passos/mambaforge/envs/gt4sd-hte/bin/python", str(TRAIN)]
    try:
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return -1e9
    # Read stack.json
    try:
        stack = json.loads((ART / "stack.json").read_text())
        return float(stack.get("rate", {}).get("r2", -1e9))
    except Exception:
        return -1e9


def main() -> None:
    grid_target = ["attn_qv", "attn_qvo", "ffn"]
    grid_r = [4, 8, 16]
    grid_alpha = [8, 16, 32]
    grid_layers = [1, 2]
    grid_dropout = [0.0, 0.1]

    results = []
    for target, r, alpha, layers, d in product(grid_target, grid_r, grid_alpha, grid_layers, grid_dropout):
        r2 = run_one(target, r, alpha, layers, d)
        results.append({"target": target, "r": r, "alpha": alpha, "layers": layers, "dropout": d, "rate_r2": r2})
        # Save incremental results
        (ART / "lora_hpo_results.json").write_text(json.dumps(results, indent=2))

    # Select best
    best = max(results, key=lambda x: x["rate_r2"]) if results else None
    if best is not None:
        (ART / "lora_hpo_best.json").write_text(json.dumps(best, indent=2))
        print("BEST:", best)
    else:
        print("No successful HPO runs")


if __name__ == "__main__":
    main()


