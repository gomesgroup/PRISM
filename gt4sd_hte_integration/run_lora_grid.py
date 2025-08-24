#!/usr/bin/env python
import itertools
import json
import os
import subprocess
from pathlib import Path

ART = Path("/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration/artifacts/rt_adapter_v4")
SCRIPT = "/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration/rt_adapter_train_v4.py"
PY = "/home/passos/mambaforge/envs/gt4sd-hte/bin/python"
ENV = os.environ.copy()
ENV["PYTHONPATH"] = "/home/passos/ml_measurable_hte_rates"
ENV["CUDA_VISIBLE_DEVICES"] = "0,1"

grid = {
    'target': ['attn_qv', 'attn_qvo'],
    'r': [4, 8, 16],
    'alpha': [8.0, 16.0, 32.0],
    'layers': [1, 2],
    'dropout': [0.0, 0.1],
}

combos = list(itertools.product(grid['target'], grid['r'], grid['alpha'], grid['layers'], grid['dropout']))

results = []
for (target, r, alpha, layers, drop) in combos:
    log = ART / f"lora_{target}_r{r}_a{int(alpha)}_L{layers}_d{drop}.log"
    cmd = [PY, SCRIPT,
           "--use-lora",
           "--lora-target", target,
           "--lora-r", str(r),
           "--lora-alpha", str(alpha),
           "--lora-dropout", str(drop),
           "--lora-last-layers", str(layers),
           "--gpus", "0,1",
    ]
    print("Running:", " ".join(cmd))
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    # After run, read stack.json for rate R2
    stack_path = ART / "stack.json"
    rate_r2 = None
    if stack_path.exists():
        try:
            data = json.loads(stack_path.read_text())
            rate_r2 = float(data.get('rate', {}).get('r2', None))
        except Exception:
            rate_r2 = None
    results.append({
        'target': target, 'r': r, 'alpha': alpha, 'layers': layers, 'dropout': drop, 'rate_r2': rate_r2,
        'log': str(log)
    })

summary = ART / "lora_grid_summary.json"
summary.write_text(json.dumps(sorted(results, key=lambda x: (x['rate_r2'] is None, -(x['rate_r2'] or -1e9))), indent=2))
print("Wrote:", summary)


