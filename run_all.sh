#!/usr/bin/env bash
# Full PRISM ML workflow, in order. Run from the repository root with the
# environment active (see README). The graph-network families need torch and
# are slow on CPU; pass --skip-graph to leave them out.
set -euo pipefail
cd "$(dirname "$0")"
GRAPH_FLAG="${1:-}"

python -m pytest -q
python scripts/benchmark_features.py                 # feature sets, fixed learner
python scripts/benchmark_models.py ${GRAPH_FLAG}     # every learner family
python scripts/holdout.py                            # selected model, reserved sets
python analysis/chemical_space.py                    # reactant diversity, train vs test
python scripts/train_final.py                        # refit on all data -> model/
