# CheMeleon HTE Rate Prediction Optimization Plan

## Current Baseline (Single Model)
- **Dataset**: `corrected_hte_rates_each_8_optuna_finite_lnk.csv` (1121 rows, filtered for finite HTE_lnk)
- **Target**: HTE_lnk (log-scale reaction rates)
- **Model**: CheMeleon-initialized ChemProp with multicomponent MPNN
- **Metrics**: R² = 0.862, MAE = 0.223, MSE = 0.0985

## Sequential Optimization Strategy

### Phase 1: Quick Wins (1-2 hours each on GPU 3)

#### 1. Ensemble Training [opt-1-ensemble]
**Rationale**: Ensembles consistently improve all metrics with minimal effort
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_ens5 \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --epochs 100 --patience 10 --warmup-epochs 5 \
  --batch-size 512 -n 16 \
  --accelerator gpu --devices 1 \
  --ensemble-size 5 \
  --metrics r2 mae rmse \
  --log
```
**Expected**: R² → 0.88+, MAE → 0.20

#### 2. Larger FFN Architecture [opt-2-ffn-arch]
**Rationale**: CheMeleon provides rich features; larger head can better map to targets
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_bigffn \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --epochs 100 --patience 10 --warmup-epochs 5 \
  --batch-size 256 -n 16 \
  --accelerator gpu --devices 1 \
  --ffn-hidden-dim 1024 --ffn-num-layers 3 \
  --dropout 0.2 --batch-norm \
  --metrics r2 mae rmse \
  --log
```
**Expected**: R² → 0.87+, better generalization

### Phase 2: Reaction Featurization (30 min each)

#### 3. Reaction Mode Comparison [opt-3-rxn-modes]
**Rationale**: Different modes capture different reaction aspects
```bash
# REAC_PROD mode
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_reac_prod \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --rxn-mode REAC_PROD \
  --epochs 50 --patience 5 --warmup-epochs 2 \
  --batch-size 512 -n 16 \
  --accelerator gpu --devices 1 \
  --metrics r2 mae rmse \
  --log

# PROD_DIFF mode
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_prod_diff \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --rxn-mode PROD_DIFF \
  --epochs 50 --patience 5 --warmup-epochs 2 \
  --batch-size 512 -n 16 \
  --accelerator gpu --devices 1 \
  --metrics r2 mae rmse \
  --log
```
**Expected**: One mode may outperform baseline by 1-2% R²

#### 4. Atom Featurizer Variants [opt-4-featurizers]
**Rationale**: Resonance-invariant features for chemical reactions
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_rigr \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --multi-hot-atom-featurizer-mode RIGR \
  --keep-h \
  --epochs 50 --patience 5 --warmup-epochs 2 \
  --batch-size 512 -n 16 \
  --accelerator gpu --devices 1 \
  --metrics r2 mae rmse \
  --log
```

### Phase 3: Feature Engineering (45 min)

#### 5. Add Numeric Descriptors [opt-5-descriptors]
**Rationale**: Controls column contains reaction conditions
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_descriptors \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --descriptors-columns Controls predicted_bias \
  --epochs 100 --patience 10 --warmup-epochs 5 \
  --batch-size 512 -n 16 \
  --accelerator gpu --devices 1 \
  --metrics r2 mae rmse \
  --log
```
**Expected**: R² → 0.87+ if descriptors are informative

### Phase 4: Training Optimization (1 hour)

#### 6. Learning Rate Schedule [opt-6-lr-schedule]
**Rationale**: Better convergence with optimized schedule
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_lr_opt \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --init-lr 1e-4 --max-lr 3e-3 --final-lr 1e-4 \
  --epochs 200 --patience 20 --warmup-epochs 10 \
  --batch-size 256 -n 16 \
  --accelerator gpu --devices 1 \
  --metrics r2 mae rmse \
  --log
```

### Phase 5: Robust Evaluation (2 hours)

#### 7. K-Fold Cross-Validation [opt-7-kfold]
**Rationale**: Reduce variance, get confidence intervals
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_kfold5 \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --epochs 100 --patience 10 --warmup-epochs 5 \
  --batch-size 512 -n 16 \
  --accelerator gpu --devices 1 \
  -k 5 --save-smiles-splits \
  --metrics r2 mae rmse \
  --log
```
**Expected**: Mean R² ± std across folds

### Phase 6: Automated Search (4-6 hours)

#### 8. Hyperparameter Optimization [opt-8-hpopt]
**Rationale**: Systematic search for optimal configuration
```bash
CUDA_VISIBLE_DEVICES=3 chemprop hpopt \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/hpopt_lnk \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --search-params ffn_hidden_dim=256,512,1024 \
  --search-params ffn_num_layers=2,3 \
  --search-params dropout=0.0,0.1,0.2,0.3 \
  --search-params rxn_mode=REAC_DIFF,REAC_PROD,PROD_DIFF \
  --search-params max_lr=0.001,0.002,0.003 \
  --search-params batch_size=256,512 \
  --epochs 50 --patience 5 --warmup-epochs 2 \
  --num-iters 20 \
  --accelerator gpu --devices 1 \
  --metrics r2 mae rmse
```

### Phase 7: Final Model (2 hours)

#### 9. Combine Best Settings [opt-9-combine]
**Analysis Script**: Analyze results from phases 1-8
```python
# scripts/analyze_chemeleon_runs.py
import polars as pl
import json
from pathlib import Path

runs_dir = Path("runs")
results = []

for run_dir in runs_dir.glob("chemeleon_lnk_*"):
    if (run_dir / "model_0/test_predictions.csv").exists():
        # Extract metrics from logs or test predictions
        results.append({
            "run": run_dir.name,
            "r2": ...,  # Parse from logs
            "mae": ...,
            "mse": ...
        })

df = pl.DataFrame(results).sort("r2", descending=True)
print(df.head(5))
```

#### 10. Production Model [opt-10-final]
**Combine best settings from all experiments**
```bash
CUDA_VISIBLE_DEVICES=3 chemprop train \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -o runs/chemeleon_lnk_production \
  --smiles-columns amine_smiles acid_smiles \
  --target-columns HTE_lnk --from-foundation CheMeleon \
  --ensemble-size 5 \
  --ffn-hidden-dim 1024 --ffn-num-layers 3 \
  --dropout 0.2 --batch-norm \
  --rxn-mode [BEST_FROM_PHASE_2] \
  --descriptors-columns Controls predicted_bias \
  --init-lr 1e-4 --max-lr 3e-3 --final-lr 1e-4 \
  --epochs 200 --patience 20 --warmup-epochs 10 \
  --batch-size 256 -n 16 \
  --accelerator gpu --devices 1 \
  -k 5 --save-smiles-splits \
  --metrics r2 mae rmse \
  --log
```

## Expected Final Performance
- **R²**: 0.90+ (from 0.862)
- **MAE**: < 0.20 (from 0.223)
- **MSE**: < 0.08 (from 0.0985)

## Monitoring Commands
```bash
# Watch training progress
tail -f "$(ls -t chemprop_logs/train/*.log | head -1)"

# GPU utilization
watch -n 1 nvidia-smi

# Compare runs
python scripts/analyze_chemeleon_runs.py
```

## Success Criteria
1. Each optimization should be evaluated independently
2. Track improvement over baseline for each change
3. Combine only improvements that stack well
4. Final model should show >5% R² improvement

## Time Estimate
- Phase 1-3: 4 hours
- Phase 4-5: 3 hours
- Phase 6-7: 6-8 hours
- **Total**: 13-15 hours of GPU time

## Notes
- Run experiments in order to build on learnings
- Save all logs for post-analysis
- Consider early stopping if no improvement after 3 phases
- Multi-GPU training possible but single GPU sufficient for this dataset size

