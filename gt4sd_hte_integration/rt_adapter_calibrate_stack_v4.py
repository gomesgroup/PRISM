#!/usr/bin/env python
"""
Calibrate and stack RT Adapter v4 without retraining.

Steps:
- Load tokenizer, frozen RT model, and saved AdapterV4 head.
- Build datasets and dataloaders for train/valid.
- Generate valid predictions in log10 space.
- Fit calibrators: linear (log), isotonic (log), isotonic (rate); pick best by rate-space R2.
- Train LightGBM on extracted features (if available) and calibrate it in log space.
- Stack MLP (calibrated) + LightGBM (calibrated) via linear regression; also compute 5-fold CV stacking.
- Save metrics and calibrator artifacts under artifacts/rt_adapter_v4/.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
import joblib
import importlib

from gt4sd_hte_integration.rt_adapter_train_v4 import (
    ExpressionBertTokenizer,  # type: ignore
    AutoConfig, AutoModelForCausalLM,  # type: ignore
    load_scaling,
    build_id_vocabs,
    RTLogIDDataset,
    collate,
    AdapterV4Head,
    RUNS,
    MODEL_DIR,
    OUT_DIR,
    DEVICE,
)


def export_features(model, head, dataloader, hte_token_id: int, d_token_ids: List[int]):
    Xs, ys = [], []
    with torch.no_grad():
        for input_ids, attention_mask, y, acid_ids, amine_ids in dataloader:
            input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
            acid_ids = acid_ids.to(DEVICE); amine_ids = amine_ids.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            feats = head.extract_features(hidden_states, input_ids, hte_token_id, d_token_ids, acid_ids, amine_ids)
            Xs.append(feats); ys.append(y.numpy())
    return np.vstack(Xs), np.concatenate(ys)


def main() -> None:
    print("[calib] Loading scaling & models...")
    scaling = load_scaling(); mean, std = scaling['mean'], scaling['std']

    tokenizer = ExpressionBertTokenizer.from_pretrained(str(RUNS))
    config = AutoConfig.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), config=config)
    model.to(DEVICE); model.eval()

    vocab = tokenizer.get_vocab()
    hte_token_id = vocab.get("<hte>"); assert hte_token_id is not None
    d_token_ids = [vocab.get(f"<d{k}>") for k in range(16)]; d_token_ids = [t for t in d_token_ids if t is not None]

    train_lines = (RUNS / "train.txt").read_text().strip().splitlines()
    valid_lines = (RUNS / "valid.txt").read_text().strip().splitlines()
    acid_vocab, amine_vocab = build_id_vocabs(train_lines + valid_lines)

    train_ds = RTLogIDDataset(train_lines, mean, std, acid_vocab, amine_vocab)
    valid_ds = RTLogIDDataset(valid_lines, mean, std, acid_vocab, amine_vocab)

    BATCH = 12
    from torch.utils.data import DataLoader  # type: ignore
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, drop_last=False,
                              collate_fn=lambda b: collate(tokenizer, b))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, drop_last=False,
                              collate_fn=lambda b: collate(tokenizer, b))

    # Instantiate head and load weights
    # Peek sizes
    input_ids, attention_mask, y, acid_ids, amine_ids = next(iter(valid_loader))
    input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    num_layers = len(outputs.hidden_states) - 1
    hidden_size = outputs.hidden_states[-1].shape[-1]

    head = AdapterV4Head(num_layers, hidden_size, num_heads=4,
                         acid_vocab_size=len(acid_vocab), amine_vocab_size=len(amine_vocab), id_emb_dim=64, dropout=0.15).to(DEVICE)
    head.load_state_dict(torch.load(OUT_DIR / "adapter_v4.pt", map_location=DEVICE))
    head.eval()

    print("[calib] Collecting valid predictions...")
    y_true_log, y_pred_log = [], []
    with torch.no_grad():
        for input_ids, attention_mask, y, acid_ids, amine_ids in valid_loader:
            input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
            acid_ids = acid_ids.to(DEVICE); amine_ids = amine_ids.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            pred_log = head(hidden_states, input_ids, hte_token_id, d_token_ids, acid_ids, amine_ids)
            y_true_log.append(y.numpy()); y_pred_log.append(pred_log.detach().cpu().numpy())
    y_true_log = np.concatenate(y_true_log); y_pred_log = np.concatenate(y_pred_log)
    y_true_rate = np.power(10.0, y_true_log)

    print("[calib] Fitting calibrators...")
    A_lin = np.vstack([y_pred_log, np.ones_like(y_pred_log)]).T
    a_lin, b_lin = np.linalg.lstsq(A_lin, y_true_log, rcond=None)[0]
    y_cal_log_lin = a_lin * y_pred_log + b_lin
    r2_rate_lin = r2_score(y_true_rate, np.power(10.0, y_cal_log_lin))

    iso_log = None; iso_rate = None
    try:
        iso_log = IsotonicRegression(out_of_bounds='clip')
        iso_log.fit(y_pred_log, y_true_log)
        y_cal_log_iso = iso_log.predict(y_pred_log)
        r2_rate_iso_fromlog = r2_score(y_true_rate, np.power(10.0, y_cal_log_iso))
    except Exception:
        y_cal_log_iso = None; r2_rate_iso_fromlog = -1e9

    try:
        iso_rate = IsotonicRegression(out_of_bounds='clip')
        iso_rate.fit(np.power(10.0, y_pred_log), y_true_rate)
        y_rate_iso = iso_rate.predict(np.power(10.0, y_pred_log))
        r2_rate_iso = r2_score(y_true_rate, y_rate_iso)
        y_cal_log_from_rate_iso = np.log10(np.maximum(y_rate_iso, 1e-30))
    except Exception:
        r2_rate_iso = -1e9; y_cal_log_from_rate_iso = None

    candidates = [
        ("linear_log", r2_rate_lin),
        ("isotonic_log", r2_rate_iso_fromlog),
        ("isotonic_rate", r2_rate_iso),
    ]
    best_type, _ = max(candidates, key=lambda t: t[1])
    calib_payload: Dict[str, object] = {"type": best_type}
    if best_type == "linear_log":
        y_mlp_cal = y_cal_log_lin
        calib_payload.update({"slope": float(a_lin), "intercept": float(b_lin)})
    elif best_type == "isotonic_log" and iso_log is not None:
        y_mlp_cal = y_cal_log_iso
        iso_log_path = OUT_DIR / "calibrator_isotonic_log.pkl"
        joblib.dump(iso_log, iso_log_path)
        calib_payload.update({"isotonic_log_model": str(iso_log_path)})
    else:
        y_mlp_cal = y_cal_log_from_rate_iso
        if iso_rate is not None:
            iso_rate_path = OUT_DIR / "calibrator_isotonic_rate.pkl"
            joblib.dump(iso_rate, iso_rate_path)
            calib_payload.update({"isotonic_rate_model": str(iso_rate_path)})

    # Metrics for chosen calibrator
    r2_log = r2_score(y_true_log, y_mlp_cal)
    mae_log = mean_absolute_error(y_true_log, y_mlp_cal)
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_mlp_cal)))
    r2_rate = r2_score(y_true_rate, np.power(10.0, y_mlp_cal))

    print("[calib] Writing metrics_isotonic_valid.json ...")
    with open(OUT_DIR / "metrics_isotonic_valid.json", "w", encoding="utf-8") as f:
        json.dump({
            "log10": {"r2": float(r2_log), "mae": float(mae_log), "rmse": float(rmse_log)},
            "rate": {"r2": float(r2_rate)},
            "calibration": calib_payload
        }, f, indent=2)

    # LightGBM on features
    print("[stack] Exporting features...")
    Xtr, ytr = export_features(model, head, train_loader, hte_token_id, d_token_ids)
    Xva, yva = export_features(model, head, valid_loader, hte_token_id, d_token_ids)

    a_lgb = 1.0; b_lgb = 0.0
    try:
        lgb = importlib.import_module("lightgbm")
        params = {'objective': 'regression', 'metric': ['l2','l1'], 'learning_rate': 0.05,
                  'num_leaves': 63, 'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
                  'min_data_in_leaf': 20, 'verbose': -1}
        booster = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=3000,
                            valid_sets=[lgb.Dataset(Xva, label=yva)], valid_names=['valid'],
                            callbacks=[lgb.early_stopping(stopping_rounds=100)])
        pred_lgb = booster.predict(Xva, num_iteration=booster.best_iteration)
        A = np.vstack([pred_lgb, np.ones_like(pred_lgb)]).T
        a_lgb, b_lgb = np.linalg.lstsq(A, yva, rcond=None)[0]
        y_lgb_cal = a_lgb * pred_lgb + b_lgb
    except Exception:
        booster = None
        y_lgb_cal = None

    print("[stack] Stacking...")
    if y_lgb_cal is not None:
        M = np.vstack([y_mlp_cal, y_lgb_cal, np.ones_like(yva)]).T
        w1, w2, b_stack = np.linalg.lstsq(M, yva, rcond=None)[0]
        y_stack_log = w1*y_mlp_cal + w2*y_lgb_cal + b_stack
        r2_stack = r2_score(yva, y_stack_log)
        mae_stack = mean_absolute_error(yva, y_stack_log)
        rmse_stack = float(np.sqrt(mean_squared_error(yva, y_stack_log)))
        y_rate = np.power(10.0, yva); y_stack_rate = np.power(10.0, y_stack_log)
        r2_stack_rate = r2_score(y_rate, y_stack_rate)

        # CV stacking
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_stack_oof = np.zeros_like(yva)
        weights_list = []
        for tr_idx, te_idx in kf.split(M):
            M_tr, M_te = M[tr_idx], M[te_idx]
            y_tr = yva[tr_idx]
            w = np.linalg.lstsq(M_tr, y_tr, rcond=None)[0]
            weights_list.append(w)
            y_stack_oof[te_idx] = M_te @ w
        r2_stack_cv = r2_score(yva, y_stack_oof)
        mae_stack_cv = mean_absolute_error(yva, y_stack_oof)
        rmse_stack_cv = float(np.sqrt(mean_squared_error(yva, y_stack_oof)))
        y_stack_rate_cv = np.power(10.0, y_stack_oof)
        r2_stack_rate_cv = r2_score(y_rate, y_stack_rate_cv)

        weights_mean = np.mean(np.stack(weights_list, axis=0), axis=0)

        print("[stack] Writing stack_isotonic.json ...")
        (OUT_DIR / "stack_isotonic.json").write_text(json.dumps({
            'log10': {'r2': float(r2_stack), 'mae': float(mae_stack), 'rmse': float(rmse_stack)},
            'rate': {'r2': float(r2_stack_rate)},
            'cv_log10': {'r2': float(r2_stack_cv), 'mae': float(mae_stack_cv), 'rmse': float(rmse_stack_cv)},
            'cv_rate': {'r2': float(r2_stack_rate_cv)},
            'weights': {'w_mlp': float(w1), 'w_lgb': float(w2), 'b': float(b_stack)},
            'cv_weights_mean': {'w_mlp': float(weights_mean[0]), 'w_lgb': float(weights_mean[1]), 'b': float(weights_mean[2])}
        }, indent=2))
    else:
        print("[stack] LightGBM unavailable - writing MLP-only metrics ...")
        (OUT_DIR / "stack_isotonic.json").write_text(json.dumps({
            'note': 'LightGBM unavailable',
            'mlp_only': {'log10': {'r2': float(r2_log), 'mae': float(mae_log), 'rmse': float(rmse_log)}, 'rate': {'r2': float(r2_rate)}}
        }, indent=2))


if __name__ == "__main__":
    try:
        main()
        print("[done]")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


