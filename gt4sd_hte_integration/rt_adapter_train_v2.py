#!/usr/bin/env python
"""
RT Adapter v2: Layer-wise aggregation, attentive pooling, Huber loss, and LightGBM stacking.

- Frozen RT (XLNet) produces hidden states for all layers.
- Features:
  - Attentive pooling over a small window around the <hte> token across layers.
  - Attentive pooling over descriptor tokens <d0>...<d15> across layers.
  - Learnable layer weights (ELMo-style) to aggregate hidden states.
- Heads:
  - MLP (Huber loss) trained on pooled features.
  - LightGBM regressor trained on pooled features.
- Stacking:
  - Linear stacking of MLP + LightGBM on validation set with calibration.

Outputs:
- artifacts/rt_adapter_v2/{adapter_v2.pt, metrics_valid.json, lgbm.txt, stack.json}
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Env/device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
REPO = Path("/home/passos/ml_measurable_hte_rates/regression-transformer")
RUNS = REPO / "runs" / "hte"
MODEL_DIR = REPO / "runs" / "best_model_final" / "model"
OUT_DIR = Path("/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration/artifacts/rt_adapter_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import RT components
sys.path.append(str(REPO))
from terminator.tokenization import ExpressionBertTokenizer  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore


class RTDataset(Dataset):
    def __init__(self, lines: List[str]):
        self.samples: List[Tuple[str, float]] = []
        for line in lines:
            line = line.strip()
            if "<hte>" not in line:
                continue
            try:
                after = line.split("<hte>", 1)[1]
                # value is first token right after <hte>, separated by space or '|'
                val_token = after.split("|")[0].strip().split()[0]
                y = float(val_token)
                self.samples.append((line, y))
            except Exception:
                continue
        assert len(self.samples) > 0, "No valid samples with <hte> value parsed."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LayerwiseAggregator(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        # Learn layer weights (ELMo-style)
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        # Token attention scorer for pooling within a set of positions
        self.token_scorer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, layer_hidden: List[torch.Tensor], positions: List[int]) -> torch.Tensor:
        """
        layer_hidden: list of length L where each entry is (T, H)
        positions: list of token indices to pool over
        Returns: aggregated vector (H,)
        """
        L = len(layer_hidden)
        H = layer_hidden[0].shape[-1]
        # Stack layers: (L, T, H)
        hs = torch.stack(layer_hidden, dim=0)  # (L, T, H)
        # Slice positions: (L, P, H)
        pos = torch.tensor(positions, dtype=torch.long, device=hs.device)
        pos = pos[(pos >= 0) & (pos < hs.shape[1])]
        if pos.numel() == 0:
            # fallback: last token
            pos = torch.tensor([hs.shape[1]-1], dtype=torch.long, device=hs.device)
        selected = hs.index_select(dim=1, index=pos)  # (L, P, H)
        # Token attention pooling within each layer
        # Compute scores per token: (L, P, 1)
        scores = self.token_scorer(selected)  # (L, P, 1)
        attn = torch.softmax(scores.squeeze(-1), dim=-1)  # (L, P)
        pooled_tokens = torch.einsum('lph,lph->lh', selected, attn.unsqueeze(-1).expand_as(selected))  # (L, H)
        # Layer weights
        layer_weights = torch.softmax(self.layer_logits, dim=-1)  # (L,)
        aggregated = torch.einsum('l,lh->h', layer_weights, pooled_tokens)  # (H,)
        return aggregated


def find_token_positions(input_ids: torch.Tensor, token_id: int) -> List[int]:
    return (input_ids == token_id).nonzero(as_tuple=False).view(-1).tolist()


def extract_pooled_features(
    model, tokenizer, batch: List[Tuple[str, float]], hte_token_id: int, d_token_ids: List[int],
    window: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    texts = [b[0] for b in batch]
    y = np.array([b[1] for b in batch], dtype=np.float32)

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # outputs.hidden_states: tuple of length L+1 (including embeddings), take last L
        hidden_states = outputs.hidden_states[1:]  # drop embeddings
        # Convert to list of (B, T, H)
        layers = [hs for hs in hidden_states]  # list length L

    B, T, H = layers[-1].shape
    num_layers = len(layers)

    # Build per-sample aggregators (shared weights is fine, but simpler: share one module)
    aggregator = LayerwiseAggregator(num_layers=num_layers, hidden_size=H).to(DEVICE)
    aggregator.token_scorer.weight.data.zero_()  # init near uniform

    features = []
    for i in range(B):
        ids = input_ids[i]
        # prepare layer list for sample i: (L, T, H) -> list of (T, H)
        sample_layers = [layer[i] for layer in layers]
        # hte pos and window
        hte_positions = find_token_positions(ids, hte_token_id)
        if len(hte_positions) == 0:
            hte_positions = [T - 1]
        pos = hte_positions[0]
        win_positions = list(range(pos, min(pos + window + 1, T)))
        pooled_hte = aggregator(sample_layers, win_positions)  # (H,)
        # descriptor token positions
        desc_positions: List[int] = []
        for tid in d_token_ids:
            desc_positions += find_token_positions(ids, tid)
        if len(desc_positions) == 0:
            desc_positions = [0]
        pooled_desc = aggregator(sample_layers, desc_positions)  # (H,)
        feat = torch.cat([pooled_hte, pooled_desc], dim=-1).detach().cpu().numpy()  # (2H,)
        features.append(feat)

    X = np.stack(features, axis=0)
    return X, y


class MLPRegressorHuber(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 768, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.loss = nn.SmoothL1Loss(beta=0.5)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_v2():
    print("=== RT Adapter v2 Training ===")
    tokenizer = ExpressionBertTokenizer.from_pretrained(str(RUNS))
    config = AutoConfig.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), config=config)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    vocab = tokenizer.get_vocab()
    hte_token_id = vocab.get("<hte>")
    assert hte_token_id is not None, "<hte> not in vocabulary"
    d_token_ids = [vocab.get(f"<d{k}>") for k in range(16)]
    d_token_ids = [tid for tid in d_token_ids if tid is not None]

    train_lines = Path(RUNS / "train.txt").read_text().strip().splitlines()
    valid_lines = Path(RUNS / "valid.txt").read_text().strip().splitlines()

    train_ds = RTDataset(train_lines)
    valid_ds = RTDataset(valid_lines)

    BATCH = 16
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False, collate_fn=lambda b: b)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, drop_last=False, collate_fn=lambda b: b)

    # Peek for input dim
    X_one, _ = extract_pooled_features(model, tokenizer, next(iter(train_loader)), hte_token_id, d_token_ids)
    input_dim = X_one.shape[1]

    reg = MLPRegressorHuber(input_dim=input_dim, hidden=768, dropout=0.15).to(DEVICE)
    opt = torch.optim.AdamW(reg.parameters(), lr=2e-4, weight_decay=2e-4)

    best_r2 = -1e9
    bad = 0
    patience = 8
    EPOCHS = 30

    for epoch in range(1, EPOCHS + 1):
        reg.train()
        train_losses = []
        for batch in train_loader:
            X, y = extract_pooled_features(model, tokenizer, batch, hte_token_id, d_token_ids)
            Xt = torch.from_numpy(X).float().to(DEVICE)
            yt = torch.from_numpy(y).float().to(DEVICE)
            pred = reg(Xt)
            loss = reg.loss(pred, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation
        reg.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for batch in valid_loader:
                X, y = extract_pooled_features(model, tokenizer, batch, hte_token_id, d_token_ids)
                Xt = torch.from_numpy(X).float().to(DEVICE)
                pred = reg(Xt).detach().cpu().numpy()
                ys.append(y)
                ps.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)

        # Linear calibration: y_true ≈ a*y_pred + b
        A = np.vstack([y_pred, np.ones_like(y_pred)]).T
        a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
        y_cal = a * y_pred + b

        r2 = r2_score(y_true, y_cal)
        mae = mean_absolute_error(y_true, y_cal)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_cal)))

        print(f"Epoch {epoch:02d} | train_huber={np.mean(train_losses):.3f} | val_R2={r2:.4f} MAE={mae:.2f} RMSE={rmse:.2f}")

        if r2 > best_r2:
            best_r2 = r2
            bad = 0
            torch.save(reg.state_dict(), OUT_DIR / "adapter_v2.pt")
            with open(OUT_DIR / "metrics_valid.json", "w") as f:
                json.dump({"r2": float(r2), "mae": float(mae), "rmse": float(rmse), "calibration": {"slope": float(a), "intercept": float(b)}}, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    # Export pooled features for LightGBM on the full train/valid
    def export_features(dataloader) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        with torch.no_grad():
            for batch in dataloader:
                X, y = extract_pooled_features(model, tokenizer, batch, hte_token_id, d_token_ids)
                Xs.append(X)
                ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    Xtr, ytr = export_features(train_loader)
    Xva, yva = export_features(valid_loader)

    # LightGBM training
    try:
        import lightgbm as lgb
        lgb_train = lgb.Dataset(Xtr, label=ytr)
        lgb_valid = lgb.Dataset(Xva, label=yva)
        params = {
            'objective': 'regression',
            'metric': ['l2', 'l1'],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 15,
            'verbose': -1,
        }
        booster = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_valid],
                            valid_names=['valid'], early_stopping_rounds=50, verbose_eval=False)
        lgb_pred = booster.predict(Xva, num_iteration=booster.best_iteration)
        # Calibrate
        A = np.vstack([lgb_pred, np.ones_like(lgb_pred)]).T
        a_lgb, b_lgb = np.linalg.lstsq(A, yva, rcond=None)[0]
        y_lgb_cal = a_lgb * lgb_pred + b_lgb
        r2_lgb = r2_score(yva, y_lgb_cal)
        mae_lgb = mean_absolute_error(yva, y_lgb_cal)
        rmse_lgb = float(np.sqrt(mean_squared_error(yva, y_lgb_cal)))
        with open(OUT_DIR / "lgbm.txt", "w") as f:
            f.write(f"R2={r2_lgb:.4f} MAE={mae_lgb:.3f} RMSE={rmse_lgb:.3f}\n")
    except Exception as e:
        booster = None
        r2_lgb = -1e9
        mae_lgb = rmse_lgb = 1e9
        with open(OUT_DIR / "lgbm.txt", "w") as f:
            f.write(f"LightGBM training failed: {e}\n")

    # Stacking: combine MLP (calibrated) + LGBM (calibrated)
    # Recompute best MLP preds on valid using saved model
    reg.load_state_dict(torch.load(OUT_DIR / "adapter_v2.pt", map_location=DEVICE))
    reg.eval()
    ps = []
    with torch.no_grad():
        for batch in valid_loader:
            X, _ = extract_pooled_features(model, tokenizer, batch, hte_token_id, d_token_ids)
            Xt = torch.from_numpy(X).float().to(DEVICE)
            pred = reg(Xt).detach().cpu().numpy()
            ps.append(pred)
    y_pred_mlp = np.concatenate(ps)
    # Load calibration for MLP
    calib = json.loads(Path(OUT_DIR / "metrics_valid.json").read_text())['calibration']
    y_mlp_cal = calib['slope'] * y_pred_mlp + calib['intercept']

    # If LGBM available
    if booster is not None:
        y_lgb = booster.predict(Xva, num_iteration=booster.best_iteration)
        y_lgb_cal = a_lgb * y_lgb + b_lgb
        # Stack: y ≈ w1*y_mlp + w2*y_lgb + b
        M = np.vstack([y_mlp_cal, y_lgb_cal, np.ones_like(yva)]).T
        w1, w2, b_stack = np.linalg.lstsq(M, yva, rcond=None)[0]
        y_stack = w1 * y_mlp_cal + w2 * y_lgb_cal + b_stack
        r2_stack = r2_score(yva, y_stack)
        mae_stack = mean_absolute_error(yva, y_stack)
        rmse_stack = float(np.sqrt(mean_squared_error(yva, y_stack)))
        with open(OUT_DIR / "stack.json", "w") as f:
            json.dump({
                'w1_mlp': float(w1), 'w2_lgb': float(w2), 'b': float(b_stack),
                'r2': float(r2_stack), 'mae': float(mae_stack), 'rmse': float(rmse_stack)
            }, f, indent=2)
        print(f"Stack | val_R2={r2_stack:.4f} MAE={mae_stack:.2f} RMSE={rmse_stack:.2f}")
    else:
        print("Stack | skipped (LGBM unavailable)")


if __name__ == "__main__":
    train_v2()
