#!/usr/bin/env python
"""
RT Adapter v3: Trainable multi-head layer-wise pooling, log10 targets, and LightGBM stacking.

- Frozen RT (XLNet) provides hidden states per layer.
- Trainable aggregator (per-head token pooling + layer weights) is trained jointly with an MLP head using Huber loss in log10 space.
- Metrics reported in both log10 and rate space (using runs/hte/scaling.json parameters).
- LightGBM trains on aggregator features; linear stacking with MLP in log10 space.

Artifacts: artifacts/rt_adapter_v3/
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

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
REPO = Path("/home/passos/ml_measurable_hte_rates/regression-transformer")
RUNS = REPO / "runs" / "hte"
MODEL_DIR = REPO / "runs" / "best_model_final" / "model"
OUT_DIR = Path("/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration/artifacts/rt_adapter_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import RT components
sys.path.append(str(REPO))
from terminator.tokenization import ExpressionBertTokenizer  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore


def load_scaling() -> Dict:
    scaling = json.loads((RUNS / "scaling.json").read_text())
    target = scaling["target"]
    return {"mean": float(target["mean"]), "std": float(target["std"]) }


class RTLogDataset(Dataset):
    def __init__(self, lines: List[str], mean: float, std: float):
        self.samples: List[Tuple[str, float, float]] = []  # (text, y_z, y_log10)
        for line in lines:
            line = line.strip()
            if "<hte>" not in line:
                continue
            try:
                after = line.split("<hte>", 1)[1]
                y_z = float(after.split("|")[0].strip().split()[0])
                y_log = y_z * std + mean
                self.samples.append((line, y_z, y_log))
            except Exception:
                continue
        assert len(self.samples) > 0, "No valid samples parsed."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, y_z, y_log = self.samples[idx]
        return text, y_log  # train in log10 space


def collate(tokenizer, batch):
    texts = [b[0] for b in batch]
    y = np.array([b[1] for b in batch], dtype=np.float32)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return enc["input_ids"], enc["attention_mask"], torch.from_numpy(y)


class MultiHeadAggregator(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # Layer weights (ELMo-style)
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        # Token scorers per head
        self.token_scorer = nn.Linear(hidden_size, num_heads, bias=False)

    def pool_positions(self, layer_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        layer_hidden: (L, T, H)
        positions: (P,) token indices within [0, T)
        Returns: (num_heads*H,) pooled vector
        """
        if positions.numel() == 0:
            positions = torch.tensor([layer_hidden.shape[1]-1], device=layer_hidden.device, dtype=torch.long)
        # select positions: (L, P, H)
        selected = layer_hidden.index_select(dim=1, index=positions)
        # scores per head: (L, P, num_heads)
        scores = self.token_scorer(selected)  # linear on last dim
        # softmax over positions
        attn = torch.softmax(scores, dim=1)  # (L, P, num_heads)
        # weighted sum over positions P -> (L, num_heads, H)
        pooled = torch.einsum('lph,lpc->lch', selected, attn)
        # combine layers via weights
        w = torch.softmax(self.layer_logits, dim=-1)  # (L,)
        combined = torch.einsum('l,lch->ch', w, pooled)  # (num_heads, H)
        return combined.reshape(self.num_heads * self.hidden_size)

    def forward(self, hidden_states: List[torch.Tensor], input_ids: torch.Tensor,
                hte_token_id: int, d_token_ids: List[int], window: int = 4) -> torch.Tensor:
        """
        hidden_states: list length L of (B, T, H)
        input_ids: (B, T)
        Returns features: (B, 2*num_heads*H)
        """
        # stack layers to (L, B, T, H)
        layers = torch.stack(hidden_states[1:], dim=0)  # drop embeddings
        L, B, T, H = layers.shape
        assert L == self.num_layers
        feats = []
        for i in range(B):
            ids = input_ids[i]
            # hte window
            pos_list = (ids == hte_token_id).nonzero(as_tuple=False).view(-1)
            if pos_list.numel() == 0:
                pos = T - 1
            else:
                pos = int(pos_list[0].item())
            win = torch.arange(pos, min(pos + window + 1, T), device=ids.device, dtype=torch.long)
            # reshape per-sample layers (L,T,H)
            lbh = layers[:, i, :, :]
            pooled_hte = self.pool_positions(lbh, win)  # (C*H)
            # descriptor positions
            desc_pos = []
            for tid in d_token_ids:
                desc_pos.append((ids == tid).nonzero(as_tuple=False).view(-1))
            if len(desc_pos) > 0:
                desc_pos = torch.cat(desc_pos, dim=0)
            else:
                desc_pos = torch.tensor([0], device=ids.device, dtype=torch.long)
            pooled_desc = self.pool_positions(lbh, desc_pos)
            feat = torch.cat([pooled_hte, pooled_desc], dim=-1)
            feats.append(feat)
        return torch.stack(feats, dim=0)  # (B, 2*C*H)


class AdapterV3Head(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.aggregator = MultiHeadAggregator(num_layers, hidden_size, num_heads)
        in_dim = 2 * num_heads * hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        self.loss = nn.SmoothL1Loss(beta=0.5)

    def forward(self, hidden_states: List[torch.Tensor], input_ids: torch.Tensor,
                hte_token_id: int, d_token_ids: List[int]) -> torch.Tensor:
        feats = self.aggregator(hidden_states, input_ids, hte_token_id, d_token_ids)
        return self.mlp(feats).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, hidden_states: List[torch.Tensor], input_ids: torch.Tensor,
                          hte_token_id: int, d_token_ids: List[int]) -> np.ndarray:
        feats = self.aggregator(hidden_states, input_ids, hte_token_id, d_token_ids)
        return feats.detach().cpu().numpy()


def train_v3():
    print("=== RT Adapter v3 Training ===")
    scaling = load_scaling()
    mean, std = scaling['mean'], scaling['std']

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

    train_lines = (RUNS / "train.txt").read_text().strip().splitlines()
    valid_lines = (RUNS / "valid.txt").read_text().strip().splitlines()

    train_ds = RTLogDataset(train_lines, mean, std)
    valid_ds = RTLogDataset(valid_lines, mean, std)

    BATCH = 12
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False,
                              collate_fn=lambda b: collate(tokenizer, b))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, drop_last=False,
                              collate_fn=lambda b: collate(tokenizer, b))

    # Peek for sizes
    input_ids, attention_mask, y = next(iter(train_loader))
    input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)
    num_layers = len(hidden_states) - 1  # without embeddings
    hidden_size = hidden_states[-1].shape[-1]

    head = AdapterV3Head(num_layers=num_layers, hidden_size=hidden_size, num_heads=4, dropout=0.15).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=2e-4)

    best_r2 = -1e9
    bad = 0
    patience = 8
    EPOCHS = 30

    for epoch in range(1, EPOCHS + 1):
        head.train()
        train_losses = []
        for input_ids, attention_mask, y in train_loader:
            input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            pred_log = head(hidden_states, input_ids, hte_token_id, d_token_ids)
            loss = head.loss(pred_log, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation
        head.eval()
        y_true_log, y_pred_log = [], []
        with torch.no_grad():
            for input_ids, attention_mask, y in valid_loader:
                input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = list(outputs.hidden_states)
                pred_log = head(hidden_states, input_ids, hte_token_id, d_token_ids)
                y_true_log.append(y.numpy())
                y_pred_log.append(pred_log.detach().cpu().numpy())
        y_true_log = np.concatenate(y_true_log)
        y_pred_log = np.concatenate(y_pred_log)

        # Calibration in log10 space
        A = np.vstack([y_pred_log, np.ones_like(y_pred_log)]).T
        a, b = np.linalg.lstsq(A, y_true_log, rcond=None)[0]
        y_cal_log = a * y_pred_log + b

        # Metrics in log10 space
        r2_log = r2_score(y_true_log, y_cal_log)
        mae_log = mean_absolute_error(y_true_log, y_cal_log)
        rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_cal_log)))

        # Metrics in rate space
        y_true_rate = np.power(10.0, y_true_log)
        y_pred_rate = np.power(10.0, y_cal_log)
        r2_rate = r2_score(y_true_rate, y_pred_rate)
        mae_rate = mean_absolute_error(y_true_rate, y_pred_rate)
        rmse_rate = float(np.sqrt(mean_squared_error(y_true_rate, y_pred_rate)))

        print(f"Epoch {epoch:02d} | train_huber={np.mean(train_losses):.3f} | val_log R2={r2_log:.4f} MAE={mae_log:.3f} RMSE={rmse_log:.3f} | val_rate R2={r2_rate:.4f}")

        if r2_log > best_r2:
            best_r2 = r2_log
            bad = 0
            torch.save(head.state_dict(), OUT_DIR / "adapter_v3.pt")
            with open(OUT_DIR / "metrics_valid.json", "w") as f:
                json.dump({
                    "log10": {"r2": float(r2_log), "mae": float(mae_log), "rmse": float(rmse_log)},
                    "rate": {"r2": float(r2_rate), "mae": float(mae_rate), "rmse": float(rmse_rate)},
                    "calibration": {"slope": float(a), "intercept": float(b)}
                }, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    # Export features for LightGBM
    head.load_state_dict(torch.load(OUT_DIR / "adapter_v3.pt", map_location=DEVICE))
    head.eval()

    def export_features(dataloader) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        with torch.no_grad():
            for input_ids, attention_mask, y in dataloader:
                input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = list(outputs.hidden_states)
                feats = head.extract_features(hidden_states, input_ids, hte_token_id, d_token_ids)
                Xs.append(feats)
                ys.append(y.numpy())
        return np.vstack(Xs), np.concatenate(ys)

    Xtr, ytr = export_features(train_loader)
    Xva, yva = export_features(valid_loader)

    # LightGBM
    try:
        import lightgbm as lgb
        lgb_train = lgb.Dataset(Xtr, label=ytr)
        lgb_valid = lgb.Dataset(Xva, label=yva)
        params = {
            'objective': 'regression',
            'metric': ['l2', 'l1'],
            'learning_rate': 0.05,
            'num_leaves': 63,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'min_data_in_leaf': 20,
            'verbose': -1,
        }
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
        )
        lgb_pred_log = booster.predict(Xva, num_iteration=booster.best_iteration)
        # Calibrate LGBM in log space
        A = np.vstack([lgb_pred_log, np.ones_like(lgb_pred_log)]).T
        a_lgb, b_lgb = np.linalg.lstsq(A, yva, rcond=None)[0]
        y_lgb_cal = a_lgb * lgb_pred_log + b_lgb
        r2_lgb = r2_score(yva, y_lgb_cal)
        mae_lgb = mean_absolute_error(yva, y_lgb_cal)
        rmse_lgb = float(np.sqrt(mean_squared_error(yva, y_lgb_cal)))
        (OUT_DIR / "lgbm.txt").write_text(f"log10: R2={r2_lgb:.4f} MAE={mae_lgb:.3f} RMSE={rmse_lgb:.3f}\n")
    except Exception as e:
        booster = None
        (OUT_DIR / "lgbm.txt").write_text(f"LightGBM failed: {e}\n")

    # Stacking (log space)
    # MLP preds (calibrated)
    ps = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in valid_loader:
            input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            pred = head(hidden_states, input_ids, hte_token_id, d_token_ids).detach().cpu().numpy()
            ps.append(pred)
    y_mlp = np.concatenate(ps)
    calib = json.loads((OUT_DIR / "metrics_valid.json").read_text())["calibration"]
    y_mlp_cal = calib['slope'] * y_mlp + calib['intercept']

    if booster is not None:
        y_lgb = booster.predict(Xva, num_iteration=getattr(booster, 'best_iteration', None))
        y_lgb_cal = a_lgb * y_lgb + b_lgb
        M = np.vstack([y_mlp_cal, y_lgb_cal, np.ones_like(yva)]).T
        w1, w2, b_stack = np.linalg.lstsq(M, yva, rcond=None)[0]
        y_stack_log = w1 * y_mlp_cal + w2 * y_lgb_cal + b_stack
        r2_stack = r2_score(yva, y_stack_log)
        mae_stack = mean_absolute_error(yva, y_stack_log)
        rmse_stack = float(np.sqrt(mean_squared_error(yva, y_stack_log)))
        # Rate space
        yva_rate = np.power(10.0, yva)
        y_stack_rate = np.power(10.0, y_stack_log)
        r2_stack_rate = r2_score(yva_rate, y_stack_rate)
        (OUT_DIR / "stack.json").write_text(json.dumps({
            'log10': {'r2': float(r2_stack), 'mae': float(mae_stack), 'rmse': float(rmse_stack)},
            'rate': {'r2': float(r2_stack_rate)},
            'weights': {'w_mlp': float(w1), 'w_lgb': float(w2), 'b': float(b_stack)}
        }, indent=2))
        print(f"Stack | log10 R2={r2_stack:.4f} | rate R2={r2_stack_rate:.4f}")
    else:
        print("Stack | skipped (LGBM unavailable)")


if __name__ == "__main__":
    train_v3()
