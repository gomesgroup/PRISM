#!/usr/bin/env python
"""
RT Adapter Training: Predict HTE directly from frozen RT hidden states.

Pipeline:
- Load RT tokenizer and model (frozen)
- Iterate train.txt / valid.txt in runs/hte
- For each sample, find <hte> position and extract last-layer hidden state at that position
- Train small MLP regressor on hidden states to predict HTE (unscaled)
- Evaluate on valid set: R^2, MAE, RMSE (and in log10)
- Save metrics and adapter weights
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Use CPU-friendly defaults; will move to CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
REPO = Path("/home/passos/ml_measurable_hte_rates/regression-transformer")
RUNS = REPO / "runs" / "hte"
MODEL_DIR = REPO / "runs" / "best_model_final" / "model"
OUT_DIR = Path("/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration/artifacts/rt_adapter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import RT components
sys.path.append(str(REPO))
from terminator.tokenization import ExpressionBertTokenizer  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore


class RTFeatureDataset(Dataset):
    def __init__(self, lines: List[str], tokenizer: ExpressionBertTokenizer, hte_token_id: int):
        self.lines = lines
        self.tokenizer = tokenizer
        self.hte_token_id = hte_token_id
        self.values = []  # float target
        self.inputs = []  # tokenized ids

        for line in self.lines:
            # parse value after <hte>
            try:
                before, after = line.split("<hte>")
            except ValueError:
                # skip if malformed
                continue
            # value until pipe or whitespace
            value_str = after.split("|")[0].strip().split()[0]
            try:
                value = float(value_str)
            except Exception:
                # skip if not float
                continue
            self.values.append(value)
            self.inputs.append(line.strip())

        assert len(self.values) == len(self.inputs) and len(self.values) > 0, "No valid samples parsed."

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.values[idx]


def extract_features(model, tokenizer, batch: List[Tuple[str, float]], hte_token_id: int) -> Tuple[np.ndarray, np.ndarray]:
    texts = [b[0] for b in batch]
    y = np.array([b[1] for b in batch], dtype=np.float32)

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # (B, T, H)

    # find position of <hte> in each sample
    features = []
    for i in range(input_ids.shape[0]):
        ids = input_ids[i]
        pos = (ids == hte_token_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            # fallback: use last token hidden state
            vec = hidden_states[i, -1, :].detach().cpu().numpy()
        else:
            idx_pos = pos[0].item()
            vec = hidden_states[i, idx_pos, :].detach().cpu().numpy()
        features.append(vec)

    X = np.stack(features, axis=0)
    return X, y


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.1):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    print("=== RT Adapter Training ===")
    # Load tokenizer and model (frozen)
    tokenizer = ExpressionBertTokenizer.from_pretrained(str(RUNS))
    config = AutoConfig.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), config=config)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    vocab = tokenizer.get_vocab()
    hte_token_id = vocab.get("<hte>")
    assert hte_token_id is not None, "<hte> token not found in vocabulary"

    # Load data
    train_lines = Path(RUNS / "train.txt").read_text().strip().splitlines()
    valid_lines = Path(RUNS / "valid.txt").read_text().strip().splitlines()

    train_ds = RTFeatureDataset(train_lines, tokenizer, hte_token_id)
    valid_ds = RTFeatureDataset(valid_lines, tokenizer, hte_token_id)

    # Dataloaders (feature extraction on the fly; could precompute later)
    BATCH = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False, collate_fn=lambda batch: batch)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, drop_last=False, collate_fn=lambda batch: batch)

    # Peek one batch to get hidden size
    batch_peek = next(iter(train_loader))
    X_one, _ = extract_features(model, tokenizer, batch_peek, hte_token_id)
    input_dim = X_one.shape[1]

    reg = MLPRegressor(input_dim=input_dim, hidden=512, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(reg.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.L1Loss()  # MAE is robust for skewed targets

    best_r2 = -1e9
    patience = 7
    bad = 0

    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        reg.train()
        train_losses = []
        for batch in train_loader:
            X, y = extract_features(model, tokenizer, batch, hte_token_id)
            Xt = torch.from_numpy(X).float().to(DEVICE)
            yt = torch.from_numpy(y).float().to(DEVICE)

            pred = reg(Xt)
            loss = loss_fn(pred, yt)
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
                X, y = extract_features(model, tokenizer, batch, hte_token_id)
                Xt = torch.from_numpy(X).float().to(DEVICE)
                pred = reg(Xt).detach().cpu().numpy()
                ys.append(y)
                ps.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        print(f"Epoch {epoch:02d} | train_mae={np.mean(train_losses):.3f} | val_R2={r2:.4f} MAE={mae:.2f} RMSE={rmse:.2f}")

        # Early stopping on R2
        if r2 > best_r2:
            best_r2 = r2
            bad = 0
            torch.save(reg.state_dict(), OUT_DIR / "adapter.pt")
            with open(OUT_DIR / "metrics_valid.json", "w") as f:
                json.dump({"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    print("\n== Final Validation Metrics ==")
    print(Path(OUT_DIR / "metrics_valid.json").read_text())


if __name__ == "__main__":
    main()
