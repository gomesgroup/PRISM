#!/usr/bin/env python
"""
RT Adapter v4: Acid/amine ID embeddings + BitFit on last 2 RT layers, log10 target, LightGBM stacking.

- Parse acid/amine SMILES from train/valid lines to build ID vocabularies; embed IDs and append to features.
- BitFit: enable gradients only for bias parameters in the last 2 transformer layers (fallback to all biases if structure differs).
- Train aggregator+MLP head in log10 space (Huber), calibrate, report metrics in log10 and rate space.
- Train LightGBM on extracted features and stack with calibrated MLP in log10 space.

Artifacts: artifacts/rt_adapter_v4/
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, cast

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
import joblib
import importlib
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO = Path("/home/passos/ml_measurable_hte_rates/regression-transformer")
RUNS = REPO / "runs" / "hte"
MODEL_DIR = REPO / "runs" / "best_model_final" / "model"
OUT_DIR = Path("/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration/artifacts/rt_adapter_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(REPO))
from terminator.tokenization import ExpressionBertTokenizer  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore


USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
LORA_TARGET = "ffn"  # one of: 'ffn', 'attn_qv', 'attn_qvo'
LORA_NUM_LAST_LAYERS = 1
WRAP_DATA_PARALLEL = True  # wrap model with DataParallel if 2+ GPUs available

def load_scaling() -> Dict:
    scaling = json.loads((RUNS / "scaling.json").read_text())
    target = scaling["target"]
    return {"mean": float(target["mean"]), "std": float(target["std"]) }


def parse_acid_amine(text: str) -> Tuple[str, str]:
    # Format: ... | <acid>.<amine>>*
    try:
        rhs = text.split("|", 1)[1].strip()
        left = rhs.split(">>", 1)[0]
        parts = left.split(".")
        acid = parts[0].strip()
        amine = parts[1].strip() if len(parts) > 1 else ""
        return acid, amine
    except Exception:
        return "", ""


def build_id_vocabs(lines: List[str]) -> Tuple[Dict[str,int], Dict[str,int]]:
    acids, amines = {}, {}
    for line in lines:
        a, m = parse_acid_amine(line)
        if a and a not in acids:
            acids[a] = len(acids)
        if m and m not in amines:
            amines[m] = len(amines)
    return acids, amines


class RTLogIDDataset(Dataset):
    def __init__(self, lines: List[str], mean: float, std: float, acid_vocab: Dict[str,int], amine_vocab: Dict[str,int]):
        self.samples: List[Tuple[str, float, int, int]] = []  # (text, y_log10, acid_id, amine_id)
        for line in lines:
            line = line.strip()
            if "<hte>" not in line:
                continue
            try:
                y_z = float(line.split("<hte>",1)[1].split("|")[0].strip().split()[0])
                y_log = y_z * std + mean
                acid, amine = parse_acid_amine(line)
                acid_id = acid_vocab.get(acid, -1)
                amine_id = amine_vocab.get(amine, -1)
                self.samples.append((line, y_log, acid_id, amine_id))
            except Exception:
                continue
        assert len(self.samples) > 0, "No valid samples."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(tokenizer, batch):
    texts = [b[0] for b in batch]
    y = np.array([b[1] for b in batch], dtype=np.float32)
    acid_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)
    amine_ids = torch.tensor([b[3] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return enc["input_ids"], enc["attention_mask"], torch.from_numpy(y), acid_ids, amine_ids


class MultiHeadAggregator(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.token_scorer = nn.Linear(hidden_size, num_heads, bias=False)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def pool_positions(self, layer_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            positions = torch.tensor([layer_hidden.shape[1]-1], device=layer_hidden.device, dtype=torch.long)
        selected = layer_hidden.index_select(dim=1, index=positions)  # (L,P,H)
        scores = self.token_scorer(selected)  # (L,P,C)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.einsum('lph,lpc->lch', selected, attn)  # (L,C,H)
        w = torch.softmax(self.layer_logits, dim=-1)  # (L,)
        combined = torch.einsum('l,lch->ch', w, pooled)  # (C,H)
        return combined.reshape(self.num_heads * self.hidden_size)

    def forward(self, hidden_states: List[torch.Tensor], input_ids: torch.Tensor,
                hte_token_id: int, d_token_ids: List[int], window: int = 4) -> torch.Tensor:
        layers = torch.stack(hidden_states[1:], dim=0)  # (L,B,T,H)
        _L,B,T,_H = layers.shape
        feats = []
        for i in range(B):
            ids = input_ids[i]
            lbh = layers[:, i, :, :]
            pos_list = (ids == hte_token_id).nonzero(as_tuple=False).view(-1)
            pos = int(pos_list[0].item()) if pos_list.numel() > 0 else T-1
            win = torch.arange(pos, min(pos+window+1, T), device=ids.device, dtype=torch.long)
            pooled_hte = self.pool_positions(lbh, win)
            desc_pos = []
            for tid in d_token_ids:
                desc_pos.append((ids == tid).nonzero(as_tuple=False).view(-1))
            if len(desc_pos) > 0:
                desc_pos = torch.cat(desc_pos, dim=0)
            else:
                desc_pos = torch.tensor([0], device=ids.device, dtype=torch.long)
            pooled_desc = self.pool_positions(lbh, desc_pos)
            feats.append(torch.cat([pooled_hte, pooled_desc], dim=-1))
        return torch.stack(feats, dim=0)  # (B, 2*C*H)


class AdapterV4Head(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, acid_vocab_size: int, amine_vocab_size: int, id_emb_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.aggregator = MultiHeadAggregator(num_layers, hidden_size, num_heads)
        self.acid_emb = nn.Embedding(max(acid_vocab_size,1), id_emb_dim)
        self.amine_emb = nn.Embedding(max(amine_vocab_size,1), id_emb_dim)
        in_dim = 2 * num_heads * hidden_size + 2 * id_emb_dim
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
                hte_token_id: int, d_token_ids: List[int], acid_ids: torch.Tensor, amine_ids: torch.Tensor) -> torch.Tensor:
        feats = self.aggregator(hidden_states, input_ids, hte_token_id, d_token_ids)
        acid_vec = self.acid_emb(torch.clamp(acid_ids, min=0))
        amine_vec = self.amine_emb(torch.clamp(amine_ids, min=0))
        x = torch.cat([feats, acid_vec, amine_vec], dim=-1)
        return self.mlp(x).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, hidden_states: List[torch.Tensor], input_ids: torch.Tensor,
                          hte_token_id: int, d_token_ids: List[int], acid_ids: torch.Tensor, amine_ids: torch.Tensor) -> np.ndarray:
        feats = self.aggregator(hidden_states, input_ids, hte_token_id, d_token_ids)
        acid_vec = self.acid_emb(torch.clamp(acid_ids, min=0))
        amine_vec = self.amine_emb(torch.clamp(amine_ids, min=0))
        x = torch.cat([feats, acid_vec, amine_vec], dim=-1)
        return x.detach().cpu().numpy()


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.use_bias = base_linear.bias is not None
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.use_bias and self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.r = r
        self.scaling = alpha / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            # Ensure LoRA params are on same device as base
            device = self.base.weight.device
            self.lora_A.to(device)
            self.lora_B.to(device)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            x_d = self.dropout(x)
            y = y + self.scaling * self.lora_B(self.lora_A(x_d))
        return y


def inject_lora_last_layers(model: nn.Module, num_last_layers: int = 1, r: int = 8, alpha: float = 16.0, dropout: float = 0.0, target: str = "ffn") -> int:
    replaced = 0
    try:
        trans = getattr(model, 'transformer', None)
        layers = getattr(trans, 'layer', None)
        if layers is None:
            return 0
        L = len(layers) if hasattr(layers, '__len__') else 0
        for idx in range(max(0, L - num_last_layers), L):
            layer = layers[idx]
            linear_names: List[str] = []
            for name, sub in layer.named_modules():
                if name == "":
                    continue
                if isinstance(sub, nn.Linear) and not isinstance(sub, LoRALinear):
                    # Filter targets
                    if target == "ffn":
                        if ("layer_1" in name) or ("layer_2" in name):
                            linear_names.append(name)
                    elif target == "attn_qv":
                        if name.startswith("rel_attn") and (name.endswith(".q") or name.endswith(".v")):
                            linear_names.append(name)
                    elif target == "attn_qvo":
                        if name.startswith("rel_attn") and (name.endswith(".q") or name.endswith(".v") or name.endswith(".o")):
                            linear_names.append(name)
                    else:
                        linear_names.append(name)
            for name in linear_names:
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                base_lin = getattr(parent, parts[-1])
                lora_lin = LoRALinear(base_lin, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, parts[-1], lora_lin)
                replaced += 1
    except Exception:
        return replaced
    return replaced


def enable_bitfit_last_layers(model, num_last_layers: int = 2) -> None:
    # disable all grads
    for p in model.parameters():
        p.requires_grad = False
    enabled = 0
    try:
        trans = getattr(model, 'transformer', None)
        layers = getattr(trans, 'layer', None)
        if layers is not None and (isinstance(layers, (list, tuple)) or hasattr(layers, '__len__')):
            L = len(layers)
            for idx in range(max(0, L - num_last_layers), L):
                for name, p in layers[idx].named_parameters():
                    if 'bias' in name:
                        p.requires_grad = True
                        enabled += 1
        else:
            # fallback: enable all biases
            for name, p in model.named_parameters():
                if 'bias' in name:
                    p.requires_grad = True
                    enabled += 1
    except Exception:
        for name, p in model.named_parameters():
            if 'bias' in name:
                p.requires_grad = True
                enabled += 1
    print(f"BitFit enabled parameters: {enabled}")


def train_v4():
    print("=== RT Adapter v4 Training ===")
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
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False,
                              collate_fn=lambda b: collate(tokenizer, b))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, drop_last=False,
                              collate_fn=lambda b: collate(tokenizer, b))

    # Peek sizes
    input_ids, attention_mask, y, acid_ids, amine_ids = next(iter(train_loader))
    input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    num_layers = len(outputs.hidden_states) - 1
    hidden_size = outputs.hidden_states[-1].shape[-1]

    head = AdapterV4Head(num_layers, hidden_size, num_heads=4,
                         acid_vocab_size=len(acid_vocab), amine_vocab_size=len(amine_vocab), id_emb_dim=64, dropout=0.15).to(DEVICE)

    # BitFit
    enable_bitfit_last_layers(model, num_last_layers=2)
    if USE_LORA:
        replaced = inject_lora_last_layers(model, num_last_layers=LORA_NUM_LAST_LAYERS, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, target=LORA_TARGET)
        print(f"LoRA injected linear layers: {replaced}")
    # Optional DataParallel on 2 GPUs
    if WRAP_DATA_PARALLEL and torch.cuda.device_count() >= 2:
        try:
            model = torch.nn.DataParallel(model, device_ids=[0,1], output_device=0)
            print("Wrapped model with DataParallel on devices [0,1]")
        except Exception as _:
            pass

    # Optimizer over head params + any enabled model params
    params = [p for p in head.parameters() if p.requires_grad] + [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=2e-4)

    best_r2 = -1e9; bad = 0; patience = 8; EPOCHS = 25

    for epoch in range(1, EPOCHS + 1):
        head.train(); model.train()  # BitFit biases train
        train_losses = []
        for input_ids, attention_mask, y, acid_ids, amine_ids in train_loader:
            input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
            y = y.to(DEVICE); acid_ids = acid_ids.to(DEVICE); amine_ids = amine_ids.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            pred_log = head(hidden_states, input_ids, hte_token_id, d_token_ids, acid_ids, amine_ids)
            # Rate-aware weighting: weights proportional to rate = 10^y, normalized to mean=1
            with torch.no_grad():
                rate = torch.pow(10.0, y)
                weights = rate / (torch.mean(rate) + 1e-8)
            loss_unreduced = nn.functional.smooth_l1_loss(pred_log, y, beta=0.5, reduction='none')
            loss = (loss_unreduced * weights).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())

        model.eval(); head.eval()
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

        # --- Calibrations ---
        # 1) Linear calibration in log10 space
        A_lin = np.vstack([y_pred_log, np.ones_like(y_pred_log)]).T
        a_lin, b_lin = np.linalg.lstsq(A_lin, y_true_log, rcond=None)[0]
        y_cal_log_lin = a_lin * y_pred_log + b_lin
        # Metrics for linear-log calibration
        r2_log_lin = r2_score(y_true_log, y_cal_log_lin)
        mae_log_lin = mean_absolute_error(y_true_log, y_cal_log_lin)
        rmse_log_lin = float(np.sqrt(mean_squared_error(y_true_log, y_cal_log_lin)))
        y_true_rate = np.power(10.0, y_true_log)
        y_rate_lin = np.power(10.0, y_cal_log_lin)
        r2_rate_lin = r2_score(y_true_rate, y_rate_lin)

        # 2) Isotonic calibration in log10 space
        try:
            iso_log = IsotonicRegression(out_of_bounds='clip')
            iso_log.fit(y_pred_log, y_true_log)
            y_cal_log_iso = iso_log.predict(y_pred_log)
            r2_log_iso = r2_score(y_true_log, y_cal_log_iso)
            mae_log_iso = mean_absolute_error(y_true_log, y_cal_log_iso)
            rmse_log_iso = float(np.sqrt(mean_squared_error(y_true_log, y_cal_log_iso)))
            y_rate_iso_fromlog = np.power(10.0, y_cal_log_iso)
            r2_rate_iso_fromlog = r2_score(y_true_rate, y_rate_iso_fromlog)
        except Exception:
            iso_log = None
            y_cal_log_iso = None
            r2_log_iso = -1e9; mae_log_iso = 1e9; rmse_log_iso = 1e9; r2_rate_iso_fromlog = -1e9

        # 3) Isotonic calibration directly in rate space
        try:
            y_pred_rate_uncal = np.power(10.0, y_pred_log)
            iso_rate = IsotonicRegression(out_of_bounds='clip')
            iso_rate.fit(y_pred_rate_uncal, y_true_rate)
            y_rate_iso = iso_rate.predict(y_pred_rate_uncal)
            r2_rate_iso = r2_score(y_true_rate, y_rate_iso)
            # Map back to log10 for consistency in stacking later
            y_cal_log_from_rate_iso = np.log10(np.maximum(y_rate_iso, 1e-30))
            r2_log_from_rate_iso = r2_score(y_true_log, y_cal_log_from_rate_iso)
            mae_log_from_rate_iso = mean_absolute_error(y_true_log, y_cal_log_from_rate_iso)
            rmse_log_from_rate_iso = float(np.sqrt(mean_squared_error(y_true_log, y_cal_log_from_rate_iso)))
        except Exception as _:
            iso_rate = None
            y_cal_log_from_rate_iso = None
            r2_rate_iso = -1e9; r2_log_from_rate_iso = -1e9; mae_log_from_rate_iso = 1e9; rmse_log_from_rate_iso = 1e9

        # Choose calibrator by best rate-space R2
        candidates = [
            ("linear_log", r2_rate_lin),
            ("isotonic_log", r2_rate_iso_fromlog),
            ("isotonic_rate", r2_rate_iso),
        ]
        best_type, _ = max(candidates, key=lambda t: t[1])
        if best_type == "linear_log":
            _y_cal_log = y_cal_log_lin
            r2_log = r2_log_lin; mae_log = mae_log_lin; rmse_log = rmse_log_lin
            r2_rate = r2_rate_lin
        elif best_type == "isotonic_log":
            _y_cal_log = y_cal_log_iso
            r2_log = r2_log_iso; mae_log = mae_log_iso; rmse_log = rmse_log_iso
            r2_rate = r2_rate_iso_fromlog
        else:
            _y_cal_log = y_cal_log_from_rate_iso
            r2_log = r2_log_from_rate_iso; mae_log = mae_log_from_rate_iso; rmse_log = rmse_log_from_rate_iso
            r2_rate = r2_rate_iso

        print(f"Epoch {epoch:02d} | train_huber={np.mean(train_losses):.3f} | val_log R2={r2_log:.4f} MAE={mae_log:.3f} RMSE={rmse_log:.3f} | val_rate R2={r2_rate:.4f} | cal={best_type}")

        if r2_log > best_r2:
            best_r2 = r2_log; bad = 0
            torch.save(head.state_dict(), OUT_DIR / "adapter_v4.pt")
            calib_payload: Dict[str, object] = {"type": best_type}
            if best_type == "linear_log":
                calib_payload.update({"slope": float(a_lin), "intercept": float(b_lin)})
            elif best_type == "isotonic_log" and iso_log is not None:
                iso_log_path = OUT_DIR / "calibrator_isotonic_log.pkl"
                joblib.dump(iso_log, iso_log_path)
                calib_payload.update({"isotonic_log_model": str(iso_log_path)})
            elif best_type == "isotonic_rate" and iso_rate is not None:
                iso_rate_path = OUT_DIR / "calibrator_isotonic_rate.pkl"
                joblib.dump(iso_rate, iso_rate_path)
                calib_payload.update({"isotonic_rate_model": str(iso_rate_path)})
            with open(OUT_DIR / "metrics_valid.json", "w", encoding="utf-8") as f:
                json.dump({
                    "log10": {"r2": float(r2_log), "mae": float(mae_log), "rmse": float(rmse_log)},
                    "rate": {"r2": float(r2_rate)},
                    "calibration": calib_payload
                }, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping"); break

    # Export features and train LightGBM
    head.load_state_dict(torch.load(OUT_DIR / "adapter_v4.pt", map_location=DEVICE)); head.eval()

    def export_features(dataloader):
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

    Xtr, ytr = export_features(train_loader)
    Xva, yva = export_features(valid_loader)

    a_lgb: float = 1.0
    b_lgb: float = 0.0
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
        r2_lgb = r2_score(yva, y_lgb_cal)
        (OUT_DIR / "lgbm.txt").write_text(f"log10: R2={r2_lgb:.4f}\n")
    except Exception as e:
        booster = None
        (OUT_DIR / "lgbm.txt").write_text(f"LightGBM failed: {e}\n")

    # Utilities for applying selected calibration to log10 predictions
    def apply_calibration_to_log_preds(y_pred_log: np.ndarray, calib_spec: Dict[str, object]) -> np.ndarray:
        ctype = str(calib_spec.get("type", "linear_log"))
        if ctype == "linear_log":
            a = float(cast(float, calib_spec.get("slope", 1.0)))
            b = float(cast(float, calib_spec.get("intercept", 0.0)))
            return a * y_pred_log + b
        elif ctype == "isotonic_log":
            model_path = calib_spec.get("isotonic_log_model", None)
            if model_path is None:
                return y_pred_log
            iso = joblib.load(str(model_path))
            return iso.predict(y_pred_log)
        elif ctype == "isotonic_rate":
            model_path = calib_spec.get("isotonic_rate_model", None)
            if model_path is None:
                return y_pred_log
            iso = joblib.load(str(model_path))
            rate_uncal = np.power(10.0, y_pred_log)
            rate_cal = iso.predict(rate_uncal)
            return np.log10(np.maximum(rate_cal, 1e-30))
        else:
            return y_pred_log

    # Stacking
    # MLP preds calibrated using best calibration
    calib = json.loads((OUT_DIR / "metrics_valid.json").read_text())["calibration"]
    y_mlp = []
    with torch.no_grad():
        for input_ids, attention_mask, _, acid_ids, amine_ids in valid_loader:
            input_ids = input_ids.to(DEVICE); attention_mask = attention_mask.to(DEVICE)
            acid_ids = acid_ids.to(DEVICE); amine_ids = amine_ids.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            pred = head(hidden_states, input_ids, hte_token_id, d_token_ids, acid_ids, amine_ids).detach().cpu().numpy()
            y_mlp.append(pred)
    y_mlp = np.concatenate(y_mlp)
    y_mlp_cal = apply_calibration_to_log_preds(y_mlp, calib)

    if booster is not None:
        y_lgb = booster.predict(Xva, num_iteration=getattr(booster, 'best_iteration', None))
        y_lgb_cal = a_lgb * y_lgb + b_lgb

        # Direct stacking on all valid
        M = np.vstack([y_mlp_cal, y_lgb_cal, np.ones_like(yva)]).T
        w1, w2, b_stack = np.linalg.lstsq(M, yva, rcond=None)[0]
        y_stack_log = w1*y_mlp_cal + w2*y_lgb_cal + b_stack
        r2_stack = r2_score(yva, y_stack_log)
        mae_stack = mean_absolute_error(yva, y_stack_log)
        rmse_stack = float(np.sqrt(mean_squared_error(yva, y_stack_log)))
        y_rate = np.power(10.0, yva); y_stack_rate = np.power(10.0, y_stack_log)
        r2_stack_rate = r2_score(y_rate, y_stack_rate)

        # K-fold CV stacking for more stable estimation
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

        (OUT_DIR / "stack.json").write_text(json.dumps({
            'log10': {'r2': float(r2_stack), 'mae': float(mae_stack), 'rmse': float(rmse_stack)},
            'rate': {'r2': float(r2_stack_rate)},
            'cv_log10': {'r2': float(r2_stack_cv), 'mae': float(mae_stack_cv), 'rmse': float(rmse_stack_cv)},
            'cv_rate': {'r2': float(r2_stack_rate_cv)},
            'weights': {'w_mlp': float(w1), 'w_lgb': float(w2), 'b': float(b_stack)},
            'cv_weights_mean': {'w_mlp': float(weights_mean[0]), 'w_lgb': float(weights_mean[1]), 'b': float(weights_mean[2])}
        }, indent=2))
        print(f"Stack v4 | log10 R2={r2_stack:.4f} | rate R2={r2_stack_rate:.4f} | CV log10 R2={r2_stack_cv:.4f} | CV rate R2={r2_stack_rate_cv:.4f}")
    else:
        print("Stack v4 | skipped (LGBM unavailable)")


if __name__ == "__main__":
    train_v4()
