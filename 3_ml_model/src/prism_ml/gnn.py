"""A graph network over the two reactants, with the package's two heads.

Message passing over bonds, a pooled molecular representation for each
reactant, and the same pair of heads the descriptor models use. It is exposed
through the same `fit` / `predict` / `predict_proba` surface as the
scikit-learn estimators, so the benchmark treats it as one more family.

One caveat belongs with any result from it. The other families are compared on
identical descriptors, so a difference between them is the learner alone. This
model reads molecular graphs instead, so a difference between it and the others
is **learner and representation together** — it is not told buried volume or a
Fukui index, it has to find whatever it needs in atoms and bonds.

`torch` is imported lazily, so the rest of the package works without it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import graph as graph_mod


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the graph model needs `torch`; install it or choose another "
            "family from config.MODEL_FAMILIES") from exc


def _build_module(hidden: int, layers: int, dropout: float, n_descriptors: int,
                  atom_features: int, encoder: str):
    torch = _torch()
    nn = torch.nn

    class MoleculeEncoder(nn.Module):
        """Message passing, then mean pool concatenated with the centre atom.

        With ``encoder="bond"`` messages travel along bonds. With
        ``encoder="attention"`` every atom attends to every other, with learned
        weights — the right form when connectivity is a complete graph, as it
        is for AIMNet2 features, since uniform averaging over a complete graph
        would collapse to a mean and discard all structure.
        """

        def __init__(self):
            super().__init__()
            self.project = nn.Linear(atom_features, hidden)
            self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
            self.attention = nn.ModuleList(
                nn.Linear(2 * hidden, 1, bias=False) for _ in range(layers)
            ) if encoder == "attention" else None
            self.dropout = nn.Dropout(dropout)
            self.out_dim = hidden * 2

        def forward(self, atoms, adjacency, mask, centre):
            keep = mask.unsqueeze(-1)
            x = torch.relu(self.project(atoms)) * keep
            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                if self.attention is None:
                    # Sum over bonded neighbours, normalised by degree so that
                    # a large molecule does not simply produce larger
                    # activations.
                    degree = adjacency.sum(-1, keepdim=True).clamp(min=1.0)
                    message = torch.bmm(adjacency, x) / degree
                else:
                    # score[i, j] = leaky_relu(W . [x_i, x_j]) and W has no
                    # bias, so it splits into W_left . x_i + W_right . x_j.
                    # Adding the two halves broadcast is identical to building
                    # the (batch, n, n, 2 * hidden) pair tensor and projecting
                    # it, but never materialises that tensor -- which dominated
                    # the cost, at 2 * hidden times the memory traffic.
                    w = self.attention[i].weight
                    half = w.size(1) // 2
                    source = x @ w[:, :half].t()
                    target = x @ w[:, half:].t()
                    scores = torch.nn.functional.leaky_relu(
                        source + target.transpose(1, 2))
                    # Mask padded atoms and non-edges before the softmax, so
                    # they receive no weight rather than a small one.
                    allowed = (adjacency > 0) & mask.unsqueeze(1).bool()
                    scores = scores.masked_fill(~allowed, float("-inf"))
                    weights = torch.softmax(scores, dim=-1)
                    weights = torch.nan_to_num(weights)
                    message = torch.bmm(weights, x)
                x = norm(x + self.dropout(torch.relu(layer(message)))) * keep
            pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
            centre_vec = x[torch.arange(x.size(0), device=x.device), centre]
            return torch.cat([pooled, centre_vec], dim=-1)

    class ReactionGNN(nn.Module):
        """One encoder per reactant, then a class head and a rate-constant head."""

        def __init__(self):
            super().__init__()
            self.acyl = MoleculeEncoder()
            self.amine = MoleculeEncoder()
            width = self.acyl.out_dim + self.amine.out_dim + n_descriptors
            def head():
                return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.class_head = head()
            self.rate_head = head()

        def forward(self, batch):
            parts = [self.acyl(batch["acyl_atoms"], batch["acyl_adjacency"],
                               batch["acyl_mask"], batch["acyl_centre"]),
                     self.amine(batch["amine_atoms"], batch["amine_adjacency"],
                                batch["amine_mask"], batch["amine_centre"])]
            if n_descriptors:
                parts.append(batch["descriptors"])
            h = torch.cat(parts, dim=-1)
            return self.class_head(h).squeeze(-1), self.rate_head(h).squeeze(-1)

    return ReactionGNN()


class GraphNetEstimator:
    """Graph network with a scikit-learn-shaped interface.

    Declares ``needs_rows`` so the caller passes the reaction table alongside
    the design matrix: the graphs are looked up by reactant id, which the
    numeric matrix does not carry.
    """

    needs_rows = True

    def __init__(self, acyl_graphs: Dict[int, graph_mod.MolGraph],
                 amine_graphs: Dict[int, graph_mod.MolGraph],
                 hidden: int = 64, layers: int = 3, dropout: float = 0.1,
                 epochs: int = 150, learning_rate: float = 3e-3,
                 batch_size: int = 128, use_descriptors: bool = False,
                 random_state: int = 0, task: str = "classification",
                 encoder: str = "bond", device: str = "auto"):
        self.acyl_graphs = acyl_graphs
        self.amine_graphs = amine_graphs
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_descriptors = use_descriptors
        self.random_state = random_state
        self.task = task
        self.encoder = encoder
        self.device = device
        self._device = None
        self.module = None
        self._n_descriptors = 0
        self._caches: Dict[str, dict] = {}

    # ---------------------------------------------------------------- device
    def _resolve_device(self):
        """"auto" means CUDA when present, else CPU; PRISM_DEVICE overrides.

        On CPU this is exactly the previous behaviour. On a GPU, CUDA kernels
        are not bit-for-bit deterministic, so scores can differ from a CPU run
        in the last digits -- compare GPU runs with GPU runs.
        """
        import os
        torch = _torch()
        choice = os.environ.get("PRISM_DEVICE", self.device)
        if choice == "auto":
            choice = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(choice)

    def _dev(self):
        """The fitted device, or CPU when a batch is built before any fit."""
        return self._device if self._device is not None else _torch().device("cpu")

    # ------------------------------------------------------------------ data
    def _cache(self, side: str):
        """Pad every reactant once, not once per minibatch per epoch.

        The graphs never change during a fit, so padding them on every batch of
        every epoch repeated the same work tens of thousands of times. Here each
        side is padded once to its widest molecule and a batch becomes an index
        into that tensor.
        """
        torch = _torch()
        store = self._caches.get(side)
        if store is None:
            graphs = self.acyl_graphs if side == "acyl" else self.amine_graphs
            ids = sorted(graphs)
            padded = graph_mod.pad_batch([graphs[i] for i in ids])
            store = {k: torch.as_tensor(v).to(self._dev()) for k, v in padded.items()}
            store["position"] = {g: i for i, g in enumerate(ids)}
            self._caches[side] = store
        return store

    def _side(self, side: str, ids: Sequence[int]):
        torch = _torch()
        store = self._cache(side)
        take = torch.as_tensor([store["position"][int(i)] for i in ids],
                               device=self._dev())
        mask = store["mask"].index_select(0, take)
        # Trim back to the widest molecule actually present, so the batch has
        # the same shape it had when it was padded from scratch.
        n = int(mask.sum(1).max().item())
        return {"atoms": store["atoms"].index_select(0, take)[:, :n],
                "adjacency": store["adjacency"].index_select(0, take)[:, :n, :n],
                "mask": mask[:, :n],
                "centre": store["centre"].index_select(0, take)}

    def _batch(self, rows: pd.DataFrame, X: np.ndarray, index: Sequence[int]):
        torch = _torch()
        index = list(index)
        batch = {f"acyl_{k}": v for k, v in
                 self._side("acyl", [rows.acyl.iloc[i] for i in index]).items()}
        batch.update({f"amine_{k}": v for k, v in
                      self._side("amine", [rows.amine.iloc[i] for i in index]).items()})
        if self.use_descriptors:
            batch["descriptors"] = torch.as_tensor(X[index], dtype=torch.float32,
                                                   device=self._dev())
        return batch

    def covers(self, rows: pd.DataFrame) -> np.ndarray:
        """Mask of reactions whose two reactants both have a graph."""
        return np.array([int(a) in self.acyl_graphs and int(m) in self.amine_graphs
                         for a, m in zip(rows.acyl, rows.amine)])

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, rows: pd.DataFrame = None):
        torch = _torch()
        if rows is None:
            raise ValueError("the graph model needs the reaction table; pass rows=")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._device = self._resolve_device()
        self._caches = {}

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._n_descriptors = X.shape[1] if self.use_descriptors else 0
        width = next(iter(self.acyl_graphs.values())).n_features
        self.module = _build_module(self.hidden, self.layers, self.dropout,
                                    self._n_descriptors, width, self.encoder
                                    ).to(self._device)
        optimiser = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        classification = self.task == "classification"
        loss_fn = (torch.nn.BCEWithLogitsLoss() if classification
                   else torch.nn.SmoothL1Loss())

        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.module.train()
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                index = order[start:start + self.batch_size]
                if len(index) < 2:
                    continue
                batch = self._batch(rows, X, index)
                class_logit, rate = self.module(batch)
                output = class_logit if classification else rate
                loss = loss_fn(output, torch.as_tensor(y[index], device=self._device))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        return self

    # --------------------------------------------------------------- predict
    def _forward(self, X: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        torch = _torch()
        self.module.eval()
        X = np.asarray(X, dtype=np.float32)
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                index = range(start, min(start + self.batch_size, len(rows)))
                class_logit, rate = self.module(self._batch(rows, X, index))
                out.append((class_logit if self.task == "classification"
                            else rate).detach().cpu().numpy())
        return np.concatenate(out)

    def predict(self, X: np.ndarray, rows: pd.DataFrame = None) -> np.ndarray:
        values = self._forward(X, rows)
        if self.task != "classification":
            return values
        return (values >= 0.0).astype(int)

    def predict_proba(self, X: np.ndarray, rows: pd.DataFrame = None) -> np.ndarray:
        logit = self._forward(X, rows)
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - p, p])
