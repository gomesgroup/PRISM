"""Two heads on one descriptor table: a reactivity call and a rate constant.

The two heads share a feature table and a selection decision. They do not share
parameters, and they do not exchange predictions: passing the classifier's
output to the rate-constant head was measured across five feature sets and
lowered cross-validated performance every time, so the coupling is
deliberately absent.

The classification target is whether the pair reacts at all:

**reactive** — a rate constant was measurable, or the reaction was too fast to
resolve (`instant`).

**unreactive** — no usable rate constant, for any of three reasons: the
amidation rate constant is smaller than that of the competing hydrolysis of
the acyl chloride; no product forms at all; or a side reaction of the acyl
chloride or the amine is favoured over amidation. The source data records all
three as `slow`, and the model does not distinguish between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from importlib import import_module

from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class Prediction:
    """What the model says about a set of reactions."""

    probability: np.ndarray     #: P(reactive)
    reactive: np.ndarray        #: the class call at the decision threshold
    log_k: np.ndarray           #: predicted log10 of the rate constant, every row
    emitted: np.ndarray         #: rows the gate lets through

    def gated_log_k(self) -> np.ndarray:
        """Rate constant where the class head says reactive, NaN elsewhere.

        This is what a caller should show a chemist: a rate constant for a pair
        the model expects not to react is not a useful number.
        """
        out = np.full(self.log_k.shape, np.nan)
        out[self.emitted] = self.log_k[self.emitted]
        return out


class ReactivityRateModel:
    """Predicts whether a pair reacts, and its rate constant if it does.

    Args:
        threshold: probability above which a pair is called reactive.
        family: learner family, resolved against :data:`config.MODEL_FAMILIES`.
            ``None`` means :data:`config.SELECTED_FAMILY`, read at call time —
            a module-level default would be frozen at import and would silently
            ignore any later change.
            A family normally supplies both heads. A *split* family instead
            names ``classifier_family`` and ``regressor_family``, and each head
            is built from that family's estimator -- so one learner can decide
            reactivity and another predict the rate constant.
        overrides: keyword arguments merged into both estimators.
        graphs: ``(acyl_graphs, amine_graphs)`` from :mod:`prism_ml.graph`,
            required only by families that read molecular structure.
    """

    def __init__(self, threshold: float = 0.5,
                 family: Optional[str] = None,
                 overrides: Optional[dict] = None,
                 graphs: Optional[tuple] = None):
        family = family or config.SELECTED_FAMILY
        if family not in config.MODEL_FAMILIES:
            raise ValueError(f"unknown model family '{family}'; "
                             f"choose from {sorted(config.MODEL_FAMILIES)}")
        self.threshold = threshold
        self.family = family
        self.overrides = dict(overrides or {})
        self.graphs = graphs
        self.scaler: Optional[StandardScaler] = None
        self.classifier = None
        self.regressor = None
        self.n_rate_rows: int = 0

    def _build(self, role: str):
        """Instantiate one head from the family specification."""
        source = self.family
        spec = config.MODEL_FAMILIES[source]
        # A split family delegates each head to the family it names.
        if f"{role}_family" in spec:
            source = spec[f"{role}_family"]
            spec = config.MODEL_FAMILIES[source]
        if spec.get("needs_graphs"):
            if self.graphs is None:
                raise ValueError(
                    f"family '{source}' reads molecular graphs; construct "
                    f"the model with graphs=(acyl_graphs, amine_graphs)")
            from .gnn import GraphNetEstimator
            acyl_graphs, amine_graphs = self.graphs
            return GraphNetEstimator(acyl_graphs, amine_graphs,
                                     **{**spec[role], **self.overrides})
        module_name, class_name, kwargs = spec[role]
        estimator = getattr(import_module(module_name), class_name)
        return estimator(**{**kwargs, **self.overrides})

    @staticmethod
    def _call(estimator, method: str, X, rows=None, *args):
        """Call an estimator, passing rows only to those that ask for them."""
        fn = getattr(estimator, method)
        if getattr(estimator, "needs_rows", False):
            return fn(X, *args, rows=rows)
        return fn(X, *args)

    def fit(self, X: np.ndarray, reactive: Sequence[int],
            logk: Sequence[float],
            rows: "Optional[object]" = None) -> "ReactivityRateModel":
        """Fit both heads.

        The classifier sees every row. The rate-constant head sees only rows
        with a measured value: ``logk`` is NaN for the rest, which are left- or
        right-censored observations rather than missing ones, and recruiting
        them would need a censored likelihood this model does not have.
        """
        X = np.asarray(X, dtype=float)
        reactive = np.asarray(reactive, dtype=int)
        logk = np.asarray(logk, dtype=float)
        if not (len(X) == len(reactive) == len(logk)):
            raise ValueError("X, reactive and logk must have the same length")

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self._rows = rows
        self.classifier = self._call(self._build("classifier"), "fit", Xs, rows, reactive)

        measured = np.isfinite(logk)
        self.n_rate_rows = int(measured.sum())
        if self.n_rate_rows < 10:
            raise ValueError(f"only {self.n_rate_rows} rows with a rate constant; "
                             f"need >= 10")
        rows_measured = rows.iloc[measured].reset_index(drop=True) if rows is not None else None
        self.regressor = self._call(self._build("regressor"), "fit",
                                    Xs[measured], rows_measured, logk[measured])
        return self

    def predict(self, X: np.ndarray, rows: "Optional[object]" = None) -> Prediction:
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("fit() before predict()")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        probability = self._call(self.classifier, "predict_proba", Xs, rows)[:, 1]
        reactive = (probability >= self.threshold).astype(int)
        return Prediction(probability=probability, reactive=reactive,
                          log_k=self._call(self.regressor, "predict", Xs, rows),
                          emitted=reactive.astype(bool))
