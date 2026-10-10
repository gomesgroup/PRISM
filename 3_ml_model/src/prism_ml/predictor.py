"""Save, load and apply the selected model.

The saved model is a directory:

    model/
      prism_model.joblib   the fitted two-head model (scaler + both heads)
      metadata.json        feature set, learners, input columns, training data,
                           software versions, and the decision threshold

`Predictor` takes descriptor rows for an acyl chloride and an amine, assembles
them in the column order the model was trained on, and returns

    p_reactive   probability the pair reacts
    reactive     1 if p_reactive >= threshold
    log10_k      predicted log10 rate constant (every row)
    log10_k_gated  the same, but blank where the pair is predicted un-reactive

A pair missing any required descriptor is refused rather than imputed.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

MODEL_FILE, META_FILE = "prism_model.joblib", "metadata.json"


def save(model, columns: Sequence[str], feature_set: str, out_dir: Path,
         training: Optional[dict] = None) -> Path:
    """Write a fitted ReactivityRateModel and the metadata needed to reuse it."""
    import joblib
    import sklearn
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE)
    spec = config.MODEL_FAMILIES[model.family]
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_set": feature_set,
        "feature_set_label": config.FEATURE_SETS[feature_set]["label"],
        "family": model.family,
        "classifier": spec.get("classifier_family", model.family),
        "regressor": spec.get("regressor_family", model.family),
        "threshold": model.threshold,
        "columns": list(columns),
        "training": training or {},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     **_descriptor_versions()},
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return out_dir


class Predictor:
    """A saved model, ready to score unseen acyl chloride / amine pairs."""

    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.columns = list(metadata["columns"])
        self.acyl_columns = [c.split(":", 1)[1] for c in self.columns if c.startswith("acyl:")]
        self.amine_columns = [c.split(":", 1)[1] for c in self.columns if c.startswith("amine:")]

    @classmethod
    def load(cls, model_dir: Path = None) -> "Predictor":
        import joblib
        model_dir = Path(model_dir or config.MODEL_DIR)
        if not (model_dir / MODEL_FILE).exists():
            raise FileNotFoundError(f"no saved model in {model_dir}; run scripts/train_final.py")
        meta = json.loads((model_dir / META_FILE).read_text())
        return cls(joblib.load(model_dir / MODEL_FILE), meta)

    def _row(self, acyl: Dict[str, float], amine: Dict[str, float]) -> np.ndarray:
        missing = ([f"acyl:{c}" for c in self.acyl_columns if _blank(acyl.get(c))] +
                   [f"amine:{c}" for c in self.amine_columns if _blank(amine.get(c))])
        if missing:
            raise ValueError(f"missing descriptors: {missing}")
        return np.array([float(acyl[c]) for c in self.acyl_columns] +
                        [float(amine[c]) for c in self.amine_columns])

    def predict_descriptors(self, acyl_rows: Sequence[Dict[str, float]],
                            amine_rows: Sequence[Dict[str, float]]) -> pd.DataFrame:
        """Score pairs given one descriptor dict per acyl chloride and amine."""
        X = np.vstack([self._row(a, m) for a, m in zip(acyl_rows, amine_rows)])
        pred = self.model.predict(X)
        return pd.DataFrame({"p_reactive": np.round(pred.probability, 4),
                             "reactive": pred.reactive,
                             "log10_k": np.round(pred.log_k, 3),
                             "log10_k_gated": np.round(pred.gated_log_k(), 3)})

    def predict_smiles(self, pairs: pd.DataFrame, with_progress: bool = True) -> pd.DataFrame:
        """Score pairs given their SMILES (columns acyl_smiles, amine_smiles).

        Descriptors are computed once per distinct molecule. A pair whose
        descriptors cannot be computed is reported with an error and no score.
        """
        from . import descriptors
        cache: Dict[tuple, object] = {}

        def describe(smiles: str, kind: str):
            key = (smiles, kind)
            if key not in cache:
                try:
                    cache[key] = descriptors.compute(smiles, kind)
                    if with_progress:
                        print(f"  descriptors: {kind:5s} {smiles}")
                except Exception as exc:  # noqa: BLE001 - reported per pair
                    cache[key] = exc
            return cache[key]

        out = []
        for _, pair in pairs.iterrows():
            acyl = describe(str(pair.acyl_smiles), "acyl")
            amine = describe(str(pair.amine_smiles), "amine")
            row = pair.to_dict()
            errors = [str(e) for e in (acyl, amine) if isinstance(e, Exception)]
            if not errors:
                try:
                    row.update(self.predict_descriptors([acyl], [amine]).iloc[0].to_dict())
                except ValueError as exc:
                    errors.append(str(exc))
            row["error"] = "; ".join(errors)
            out.append(row)
        frame = pd.DataFrame(out)
        if "reactive" in frame:
            frame["reactive"] = frame["reactive"].astype("Int64")   # 0/1, blank if unscored
        return frame


def _descriptor_versions() -> dict:
    """Versions of the descriptor stack, when installed -- conformers and xTB
    values can shift between releases, which shifts predictions."""
    from importlib import metadata
    out = {}
    for dist in ("rdkit", "morfeus-ml", "xtb"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))
