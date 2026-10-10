"""The saved-model layer: round trip, column order, and refusal to impute."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_ml import predictor  # noqa: E402
from prism_ml.model import ReactivityRateModel  # noqa: E402

ACYL, AMINE = ["a1", "a2", "a3"], ["m1", "m2"]
COLUMNS = [f"acyl:{c}" for c in ACYL] + [f"amine:{c}" for c in AMINE]


def _fitted(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, len(COLUMNS)))
    reactive = (X[:, 0] + X[:, 3] > 0).astype(int)
    logk = np.where(reactive == 1, X[:, 1] - X[:, 4], np.nan)
    model = ReactivityRateModel(family="boosted_trees").fit(X, reactive, logk)
    predictor.save(model, COLUMNS, "tier2_reduced", tmp_path, {"reactions": 150})
    return model, X


def test_round_trip_reproduces_predictions(tmp_path):
    model, X = _fitted(tmp_path)
    loaded = predictor.Predictor.load(tmp_path)
    assert loaded.columns == COLUMNS
    acyl = [dict(zip(ACYL, row[:3])) for row in X[:5]]
    amine = [dict(zip(AMINE, row[3:])) for row in X[:5]]
    got = loaded.predict_descriptors(acyl, amine)
    np.testing.assert_allclose(got.p_reactive, np.round(model.predict(X[:5]).probability, 4))
    np.testing.assert_allclose(got.log10_k, np.round(model.predict(X[:5]).log_k, 3))


def test_gated_rate_is_blank_where_unreactive(tmp_path):
    _, X = _fitted(tmp_path)
    loaded = predictor.Predictor.load(tmp_path)
    got = loaded.predict_descriptors([dict(zip(ACYL, r[:3])) for r in X],
                                     [dict(zip(AMINE, r[3:])) for r in X])
    assert got.loc[got.reactive == 0, "log10_k_gated"].isna().all()
    assert got.loc[got.reactive == 1, "log10_k_gated"].notna().all()


def test_missing_descriptor_is_refused_not_imputed(tmp_path):
    _fitted(tmp_path)
    loaded = predictor.Predictor.load(tmp_path)
    with pytest.raises(ValueError, match="missing descriptors"):
        loaded.predict_descriptors([{"a1": 0.1, "a2": 0.2}], [{"m1": 0.0, "m2": 0.0}])


def test_shipped_model_matches_its_metadata():
    """The model in model/ must load and agree with what its metadata claims."""
    shipped = Path(__file__).resolve().parents[1] / "model"
    if not (shipped / predictor.MODEL_FILE).exists():
        pytest.skip("no shipped model")
    loaded = predictor.Predictor.load(shipped)
    assert loaded.model.family == loaded.metadata["family"]
    assert len(loaded.columns) == len(loaded.model.scaler.mean_)
