"""A split model must be exactly its two borrowed heads, nothing more.

The split family's whole claim is that its classifier *is* the classifier of
one family and its regressor *is* the regressor of another. That is checked
directly: each head's predictions must equal, to the bit, those of a model
built from the family it delegates to.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_ml import config  # noqa: E402
from prism_ml.model import ReactivityRateModel  # noqa: E402

SPLIT = "gb_exact_x_histgb"


def _data(n=160, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    reactive = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    logk = np.where(reactive == 1, X[:, 2] - X[:, 3] + rng.normal(scale=0.2, size=n), np.nan)
    return X, reactive, logk


def test_split_family_declares_two_existing_families():
    spec = config.MODEL_FAMILIES[SPLIT]
    assert spec["classifier_family"] in config.MODEL_FAMILIES
    assert spec["regressor_family"] in config.MODEL_FAMILIES
    assert spec["classifier_family"] != spec["regressor_family"]


def test_split_heads_reproduce_their_source_families():
    X, reactive, logk = _data()
    spec = config.MODEL_FAMILIES[SPLIT]
    split = ReactivityRateModel(family=SPLIT).fit(X, reactive, logk).predict(X)
    clf = ReactivityRateModel(family=spec["classifier_family"]).fit(X, reactive, logk).predict(X)
    reg = ReactivityRateModel(family=spec["regressor_family"]).fit(X, reactive, logk).predict(X)
    np.testing.assert_array_equal(split.probability, clf.probability)
    np.testing.assert_array_equal(split.reactive, clf.reactive)
    np.testing.assert_array_equal(split.log_k, reg.log_k)
    # and it really is a mix: neither source alone gives both outputs
    assert not np.array_equal(clf.log_k, reg.log_k)
