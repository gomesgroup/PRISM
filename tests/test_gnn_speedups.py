"""The two optimisations in the graph model must not change its arithmetic.

Both changes below were made for speed alone. Each is claimed to be exactly
equivalent to the slower form it replaced, so each gets a test that compares it
against that slower form rather than against a stored number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Optional dependencies: skip, rather than fail to collect, without them.
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_ml import config, graph  # noqa: E402
from prism_ml.gnn import GraphNetEstimator  # noqa: E402


def test_factorised_attention_matches_the_pair_tensor():
    """score[i, j] = W . [x_i, x_j] split into two halves of W.

    The model used to build the full (batch, n, n, 2 * hidden) pair tensor and
    project it. Because the layer has no bias the projection is separable, and
    this checks the separated form against the original one.
    """
    functional = torch.nn.functional
    torch.manual_seed(0)
    batch, n, hidden = 3, 7, 16
    x = torch.randn(batch, n, hidden)
    weight = torch.randn(1, 2 * hidden)

    left = x.unsqueeze(2).expand(-1, -1, n, -1)
    pairs = torch.cat([left, left.transpose(1, 2)], dim=-1)
    reference = functional.leaky_relu(functional.linear(pairs, weight)).squeeze(-1)

    source = x @ weight[:, :hidden].t()
    target = x @ weight[:, hidden:].t()
    factorised = functional.leaky_relu(source + target.transpose(1, 2))

    assert torch.allclose(reference, factorised, atol=1e-6)


def _toy_graphs(count, width=4):
    rng = np.random.default_rng(0)
    out = {}
    for i in range(1, count + 1):
        n = 3 + i % 4
        adjacency = np.zeros((n, n), dtype=np.float32)
        for a in range(n - 1):
            adjacency[a, a + 1] = adjacency[a + 1, a] = 1.0
        out[i] = graph.MolGraph(atoms=rng.normal(size=(n, width)).astype(np.float32),
                                adjacency=adjacency, centre=0, source="test")
    return out


def test_cached_padding_matches_padding_from_scratch():
    """Padding once and indexing must equal padding the batch each time."""
    acyl, amine = _toy_graphs(5), _toy_graphs(6)
    rows = pd.DataFrame({"acyl": [1, 3, 5, 2], "amine": [2, 2, 6, 4]})
    estimator = GraphNetEstimator(acyl, amine)

    index = [0, 1, 2, 3]
    cached = estimator._batch(rows, np.zeros((4, 1), dtype=np.float32), index)
    expected = {
        "acyl": graph.pad_batch([acyl[int(rows.acyl.iloc[i])] for i in index]),
        "amine": graph.pad_batch([amine[int(rows.amine.iloc[i])] for i in index]),
    }
    for side, padded in expected.items():
        for key, value in padded.items():
            got = cached[f"{side}_{key}"].numpy()
            assert got.shape == value.shape, f"{side}_{key} shape"
            assert np.array_equal(got, value), f"{side}_{key} values"
