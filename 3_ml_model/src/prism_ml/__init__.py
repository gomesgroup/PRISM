"""Reactivity and rate-constant models for PRISM amide couplings.

Predict for an unseen acyl chloride / amine pair with the saved model:

    from prism_ml.predictor import Predictor
    model = Predictor.load()                       # model/ in the repository
    pairs = pd.DataFrame({"acyl_smiles": ["O=C(Cl)c1ccccc1"],
                          "amine_smiles": ["Nc1ccccc1"]})
    model.predict_smiles(pairs)

The workflow that produced and evaluated that model lives in scripts/.
"""

from . import benchmark, config, data, features, holdout, metrics, model, plotting, splits
from .model import Prediction, ReactivityRateModel

__version__ = "1.0.0"
__all__ = ["benchmark", "config", "data", "features", "holdout", "metrics", "model",
           "plotting", "splits", "Prediction", "ReactivityRateModel", "__version__"]
