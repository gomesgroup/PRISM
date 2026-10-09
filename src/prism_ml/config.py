"""Paths, feature-set definitions and the fixed model hyperparameters.

Everything a reader needs in order to know *what was fitted* lives here, so
that the rest of the package contains mechanism rather than choices.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths -----
# Every path is resolved from the package root, so the package is
# self-contained: it can be cloned and run without anything beside it. Each
# location honours an environment override for a checkout that keeps its data
# elsewhere.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PRISM_DATA_ROOT", PACKAGE_ROOT / "data"))

#: The Excel workbooks the models are trained and evaluated on.
DATA_DIR = Path(os.environ.get("PRISM_DATA_DIR", DATA_ROOT / "datasheets"))
#: Optimised reactant geometries (acyls/, amines/), used only to regenerate
#: the AIMNet2 atom features for the graph-network benchmark.
XYZ_DIR = Path(os.environ.get("PRISM_XYZ_DIR", DATA_ROOT / "xyz"))

#: AIMNet2 atom features, written by scripts/extract_aimnet_features.py.
AIMNET_NPZ = Path(os.environ.get("PRISM_AIMNET_NPZ", DATA_ROOT / "aimnet2_atoms.npz"))
#: The TorchScript AIMNet2 model those features are computed with: the
#: wB97M-D3/CPCM checkpoint distributed with LoQI (data/aimnet2/README.md).
#: Needed only to regenerate AIMNET_NPZ from geometry; the tracked features run
#: without it. Override with PRISM_AIMNET_MODEL to use a checkpoint elsewhere.
AIMNET_MODEL = Path(os.environ.get(
    "PRISM_AIMNET_MODEL", DATA_ROOT / "aimnet2" / "wb97m_cpcms_v2_0.jpt"))

#: Reactions with a fitted rate constant (1.5x hydrolysis control), one row per
#: reaction; built by scripts/build_rate_tables.py.
RATES_XLSX = Path(os.environ.get("PRISM_RATES_XLSX", DATA_DIR / "PRISM_rates.xlsx"))
#: Replicate batches of one reactant, recorded in the rate workbooks under the
#: name below and, for 2B and 3C, under the labels of other acyl chlorides. Rows
#: with these names belong to the reactant given here; replicate rows of a pair
#: are combined by prism_ml.data.combine_rate_rows / combine_class_rows.
REPLICATE_BATCHES = {"4-fluorobenzoyl chloride-1A": 1,
                     "4-fluorobenzoyl chloride-2B": 1,
                     "4-fluorobenzoyl chloride-3C": 1}
#: Reactions without one, carrying a reactivity class instead.
EXCLUDED_XLSX = Path(os.environ.get("PRISM_EXCLUDED_XLSX",
                                    DATA_DIR / "PRISM_rates_unmeasurable.xlsx"))
#: Per-reactant descriptors behind every reported result (benchmark, holdout).
FEATURES_XLSX = DATA_DIR / "PRISM_substrate_features.xlsx"
#: The same reactants' descriptors recomputed from SMILES by
#: prism_ml.descriptors (scripts/compute_descriptors.py). The saved model is
#: trained on these, because that is exactly what predict.py computes for an unseen
#: molecule -- the stored table above came from a different conformer
#: pipeline and does not mix with on-the-fly descriptors.
FEATURES_SMILES_XLSX = DATA_DIR / "PRISM_substrate_features_smiles.xlsx"
#: Descriptors of the excluded substrate scope: sheets of the same workbook.
#: They were computed separately, with the stored table's pipeline (not from
#: SMILES), and are used only to score the excluded scope.
SCOPE_FEATURE_SHEETS = {"acyl": "excluded_acyl", "amine": "excluded_amine"}
#: Reactant identities and SMILES; the `excluded` sheet is the substrate scope.
SCOPE_XLSX = DATA_DIR / "PRISM_substrate_scope.xlsx"
#: Drug-like amines screened separately.
DRUG_SCOPE_XLSX = DATA_DIR / "PRISM_rates_drug_scope.xlsx"

#: Where results and the saved model are written.
RESULTS_DIR = PACKAGE_ROOT / "results"
MODEL_DIR = Path(os.environ.get("PRISM_MODEL_DIR", PACKAGE_ROOT / "model"))

#: Method comparisons (NMR, Hammett) live apart from the ML workflow.
ANALYSIS_ROOT = PACKAGE_ROOT / "analysis"
ANALYSIS_DIR = Path(os.environ.get("PRISM_ANALYSIS_DIR", ANALYSIS_ROOT / "results"))
NMR_XLSX = ANALYSIS_ROOT / "data" / "PRISM_vs_nmr.xlsx"
HAMMETT_XLSX = ANALYSIS_ROOT / "data" / "PRISM_vs_hammett.xlsx"

# ------------------------------------------------------------- features -----
# Regular expressions selecting descriptor columns, by what the descriptor
# costs to obtain. The tiers are cumulative: each contains everything above it.
#
#   tier1      one 3D conformer; buried volume, Sterimol, SASA,
#              pyramidalization, dispersion. No quantum chemistry.
#   tier2      + one xTB single point: frontier orbitals, global reactivity
#              indices, atomic partial charges.
#   tier3      + two further single points (the N+1 and N-1 electron states)
#              for Fukui indices.
#   tier_all   + a Hessian: local force constants and frequencies.
#
# `tier2_reduced` is tier2 without the electron affinity and the chemical
# hardness derived from it. Both depend on the anion SCF, the least numerically
# stable part of an xTB run, and both were measured to carry no usable
# classification signal while costing accuracy on unseen substrates.
GEOMETRY = r"BV_|L_|B1_|B5_|SA$|SV$|PYR$|DISP_|num_Hs$"
XTB_SP = r"IP$|EA$|HOMO$|LUMO$|global_E$|global_N$|hardness$|Charges_"
XTB_SP_REDUCED = r"IP$|HOMO$|LUMO$|global_E$|global_N$|Charges_"

FEATURE_SETS = {
    "tier1": dict(
        include=rf"^({GEOMETRY})",
        label="Tier 1",
        cost="1 conformer",
        description="Geometry only: buried volume, Sterimol, SASA, "
                    "pyramidalization, dispersion."),
    "tier2": dict(
        include=rf"^({GEOMETRY}|{XTB_SP})",
        label="Tier 2",
        cost="+1 xTB SP",
        description="Tier 1 plus one xTB single point: IP, EA, HOMO, LUMO, "
                    "global electrophilicity and nucleophilicity, hardness, "
                    "atomic charges."),
    "tier2_reduced": dict(
        include=rf"^({GEOMETRY}|{XTB_SP_REDUCED})",
        label="Tier 2 (reduced)",
        cost="+1 xTB SP",
        description="Tier 2 without electron affinity and hardness, both of "
                    "which derive from the anion SCF."),
    "tier3_reduced": dict(
        include=r"^(?!FCS_|FREQS_|pka_|EA$|hardness$).",
        label="Tier 3 (reduced)",
        cost="+3 xTB SP",
        description="Tier 3 without electron affinity and hardness, the same "
                    "two columns Tier 2 (reduced) drops."),
    "tier3": dict(
        include=r"^(?!FCS_|FREQS_|pka_).",
        label="Tier 3",
        cost="+3 xTB SP",
        description="Tier 2 plus Fukui indices, which need the N+1 and N-1 "
                    "electron states."),
    # Defined for completeness and excluded from the benchmark. Its only
    # distinguishing columns are the Hessian block, which was never computed
    # for one amine — and a Tier-all model with that block removed is Tier 3
    # exactly, column for column. Benchmarking it would therefore either drop
    # that reactant's reactions, making the comparison unequal, or reduce to a
    # duplicate of Tier 3.
    "tier_all": dict(
        include=r"^(?!pka_).",
        label="Tier all",
        cost="+Hessian",
        description="Tier 3 plus local force constants and vibrational "
                    "frequencies from a Hessian. Not benchmarked: see above."),
}

#: Candidate order used in tables and figures. Every set here describes every
#: reaction in the pool, so the comparison needs no subsetting and the whole
#: dataset is used.
BENCHMARK_ORDER = ["tier1", "tier2", "tier2_reduced", "tier3", "tier3_reduced"]

#: The feature set the study selected. Named here once so that changing the
#: recommendation is a one-line edit rather than a search.
SELECTED = "tier2_reduced"

#: Acid/base constants are available but sparse. Opt in with --with-pka; the
#: benchmark reports how many reactions survive, which is the argument for or
#: against using them.
PKA_PATTERN = r"^pka_"

# --------------------------------------------------------------- model ------
#: HistGradientBoosting settings. Deterministic at this data size: no
#: subsampling, and early stopping is inactive below 10k rows.
BOOSTING = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
                random_state=0)

#: Learner families compared under the same folds and the same feature set.
#: Each entry names the classifier and regressor to use and the keyword
#: arguments for each; `prism_ml.model` resolves the names against
#: scikit-learn. Every estimator is either deterministic at this data size or
#: given a fixed seed, so re-running changes nothing.
#:
#: The list spans the families a reader would reasonably ask about — one
#: additive ensemble, two averaging ensembles, a linear model, a local method,
#: and the uninformative floor — rather than many variants of one idea.
MODEL_FAMILIES = {
    "boosted_trees": dict(
        label="Gradient-boosted trees",
        classifier=("sklearn.ensemble", "HistGradientBoostingClassifier", BOOSTING),
        regressor=("sklearn.ensemble", "HistGradientBoostingRegressor", BOOSTING),
        note="Additive ensemble; trees fitted in sequence on the residual."),
    # Further boosting implementations, each on the same budget as
    # `boosted_trees` -- 200 rounds, learning rate 0.1 where the library takes
    # one, a fixed seed -- and otherwise its own defaults. None is tuned, so a
    # difference between them is the implementation, not the effort spent on it.
    # XGBoost, LightGBM and CatBoost are optional; their import is deferred to
    # fit time, so the package runs without them unless one is requested.
    "gradient_boosting": dict(
        label="Gradient boosting (exact)",
        classifier=("sklearn.ensemble", "GradientBoostingClassifier",
                    dict(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0)),
        regressor=("sklearn.ensemble", "GradientBoostingRegressor",
                   dict(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0)),
        note="Classic gradient boosting with exact split search."),
    "adaboost": dict(
        label="AdaBoost",
        classifier=("sklearn.ensemble", "AdaBoostClassifier",
                    dict(n_estimators=200, random_state=0)),
        regressor=("sklearn.ensemble", "AdaBoostRegressor",
                   dict(n_estimators=200, random_state=0)),
        note="Reweights the examples previous learners got wrong."),
    "xgboost": dict(
        label="XGBoost",
        classifier=("xgboost", "XGBClassifier",
                    dict(n_estimators=200, learning_rate=0.1, tree_method="hist",
                         random_state=0, n_jobs=1, verbosity=0)),
        regressor=("xgboost", "XGBRegressor",
                   dict(n_estimators=200, learning_rate=0.1, tree_method="hist",
                        random_state=0, n_jobs=1, verbosity=0)),
        note="Regularised gradient boosting; second-order split gain."),
    "lightgbm": dict(
        label="LightGBM",
        classifier=("lightgbm", "LGBMClassifier",
                    dict(n_estimators=200, learning_rate=0.1, num_leaves=15,
                         random_state=0, n_jobs=1, deterministic=True, verbose=-1)),
        regressor=("lightgbm", "LGBMRegressor",
                   dict(n_estimators=200, learning_rate=0.1, num_leaves=15,
                        random_state=0, n_jobs=1, deterministic=True, verbose=-1)),
        note="Histogram boosting with leaf-wise growth; 15 leaves as boosted_trees."),
    "catboost": dict(
        label="CatBoost",
        classifier=("catboost", "CatBoostClassifier",
                    dict(iterations=200, learning_rate=0.1, depth=6, random_seed=0,
                         thread_count=1, verbose=0, allow_writing_files=False)),
        regressor=("catboost", "CatBoostRegressor",
                   dict(iterations=200, learning_rate=0.1, depth=6, random_seed=0,
                        thread_count=1, verbose=0, allow_writing_files=False)),
        note="Ordered boosting on symmetric trees; resists target leakage."),
    # A split model: each head borrows the estimator of another family. The
    # classifier is picked on CV accuracy and the regressor on CV R2, instead of
    # one learner chosen on their average -- which hid the strongest classifier
    # (exact gradient boosting) behind its weak rate head.
    "gb_exact_x_histgb": dict(
        label="Split: GB (exact) + HistGB",
        classifier_family="gradient_boosting",
        regressor_family="boosted_trees",
        note="Exact gradient boosting decides reactivity; histogram gradient "
             "boosting predicts the rate constant."),
    "random_forest": dict(
        label="Random forest",
        classifier=("sklearn.ensemble", "RandomForestClassifier",
                    dict(n_estimators=400, random_state=0, n_jobs=1)),
        regressor=("sklearn.ensemble", "RandomForestRegressor",
                   dict(n_estimators=400, random_state=0, n_jobs=1)),
        note="Averaging ensemble over bootstrapped trees."),
    "extra_trees": dict(
        label="Extremely randomised trees",
        classifier=("sklearn.ensemble", "ExtraTreesClassifier",
                    dict(n_estimators=400, random_state=0, n_jobs=1)),
        regressor=("sklearn.ensemble", "ExtraTreesRegressor",
                   dict(n_estimators=400, random_state=0, n_jobs=1)),
        note="As above, with split points drawn at random rather than optimised."),
    "linear": dict(
        label="Linear",
        classifier=("sklearn.linear_model", "LogisticRegression",
                    dict(max_iter=5000, random_state=0)),
        regressor=("sklearn.linear_model", "RidgeCV", dict()),
        note="Logistic regression and ridge regression on the same features."),
    "nearest_neighbours": dict(
        label="k-nearest neighbours",
        classifier=("sklearn.neighbors", "KNeighborsClassifier", dict(n_neighbors=5)),
        regressor=("sklearn.neighbors", "KNeighborsRegressor", dict(n_neighbors=5)),
        note="Local method; predicts from the five most similar training pairs."),
    # Reads molecular graphs rather than the descriptor table, so a difference
    # between this and the others is learner *and* representation together.
    # Needs torch and rdkit; `needs_graphs` tells the model to construct it
    # from the graphs rather than from a scikit-learn class path.
    "graph_network": dict(
        label="Graph network (bonds)",
        needs_graphs=True,
        graph_source="smiles",
        classifier=dict(task="classification", epochs=80, hidden=64, layers=3,
                        encoder="bond"),
        regressor=dict(task="regression", epochs=80, hidden=64, layers=3,
                       encoder="bond"),
        note="Message passing over bonds; built from SMILES."),
    # Atom features from AIMNet2, a neural potential run over optimised
    # geometries: 256 learned atomic features, 256 AIM features and the partial
    # charge per atom. They already encode each atom's environment, so the
    # encoder attends over a complete graph rather than along bonds.
    "graph_network_aimnet": dict(
        label="Graph network (AIMNet2)",
        needs_graphs=True,
        graph_source="aimnet2",
        classifier=dict(task="classification", epochs=80, hidden=64, layers=2,
                        encoder="attention"),
        regressor=dict(task="regression", epochs=80, hidden=64, layers=2,
                       encoder="attention"),
        note="Attention over AIMNet2 atom features from optimised geometries."),
    "uninformative": dict(
        label="Uninformative baseline",
        classifier=("sklearn.dummy", "DummyClassifier", dict(strategy="prior")),
        regressor=("sklearn.dummy", "DummyRegressor", dict(strategy="mean")),
        note="Majority class and training mean; the floor any model must clear."),
}

#: The learner the study selected: highest mean joint score in repeated
#: class-stratified validation on the selected feature set (SELECTION_CV).
SELECTED_FAMILY = "lightgbm"
#: The learner held fixed while feature sets were compared. Tier 2 (reduced)
#: was chosen with this learner, so the feature benchmark keeps it; the learner
#: comparison then runs on the chosen feature set.
FEATURE_BENCHMARK_FAMILY = "boosted_trees"

#: The held-out test set, fixed. Chosen with the class-stratified rule below
#: (prism_ml.splits.build_split) during development; the rule's picks depend on
#: the rate tables, so the set is pinned. Set to None to apply the rule to the
#: current data instead.
FIXED_TEST_SET = {"acyl": (2, 18, 25), "amine": (18, 21, 23)}

#: Reactants held out of the test set by fiat rather than by the split rule.
#: Amine 34's descriptors were generated by a different pipeline from every
#: other reactant's, so scoring it as an unseen reactant measures that
#: mismatch as much as it measures chemistry.
FORCE_TRAIN_AMINES = (34,)
FORCE_TRAIN_ACYLS = ()

#: Split rule: a class with at most SMALL_CLASS members contributes no test
#: representative and stays wholly in the pool, so leave-one-reactant-out can
#: still hold one of its members out with a sibling in training. A class with
#: more than LARGE_CLASS members contributes two; the rest contribute one.
SMALL_CLASS = 2
LARGE_CLASS = 10

#: Model selection. Candidates are ranked by repeated class-stratified
#: validation inside the training pool: STRATIFIED_DRAWS draws (seeds from
#: STRATIFIED_FIRST_SEED), each holding out reactants by the same rule as the
#: test set but with members chosen at random. Leave-one-reactant-out scores are
#: reported alongside. "loro" ranks by those instead.
SELECTION_CV = "stratified"
STRATIFIED_DRAWS = 20
STRATIFIED_FIRST_SEED = 1

#: Leave-one-reactant-out settings.
MIN_FOLD_ROWS = 5
PAIR_FOLDS = 40
FOLD_SEED = 0
