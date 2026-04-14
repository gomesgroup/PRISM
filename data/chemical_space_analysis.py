"""
Chemical Space Analysis for Amines and Acyl Chlorides
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)

AMINES_CSV          = "features/descriptors_amines.csv"          
ACYL_CSV            = "features/descriptors_acyl_chlorides.csv"  
AMINES_SMILES_COL   = "smiles"              
ACYL_SMILES_COL     = "smiles"              
AMINES_LABEL_COL    = "amines"              
ACYL_LABEL_COL      = "acyl_chlorides"      
SPLIT_COL           = "split_type"
OUTPUT_DIR          = "chemical_space_output"

def load_reactants(csv_path: str, smiles_col: str, label_col: str = None) -> pd.DataFrame:
    """
    Load a reactant CSV. Validates the SMILES column exists and drops rows
    with unparseable SMILES, printing a warning for each.

    Parameters
    ----------
    csv_path  : Path to the CSV file.
    smiles_col: Name of the column containing SMILES strings.
    label_col : Optional column used as compound labels.

    Returns
    -------
    df : Cleaned DataFrame with a 'smiles' and 'label' column.
    """
    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={smiles_col: "smiles"})

    if label_col and label_col in df.columns:
        df = df.rename(columns={label_col: "label"})
    else:
        df["label"] = df["smiles"]  # fall back to SMILES as label

    invalid = [i for i, smi in df["smiles"].items()
               if Chem.MolFromSmiles(str(smi)) is None]
    if invalid:
        print(f"  [!] {len(invalid)} unparseable SMILES dropped from {csv_path}: "
              f"{df.loc[invalid, 'smiles'].tolist()}")
        df = df.drop(index=invalid).reset_index(drop=True)

    print(f"  [✓] Loaded {len(df)} compounds from {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. FINGERPRINTS
# ─────────────────────────────────────────────────────────────

def smiles_to_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    """Convert SMILES to a Morgan fingerprint (radius=2, 2048 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprint(mol)


def build_fingerprints(df: pd.DataFrame) -> list:
    return [smiles_to_fp(smi) for smi in df["smiles"]]


def compute_tanimoto_matrix(fps: list) -> np.ndarray:
    n = len(fps)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            mat[i, j] = mat[j, i] = sim
    return mat


# ─────────────────────────────────────────────────────────────
# 3. TANIMOTO HEATMAP  (per class)
# ─────────────────────────────────────────────────────────────

def plot_tanimoto_heatmap(
    df: pd.DataFrame,
    fps: list,
    title: str,
    output_path: str,
    figsize: tuple = (10, 8),
) -> np.ndarray:
    """
    Plot and save a Tanimoto similarity heatmap for a single reactant class.

    Returns
    -------
    sim_matrix : np.ndarray  (N × N)
    """
    sim_matrix = compute_tanimoto_matrix(fps)
    labels = df["label"].tolist()
    tick_fs = max(4, min(9, 120 // max(len(labels), 1)))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        sim_matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Tanimoto Similarity", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=tick_fs)
    ax.tick_params(axis="y", rotation=0,  labelsize=tick_fs)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Heatmap saved → {output_path}")
    return sim_matrix


# ─────────────────────────────────────────────────────────────
# 4. DIVERSITY SCORES  (per class)
# ─────────────────────────────────────────────────────────────

def compute_diversity_scores(fps: list, label: str) -> dict:
    """
    Compute mean ± SD pairwise Tanimoto diversity for a fingerprint set.

    Returns
    -------
    scores : dict  (mean, std, min, max diversity + n_pairs)
    """
    diversities = [
        1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps))
        for j in range(i + 1, len(fps))
    ]
    arr = np.array(diversities)
    scores = {
        "mean_diversity": float(np.mean(arr)),
        "std_diversity":  float(np.std(arr)),
        "min_diversity":  float(np.min(arr)),
        "max_diversity":  float(np.max(arr)),
        "n_pairs":        len(arr),
    }
    print(f"\n  [{label}] Diversity Scores:")
    for k, v in scores.items():
        print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
    return scores


def plot_diversity_distribution(
    fps_dict: dict,
    output_path: str,
    figsize: tuple = (9, 5),
):
    """
    Overlay pairwise diversity histograms for multiple reactant classes
    for direct visual comparison.

    Parameters
    ----------
    fps_dict : {"Amines": fps_list, "Acyl Chlorides": fps_list}
    """
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=figsize)

    for (label, fps), color in zip(fps_dict.items(), colors):
        divs = [
            1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for i in range(len(fps))
            for j in range(i + 1, len(fps))
        ]
        mean_d = np.mean(divs)
        ax.hist(divs, bins=30, alpha=0.55, label=label, color=color, edgecolor="white")
        ax.axvline(mean_d, color=color, linestyle="--", linewidth=1.8,
                   label=f"{label} mean = {mean_d:.3f}")

    ax.set_xlabel("Pairwise Tanimoto Diversity (1 – Similarity)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Pairwise Diversity Distribution by Reactant Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Diversity distribution saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST — TANIMOTO & DOMAIN OF APPLICABILITY
# ─────────────────────────────────────────────────────────────

def _normalize_split(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def split_train_test_fps(df: pd.DataFrame, fps: list, split_col: str = SPLIT_COL):
    """
    Return (train_df, train_fps), (test_df, test_fps) using split_col.
    Expected values: 'train', 'test' (case-insensitive).
    """
    if split_col not in df.columns:
        return None

    splits = df[split_col].map(_normalize_split)
    tr_mask = splits == "train"
    te_mask = splits == "test"
    if not tr_mask.any() or not te_mask.any():
        print(f"  [!] {split_col}: need both 'train' and 'test' rows; skipping DOA.")
        return None

    tr_idx = df.index[tr_mask].tolist()
    te_idx = df.index[te_mask].tolist()
    train_df = df.loc[tr_idx].reset_index(drop=True)
    test_df = df.loc[te_idx].reset_index(drop=True)
    train_fps = [fps[i] for i in tr_idx]
    test_fps = [fps[i] for i in te_idx]
    return (train_df, train_fps), (test_df, test_fps)


def max_tanimoto_to_train_per_test(test_fps: list, train_fps: list) -> np.ndarray:
    """For each test fingerprint, max Tanimoto similarity to any train structure."""
    if not test_fps or not train_fps:
        return np.array([])
    out = []
    for tf in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(tf, train_fps)
        out.append(max(sims))
    return np.array(out, dtype=np.float64)


def pairwise_tanimoto_within(fps: list) -> list:
    """Upper triangle pairwise similarities (i < j), excluding self."""
    sims = []
    n = len(fps)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return sims


def pairwise_tanimoto_cross(fps_a: list, fps_b: list) -> list:
    """All ordered pairs (a in A, b in B), a != b if A and B share structures."""
    sims = []
    for fa in fps_a:
        sims.extend(DataStructs.BulkTanimotoSimilarity(fa, fps_b))
    return sims


def compute_diversity_tanimoto_metrics(train_fps: list, test_fps: list) -> dict:
    """
    Tanimoto-based metrics for train vs test domain overlap.

    - max_sim_to_train: per-test max similarity to training set (nearest train neighbor).
    - train_train, test_test, train_test: pairwise similarity distributions.
    """
    max_sim = max_tanimoto_to_train_per_test(test_fps, train_fps)
    train_train = pairwise_tanimoto_within(train_fps)
    test_test = pairwise_tanimoto_within(test_fps)
    train_test = pairwise_tanimoto_cross(test_fps, train_fps)

    def _summ(name: str, arr: list) -> dict:
        a = np.asarray(arr, dtype=np.float64)
        return {
            f"{name}_mean": float(np.mean(a)),
            f"{name}_std": float(np.std(a)),
            f"{name}_median": float(np.median(a)),
            f"{name}_min": float(np.min(a)),
            f"{name}_max": float(np.max(a)),
        }

    metrics = {
        "n_train": len(train_fps),
        "n_test": len(test_fps),
        "max_sim_to_train_mean": float(np.mean(max_sim)),
        "max_sim_to_train_std": float(np.std(max_sim)),
        "max_sim_to_train_median": float(np.median(max_sim)),
        "max_sim_to_train_min": float(np.min(max_sim)),
        "max_sim_to_train_max": float(np.max(max_sim)),
        "frac_test_max_sim_below_0.3": float(np.mean(max_sim < 0.3)),
        "frac_test_max_sim_below_0.5": float(np.mean(max_sim < 0.5)),
        "frac_test_max_sim_below_0.7": float(np.mean(max_sim < 0.7)),
    }
    metrics.update(_summ("pairwise_train_train", train_train))
    metrics.update(_summ("pairwise_test_test", test_test))
    metrics.update(_summ("pairwise_train_test", train_test))

    return {
        "metrics": metrics,
        "max_sim_to_train": max_sim,
        "train_train": train_train,
        "test_test": test_test,
        "train_test": train_test,
    }


def plot_diversity_max_similarity_to_train(
    amine_bundle: dict,
    acyl_bundle: dict,
    output_path: str,
    figsize: tuple = (11, 5),
):
    """
    Histogram of nearest-train-neighbor Tanimoto for each test compound.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (title, bundle) in zip(
        axes,
        [("Amines — max Tanimoto to train (per test)", amine_bundle),
         ("Acyl chlorides — max Tanimoto to train (per test)", acyl_bundle)],
    ):
        if bundle is None:
            ax.set_visible(False)
            continue
        m = bundle["max_sim_to_train"]
        ax.hist(m, bins=min(20, max(8, len(m))), color="#1976D2", alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(m), color="crimson", linestyle="--", linewidth=2, label=f"mean = {np.mean(m):.3f}")
        ax.axvline(np.median(m), color="darkgreen", linestyle=":", linewidth=2, label=f"median = {np.median(m):.3f}")
        ax.set_xlabel("Max Tanimoto similarity to training set", fontsize=11)
        ax.set_ylabel("Count (test compounds)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle(
        "Tanimoto Similarity Overlap of the Test Set to the Training Set\n"
        "(higher = closer to at least one train structure; lower = more out-of-sample)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] DOA max-similarity-to-train plot → {output_path}")


def plot_diversity_train_test_heatmap(
    train_fps: list,
    test_fps: list,
    train_labels: list,
    test_labels: list,
    title: str,
    output_path: str,
    figsize: tuple = (10, 6),
):
    """test × train Tanimoto matrix (rows = test, cols = train)."""
    n_te, n_tr = len(test_fps), len(train_fps)
    mat = np.zeros((n_te, n_tr))
    for i, tf in enumerate(test_fps):
        mat[i, :] = DataStructs.BulkTanimotoSimilarity(tf, train_fps)

    mean_tt = float(np.mean(mat))

    tr_fs = max(5, min(8, 200 // max(n_tr, 1)))
    te_fs = max(5, min(8, 200 // max(n_te, 1)))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        xticklabels=train_labels,
        yticklabels=test_labels,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Tanimoto", "shrink": 0.85},
    )
    ax.set_xlabel("Training compounds", fontsize=11)
    ax.set_ylabel("Test compounds", fontsize=11)
    ax.set_title(
        f"{title}\n(mean Tanimoto over test×train sets = {mean_tt:.3f})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(axis="x", rotation=90, labelsize=tr_fs)
    ax.tick_params(axis="y", rotation=0, labelsize=te_fs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] DOA train×test heatmap → {output_path}")


def scaffold_overlap_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Murcko scaffold: how many test compounds share a scaffold seen in training."""
    tr_scaf = set(train_df["smiles"].map(get_murcko_scaffold))
    te_scaf = test_df["smiles"].map(get_murcko_scaffold).tolist()
    in_train = [s in tr_scaf for s in te_scaf]
    te_unique = set(te_scaf)
    novel_scaffolds = len(te_unique - tr_scaf)
    return {
        "frac_test_compounds_scaffold_in_train": float(np.mean(in_train)) if in_train else np.nan,
        "n_distinct_test_scaffolds": len(te_unique),
        "n_test_scaffolds_not_in_train": int(novel_scaffolds),
    }


def write_diversity_summary_text(
    amine_bundle: dict,
    acyl_bundle: dict,
    path: str,
    amine_train_test: tuple = None,
    acyl_train_test: tuple = None,
) -> str:
    lines = [
        "Train and Test dataset diversity analysis using Tanimoto on Morgan fingerprints (r=2, 2048 bits)",
        "=" * 72,
        "",
        "Interpretation:",
        "  • max Tanimoto to train: for each test compound, similarity to the nearest training structure.",
        "    Values near 1.0 indicate the test case is well covered by the training library; lower values",
        "    suggest greater structural novelty (more 'out-of-sample' in fingerprint space).",
        "  • Pairwise train–test vs train–train: if train–test similarities are systematically lower than",
        "    train–train, the held-out set explores regions less dense in the training manifold.",
        "  • Murcko scaffolds: fraction of test compounds whose scaffold appears in training complements",
        "    fingerprint similarity (same scaffold can still differ in substitution).",
        "",
    ]

    def _block(name: str, bundle: dict, tr_te: tuple = None):
        if bundle is None:
            return [f"{name}: (no split data)", ""]
        met = bundle["metrics"]
        bl = [
            f"{name}",
            "-" * len(name),
            f"  Compounds: train = {met['n_train']}, test = {met['n_test']}",
            f"  Max Tanimoto to train (per test): mean = {met['max_sim_to_train_mean']:.4f}, "
            f"std = {met['max_sim_to_train_std']:.4f}, median = {met['max_sim_to_train_median']:.4f}",
            f"  Range of max-sim-to-train: [{met['max_sim_to_train_min']:.4f}, {met['max_sim_to_train_max']:.4f}]",
            f"  Pairwise mean Tanimoto — train–train: {met['pairwise_train_train_mean']:.4f}; "
            f"test–test: {met['pairwise_test_test_mean']:.4f}; train–test: {met['pairwise_train_test_mean']:.4f}",
        ]
        if tr_te is not None:
            tr_df, te_df = tr_te
            st = scaffold_overlap_stats(tr_df, te_df)
            bl.extend([
                "  Murcko scaffolds (test vs train):",
                "    fraction of test compounds whose scaffold appears in train: "
                f"{st['frac_test_compounds_scaffold_in_train']:.3f}",
                "    distinct test scaffolds: "
                f"{st['n_distinct_test_scaffolds']}; "
                "not seen in train (among distinct): "
                f"{st['n_test_scaffolds_not_in_train']}",
            ])
        bl.append("")
        return bl

    lines.extend(_block("Amines", amine_bundle, amine_train_test))
    lines.extend(_block("Acyl chlorides", acyl_bundle, acyl_train_test))

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [✓] DOA summary written → {path}")
    return text


def analyze_train_test_doa(
    amine_df: pd.DataFrame,
    amine_fps: list,
    acyl_df: pd.DataFrame,
    acyl_fps: list,
    output_dir: str,
    split_col: str = SPLIT_COL,
) -> dict:
    """
    Train vs test Tanimoto overlap: heatmaps, max-similarity-to-train plot, summary text.
    Returns dict with 'amines' and 'acyl_chlorides' bundles (or None if no split).
    """
    os.makedirs(output_dir, exist_ok=True)

    amine_split = split_train_test_fps(amine_df, amine_fps, split_col)
    acyl_split = split_train_test_fps(acyl_df, acyl_fps, split_col)

    amine_bundle = None
    acyl_bundle = None
    amine_train_test_dfs = None
    acyl_train_test_dfs = None

    if amine_split is not None:
        (tr_df, tr_fps), (te_df, te_fps) = amine_split
        amine_train_test_dfs = (tr_df, te_df)
        amine_bundle = compute_diversity_tanimoto_metrics(tr_fps, te_fps)
        plot_diversity_train_test_heatmap(
            tr_fps,
            te_fps,
            tr_df["label"].astype(str).tolist(),
            te_df["label"].astype(str).tolist(),
            "Amines: Tanimoto test × train",
            os.path.join(output_dir, "diversity_heatmap_amines_test_x_train.png"),
        )

    if acyl_split is not None:
        (tr_df, tr_fps), (te_df, te_fps) = acyl_split
        acyl_train_test_dfs = (tr_df, te_df)
        acyl_bundle = compute_diversity_tanimoto_metrics(tr_fps, te_fps)
        plot_diversity_train_test_heatmap(
            tr_fps,
            te_fps,
            tr_df["label"].astype(str).tolist(),
            te_df["label"].astype(str).tolist(),
            "Acyl chlorides: Tanimoto test × train",
            os.path.join(output_dir, "diversity_heatmap_acyl_test_x_train.png"),
        )

    if amine_bundle is None and acyl_bundle is None:
        return {"amines": None, "acyl_chlorides": None, "summary_text": ""}

    plot_diversity_max_similarity_to_train(
        amine_bundle,
        acyl_bundle,
        os.path.join(output_dir, "diversity_max_tanimoto_to_train.png"),
    )

    summary_path = os.path.join(output_dir, "diversity_train_test_summary.txt")
    summary_text = write_diversity_summary_text(
        amine_bundle,
        acyl_bundle,
        summary_path,
        amine_train_test=amine_train_test_dfs,
        acyl_train_test=acyl_train_test_dfs,
    )

    return {
        "amines": amine_bundle,
        "acyl_chlorides": acyl_bundle,
        "summary_text": summary_text,
    }


def get_murcko_scaffold(smiles: str) -> str:
    """Return canonical Murcko scaffold SMILES; empty string for acyclics."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold else ""


def classify_amine(smiles: str) -> str:
    """Classify amine functional type via SMARTS matching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    patterns = {
        "heteroaromatic":      "[n;H1]",
        "aromatic_primary":    "[NH2]c",
        "aromatic_secondary":  "[NH]c",
        "aliphatic_primary":   "[NH2;!$([NH2]c)]",
        "aliphatic_secondary": "[NH;!$([NH]c);!$([NH]=*)]",
        "aliphatic_tertiary":  "[N;H0;!$(N=*);!$(Nc)]",
    }
    for label, smarts in patterns.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return label
    return "other"

def _is_heteroaromatic_carbon(mol, atom_idx: int) -> bool:
    """
    Return True if the atom at atom_idx is an aromatic carbon that belongs
    to a ring containing at least one heteroatom (N, O, or S) anywhere in
    that ring — regardless of position relative to atom_idx.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if not (atom.GetIsAromatic() and atom.GetAtomicNum() == 6):
        return False

    ring_info = mol.GetRingInfo()
    heteroatoms = {7, 8, 16}  # N, O, S atomic numbers

    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() in heteroatoms
                   for i in ring):
                return True
    return False

def classify_acyl_chloride(smiles: str) -> str:
    """
    Classify an acyl chloride by the carbon directly attached to the carbonyl.

    Categories (evaluated most-specific → least-specific):
      heteroaromatic     – carbonyl on a heteroaromatic ring (pyridine, furan, thiophene, etc.)
      aromatic           – carbonyl on a carbocyclic aromatic ring (benzoyl-type)
      alpha_halogenated  – sp3 alpha carbon bearing a halogen (e.g. chloroacetyl chloride)
      alkenyl            – alpha carbon is sp2 non-aromatic (vinyl/acryloyl-type)
      aliphatic_tertiary – alpha carbon is sp3 with no H (pivaloyl-type)
      aliphatic_secondary– alpha carbon is sp3 with one H (isobutyryl-type)
      aliphatic_primary  – alpha carbon is sp3 with two or three H (acetyl/propionyl-type)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"

    # SMARTS to find the alpha carbon (atom directly bonded to the carbonyl C)
    acyl_smarts = Chem.MolFromSmarts("[C](=O)(Cl)[#6]")
    match = mol.GetSubstructMatch(acyl_smarts)
    if match:
        alpha_idx = match[3]  # index 3 = the alpha carbon
        if _is_heteroaromatic_carbon(mol, alpha_idx):
            return "heteroaromatic"

    patterns = {
        "aromatic":            "[C](=O)(Cl)c",
        "alkenyl":             "[C](=O)(Cl)[CX3]=[CX3]", # olefin
        "aliphatic_tertiary":  "[C](=O)(Cl)[CX4;H0]([CX4])([CX4])[CX4]",
        "aliphatic_secondary": "[C](=O)(Cl)[CX4;H1]([C])[C]",
        "aliphatic_primary":   "[C](=O)(Cl)[CX4;H2,H3]",
    }

    for label, smarts in patterns.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return label
    return "other"


def compute_structural_metrics(df: pd.DataFrame, role: str) -> dict:
    """
    Compute scaffold and (for amines) functional-type diversity.

    Parameters
    ----------
    df   : DataFrame with a 'smiles' column.
    role : "amine" or "acyl_chloride" — controls whether amine typing is run.

    Returns
    -------
    metrics : dict
    """
    df = df.copy()
    df["scaffold"] = df["smiles"].apply(get_murcko_scaffold)

    metrics = {
        "n_compounds":           len(df),
        "distinct_scaffolds":    int(df["scaffold"].nunique()),
        "acyclic_count":         int((df["scaffold"] == "").sum()),
        "scaffold_distribution": df["scaffold"].value_counts().to_dict(),
    }

    if "amine" in role.lower():
        df["amine_type"] = df["smiles"].apply(classify_amine)
        type_counts = df["amine_type"].value_counts().to_dict()
        metrics["amine_type_breakdown"] = type_counts
        metrics["unique_amine_types"]   = len(type_counts)
        
    if "acyl_chloride" in role.lower():
        df["acyl_chloride_type"] = df["smiles"].apply(classify_acyl_chloride)
        type_counts = df["acyl_chloride_type"].value_counts().to_dict()
        metrics["acyl_chloride_type_breakdown"] = type_counts
        metrics["unique_acyl_chloride_types"]   = len(type_counts)

    print(f"\n  [{role}] Structural Metrics:")
    for k, v in metrics.items():
        if k != "scaffold_distribution":
            print(f"      {k}: {v}")
    return metrics, df

def results_summarized(
    amine_metrics:   dict,
    acyl_metrics:    dict,
    amine_diversity: dict,
    acyl_diversity:  dict,
) -> str:
    n_a      = amine_metrics["n_compounds"]
    n_ac     = acyl_metrics["n_compounds"]
    n_combos = n_a * n_ac
    scaf_a   = amine_metrics["distinct_scaffolds"]
    scaf_ac  = acyl_metrics["distinct_scaffolds"]
    types_str_amines = ", ".join(
        f"{v} {k.replace('_', ' ')}"
        for k, v in amine_metrics.get("amine_type_breakdown", {}).items()
    )
    types_str_acyl_chlorides = ", ".join(
        f"{v} {k.replace('_', ' ')}"
        for k, v in acyl_metrics.get("acyl_chloride_type_breakdown", {}).items()
    )
    n_types_amines = amine_metrics.get("unique_amine_types", "N/A")
    n_types_acyl_chlorides = acyl_metrics.get("unique_acyl_chloride_types", "N/A")

    def f(d, key): return f"{d[key]:.3f}" if key in d else "N/A"

    text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHEMICAL SPACE ANALYSIS RESULTS 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The reactant library comprised {n_a} amines and {n_ac} acyl chlorides,
with a total of {n_combos} unique amide combinations ({n_a} × {n_ac}).

Amine chemical space: The {n_a} amines spanned {n_types_amines} functional
classes ({types_str_amines}) across {scaf_a} distinct Murcko scaffolds,
reflecting systematic variation in electronic character and nucleophilicity.
Mean pairwise Tanimoto diversity: {f(amine_diversity, 'mean_diversity')} ± {f(amine_diversity, 'std_diversity')} SD
(range: {f(amine_diversity, 'min_diversity')} – {f(amine_diversity, 'max_diversity')}).

Acyl chloride chemical space: The {n_ac} acyl chlorides spanned {n_types_acyl_chlorides} functional classes ({types_str_acyl_chlorides}) and {scaf_ac}
distinct Murcko scaffolds, providing systematic steric and electronic
variation at the carbonyl electrophile.
Mean pairwise Tanimoto diversity: {f(acyl_diversity, 'mean_diversity')} ± {f(acyl_diversity, 'std_diversity')} SD
(range: {f(acyl_diversity, 'min_diversity')} – {f(acyl_diversity, 'max_diversity')}).

Together, the library achieves combinatorial expansion with systematic
electronic and steric modification rather than clustering around a
narrow structural core.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(text)
    return text

def analyze_chemical_space(
    amines_csv:        str,
    acyl_csv:          str,
    amines_smiles_col: str = "smiles",
    acyl_smiles_col:   str = "smiles",
    amines_label_col:  str = None,
    acyl_label_col:    str = None,
    output_dir:        str = "chemical_space_output",
) -> dict:
    """
    Master function: reads two separate CSV files and runs all chemical space
    analyses independently for amines and acyl chlorides, then overlays
    diversity distributions for comparison.

    Parameters
    ----------
    amines_csv        : Path to the amines CSV file.
    acyl_csv          : Path to the acyl chlorides CSV file.
    amines_smiles_col : SMILES column name in the amines CSV.
    acyl_smiles_col   : SMILES column name in the acyl chlorides CSV.
    amines_label_col  : Name/ID column in the amines CSV (None → use SMILES).
    acyl_label_col    : Name/ID column in the acyl chlorides CSV (None → use SMILES).
    output_dir        : Directory where all output figures are saved.

    Returns
    -------
    results : dict with keys
        'amine_metrics'    – structural metrics for amines
        'acyl_metrics'     – structural metrics for acyl chlorides
        'amine_diversity'  – diversity scores for amines
        'acyl_diversity'   – diversity scores for acyl chlorides
        'amine_sim_matrix' – N×N Tanimoto similarity matrix (amines)
        'acyl_sim_matrix'  – N×N Tanimoto similarity matrix (acyl chlorides)
        'results_text'     – Chemical Space Analysis Results summary

    Output files
    ------------
    heatmap_amines.png                    – Tanimoto heatmap for amines
    heatmap_acyl_chlorides.png            – Tanimoto heatmap for acyl chlorides
    diversity_distribution_comparison.png – overlaid diversity histograms
    diversity_max_tanimoto_to_train.png          – nearest-train Tanimoto per test compound
    diversity_heatmap_*_test_x_train.png         – test × train matrices (title shows mean Tanimoto)
    diversity_train_test_summary.txt             – DOA metrics and interpretation
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Chemical Space Analysis")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    amine_df = load_reactants(amines_csv, amines_smiles_col, amines_label_col)
    acyl_df  = load_reactants(acyl_csv,   acyl_smiles_col,   acyl_label_col)

    print("\n[2/6] Computing fingerprints...")
    amine_fps = build_fingerprints(amine_df)
    acyl_fps  = build_fingerprints(acyl_df)

    print("\n[3/6] Generating heatmaps...")
    amine_sim = plot_tanimoto_heatmap(
        amine_df, amine_fps,
        title="Tanimoto Similarity – Amines",
        output_path=os.path.join(output_dir, "heatmap_amines.png"),
    )
    acyl_sim = plot_tanimoto_heatmap(
        acyl_df, acyl_fps,
        title="Tanimoto Similarity – Acyl Chlorides",
        output_path=os.path.join(output_dir, "heatmap_acyl_chlorides.png"),
    )

    print("\n[4/6] Computing diversity scores...")
    amine_diversity = compute_diversity_scores(amine_fps, "Amines")
    acyl_diversity = compute_diversity_scores(acyl_fps,  "Acyl Chlorides")

    plot_diversity_distribution(
        fps_dict={"Amines": amine_fps, "Acyl Chlorides": acyl_fps},
        output_path=os.path.join(output_dir, "diversity_distribution_comparison.png"),
    )

    print("\n[5/6] Computing structural metrics...")
    amine_metrics, amine_df = compute_structural_metrics(amine_df, role="amine")
    acyl_metrics, acyl_df  = compute_structural_metrics(acyl_df,  role="acyl_chloride")
    
    # save
    # amine_df.to_csv(os.path.join(output_dir, "amine_df.csv"), index=False)
    # acyl_df.to_csv(os.path.join(output_dir, "acyl_df.csv"), index=False)

    # ── Chemical Space Analysis Results summary ────────────────────────────
    results_text = results_summarized(
        amine_metrics, acyl_metrics,
        amine_diversity, acyl_diversity,
    )

    print("\n[6/6] Train/test domain-of-applicability (Tanimoto heatmaps & max-sim to train)...")
    diversity_results = analyze_train_test_doa(
        amine_df, amine_fps,
        acyl_df, acyl_fps,
        output_dir=output_dir,
        split_col=SPLIT_COL,
    )
    if diversity_results.get("summary_text"):
        sp = os.path.join(os.path.abspath(output_dir), "diversity_train_test_summary.txt")
        print(f"  See DOA metrics and interpretation in: {sp}")

    print("\n[✓] All analyses complete. Outputs saved to:",
          os.path.abspath(output_dir))

    return {
        "amine_metrics":    amine_metrics,
        "acyl_metrics":     acyl_metrics,
        "amine_diversity":  amine_diversity,
        "acyl_diversity":   acyl_diversity,
        "amine_sim_matrix": amine_sim,
        "acyl_sim_matrix":  acyl_sim,
        "results_text":     results_text,
        "doa":              diversity_results,
    }

if __name__ == "__main__":
    results = analyze_chemical_space(
        amines_csv        = AMINES_CSV,
        acyl_csv          = ACYL_CSV,
        amines_smiles_col = AMINES_SMILES_COL,
        acyl_smiles_col   = ACYL_SMILES_COL,
        amines_label_col  = AMINES_LABEL_COL,
        acyl_label_col    = ACYL_LABEL_COL,
        output_dir        = OUTPUT_DIR,
    )
