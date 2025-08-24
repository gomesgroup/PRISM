import polars as pl
from pathlib import Path
import numpy as np


def main():
    src = Path("data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv")
    out = Path("data/rates/nmr_lnk_smiles.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(src)

    # Keep rows with valid positive NMR
    if "NMR_rate" not in df.columns:
        raise SystemExit("Input does not contain NMR_rate column")

    df = df.filter(pl.col("NMR_rate").is_not_null() & (pl.col("NMR_rate") > 0))

    # Compute ln NMR
    df = df.with_columns(
        pl.col("NMR_rate").map_elements(function=lambda x: float(np.log(x)) if x is not None and x > 0 else None, return_dtype=pl.Float64).alias("NMR_lnk")
    )

    # Columns to keep
    keep_cols = [c for c in ["amine_smiles", "acid_smiles", "NMR_lnk", "test splits"] if c in df.columns]
    out_df = df.select(keep_cols)
    out_df.write_csv(out)
    print(f"Wrote {len(out_df)} rows to {out}")


if __name__ == "__main__":
    main()


