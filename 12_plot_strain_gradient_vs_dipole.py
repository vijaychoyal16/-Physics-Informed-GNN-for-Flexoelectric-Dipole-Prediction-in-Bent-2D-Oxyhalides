#!/usr/bin/env python3
"""
12_plot_strain_gradient_vs_dipole.py

Plot strain gradient vs Px, Py, Pz for both DFT and GNN.

Inputs:
    id_prop.csv                 -> DFT target values
    strain_gradients.csv        -> strain gradient for each structure
    predictions/*_total_prediction.csv  -> GNN predictions

Outputs:
    comparison_plots/strain_gradient_vs_Px.png
    comparison_plots/strain_gradient_vs_Py.png
    comparison_plots/strain_gradient_vs_Pz.png
    comparison_plots/strain_gradient_comparison.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def load_dft_targets(id_prop_csv: Path) -> pd.DataFrame:
    if not id_prop_csv.exists():
        raise FileNotFoundError(f"id_prop.csv not found: {id_prop_csv}")
    df = pd.read_csv(id_prop_csv)

    required = ["id", "Px", "Py", "Pz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {id_prop_csv}: {missing}")

    return df.rename(
        columns={
            "Px": "dft_Px",
            "Py": "dft_Py",
            "Pz": "dft_Pz",
        }
    )


def load_strain_gradients(strain_csv: Path) -> pd.DataFrame:
    if not strain_csv.exists():
        raise FileNotFoundError(f"strain gradient file not found: {strain_csv}")
    df = pd.read_csv(strain_csv)

    required = ["id", "strain_gradient"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {strain_csv}: {missing}")

    return df


def load_predictions(prediction_dir: Path) -> pd.DataFrame:
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    rows = []

    for csv_file in prediction_dir.glob("*_total_prediction.csv"):
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue

        row = df.iloc[0]

        structure_id = row["id"]

        rows.append(
            {
                "id": structure_id,
                "gnn_Px": float(row["pred_Px"]),
                "gnn_Py": float(row["pred_Py"]),
                "gnn_Pz": float(row["pred_Pz"]),
            }
        )

    if len(rows) == 0:
        raise RuntimeError(f"No *_total_prediction.csv files found in {prediction_dir}")

    return pd.DataFrame(rows)


def merge_all(dft_df, strain_df, pred_df):
    merged = pd.merge(dft_df, strain_df, on="id", how="inner")
    merged = pd.merge(merged, pred_df, on="id", how="inner")

    merged = merged.sort_values("strain_gradient").reset_index(drop=True)
    return merged


def plot_component(df: pd.DataFrame, component: str, output_path: Path):
    import matplotlib.pyplot as plt

    dft_col = f"dft_{component}"
    gnn_col = f"gnn_{component}"

    plt.figure(figsize=(8, 6))

    plt.plot(df["strain_gradient"], df[dft_col], marker="o", label=f"DFT {component}")
    plt.plot(df["strain_gradient"], df[gnn_col], marker="s", label=f"GNN {component}")

    plt.xlabel("Strain gradient (m$^{-1}$)")
    plt.ylabel(f"{component} (eÅ)")
    plt.title(f"Strain gradient vs {component}: DFT vs GNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_components(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))

    for comp, marker1, marker2 in [("Px", "o", "s"), ("Py", "^", "D"), ("Pz", "v", "x")]:
        plt.plot(df["strain_gradient"], df[f"dft_{comp}"], marker=marker1, label=f"DFT {comp}")
        plt.plot(df["strain_gradient"], df[f"gnn_{comp}"], marker=marker2, linestyle="--", label=f"GNN {comp}")

    plt.xlabel("Strain gradient (m$^{-1}$)")
    plt.ylabel("Dipole component (eÅ)")
    plt.title("Strain gradient vs dipole components: DFT vs GNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_summary(df: pd.DataFrame):
    print("=" * 70)
    print("Merged comparison table")
    print("=" * 70)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Plot strain gradient vs Px/Py/Pz for DFT and GNN."
    )

    parser.add_argument(
        "--id_prop_csv",
        type=str,
        default="id_prop.csv",
        help="DFT target CSV."
    )

    parser.add_argument(
        "--strain_csv",
        type=str,
        default="strain_gradients.csv",
        help="CSV containing id and strain_gradient."
    )

    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="predictions",
        help="Directory containing *_total_prediction.csv files."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_plots",
        help="Output directory."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dft_df = load_dft_targets(Path(args.id_prop_csv))
    strain_df = load_strain_gradients(Path(args.strain_csv))
    pred_df = load_predictions(Path(args.prediction_dir))

    merged = merge_all(dft_df, strain_df, pred_df)

    merged_csv = output_dir / "strain_gradient_comparison.csv"
    merged.to_csv(merged_csv, index=False)

    plot_component(merged, "Px", output_dir / "strain_gradient_vs_Px.png")
    plot_component(merged, "Py", output_dir / "strain_gradient_vs_Py.png")
    plot_component(merged, "Pz", output_dir / "strain_gradient_vs_Pz.png")
    plot_all_components(merged, output_dir / "strain_gradient_vs_all_components.png")

    print_summary(merged)
    print()
    print("Saved files:")
    print(f"  {merged_csv}")
    print(f"  {output_dir / 'strain_gradient_vs_Px.png'}")
    print(f"  {output_dir / 'strain_gradient_vs_Py.png'}")
    print(f"  {output_dir / 'strain_gradient_vs_Pz.png'}")
    print(f"  {output_dir / 'strain_gradient_vs_all_components.png'}")


if __name__ == "__main__":
    main()
