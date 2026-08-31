#!/usr/bin/env python3
"""
13_plot_training_loss.py

Plot training and validation loss/MAE from training_log.csv.

Input:
    training_log.csv

Output:
    loss_plots/train_val_loss.png
    loss_plots/train_val_mae.png
    loss_plots/component_val_mae.png
"""

import argparse
from pathlib import Path

import pandas as pd


def load_log(log_csv: Path) -> pd.DataFrame:
    if not log_csv.exists():
        raise FileNotFoundError(f"Training log not found: {log_csv}")

    df = pd.read_csv(log_csv)

    required = ["epoch", "train_loss", "train_mae", "val_loss", "val_mae"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {log_csv}: {missing}")

    return df


def plot_loss(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mae(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_mae"], label="Train MAE")
    plt.plot(df["epoch"], df["val_mae"], label="Validation MAE")

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training and validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_component_val_mae(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    columns = ["val_mae_x", "val_mae_y", "val_mae_z"]
    existing = [col for col in columns if col in df.columns]

    if not existing:
        print("[WARNING] Component MAE columns not found. Skipping component plot.")
        return

    plt.figure(figsize=(8, 6))

    label_map = {
        "val_mae_x": "Validation MAE Px",
        "val_mae_y": "Validation MAE Py",
        "val_mae_z": "Validation MAE Pz",
    }

    for col in existing:
        plt.plot(df["epoch"], df[col], label=label_map[col])

    plt.xlabel("Epoch")
    plt.ylabel("Component MAE")
    plt.title("Validation MAE for Px, Py, Pz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_summary(df: pd.DataFrame):
    best_val = df.loc[df["val_mae"].idxmin()]

    print("=" * 70)
    print("Training log summary")
    print("=" * 70)
    print(f"Total epochs: {len(df)}")
    print()
    print("Best validation MAE:")
    print(f"  epoch     = {int(best_val['epoch'])}")
    print(f"  val_mae   = {best_val['val_mae']:.8f}")
    print(f"  train_mae = {best_val['train_mae']:.8f}")

    if "val_mae_z" in df.columns:
        print(f"  val_mae_z = {best_val['val_mae_z']:.8f}")

    print()
    print("Last epoch:")
    last = df.iloc[-1]
    print(f"  epoch     = {int(last['epoch'])}")
    print(f"  train_mae = {last['train_mae']:.8f}")
    print(f"  val_mae   = {last['val_mae']:.8f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss curves."
    )

    parser.add_argument(
        "--log_csv",
        type=str,
        default="training_log.csv",
        help="Training log CSV from 04_train.py.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="loss_plots",
        help="Output directory for plots.",
    )

    args = parser.parse_args()

    log_csv = Path(args.log_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_log(log_csv)

    loss_png = output_dir / "train_val_loss.png"
    mae_png = output_dir / "train_val_mae.png"
    component_png = output_dir / "component_val_mae.png"

    plot_loss(df, loss_png)
    plot_mae(df, mae_png)
    plot_component_val_mae(df, component_png)

    print_summary(df)

    print()
    print("Saved plots:")
    print(f"  {loss_png}")
    print(f"  {mae_png}")
    print(f"  {component_png}")


if __name__ == "__main__":
    main()
