#!/usr/bin/env python3
"""
05_evaluate.py

Evaluate a trained PolarGNN model on all graphs listed in:

    processed/graph_dataset.csv

Input:
    processed/graph_dataset.csv
    checkpoints/best_model.pt

Output:
    evaluation_results/evaluation_predictions.csv
    evaluation_results/evaluation_metrics.csv
    evaluation_results/dft_vs_gnn_plot.png

Notes:
    With only one structure and dummy target [0, 0, 0], this is only a
    pipeline test. For real evaluation, use many structures with real DFT
    Px, Py, Pz labels.
"""

import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import pandas as pd
import torch
from torch_geometric.loader import DataLoader


# Load PolarGNN from 03_model.py
model_module = SourceFileLoader("model_module", "03_model.py").load_module()
PolarGNN = model_module.PolarGNN


def load_graph(path: Path):
    """Load one PyTorch Geometric graph file."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_dataset(graph_csv: Path):
    """Load all graphs listed in graph_dataset.csv."""
    if not graph_csv.exists():
        raise FileNotFoundError(
            f"Graph dataset CSV not found: {graph_csv}\n"
            f"Run 02_build_graph.py first."
        )

    df = pd.read_csv(graph_csv)

    if "graph_path" not in df.columns:
        raise ValueError(f"'graph_path' column not found in {graph_csv}")

    graphs = []

    for _, row in df.iterrows():
        graph_path = Path(row["graph_path"])

        if not graph_path.exists():
            print(f"[WARNING] Missing graph file: {graph_path}")
            continue

        data = load_graph(graph_path)
        data.structure_id = str(row["id"])
        graphs.append(data)

    if len(graphs) == 0:
        raise RuntimeError("No graphs were loaded.")

    return graphs


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained PolarGNN model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run 04_train.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    args = checkpoint.get("args", {})

    hidden_dim = int(args.get("hidden_dim", 128))
    num_layers = int(args.get("num_layers", 3))

    model = PolarGNN(
        max_atomic_number=100,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, graphs, batch_size: int, device: torch.device):
    """Evaluate model and return predictions, targets, and metrics."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    rows = []

    all_pred = []
    all_target = []

    for batch in loader:
        batch = batch.to(device)

        pred, atomic_dipole = model(batch)
        target = batch.y.view(pred.shape)

        pred_cpu = pred.cpu()
        target_cpu = target.cpu()

        all_pred.append(pred_cpu)
        all_target.append(target_cpu)

        batch_size_now = pred_cpu.shape[0]

        for i in range(batch_size_now):
            px_pred, py_pred, pz_pred = pred_cpu[i].tolist()
            px_true, py_true, pz_true = target_cpu[i].tolist()

            rows.append(
                {
                    "sample_index": len(rows),
                    "target_Px": px_true,
                    "target_Py": py_true,
                    "target_Pz": pz_true,
                    "pred_Px": px_pred,
                    "pred_Py": py_pred,
                    "pred_Pz": pz_pred,
                    "abs_error_Px": abs(px_pred - px_true),
                    "abs_error_Py": abs(py_pred - py_true),
                    "abs_error_Pz": abs(pz_pred - pz_true),
                }
            )

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    error = all_pred - all_target
    abs_error = torch.abs(error)

    mae_xyz = torch.mean(abs_error, dim=0)
    rmse_xyz = torch.sqrt(torch.mean(error ** 2, dim=0))

    mae_total = torch.mean(abs_error)
    rmse_total = torch.sqrt(torch.mean(error ** 2))

    # R2 score for all components together
    ss_res = torch.sum((all_target - all_pred) ** 2)
    ss_tot = torch.sum((all_target - torch.mean(all_target, dim=0)) ** 2)

    if float(ss_tot) > 1e-20:
        r2 = 1.0 - float(ss_res / ss_tot)
    else:
        r2 = float("nan")

    metrics = {
        "num_samples": int(all_pred.shape[0]),
        "mae_total": float(mae_total),
        "rmse_total": float(rmse_total),
        "mae_Px": float(mae_xyz[0]),
        "mae_Py": float(mae_xyz[1]),
        "mae_Pz": float(mae_xyz[2]),
        "rmse_Px": float(rmse_xyz[0]),
        "rmse_Py": float(rmse_xyz[1]),
        "rmse_Pz": float(rmse_xyz[2]),
        "r2_all_components": r2,
    }

    return pd.DataFrame(rows), metrics, all_pred, all_target


def save_plot(pred: torch.Tensor, target: torch.Tensor, output_path: Path):
    """
    Save DFT target vs GNN prediction scatter plot.

    Uses matplotlib only.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping plot.")
        return

    pred_np = pred.numpy()
    target_np = target.numpy()

    components = ["Px", "Py", "Pz"]

    plt.figure(figsize=(6, 6))

    for i, name in enumerate(components):
        plt.scatter(
            target_np[:, i],
            pred_np[:, i],
            label=name,
            alpha=0.8,
        )

    min_value = min(target_np.min(), pred_np.min())
    max_value = max(target_np.max(), pred_np.max())

    # Avoid identical axis range for dummy one-sample zero target
    if abs(max_value - min_value) < 1e-12:
        min_value -= 1.0
        max_value += 1.0

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel("DFT target")
    plt.ylabel("GNN prediction")
    plt.title("DFT target vs GNN prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PolarGNN model."
    )

    parser.add_argument(
        "--graph_csv",
        type=str,
        default="processed/graph_dataset.csv",
        help="CSV generated by 02_build_graph.py.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Trained model checkpoint from 04_train.py.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Folder to save evaluation results.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for evaluation.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    graph_csv = Path(args.graph_csv)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_dataset(graph_csv)
    model, checkpoint = load_model(checkpoint_path, device)

    predictions_df, metrics, pred, target = evaluate_model(
        model=model,
        graphs=graphs,
        batch_size=args.batch_size,
        device=device,
    )

    predictions_csv = output_dir / "evaluation_predictions.csv"
    metrics_csv = output_dir / "evaluation_metrics.csv"
    plot_path = output_dir / "dft_vs_gnn_plot.png"

    predictions_df.to_csv(predictions_csv, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    save_plot(pred, target, plot_path)

    print("=" * 70)
    print("Evaluation finished")
    print("=" * 70)
    print(f"Number of samples: {metrics['num_samples']}")
    print()
    print("MAE:")
    print(f"  total = {metrics['mae_total']:.8f}")
    print(f"  Px    = {metrics['mae_Px']:.8f}")
    print(f"  Py    = {metrics['mae_Py']:.8f}")
    print(f"  Pz    = {metrics['mae_Pz']:.8f}")
    print()
    print("RMSE:")
    print(f"  total = {metrics['rmse_total']:.8f}")
    print(f"  Px    = {metrics['rmse_Px']:.8f}")
    print(f"  Py    = {metrics['rmse_Py']:.8f}")
    print(f"  Pz    = {metrics['rmse_Pz']:.8f}")
    print()
    print(f"R2 all components = {metrics['r2_all_components']}")
    print()
    print(f"Saved predictions: {predictions_csv}")
    print(f"Saved metrics:     {metrics_csv}")
    print(f"Saved plot:        {plot_path}")
    print()
    print("First few predictions:")
    print(predictions_df.head())


if __name__ == "__main__":
    main()
