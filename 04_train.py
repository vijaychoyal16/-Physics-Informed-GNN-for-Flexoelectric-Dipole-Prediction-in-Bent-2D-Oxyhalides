#!/usr/bin/env python3
"""
04_train.py

Train the PolarGNN model.

Input:
    processed/graph_dataset.csv
    processed/graphs/*.pt

Output:
    checkpoints/best_model.pt
    checkpoints/last_model.pt
    training_log.csv

Important:
    With only one structure and dummy labels, this script is only a pipeline test.
    For real training, you need many structures with real Px, Py, Pz labels.
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from importlib.machinery import SourceFileLoader


# Load model from 03_model.py even though filename starts with a number
model_module = SourceFileLoader("model_module", "03_model.py").load_module()
PolarGNN = model_module.PolarGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_dataset(graph_csv: Path):
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
        raise RuntimeError("No graph files were loaded.")

    return graphs


def split_dataset(graphs, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split graphs into train, validation, and test sets.

    If there is only one graph, use the same graph for train/val/test.
    This is only for pipeline testing.
    """

    random.seed(seed)
    graphs = graphs.copy()
    random.shuffle(graphs)

    n = len(graphs)

    if n == 1:
        print("[WARNING] Only one graph found. Using it for train/val/test.")
        return graphs, graphs, graphs

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]

    if len(test_graphs) == 0:
        test_graphs = val_graphs

    return train_graphs, val_graphs, test_graphs


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    count = 0

    for batch in loader:
        batch = batch.to(device)

        pred, atomic_dipole = model(batch)

        target = batch.y.view(pred.shape)

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        mae = torch.mean(torch.abs(pred.detach() - target.detach()))

        batch_size = pred.shape[0]
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        count += batch_size

    return total_loss / count, total_mae / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    count = 0

    all_pred = []
    all_target = []

    for batch in loader:
        batch = batch.to(device)

        pred, atomic_dipole = model(batch)
        target = batch.y.view(pred.shape)

        loss = loss_fn(pred, target)
        mae = torch.mean(torch.abs(pred - target))

        batch_size = pred.shape[0]
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        count += batch_size

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    component_mae = torch.mean(torch.abs(all_pred - all_target), dim=0)

    return {
        "loss": total_loss / count,
        "mae": total_mae / count,
        "mae_x": component_mae[0].item(),
        "mae_y": component_mae[1].item(),
        "mae_z": component_mae[2].item(),
        "pred": all_pred,
        "target": all_target,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_mae,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train PolarGNN for dipole / polarization prediction."
    )

    parser.add_argument(
        "--graph_csv",
        type=str,
        default="processed/graph_dataset.csv",
        help="CSV generated by 02_build_graph.py.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Folder to save model checkpoints.",
    )

    parser.add_argument(
        "--log_csv",
        type=str,
        default="training_log.csv",
        help="CSV file for training log.",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of GNN.",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GNN layers.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4000,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Training device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    graphs = load_dataset(Path(args.graph_csv))

    print("=" * 70)
    print("Dataset loaded")
    print("=" * 70)
    print(f"Total graphs: {len(graphs)}")

    train_graphs, val_graphs, test_graphs = split_dataset(
        graphs,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
    )

    print(f"Train graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PolarGNN(
        max_atomic_number=100,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.L1Loss()

    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"

    best_val_mae = float("inf")
    log_rows = []

    print()
    print("=" * 70)
    print("Training started")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_result = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = val_result["loss"]
        val_mae = val_result["mae"]

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_mae_x": val_result["mae_x"],
                "val_mae_y": val_result["mae_y"],
                "val_mae_z": val_result["mae_z"],
            }
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_mae=val_mae,
            args=args,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_mae=val_mae,
                args=args,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | "
                f"train MAE = {train_mae:.6f} | "
                f"val MAE = {val_mae:.6f} | "
                f"val MAE xyz = "
                f"({val_result['mae_x']:.6f}, "
                f"{val_result['mae_y']:.6f}, "
                f"{val_result['mae_z']:.6f})"
            )

    pd.DataFrame(log_rows).to_csv(args.log_csv, index=False)

    test_result = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)
    print(f"Best validation MAE: {best_val_mae:.6f}")
    print(f"Test MAE: {test_result['mae']:.6f}")
    print(
        f"Test MAE xyz: "
        f"({test_result['mae_x']:.6f}, "
        f"{test_result['mae_y']:.6f}, "
        f"{test_result['mae_z']:.6f})"
    )
    print(f"Best model saved to: {best_path.resolve()}")
    print(f"Last model saved to: {last_path.resolve()}")
    print(f"Training log saved to: {Path(args.log_csv).resolve()}")

    print()
    print("Example prediction on test set:")
    print("Target:")
    print(test_result["target"])
    print("Prediction:")
    print(test_result["pred"])


if __name__ == "__main__":
    main()
