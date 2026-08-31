#!/usr/bin/env python3
"""
06_predict.py

Use a trained PolarGNN model to predict total dipole / polarization
and atomic local dipoles for one graph file.

Input:
    checkpoints/best_model.pt
    processed/graphs/In2O3.pt

Output:
    printed prediction
    predictions/In2O3_total_prediction.csv
    predictions/In2O3_atomic_dipoles.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from importlib.machinery import SourceFileLoader


# Load PolarGNN from 03_model.py
model_module = SourceFileLoader("model_module", "03_model.py").load_module()
PolarGNN = model_module.PolarGNN


def load_graph(graph_path: Path):
    """
    Load graph .pt file.
    """

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    try:
        data = torch.load(graph_path, weights_only=False)
    except TypeError:
        data = torch.load(graph_path)

    return data


def load_model(checkpoint_path: Path, device: torch.device):
    """
    Load trained PolarGNN model from checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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


def predict(
    graph_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
):
    """
    Predict total and atomic dipoles for one graph.
    """

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Using CPU.")
        device = "cpu"

    device = torch.device(device)

    data = load_graph(graph_path)
    data = data.to(device)

    model, checkpoint = load_model(checkpoint_path, device)

    with torch.no_grad():
        total_dipole, atomic_dipole = model(data)

    total_dipole = total_dipole.cpu().view(-1)
    atomic_dipole = atomic_dipole.cpu()

    target = data.y.cpu().view(-1) if hasattr(data, "y") else None

    structure_id = graph_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    total_csv = output_dir / f"{structure_id}_total_prediction.csv"
    atomic_csv = output_dir / f"{structure_id}_atomic_dipoles.csv"

    total_row = {
        "id": structure_id,
        "pred_Px": float(total_dipole[0]),
        "pred_Py": float(total_dipole[1]),
        "pred_Pz": float(total_dipole[2]),
    }

    if target is not None:
        total_row.update(
            {
                "target_Px": float(target[0]),
                "target_Py": float(target[1]),
                "target_Pz": float(target[2]),
                "abs_error_Px": float(abs(total_dipole[0] - target[0])),
                "abs_error_Py": float(abs(total_dipole[1] - target[1])),
                "abs_error_Pz": float(abs(total_dipole[2] - target[2])),
            }
        )

    if hasattr(data, "volume"):
        volume = float(data.volume.cpu().view(-1)[0])
        total_row["volume_A3"] = volume

        # Convert dipole e Angstrom / Angstrom^3 to e / Angstrom^2
        # Then e / Angstrom^2 to C/m^2 using factor 16.02176634
        pol_e_per_A2 = total_dipole / volume
        pol_C_per_m2 = pol_e_per_A2 * 16.02176634

        total_row.update(
            {
                "pred_pol_x_e_per_A2": float(pol_e_per_A2[0]),
                "pred_pol_y_e_per_A2": float(pol_e_per_A2[1]),
                "pred_pol_z_e_per_A2": float(pol_e_per_A2[2]),
                "pred_pol_x_C_per_m2": float(pol_C_per_m2[0]),
                "pred_pol_y_C_per_m2": float(pol_C_per_m2[1]),
                "pred_pol_z_C_per_m2": float(pol_C_per_m2[2]),
            }
        )

    pd.DataFrame([total_row]).to_csv(total_csv, index=False)

    atomic_rows = []

    z = data.z.cpu()
    pos = data.pos.cpu()

    for i in range(atomic_dipole.shape[0]):
        px = float(atomic_dipole[i, 0])
        py = float(atomic_dipole[i, 1])
        pz = float(atomic_dipole[i, 2])
        mag = float(torch.sqrt(torch.sum(atomic_dipole[i] ** 2)))

        atomic_rows.append(
            {
                "atom_index": i,
                "Z": int(z[i]),
                "x_A": float(pos[i, 0]),
                "y_A": float(pos[i, 1]),
                "z_A": float(pos[i, 2]),
                "atomic_px": px,
                "atomic_py": py,
                "atomic_pz": pz,
                "atomic_p_magnitude": mag,
            }
        )

    pd.DataFrame(atomic_rows).to_csv(atomic_csv, index=False)

    print("=" * 70)
    print("Prediction finished")
    print("=" * 70)
    print(f"Graph file: {graph_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()
    print("Predicted total dipole / polarization-like vector:")
    print(f"Px = {float(total_dipole[0]): .8f}")
    print(f"Py = {float(total_dipole[1]): .8f}")
    print(f"Pz = {float(total_dipole[2]): .8f}")

    if target is not None:
        print()
        print("Target:")
        print(f"Px = {float(target[0]): .8f}")
        print(f"Py = {float(target[1]): .8f}")
        print(f"Pz = {float(target[2]): .8f}")

    if hasattr(data, "volume"):
        print()
        print("Predicted polarization estimate:")
        print(f"Px = {float(pol_C_per_m2[0]): .8e} C/m^2")
        print(f"Py = {float(pol_C_per_m2[1]): .8e} C/m^2")
        print(f"Pz = {float(pol_C_per_m2[2]): .8e} C/m^2")

    print()
    print(f"Saved total prediction: {total_csv}")
    print(f"Saved atomic dipoles:    {atomic_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict total and atomic dipoles using trained PolarGNN."
    )

    parser.add_argument(
        "--graph_path",
        type=str,
        default="processed/graphs/In2O3.pt",
        help="Path to graph .pt file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Folder to save prediction CSV files.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for prediction.",
    )

    args = parser.parse_args()

    predict(
        graph_path=Path(args.graph_path),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device=args.device,
    )


if __name__ == "__main__":
    main()
