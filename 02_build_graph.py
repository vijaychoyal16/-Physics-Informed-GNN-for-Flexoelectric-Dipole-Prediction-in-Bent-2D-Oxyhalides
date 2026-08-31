#!/usr/bin/env python3
"""
02_build_graph.py

Convert crystal structures into PyTorch Geometric graph files.

Input:
    processed/dataset.csv

Output:
    processed/graphs/<id>.pt
    processed/graph_dataset.csv

Each graph contains:
    z: atomic numbers, shape [num_atoms]
    pos: Cartesian positions in Angstrom, shape [num_atoms, 3]
    cell: lattice vectors, shape [3, 3]
    edge_index: graph edges, shape [2, num_edges]
    edge_dist: edge distances, shape [num_edges, 1]
    edge_unit: edge unit vectors, shape [num_edges, 3]
    y: target dipole or polarization vector [Px, Py, Pz], shape [1, 3]
    volume: cell volume in Angstrom^3
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data


def build_graph_from_structure(
    structure_path: str,
    target_vector,
    cutoff: float = 5.0,
) -> Data:
    """
    Build a periodic neighbor graph from a crystal structure.

    Parameters
    ----------
    structure_path : str
        Path to POSCAR / CIF / VASP structure file.

    target_vector : list or tuple
        Target [Px, Py, Pz].

    cutoff : float
        Neighbor cutoff radius in Angstrom.

    Returns
    -------
    data : torch_geometric.data.Data
        PyTorch Geometric graph object.
    """

    atoms = read(structure_path)

    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.cell.array
    volume = atoms.get_volume()

    z = torch.tensor(atomic_numbers, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    cell_tensor = torch.tensor(cell, dtype=torch.float)

    # ASE periodic neighbor list
    # i = source atom index
    # j = target atom index
    # S = periodic cell shift vector
    i, j, S = neighbor_list("ijS", atoms, cutoff)

    if len(i) == 0:
        raise RuntimeError(
            f"No neighbor edges found for {structure_path}. "
            f"Try increasing cutoff."
        )

    i = torch.tensor(i, dtype=torch.long)
    j = torch.tensor(j, dtype=torch.long)
    S = torch.tensor(S, dtype=torch.float)

    # Periodic shift in Cartesian coordinates
    shifts = S @ cell_tensor

    # Vector from central atom j to neighbor atom i with PBC shift
    # r_ij = R_i + shift - R_j
    edge_vec = pos[i] + shifts - pos[j]

    edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)
    edge_unit = edge_vec / edge_dist.clamp(min=1e-8)

    edge_index = torch.stack([i, j], dim=0)

    y = torch.tensor(target_vector, dtype=torch.float).view(1, 3)

    data = Data(
        z=z,
        pos=pos,
        cell=cell_tensor,
        edge_index=edge_index,
        edge_dist=edge_dist,
        edge_unit=edge_unit,
        y=y,
        volume=torch.tensor([volume], dtype=torch.float),
    )

    data.num_atoms = len(atoms)

    return data


def build_all_graphs(
    dataset_csv: Path,
    output_graph_dir: Path,
    output_csv: Path,
    cutoff: float = 5.0,
):
    """
    Convert every structure in dataset.csv into a graph file.
    """

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    output_graph_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)

    required_columns = ["id", "structure_path", "Px", "Py", "Pz"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {dataset_csv}: {missing}\n"
            f"Required columns: {required_columns}"
        )

    graph_rows = []

    for index, row in df.iterrows():
        structure_id = str(row["id"])
        structure_path = str(row["structure_path"])

        target = [
            float(row["Px"]),
            float(row["Py"]),
            float(row["Pz"]),
        ]

        try:
            data = build_graph_from_structure(
                structure_path=structure_path,
                target_vector=target,
                cutoff=cutoff,
            )

            graph_path = output_graph_dir / f"{structure_id}.pt"
            torch.save(data, graph_path)

            graph_rows.append(
                {
                    "id": structure_id,
                    "graph_path": str(graph_path.resolve()),
                    "structure_path": structure_path,
                    "Px": target[0],
                    "Py": target[1],
                    "Pz": target[2],
                    "num_atoms": int(data.num_atoms),
                    "num_edges": int(data.edge_index.shape[1]),
                    "volume_A3": float(data.volume.item()),
                    "cutoff_A": cutoff,
                }
            )

            print(
                f"[OK] {structure_id}: "
                f"atoms={data.num_atoms}, "
                f"edges={data.edge_index.shape[1]}, "
                f"graph={graph_path}"
            )

        except Exception as exc:
            print(f"[WARNING] Failed to build graph for {structure_id}: {exc}")

    if len(graph_rows) == 0:
        raise RuntimeError("No graph files were created.")

    graph_df = pd.DataFrame(graph_rows)
    graph_df.to_csv(output_csv, index=False)

    print()
    print("=" * 70)
    print("Graph building finished")
    print("=" * 70)
    print(f"Number of graphs: {len(graph_df)}")
    print(f"Graph folder: {output_graph_dir.resolve()}")
    print(f"Graph dataset CSV: {output_csv.resolve()}")
    print()
    print("First few rows:")
    print(graph_df.head())


def inspect_graph(graph_path: Path):
    """
    Print information from one saved graph.
    """

    data = torch.load(graph_path, weights_only=False)

    print("=" * 70)
    print(f"Graph inspection: {graph_path}")
    print("=" * 70)
    print(data)
    print()
    print(f"Atomic numbers z shape: {data.z.shape}")
    print(f"Positions pos shape: {data.pos.shape}")
    print(f"Cell shape: {data.cell.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge distance shape: {data.edge_dist.shape}")
    print(f"Edge unit vector shape: {data.edge_unit.shape}")
    print(f"Target y: {data.y}")
    print(f"Volume: {data.volume.item():.6f} Angstrom^3")
    print(f"Number of atoms: {data.num_atoms}")
    print(f"Number of edges: {data.edge_index.shape[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Build PyTorch Geometric graph files from crystal structures."
    )

    parser.add_argument(
        "--dataset_csv",
        type=str,
        default="processed/dataset.csv",
        help="Input dataset CSV from 01_prepare_dataset.py.",
    )

    parser.add_argument(
        "--output_graph_dir",
        type=str,
        default="processed/graphs",
        help="Folder where graph .pt files will be saved.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="processed/graph_dataset.csv",
        help="Output CSV listing all generated graph files.",
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Neighbor cutoff radius in Angstrom.",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the first generated graph after building.",
    )

    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    output_graph_dir = Path(args.output_graph_dir)
    output_csv = Path(args.output_csv)

    build_all_graphs(
        dataset_csv=dataset_csv,
        output_graph_dir=output_graph_dir,
        output_csv=output_csv,
        cutoff=args.cutoff,
    )

    if args.inspect:
        df = pd.read_csv(output_csv)
        first_graph = Path(df.iloc[0]["graph_path"])
        inspect_graph(first_graph)


if __name__ == "__main__":
    main()
