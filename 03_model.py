#!/usr/bin/env python3
"""
03_model.py

Simple polarization / dipole-moment GNN model.

Input graph from 02_build_graph.py:
    data.z          atomic numbers, shape [num_atoms]
    data.edge_index graph edges, shape [2, num_edges]
    data.edge_dist  edge distances, shape [num_edges, 1]
    data.edge_unit  edge unit vectors, shape [num_edges, 3]
    data.batch      atom-to-structure index, added by PyG DataLoader

Output:
    total_dipole    predicted [Px, Py, Pz], shape [num_graphs, 3]
    atomic_dipole   local atomic dipoles, shape [num_atoms, 3]

Model idea:
    For each edge i -> j:
        scalar coefficient c_ij is predicted by an MLP.
        edge dipole contribution = c_ij * edge_unit_ij

    For each atom j:
        atomic_dipole_j = sum_i c_ij * edge_unit_ij

    For each structure:
        total_dipole = sum_j atomic_dipole_j
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader


class EdgeDipoleConv(MessagePassing):
    """
    Edge-based message-passing layer for dipole prediction.

    It predicts a scalar c_ij for each edge and multiplies it by the
    edge unit vector. This gives a vector contribution for each edge.
    """

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.scalar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_dist, edge_unit):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_atoms, hidden_dim]

        edge_index : torch.Tensor
            Edge indices, shape [2, num_edges]

        edge_dist : torch.Tensor
            Edge distances, shape [num_edges, 1]

        edge_unit : torch.Tensor
            Edge unit vectors, shape [num_edges, 3]

        Returns
        -------
        atomic_dipole : torch.Tensor
            Atomic dipole vectors, shape [num_atoms, 3]
        """

        source, target = edge_index

        edge_features = torch.cat(
            [
                x[source],
                x[target],
                edge_dist,
            ],
            dim=-1,
        )

        hidden_edge = self.edge_mlp(edge_features)

        scalar_cij = self.scalar_head(hidden_edge)

        edge_dipole = scalar_cij * edge_unit

        atomic_dipole = self.propagate(
            edge_index=edge_index,
            edge_dipole=edge_dipole,
            size=(x.size(0), x.size(0)),
        )

        return atomic_dipole

    def message(self, edge_dipole):
        return edge_dipole


class PolarGNN(nn.Module):
    """
    Simple GNN for total dipole / polarization vector prediction.
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.max_atomic_number = max_atomic_number
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)

        self.convs = nn.ModuleList()
        self.node_updates = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(EdgeDipoleConv(hidden_dim))

            self.node_updates.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )

        self.final_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        """
        Parameters
        ----------
        data : torch_geometric.data.Data or Batch

        Returns
        -------
        total_dipole : torch.Tensor
            Shape [num_graphs, 3]

        atomic_dipole : torch.Tensor
            Shape [num_atoms, 3]
        """

        z = data.z
        edge_index = data.edge_index
        edge_dist = data.edge_dist
        edge_unit = data.edge_unit

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        x = self.atom_embedding(z)

        atomic_dipole_total = torch.zeros(
            z.size(0),
            3,
            dtype=x.dtype,
            device=x.device,
        )

        for conv, update in zip(self.convs, self.node_updates):
            atomic_dipole_layer = conv(
                x=x,
                edge_index=edge_index,
                edge_dist=edge_dist,
                edge_unit=edge_unit,
            )

            atomic_dipole_total = atomic_dipole_total + atomic_dipole_layer

            x = update(torch.cat([x, atomic_dipole_layer], dim=-1))

        atomic_dipole_total = self.final_scale * atomic_dipole_total

        num_graphs = int(batch.max().item()) + 1

        total_dipole = torch.zeros(
            num_graphs,
            3,
            dtype=atomic_dipole_total.dtype,
            device=atomic_dipole_total.device,
        )

        total_dipole.index_add_(0, batch, atomic_dipole_total)

        return total_dipole, atomic_dipole_total


def load_graph(graph_path: Path):
    """
    Load one graph file safely.
    """

    try:
        data = torch.load(graph_path, weights_only=False)
    except TypeError:
        data = torch.load(graph_path)

    return data


def test_model_on_graph(
    graph_path: Path,
    hidden_dim: int = 128,
    num_layers: int = 3,
    device: str = "cpu",
):
    """
    Quick test to check whether the model can read a graph and run forward.
    """

    data = load_graph(graph_path)

    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    model = PolarGNN(
        max_atomic_number=100,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    model.eval()

    with torch.no_grad():
        total_dipole, atomic_dipole = model(batch)

    print("=" * 70)
    print("Model forward test finished")
    print("=" * 70)
    print(f"Graph file: {graph_path}")
    print(f"Number of atoms: {batch.z.shape[0]}")
    print(f"Number of edges: {batch.edge_index.shape[1]}")
    print(f"Target y shape: {batch.y.shape}")
    print(f"Target y: {batch.y}")
    print(f"Predicted total dipole shape: {total_dipole.shape}")
    print(f"Predicted total dipole: {total_dipole}")
    print(f"Atomic dipole shape: {atomic_dipole.shape}")
    print(f"Device: {device}")


def main():
    parser = argparse.ArgumentParser(
        description="Define and test PolarGNN model."
    )

    parser.add_argument(
        "--graph_path",
        type=str,
        default="processed/graphs/In2O3.pt",
        help="Path to one graph .pt file for testing.",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size.",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GNN layers.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model test.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Using CPU.")
        args.device = "cpu"

    graph_path = Path(args.graph_path)

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_path}\n"
            f"Run 02_build_graph.py first."
        )

    test_model_on_graph(
        graph_path=graph_path,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
