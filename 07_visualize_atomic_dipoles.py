#!/usr/bin/env python3
"""
07_visualize_atomic_dipoles.py

Visualize atomic dipoles predicted by 06_predict.py, with different
atom species shown using different colors and markers.

Input:
    predictions/<prefix>_atomic_dipoles.csv

Output:
    visualization/<prefix>_atomic_dipoles_3d.png
    visualization/<prefix>_atomic_dipoles_xy.png
    visualization/<prefix>_atomic_dipoles_xz.png
    visualization/<prefix>_atomic_dipoles_yz.png

CSV columns expected:
    atom_index
    Z
    x_A
    y_A
    z_A
    atomic_px
    atomic_py
    atomic_pz
    atomic_p_magnitude
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# Periodic table symbols: index = atomic number
CHEMICAL_SYMBOLS = [
    "X",  # 0 placeholder
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]


def z_to_symbol(z: int) -> str:
    z = int(z)
    if 0 < z < len(CHEMICAL_SYMBOLS):
        return CHEMICAL_SYMBOLS[z]
    return f"Z={z}"

def _build_species_styles(species_list):
    atom_colors = [
        "#1b1b1b",  # black
        "#8b0000",  # dark red
        "#003366",  # dark blue
        "#006400",  # dark green
        "#4b0082",  # indigo
        "#8b4513",  # brown
    ]

    arrow_colors = [
        "#ff0000",  # red arrows
        "#0000ff",  # blue arrows
        "#00aa00",  # green arrows
        "#ff8c00",  # orange arrows
        "#8a2be2",  # purple arrows
        "#00bcd4",  # cyan arrows
    ]

    styles = {}
    for i, sp in enumerate(sorted(species_list)):
        styles[sp] = {
            "atom_color": atom_colors[i % len(atom_colors)],
            "arrow_color": arrow_colors[i % len(arrow_colors)],
            "marker": "o",
        }

    return styles



def load_atomic_dipoles(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Atomic dipole CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "atom_index",
        "Z",
        "x_A",
        "y_A",
        "z_A",
        "atomic_px",
        "atomic_py",
        "atomic_pz",
        "atomic_p_magnitude",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.copy()
    df["species"] = df["Z"].astype(int).apply(z_to_symbol)
    return df


def _safe_scale(values: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale vectors so arrows are visible but not too huge.
    """
    if len(values) == 0:
        return values

    mags = np.linalg.norm(values, axis=1)
    max_mag = np.max(mags)

    if max_mag < 1e-20:
        return values

    return values / max_mag * scale_factor


def _build_species_styles(species_list):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]

    styles = {}
    for i, sp in enumerate(sorted(species_list)):
        styles[sp] = {
            "color": cmap(i % 20),
            "marker": markers[i % len(markers)],
        }
    return styles


def print_summary(df: pd.DataFrame):
    mag = df["atomic_p_magnitude"].to_numpy()

    print("=" * 70)
    print("Atomic dipole summary")
    print("=" * 70)
    print(f"Number of atoms: {len(df)}")
    print(f"Minimum |p|: {mag.min():.8e}")
    print(f"Maximum |p|: {mag.max():.8e}")
    print(f"Mean    |p|: {mag.mean():.8e}")
    print()

    species_counts = df["species"].value_counts().sort_index()
    print("Species counts:")
    for sp, count in species_counts.items():
        print(f"  {sp}: {count}")
    print()

    largest = df.sort_values("atomic_p_magnitude", ascending=False).head(10)
    print("Top 10 atoms by dipole magnitude:")
    print(
        largest[
            [
                "atom_index",
                "species",
                "Z",
                "x_A",
                "y_A",
                "z_A",
                "atomic_px",
                "atomic_py",
                "atomic_pz",
                "atomic_p_magnitude",
            ]
        ].to_string(index=False)
    )


def plot_3d(
    df: pd.DataFrame,
    output_path: Path,
    scale_factor: float = 1.5,
    min_magnitude: float = 0.0,
    show_atom_index: bool = False,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if min_magnitude > 0:
        df = df[df["atomic_p_magnitude"] >= min_magnitude].copy()

    x = df["x_A"].to_numpy()
    y = df["y_A"].to_numpy()
    z = df["z_A"].to_numpy()

    dip = df[["atomic_px", "atomic_py", "atomic_pz"]].to_numpy()
    dip_scaled = _safe_scale(dip, scale_factor)

    df = df.copy()
    df["u"] = dip_scaled[:, 0]
    df["v"] = dip_scaled[:, 1]
    df["w"] = dip_scaled[:, 2]

    styles = _build_species_styles(df["species"].unique())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for sp, group in df.groupby("species"):
        style = styles[sp]
        xs = group["x_A"].to_numpy()
        ys = group["y_A"].to_numpy()
        zs = group["z_A"].to_numpy()
        us = group["u"].to_numpy()
        vs = group["v"].to_numpy()
        ws = group["w"].to_numpy()

        ax.scatter(
            xs, ys, zs,
            color=style["atom_color"],
            s=40,
            alpha=0.9,
            label=sp,
        )

        ax.quiver(
            xs, ys, zs,
            us, vs, ws,
            color=style["arrow_color"],
            length=1.0,
            normalize=False,
            linewidth=1.4,
            arrow_length_ratio=0.25,
        )

        if show_atom_index:
            for _, row in group.iterrows():
                ax.text(
                    row["x_A"], row["y_A"], row["z_A"],
                    str(int(row["atom_index"])),
                    fontsize=7,
                )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title("3D Atomic Dipole Visualization (species-colored)")

    if len(x) > 0:
        ax.set_box_aspect((np.ptp(x) + 1e-8, np.ptp(y) + 1e-8, np.ptp(z) + 1e-8))

    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_2d_projection(
    df: pd.DataFrame,
    plane: str,
    output_path: Path,
    scale_factor: float = 1.5,
    min_magnitude: float = 0.0,
    show_atom_index: bool = False,
):
    import matplotlib.pyplot as plt

    valid_planes = {"xy", "xz", "yz"}
    if plane not in valid_planes:
        raise ValueError(f"plane must be one of {valid_planes}")

    if min_magnitude > 0:
        df = df[df["atomic_p_magnitude"] >= min_magnitude].copy()

    coord_map = {
        "xy": ("x_A", "y_A", "atomic_px", "atomic_py", "x (Å)", "y (Å)"),
        "xz": ("x_A", "z_A", "atomic_px", "atomic_pz", "x (Å)", "z (Å)"),
        "yz": ("y_A", "z_A", "atomic_py", "atomic_pz", "y (Å)", "z (Å)"),
    }

    c1, c2, d1, d2, label1, label2 = coord_map[plane]

    dip = df[[d1, d2]].to_numpy()
    dip_scaled = _safe_scale(dip, scale_factor)

    df = df.copy()
    df["u"] = dip_scaled[:, 0]
    df["v"] = dip_scaled[:, 1]

    styles = _build_species_styles(df["species"].unique())

    plt.figure(figsize=(8, 7))

    for sp, group in df.groupby("species"):
        style = styles[sp]
        x = group[c1].to_numpy()
        y = group[c2].to_numpy()
        u = group["u"].to_numpy()
        v = group["v"].to_numpy()

        plt.scatter(
            x, y,
            color=style["atom_color"],
            s=40,
            alpha=0.9,
            label=sp,
        )

        plt.quiver(
            x, y, u, v,
            color=style["arrow_color"],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
        )

        if show_atom_index:
            for _, row in group.iterrows():
                plt.text(
                    row[c1], row[c2],
                    str(int(row["atom_index"])),
                    fontsize=7,
                )

    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(f"Atomic Dipole Projection on {plane.upper()} Plane (species-colored)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize atomic dipoles from prediction CSV with species-colored arrows."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="predictions/In2O3_atomic_dipoles.csv",
        help="Atomic dipole CSV from 06_predict.py",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization",
        help="Directory for saving plots",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="In2O3",
        help="Prefix for output image files",
    )

    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.5,
        help="Arrow scaling factor for quiver plots",
    )

    parser.add_argument(
        "--min_magnitude",
        type=float,
        default=0.0,
        help="Only show atoms with |dipole| >= this value",
    )

    parser.add_argument(
        "--show_atom_index",
        action="store_true",
        help="Show atom index labels on plots",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_atomic_dipoles(input_csv)
    print_summary(df)

    out_3d = output_dir / f"{args.prefix}_atomic_dipoles_3d.png"
    out_xy = output_dir / f"{args.prefix}_atomic_dipoles_xy.png"
    out_xz = output_dir / f"{args.prefix}_atomic_dipoles_xz.png"
    out_yz = output_dir / f"{args.prefix}_atomic_dipoles_yz.png"

    plot_3d(
        df,
        out_3d,
        scale_factor=args.scale_factor,
        min_magnitude=args.min_magnitude,
        show_atom_index=args.show_atom_index,
    )
    plot_2d_projection(
        df, "xy", out_xy,
        scale_factor=args.scale_factor,
        min_magnitude=args.min_magnitude,
        show_atom_index=args.show_atom_index,
    )
    plot_2d_projection(
        df, "xz", out_xz,
        scale_factor=args.scale_factor,
        min_magnitude=args.min_magnitude,
        show_atom_index=args.show_atom_index,
    )
    plot_2d_projection(
        df, "yz", out_yz,
        scale_factor=args.scale_factor,
        min_magnitude=args.min_magnitude,
        show_atom_index=args.show_atom_index,
    )

    print()
    print("Saved plots:")
    print(f"  {out_3d}")
    print(f"  {out_xy}")
    print(f"  {out_xz}")
    print(f"  {out_yz}")


if __name__ == "__main__":
    main()