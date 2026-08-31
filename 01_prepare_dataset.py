#!/usr/bin/env python3
"""
01_prepare_dataset.py

Prepare a dataset file for polarization / dipole-moment GNN training.

Input:
    1. A folder containing POSCAR / CONTCAR / CIF / VASP structure files
    2. A CSV file containing target dipole or polarization labels

Example id_prop.csv:
    id,Px,Py,Pz
    struct_001,0.12,-0.01,1.45
    struct_002,0.08,0.02,1.31

Output:
    processed/dataset.csv

The output CSV contains:
    id, structure_path, Px, Py, Pz, num_atoms, volume
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from ase.io import read


SUPPORTED_EXTENSIONS = [
    ".vasp",
    ".poscar",
    ".contcar",
    ".cif",
    ".xyz",
]


def find_structure_file(structure_dir: Path, structure_id: str) -> Path:
    """
    Find a structure file matching a given structure ID.

    Examples:
        struct_001.vasp
        struct_001.cif
        POSCAR_struct_001
    """

    # First try exact matches with supported extensions
    for ext in SUPPORTED_EXTENSIONS:
        candidate = structure_dir / f"{structure_id}{ext}"
        if candidate.exists():
            return candidate

    # Try uppercase variants
    for ext in SUPPORTED_EXTENSIONS:
        candidate = structure_dir / f"{structure_id}{ext.upper()}"
        if candidate.exists():
            return candidate

    # Try loose matching
    matches = []
    for file in structure_dir.iterdir():
        if file.is_file() and structure_id in file.name:
            matches.append(file)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple structure files found for ID '{structure_id}': "
            f"{[m.name for m in matches]}"
        )

    raise FileNotFoundError(
        f"No structure file found for ID '{structure_id}' in {structure_dir}"
    )


def validate_label_file(label_csv: Path) -> pd.DataFrame:
    """
    Read and validate id_prop.csv.
    Required columns:
        id, Px, Py, Pz
    """

    if not label_csv.exists():
        raise FileNotFoundError(f"Label file not found: {label_csv}")

    df = pd.read_csv(label_csv)

    required_columns = ["id", "Px", "Py", "Pz"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {label_csv}: {missing}\n"
            f"Required columns are: {required_columns}"
        )

    df["id"] = df["id"].astype(str)

    for col in ["Px", "Py", "Pz"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def prepare_dataset(
    structure_dir: Path,
    label_csv: Path,
    output_dir: Path,
    output_name: str = "dataset.csv",
) -> Path:
    """
    Create a clean dataset CSV linking structure paths and target labels.
    """

    structure_dir = structure_dir.resolve()
    label_csv = label_csv.resolve()
    output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = validate_label_file(label_csv)

    rows = []

    for _, row in labels.iterrows():
        structure_id = str(row["id"])

        try:
            structure_path = find_structure_file(structure_dir, structure_id)

            atoms = read(structure_path)
            num_atoms = len(atoms)
            volume = atoms.get_volume()

            rows.append(
                {
                    "id": structure_id,
                    "structure_path": str(structure_path.resolve()),
                    "Px": float(row["Px"]),
                    "Py": float(row["Py"]),
                    "Pz": float(row["Pz"]),
                    "num_atoms": int(num_atoms),
                    "volume_A3": float(volume),
                }
            )

        except Exception as exc:
            print(f"[WARNING] Skipping {structure_id}: {exc}")

    if len(rows) == 0:
        raise RuntimeError(
            "No valid structures were processed. "
            "Check your structure filenames and id_prop.csv IDs."
        )

    dataset = pd.DataFrame(rows)

    output_path = output_dir / output_name
    dataset.to_csv(output_path, index=False)

    print("=" * 70)
    print("Dataset preparation finished")
    print("=" * 70)
    print(f"Number of structures: {len(dataset)}")
    print(f"Output file: {output_path}")
    print()
    print("First few rows:")
    print(dataset.head())

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare structure-label dataset for polarization GNN."
    )

    parser.add_argument(
        "--structure_dir",
        type=str,
        default="raw_structures",
        help="Folder containing POSCAR/CIF/VASP structure files.",
    )

    parser.add_argument(
        "--label_csv",
        type=str,
        default="id_prop.csv",
        help="CSV file containing id, Px, Py, Pz columns.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Folder where processed dataset CSV will be saved.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="dataset.csv",
        help="Name of output dataset CSV file.",
    )

    args = parser.parse_args()

    prepare_dataset(
        structure_dir=Path(args.structure_dir),
        label_csv=Path(args.label_csv),
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()