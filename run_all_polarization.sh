#!/usr/bin/env python3
"""
08_calculate_polarization.py

Convert predicted dipole moment to polarization.

Input:
    predictions/In2O3_total_prediction.csv

Optional input:
    predictions/In2O3_atomic_dipoles.csv

Output:
    polarization/In2O3_polarization.csv
    polarization/In2O3_atomic_polarization.csv

Main formula:
    P = dipole / volume

If dipole is in e Angstrom and volume is in Angstrom^3:

    P = e Angstrom / Angstrom^3
      = e / Angstrom^2

Unit conversion:
    1 e / Angstrom^2 = 16.02176634 C/m^2

Important:
    This script assumes the predicted values are dipole moments in e Angstrom.
    If your target labels are already polarization values, do not divide by volume again.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


E_PER_A2_TO_C_PER_M2 = 16.02176634


def load_total_prediction(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Total prediction CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["id", "pred_Px", "pred_Py", "pred_Pz"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}"
        )

    if "volume_A3" not in df.columns:
        raise ValueError(
            f"'volume_A3' column not found in {csv_path}.\n"
            f"Run 06_predict.py again, or provide --volume manually."
        )

    return df


def calculate_total_polarization(
    df: pd.DataFrame,
    manual_volume: float = None,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        structure_id = row["id"]

        if manual_volume is not None:
            volume = float(manual_volume)
        else:
            volume = float(row["volume_A3"])

        if volume <= 0:
            raise ValueError(f"Invalid volume for {structure_id}: {volume}")

        dipole = np.array(
            [
                float(row["pred_Px"]),
                float(row["pred_Py"]),
                float(row["pred_Pz"]),
            ]
        )

        pol_e_A2 = dipole / volume
        pol_C_m2 = pol_e_A2 * E_PER_A2_TO_C_PER_M2

        pol_mag_e_A2 = float(np.linalg.norm(pol_e_A2))
        pol_mag_C_m2 = float(np.linalg.norm(pol_C_m2))

        result = {
            "id": structure_id,
            "volume_A3": volume,

            "dipole_x_eA": dipole[0],
            "dipole_y_eA": dipole[1],
            "dipole_z_eA": dipole[2],
            "dipole_magnitude_eA": float(np.linalg.norm(dipole)),

            "polarization_x_e_per_A2": pol_e_A2[0],
            "polarization_y_e_per_A2": pol_e_A2[1],
            "polarization_z_e_per_A2": pol_e_A2[2],
            "polarization_magnitude_e_per_A2": pol_mag_e_A2,

            "polarization_x_C_per_m2": pol_C_m2[0],
            "polarization_y_C_per_m2": pol_C_m2[1],
            "polarization_z_C_per_m2": pol_C_m2[2],
            "polarization_magnitude_C_per_m2": pol_mag_C_m2,
        }

        if all(
            col in row.index
            for col in ["target_Px", "target_Py", "target_Pz"]
        ):
            target_dipole = np.array(
                [
                    float(row["target_Px"]),
                    float(row["target_Py"]),
                    float(row["target_Pz"]),
                ]
            )

            target_pol_e_A2 = target_dipole / volume
            target_pol_C_m2 = target_pol_e_A2 * E_PER_A2_TO_C_PER_M2

            result.update(
                {
                    "target_dipole_x_eA": target_dipole[0],
                    "target_dipole_y_eA": target_dipole[1],
                    "target_dipole_z_eA": target_dipole[2],

                    "target_polarization_x_C_per_m2": target_pol_C_m2[0],
                    "target_polarization_y_C_per_m2": target_pol_C_m2[1],
                    "target_polarization_z_C_per_m2": target_pol_C_m2[2],

                    "abs_error_polarization_x_C_per_m2": abs(
                        pol_C_m2[0] - target_pol_C_m2[0]
                    ),
                    "abs_error_polarization_y_C_per_m2": abs(
                        pol_C_m2[1] - target_pol_C_m2[1]
                    ),
                    "abs_error_polarization_z_C_per_m2": abs(
                        pol_C_m2[2] - target_pol_C_m2[2]
                    ),
                }
            )

        rows.append(result)

    return pd.DataFrame(rows)


def calculate_atomic_polarization(
    atomic_csv: Path,
    volume_A3: float,
    output_csv: Path,
):
    """
    Convert atomic dipoles to atomic polarization contribution.

    Each atomic contribution:
        P_atom = p_atom / volume

    This is useful for visualizing local contributions, but remember that
    local atomic polarization is model-defined and not uniquely observable.
    """

    if not atomic_csv.exists():
        print(f"[WARNING] Atomic dipole CSV not found: {atomic_csv}")
        print("[WARNING] Skipping atomic polarization calculation.")
        return None

    df = pd.read_csv(atomic_csv)

    required = ["atomic_px", "atomic_py", "atomic_pz"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {atomic_csv}: {missing}"
        )

    if volume_A3 <= 0:
        raise ValueError(f"Invalid volume: {volume_A3}")

    dipole = df[["atomic_px", "atomic_py", "atomic_pz"]].to_numpy()

    pol_e_A2 = dipole / volume_A3
    pol_C_m2 = pol_e_A2 * E_PER_A2_TO_C_PER_M2

    df["atomic_pol_x_e_per_A2"] = pol_e_A2[:, 0]
    df["atomic_pol_y_e_per_A2"] = pol_e_A2[:, 1]
    df["atomic_pol_z_e_per_A2"] = pol_e_A2[:, 2]

    df["atomic_pol_x_C_per_m2"] = pol_C_m2[:, 0]
    df["atomic_pol_y_C_per_m2"] = pol_C_m2[:, 1]
    df["atomic_pol_z_C_per_m2"] = pol_C_m2[:, 2]

    df["atomic_pol_magnitude_C_per_m2"] = np.linalg.norm(
        pol_C_m2,
        axis=1,
    )

    df.to_csv(output_csv, index=False)

    return df


def print_summary(total_df: pd.DataFrame):
    print("=" * 70)
    print("Polarization calculation finished")
    print("=" * 70)

    for _, row in total_df.iterrows():
        print(f"Structure: {row['id']}")
        print(f"Volume: {row['volume_A3']:.8f} Å^3")
        print()
        print("Dipole moment:")
        print(f"  px = {row['dipole_x_eA']:.10e} eÅ")
        print(f"  py = {row['dipole_y_eA']:.10e} eÅ")
        print(f"  pz = {row['dipole_z_eA']:.10e} eÅ")
        print(f"  |p| = {row['dipole_magnitude_eA']:.10e} eÅ")
        print()
        print("Polarization:")
        print(f"  Px = {row['polarization_x_C_per_m2']:.10e} C/m^2")
        print(f"  Py = {row['polarization_y_C_per_m2']:.10e} C/m^2")
        print(f"  Pz = {row['polarization_z_C_per_m2']:.10e} C/m^2")
        print(
            f"  |P| = "
            f"{row['polarization_magnitude_C_per_m2']:.10e} C/m^2"
        )
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert predicted dipole moment to polarization."
    )

    parser.add_argument(
        "--total_csv",
        type=str,
        default="predictions/In2O3_total_prediction.csv",
        help="CSV from 06_predict.py containing total prediction.",
    )

    parser.add_argument(
        "--atomic_csv",
        type=str,
        default="predictions/In2O3_atomic_dipoles.csv",
        help="Optional CSV from 06_predict.py containing atomic dipoles.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="polarization",
        help="Directory for output CSV files.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="In2O3",
        help="Prefix for output files.",
    )

    parser.add_argument(
        "--volume",
        type=float,
        default=None,
        help="Manual volume in Angstrom^3. Overrides volume_A3 in CSV.",
    )

    parser.add_argument(
        "--skip_atomic",
        action="store_true",
        help="Skip atomic polarization calculation.",
    )

    args = parser.parse_args()

    total_csv = Path(args.total_csv)
    atomic_csv = Path(args.atomic_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_df_raw = load_total_prediction(total_csv)
    total_df = calculate_total_polarization(
        total_df_raw,
        manual_volume=args.volume,
    )

    total_output = output_dir / f"{args.prefix}_polarization.csv"
    total_df.to_csv(total_output, index=False)

    print_summary(total_df)

    if not args.skip_atomic:
        volume_A3 = float(total_df.iloc[0]["volume_A3"])
        atomic_output = output_dir / f"{args.prefix}_atomic_polarization.csv"

        atomic_df = calculate_atomic_polarization(
            atomic_csv=atomic_csv,
            volume_A3=volume_A3,
            output_csv=atomic_output,
        )

        if atomic_df is not None:
            print(f"Atomic polarization saved to: {atomic_output}")

    print(f"Total polarization saved to: {total_output}")


if __name__ == "__main__":
    main()
