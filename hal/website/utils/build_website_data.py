import glob
import json
import os
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd

code_types = {
    "surface_codes": "Surface code",
    "directional_codes": "Directional code",
    "gt_codes": "BB code",
    "narrow_gt_codes": "Narrow BB code",
    "gross_codes": "Gross code",
    "tile_codes": "Tile code",
    "radial_codes": "Radial code",
    "quantum_tanner_codes": "Tanner code",
}


def process_layout_folders(
    source_dir: str, website_dir: str, group: str | None = None
) -> None:  # pylint: disable=too-many-branches,too-many-locals
    """
    Scans for new layout folders and updates the website's assets and data.csv
    without overwriting existing entries.
    """
    code_type = code_types.get(group, "Unknown code type")
    assets_dir = os.path.join(website_dir, "assets")
    csv_path = os.path.join(website_dir, "data.csv")

    # --- 1. Load existing data or create an empty DataFrame ---
    try:
        existing_df = pd.read_csv(csv_path)
        print(f"Loaded {len(existing_df)} existing entries from {csv_path}")
    except FileNotFoundError:
        existing_df = pd.DataFrame(
            columns=[
                "hover_label",
                "code_type",
                "nkd",
                "logical_efficiency",
                "weight",
                "num_layers",
                "avg_coupler_length",
                "max_avg_face_switches",
                "avg_tsvs_per_edge",
                "hardware_cost",
                "layout_dir_path",
            ]
        )
        print("No existing data.csv found. A new one will be created.")

    # Create a set of already processed directories for fast lookups
    processed_dirs = set(existing_df["layout_dir_path"])
    
    folder_name_pattern = re.compile(
        r".*?\[\[\s*(?P<n>\d+)\s*,\s*(?P<k>\d+)\s*,\s*(?P<d>\d+)\s*\]\]"
    )

    new_codes_data = []

    # --- 2. Iterate through source folders and check against existing data ---
    for folder_name in os.listdir(source_dir):
        source_folder_path = os.path.join(source_dir, folder_name)

        # Construct the expected asset path for this folder
        expected_asset_path = f"assets/{folder_name}"

        if not os.path.isdir(source_folder_path):
            continue

        # Check if this layout has already been processed
        if expected_asset_path in processed_dirs:
            continue  # Skip to the next folder

        # --- If we reach here, it's a NEW layout that needs processing ---
        match = folder_name_pattern.match(folder_name)
        if not match:
            print(f"--> Skipping folder (name does not match pattern): {folder_name}")
            continue

        print(f"\nProcessing NEW folder: {folder_name}")

        # Create destination directory in assets
        dest_asset_dir = os.path.join(assets_dir, folder_name)
        os.makedirs(dest_asset_dir, exist_ok=True)

        # Find and copy all layer images; backward compatibility for naming of tiers
        escaped_source_path = glob.escape(source_folder_path)
        image_files = sorted(glob.glob(os.path.join(escaped_source_path, "layer_*.png")))
        image_files_tiers = sorted(glob.glob(os.path.join(escaped_source_path, "tier_*.png")))

        if not image_files:
            if not image_files_tiers:
                print(f"--> No 'layer_*.png' or 'tier_*.png' images found in {folder_name}. Skipping.")
                os.rmdir(dest_asset_dir)
                continue
            else:
                image_files = image_files_tiers
                tiers_string = "tiers"
        else:
            tiers_string = "layers"

        for img_path in image_files:
            shutil.copy(img_path, dest_asset_dir)
        print(f"  - Copied {len(image_files)} layer images to {dest_asset_dir}")

        # Extract data for the CSV; Normalize captures from either ordering
        m = folder_name_pattern.match(folder_name)        
        if m:
            n, k, d = map(int, (m["n"], m["k"], m["d"]))

        if n == 0:
            print(f"--> Skipping {folder_name} due to n=0.")
            continue
        logical_efficiency = (k * d**2) / n

        benchmark_file = os.path.join(source_folder_path, "benchmark.csv")
        try:
            benchmark_df = pd.read_csv(benchmark_file)
            num_layers = benchmark_df[f"num_{tiers_string}"].iloc[-1]
            avg_coupler_length = benchmark_df["avg_coupler_length"].iloc[-1]
            max_avg_face_switches = benchmark_df["max_avg_face_switches"].iloc[-1]
            avg_tsvs_per_edge = benchmark_df["avg_tsvs_per_edge"].iloc[-1]
            hardware_cost = benchmark_df["hardware_complexity"].iloc[-1]
        except (FileNotFoundError, KeyError, IndexError) as e:
            print(f"--> Could not read hardware_cost from {benchmark_file}. Error: {e}")
            continue

        if code_type in {"BB code", "Narrow BB code", "Gross code"}:
            weight = 6
        elif code_type == "Radial code":
            r = np.sqrt(int(k) / 2) + 1
            weight = 2 * r
        elif code_type == "Tile code":
            if k <= 8:
                weight = 6
            elif k == 12 or (k == 18 and d < 23):
                weight = 8
            elif k == 18 and d == 23:
                weight = 10

        # Ensure weight is set for all code types
        weight = weight if "weight" in locals() else None
        # Add the new data to a temporary list
        new_codes_data.append(
            {
                "hover_label": f"{code_type} [[{n}, {k}, {d}]]",
                "code_type": code_type,
                "weight": weight,
                "nkd": (n, k, d),
                "logical_efficiency": logical_efficiency,
                "num_layers": num_layers,
                "avg_coupler_length": avg_coupler_length,
                "max_avg_face_switches": max_avg_face_switches,
                "avg_tsvs_per_edge": avg_tsvs_per_edge,
                "hardware_cost": hardware_cost,
                "layout_dir_path": expected_asset_path,
            }
        )

    # --- 2b. Handle surface and codes stored as JSONs only ---
    for category in ["surface_codes"]:
        category_path = source_dir
        if category not in category_path:
            continue

        for folder_name in os.listdir(category_path):
            folder_path = os.path.join(category_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            json_path = os.path.join(folder_path, "code_params.json")
            if not os.path.isfile(json_path):
                print(f"--> Skipping {folder_path}: no code_params.json found.")
                continue

            expected_asset_path = f"assets/{category}/{folder_name}"
            if expected_asset_path in processed_dirs:
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
            except Exception as e:
                print(f"--> Failed to read {json_path}. Error: {e}")
                continue

            match = re.match(r"\[\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]\]", folder_name)
            if not match:
                print(f"--> Skipping malformed folder name: {folder_name}")
                continue

            n, k, d = map(int, match.groups())
            if n == 0:
                continue
            logical_efficiency = (k * d**2) / n

            hover_label = None
            weight = None
            if code_type == "Surface code":
                weight = 4
                hover_label = f"{code_type} [[n, 1, sqrt(n)]]"
            elif code_type == "Directional code":
                weight = 3 if k <= 4 else 4
                hover_label = f"{code_type} [[{n}, {k}, {d}]]"
            else:
                hover_label = f"{code_type} [[{n}, {k}, {d}]]"
                weight = weight or 0

            new_codes_data.append(
                {
                    "hover_label": hover_label,
                    "code_type": code_type,
                    "weight": weight,
                    "nkd": (n, k, d),
                    "logical_efficiency": logical_efficiency,
                    "num_layers": params["num_layers"],
                    "avg_coupler_length": params["avg_coupler_length"],
                    "max_avg_face_switches": params["max_avg_face_switches"],
                    "avg_tsvs_per_edge": params["avg_tsvs_per_edge"],
                    "hardware_cost": params["hardware_complexity"],
                    "layout_dir_path": expected_asset_path,
                }
            )

    # --- 3. Append new data to the existing DataFrame and save ---
    if not new_codes_data:
        print("\nNo new layouts found to process. 'data.csv' is already up to date.")
        return

    new_df = pd.DataFrame(new_codes_data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    combined_df.to_csv(csv_path, index=False)
    print(
        f"\nAppended {len(new_codes_data)} new entries. 'data.csv' now has {len(combined_df)} total entries."
    )
