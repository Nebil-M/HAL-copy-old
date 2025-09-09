import os
import shutil
from pathlib import Path

import pandas as pd

from hal.website.utils.build_website_data import process_layout_folders


def test_process_layout_folders_adds_new_entries(tmp_path):
    # Arrange: fake source folder with one new layout containing images and benchmark.csv
    src = tmp_path / "20250716_11h01m28s_RadialCode_[[24,2,6]]"
    src.mkdir(parents=True)
    # Minimal layer images
    for i in range(3):
        (src / f"layer_{i:02d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # Minimal benchmark.csv with required columns and one row
    bench = pd.DataFrame(
        {
            "num_layers": [3],
            "avg_coupler_length": [1.23],
            "max_avg_face_switches": [2.34],
            "avg_tsvs_per_edge": [0.12],
            "hardware_complexity": [42.0],
        }
    )
    bench.to_csv(src / "benchmark.csv", index=False)

    website_dir = tmp_path / "website"
    (website_dir / "assets").mkdir(parents=True)

    # Act
    process_layout_folders(
        source_dir=str(tmp_path), website_dir=str(website_dir), code_type="Radial code"
    )

    # Assert: assets copied and CSV updated
    copied_assets = website_dir / "assets" / src.name
    assert copied_assets.is_dir()
    images = list(copied_assets.glob("layer_*.png"))
    assert len(images) == 3

    csv_path = website_dir / "data.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    # One row appended
    assert len(df) == 1
    # Path is recorded as relative assets path
    assert df.loc[0, "layout_dir_path"].endswith(f"assets/{src.name}")
    # Columns present
    for col in [
        "hover_label",
        "code_type",
        "nkd",
        "logical_efficiency",
        "num_layers",
        "avg_coupler_length",
        "max_avg_face_switches",
        "avg_tsvs_per_edge",
        "hardware_cost",
    ]:
        assert col in df.columns
