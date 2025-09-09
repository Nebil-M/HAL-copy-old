import numpy as np

from hal.codes.generalized_toric_codes import GeneralizedToricCode
from hal.codes.radial_codes import RadialCode
from hal.codes.tile_codes import TileCode
from hal.hal import HardwareAwareLayout


def test_integration_place_route_benchmark(tmp_path) -> None:
    code = GeneralizedToricCode.from_paper_parameters(
        a=1, b=1, c=1, d=1, lattice_vectors=((0, 3), (4, 2))
    )
    hal = HardwareAwareLayout(
        name="integration", directory_path=str(tmp_path), tanner_graph=code.graph
    )
    hal.place(layout="community", mps_edge_order="length_desc", grid_size=40)
    hal.route()
    df = hal.benchmark()
    assert "hardware_complexity" in df.columns
    assert df.shape[0] == 1


def test_integration_route_radial_code(tmp_path) -> None:
    # Small deterministic radial code
    code = RadialCode(s=2, r=2)
    hal = HardwareAwareLayout(
        name="radial_test", directory_path=str(tmp_path), tanner_graph=code.graph
    )
    hal.place(grid_size=120)
    hal.route()
    df = hal.benchmark()
    assert "hardware_complexity" in df.columns


def test_integration_route_tile_code(tmp_path) -> None:
    # Construct a tiny tile code via paper parameters
    # Using simple degrees and small matrix size for speed
    f_ab = (1, 1)
    g_ab = (1, 1)
    matrix_size = (2, 2)

    tile_code = TileCode.init_from_generalized_toric(
        f_ab=f_ab, g_ab=g_ab, matrix_size=matrix_size, timeout=0.1
    )
    hal = HardwareAwareLayout(
        name="tile_test",
        directory_path=str(tmp_path),
        tanner_graph=tile_code.tanner_graph,
    )
    hal.place(grid_size=120)
    hal.route()
    df = hal.benchmark()
    assert "hardware_complexity" in df.columns
