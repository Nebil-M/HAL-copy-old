import numpy as np

from hal.codes.generalized_toric_codes import GeneralizedToricCode
from hal.hal import HardwareAwareLayout


def test_place_generates_tiers_and_grids(tmp_path) -> None:
    code = GeneralizedToricCode.from_paper_parameters(
        a=-1, b=0, c=3, d=2, lattice_vectors=((0, 3), (9, 0))
    )
    hal = HardwareAwareLayout(name="test", directory_path=str(tmp_path), tanner_graph=code.graph)
    hal.place(grid_size=500)

    assert hal.tiers is not None
    assert len(hal.tiers) >= 1
    tier0 = hal.tiers[0]
    assert tier0.grid is not None
    assert tier0.expanded_grid is not None
