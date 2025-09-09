import numpy as np

from hal.codes.generalized_toric_codes import GeneralizedToricCode
from hal.hal import HardwareAwareLayout


def test_route_produces_routes(tmp_path) -> None:
    code = GeneralizedToricCode.from_paper_parameters(
        a=0, b=1, c=1, d=0, lattice_vectors=((0, 3), (3, 0))
    )
    hal = HardwareAwareLayout(name="test", directory_path=str(tmp_path), tanner_graph=code.graph)
    hal.place(layout="community", mps_edge_order="length_desc", grid_size=40)
    hal.route()

    # At least one tier with routed edges and routes assigned
    routed_any = False
    for tier in hal.tiers:
        for u, v in tier.G.edges:
            if "route" in tier.G.edges[u, v]:
                routed_any = True
                break
        if routed_any:
            break
    assert routed_any
