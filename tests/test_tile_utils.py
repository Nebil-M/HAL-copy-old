from hal.codes.tile_codes import Tile


def test_transform_roundtrip_and_orientation() -> None:
    # Grid coords: even x -> vertical edge, odd x -> horizontal edge
    grid_coords = [(0, 1), (1, 0), (3, 4), (2, 5), (5, 6)]
    for coord in grid_coords:
        edge = Tile.transform_grid_to_edge_coord(coord)
        # Edge is (x, y, orientation)
        x, y, orient = edge
        if coord[0] % 2 == 0:
            assert orient == "v"
        else:
            assert orient == "h"

        # Roundtrip back to grid coordinate
        roundtrip = Tile.transform_edge_to_grid_coord(edge)
        assert roundtrip == coord


def test_get_optimal_check_qubit_coord_matches_reference() -> None:
    # Small synthetic tile grid
    tile_grid_size = (6, 6)
    data_qubits = [(1, 1), (1, 3), (3, 1), (3, 3)]  # diamond around center

    # Function under test
    best = Tile.get_optimal_check_qubit_coord(tile_grid_size, data_qubits)

    # Basic properties
    assert 0 <= best[0] < tile_grid_size[0]
    assert 0 <= best[1] < tile_grid_size[1]
    assert (best[0] + best[1]) % 2 == 0  # parity constraint

    # Reference selection: maximize neighbor count (Manhattan distance 1),
    # then minimize total Euclidean distance
    def evaluate(coord):
        i, j = coord
        neighbors = sum(1 for x, y in data_qubits if abs(x - i) + abs(y - j) == 1)
        total_dist = sum(((x - i) ** 2 + (y - j) ** 2) ** 0.5 for x, y in data_qubits)
        return neighbors, total_dist

    candidates = [
        (i, j)
        for i in range(tile_grid_size[0])
        for j in range(tile_grid_size[1])
        if (i + j) % 2 == 0
    ]

    # Compute the best according to the same tie-break
    max_neighbors = max(evaluate(c)[0] for c in candidates)
    tied = [c for c in candidates if evaluate(c)[0] == max_neighbors]
    ref = min(tied, key=lambda c: evaluate(c)[1])

    assert best == ref
