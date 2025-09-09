import numpy as np

from hal.routing.astar import AStar2DRouter, AStar3DRouter, astar


def test_astar_2d_simple_path():
    # 0 = free, 2 = obstacle, 10 = preferred-avoid (treated as blocked unless goal)
    grid = np.zeros((7, 7), dtype=int)
    # Place a vertical wall with one gap
    grid[1:6, 3] = 2
    grid[3, 3] = 0  # gap at (3,3)

    start = (1, 1)
    goal = (5, 5)
    path = astar(grid, start, goal)
    assert path, "Expected a valid path"
    # Path should pass through the gap column x=3 at y=3
    assert (3, 3) in path


def test_AStar3DRouter_routes_with_layers():
    grid = np.zeros((5, 5, 3), dtype=int)
    # Block middle layer heavily to force layer changes
    grid[:, :, 1] = 2
    # But leave a vertical shaft at (2,2) through all layers
    grid[2, 2, 1] = 0

    router = AStar3DRouter(grid)
    start = (0, 0, 0)
    goal = (4, 4, 2)
    path = router.route(start, goal)
    assert path, "Expected a valid 3D path"
    # The path should be within grid bounds and end at the goal
    assert path[0] == start and path[-1] == goal
