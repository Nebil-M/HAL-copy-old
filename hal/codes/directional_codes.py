import galois
import networkx as nx
import numpy as np
import sympy
from qldpc.codes.common import CSSCode
from qldpc.objects import Node


class DirectionalCode:
    """
    Generates a directional code instance based on the provided v1 and v2 vectors.

    This class implements the construction of directional codes as described in the paper
    "Directional Codes: a new family of quantum LDPC codes on hexagonal- and
    square-grid connectivity hardware". It can generate the parity check matrix and
    a Tanner graph representation of the code.

    Args:
        v1 (tuple[int, int]): The first vector defining the parallelogram for the torus.
        v2 (tuple[int, int]): The second vector defining the parallelogram for the torus.
        layout (str, optional): The layout of X and Z stabilizers.
            Defaults to "Layout 1".
    """

    def __init__(
        self, v1: tuple[int, int], v2: tuple[int, int], grid_type: str, layout_type: int = 1
    ) -> None:
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.grid_type = grid_type
        self.layout_type = layout_type

        # Determine grid dimensions from the bounding box of the parallelogram
        # defined by the origin, v1, v2, and v1+v2.
        corners_x = [0, self.v1[0], self.v2[0], self.v1[0] + self.v2[0]]
        corners_y = [0, self.v1[1], self.v2[1], self.v1[1] + self.v2[1]]

        # Grid dimensions span from 0 to the maximum coordinate in each axis.
        self.width = max(corners_x) + 1
        self.height = max(corners_y) + 1

        if self.width <= 1 or self.height <= 1:
            raise ValueError(
                "Vectors v1 and v2 must result in a grid with width and height greater than 1 for toric conditions."
            )

        self.graph = nx.Graph()
        self.positions = {}
        self.coord_to_node = {}
        self._generate_nodes()
        self._generate_connectivity()

    def _generate_nodes(self) -> None:
        """Creates qldpc.Node objects, adds them to the graph, and assigns positions."""
        for y in range(self.height):
            for x in range(self.width):
                index = y * self.width + x
                is_data = (x + y) % 2 == 0
                node = Node(index, is_data=is_data)

                self.graph.add_node(node)
                self.positions[node] = (x, y)
                self.coord_to_node[(x, y)] = node

    def _generate_connectivity(self) -> None:
        """Adds edges to the graph based on the specified grid type."""
        if self.grid_type == "square":
            self._connect_square()
        elif self.grid_type == "hex":
            self._connect_hex()
        else:
            raise ValueError("grid_type must be 'square' or 'hex'.")

    def _connect_square(self) -> None:
        """Connects nodes for a square-grid with toric boundary conditions."""
        for y in range(self.height):
            for x in range(self.width):
                current_node = self.coord_to_node[(x, y)]

                # Connect horizontally with wrap-around
                right_coord = ((x + 1) % self.width, y)
                right_neighbor = self.coord_to_node[right_coord]
                self.graph.add_edge(current_node, right_neighbor)

                # Connect vertically with wrap-around
                bottom_coord = (x, (y + 1) % self.height)
                bottom_neighbor = self.coord_to_node[bottom_coord]
                self.graph.add_edge(current_node, bottom_neighbor)

    def _connect_hex(self) -> None:
        """
        Connects nodes for a hexagonal-style grid with toric boundary conditions.

        This implementation creates a 'brickwork' pattern, ensuring every node has
        a degree of 3, using only horizontal and vertical connections on the grid.
        """
        for y in range(self.height):
            for x in range(self.width):
                current_node = self.coord_to_node[(x, y)]

                # 1. Add universal vertical connections.
                # Every node connects to the one below it. This provides two
                # connections for each node (one up, one down) after the loop completes.
                bottom_coord = (x, (y + 1) % self.height)
                bottom_neighbor = self.coord_to_node[bottom_coord]
                self.graph.add_edge(current_node, bottom_neighbor)

                # 2. Add staggered horizontal connections for the third neighbor.
                # The connection depends on the checkerboard parity of the node (x+y).
                # This ensures each node gets exactly one horizontal connection.
                add_horizontal_link = False
                if self.layout_type in [1, 2]:
                    # Stagger based on even parity
                    if (x + y) % 2 == 0:
                        add_horizontal_link = True
                elif self.layout_type == 3:
                    # Stagger based on odd parity (creates a shifted lattice)
                    if (x + y) % 2 != 0:
                        add_horizontal_link = True

                if add_horizontal_link:
                    right_coord = ((x + 1) % self.width, y)
                    right_neighbor = self.coord_to_node[right_coord]
                    self.graph.add_edge(current_node, right_neighbor)
