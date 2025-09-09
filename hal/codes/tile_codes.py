from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, cast

import numpy as np
from qec.code_constructions import CSSCode as CSSCode_w_distance

from hal.utils.graph_utils import generate_tanner_graph


class Qubit:
    """Representation of a qubit within a tile.

    Attributes
    ----------
    coord : tuple[int, int]
        Absolute grid coordinate.
    qubit_type : Literal["data", "X", "Z"]
        Type of the qubit (data or check basis).
    parent_tile : Tile | None
        Reference to the containing tile.
    connected_qubits : list[Qubit]
        Adjacency list of connected qubits.
    id : int | None
        Unique identifier assigned when generating parity-check matrices.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        qubit_type: Literal["data", "X", "Z"],
        parent_tile=None,
    ):

        # extract init parameters
        self.coord = coord
        self.qubit_type = qubit_type
        self.parent_tile = parent_tile

        self.connected_qubits = []  # List of nodes connected to this node

        self.id = None  # unique identifier, set when calling parity check matrix generation

    @property
    def is_fully_checked(self) -> bool:
        """
        Check if the qubit is fully checked.
        A data qubit is fully checked if it has at least one z- and one x-check qubit connected to it.
        A check qubit is itself not checked. Print and return.
        """
        if self.qubit_type == "data":
            has_x_check = any(q.qubit_type == "X" for q in self.connected_qubits)
            has_z_check = any(q.qubit_type == "Z" for q in self.connected_qubits)
            return has_x_check and has_z_check
        print(f"Qubit at {self.coord} is a check qubit. It is not connected to other check qubits.")
        return False

    @property
    def relative_coord(self) -> Tuple[int, int]:
        """
        Get the relative coordinates of the qubit in the tile.
        The coordinates are represented as a tuple of (x, y).
        """
        if self.parent_tile is not None:
            return (
                self.coord[0] - self.parent_tile.origin[0],
                self.coord[1] - self.parent_tile.origin[1],
            )
        return self.coord

    @property
    def has_empty_support(self) -> bool:
        """
        Return whether the check qubit has empty support.
        """
        if self.qubit_type == "data":
            print(f"Qubit at {self.coord} is a data qubit. This function is used for check qubits.")
            return False
        return len(self.connected_qubits) == 0

    @property
    def is_data(self) -> bool:
        """
        Check if the qubit is a data qubit.
        """
        return self.qubit_type == "data"


class TileParamsType(TypedDict):
    bounding_box: Tuple[int, int]
    data_qubit_coords: List[Tuple[int, int]]
    basis: Literal["X", "Z"]
    origin: Tuple[int, int]
    check_qubit_coord: Optional[Tuple[int, int]]


EdgeCoordType = Tuple[int, int, Literal["h", "v"]]
GridCoordType = Tuple[int, int]
BasisType = Literal["X", "Z"]


class Tile:
    """Tile representing a single stabilizer with support on neighboring edges.

    Attributes
    ----------
    bounding_box : tuple[int, int]
        Dimensions of the tile in edge coordinates.
    data_qubit_coords : list[tuple[int, int]]
        Relative grid coordinates of data qubits in the tile.
    basis : Literal["X", "Z"]
        Stabilizer basis.
    origin : tuple[int, int]
        Origin of the tile in grid coordinates within the tile matrix.
    check_qubit_coord : tuple[int, int] | None
        Relative coordinate of the check qubit; computed if not provided.
    parent_tile_matrix : TileMatrix | None
        Container matrix reference for shared qubit lists.
    tile_grid_size : tuple[int, int]
        Derived grid size for this tile (2× bounding_box).
    check_qubit : Qubit
        The check qubit instance created for this tile.
    """

    def __init__(
        self,
        bounding_box: Tuple[int, int],
        data_qubit_coords: List[GridCoordType],
        basis: BasisType,
        origin: GridCoordType = (0, 0),
        check_qubit_coord: Optional[GridCoordType] = None,
        parent_tile_matrix=None,
    ):

        # extract init parameters
        self.bounding_box = bounding_box
        self.data_qubit_coords = data_qubit_coords
        self.basis = basis
        self.origin = origin
        self.check_qubit_coord = check_qubit_coord
        self.parent_tile_matrix = parent_tile_matrix
        self.tile_grid_size = (2 * bounding_box[0], 2 * bounding_box[1])

        # instantiate check qubit, shift it according to origin
        if self.check_qubit_coord is not None:
            assert (
                0 <= self.check_qubit_coord[0] < self.tile_grid_size[0]
                and 0 <= self.check_qubit_coord[1] < self.tile_grid_size[1]
            ), f"Check qubit coordinate {self.check_qubit_coord} exceeds tile grid size {self.tile_grid_size}."
            assert (
                np.sum(self.check_qubit_coord) % 2 == 0
            ), f"Check qubit coordinate {self.check_qubit_coord} must have even sum of coordinates."
        else:
            self.check_qubit_coord = self.get_optimal_check_qubit_coord(
                self.tile_grid_size, data_qubit_coords
            )

        self.absolute_check_qubit_coord = (
            self.check_qubit_coord[0] + origin[0],
            self.check_qubit_coord[1] + origin[1],
        )

        self.check_qubit = Qubit(
            coord=self.absolute_check_qubit_coord,
            qubit_type=self.basis,
            parent_tile=self,
        )
        if self.parent_tile_matrix is not None:
            # add the check qubit to the global check qubit list of the parent tile matrix
            self.parent_tile_matrix.global_check_qubit_list.append(self.check_qubit)

        # assert that the data qubit coordinates are within the tile grid size
        for data_qubit_coord in self.data_qubit_coords:
            if not (
                0 <= data_qubit_coord[0] < self.tile_grid_size[0]
                and 0 <= data_qubit_coord[1] < self.tile_grid_size[1]
            ):
                raise ValueError(
                    f"Data qubit coordinate {data_qubit_coord} exceeds tile grid size {self.tile_grid_size}."
                )

        # create the tile
        for i, absolute_data_qubit_coord in enumerate(self.absolute_data_qubit_coords):

            # if there is a reference to a parent tile matrix, check if the data qubit already exists in the global data qubit list
            if self.parent_tile_matrix is not None:

                # check if data qubit is within the parent tile matrix bounding box
                # if it isn't, delete the coordinate from this tile and skip this qubit
                is_within_tile_matrix = (
                    0 <= absolute_data_qubit_coord[0] < self.parent_tile_matrix.matrix_grid_size[0]
                    and 0
                    <= absolute_data_qubit_coord[1]
                    < self.parent_tile_matrix.matrix_grid_size[1]
                )
                if not is_within_tile_matrix:
                    self.data_qubit_coords[i] = None
                    continue

                # check if the data qubit is already in the global qubit list
                qubit_already_exists = False
                for existing_qubit in self.parent_tile_matrix.global_data_qubit_list:
                    if existing_qubit.coord == absolute_data_qubit_coord:
                        # if it exists, skip creating a new qubit
                        qubit_already_exists = True
                        qubit = existing_qubit
                        break

                # if it deoes not exist, create a new qubit and add it to the global data qubit list
                if is_within_tile_matrix and not qubit_already_exists:
                    qubit = Qubit(
                        coord=absolute_data_qubit_coord,
                        qubit_type="data",
                        parent_tile=self,
                    )
                    self.parent_tile_matrix.global_data_qubit_list.append(qubit)

            # if there is no reference to a parent tile matrix, create a new qubit
            else:
                # if there is no reference to a parent tile matrix, create a new qubit
                qubit = Qubit(coord=absolute_data_qubit_coord, qubit_type="data", parent_tile=self)

            qubit.connected_qubits.append(self.check_qubit)
            self.check_qubit.connected_qubits.append(qubit)

        # delete the data qubit coordinates that are None
        self.data_qubit_coords = [coord for coord in self.data_qubit_coords if coord is not None]

    @staticmethod
    def get_optimal_check_qubit_coord(
        tile_grid_size: Tuple[int, int],
        data_qubit_coords: List[GridCoordType],
        blocked_check_qubit_coord: Optional[GridCoordType] = None,
    ) -> GridCoordType:
        """
        Return the coordinate within the tile grid where a check qubit should be placed
        to minimize the distance to all data qubits. The distance is calculated using the specified distance metric.

        Only coordinates with even (i + j) are considered valid.
        """

        coord_evaluation_dict = {}
        max_num_neighbors = -1
        lowest_total_distance = float("inf")

        # get coordinates with most neighboring data qubits in its support
        for i in range(tile_grid_size[0]):
            for j in range(tile_grid_size[1]):
                if (i + j) % 2 == 1:
                    continue

                if blocked_check_qubit_coord is not None and (i, j) == blocked_check_qubit_coord:
                    # skip the blocked check qubit coordinate
                    continue

                # count the number of data qubits in the support of the check qubit
                num_neighbors = sum(1 for x, y in data_qubit_coords if abs(x - i) + abs(y - j) == 1)

                max_num_neighbors = max(max_num_neighbors, num_neighbors)

                # calculate euclidean distance to all data qubits
                total_distance = sum(
                    ((x - i) ** 2 + (y - j) ** 2) ** 0.5 for x, y in data_qubit_coords
                )

                coord_evaluation_dict[(i, j)] = (num_neighbors, total_distance)

        # go through the elements in the dictionary with the max number of neighbors
        # pick the one with the lowest total distance
        best_coord = None
        for coord, (num_neighbors, total_distance) in coord_evaluation_dict.items():
            if num_neighbors == max_num_neighbors:
                if best_coord is None or total_distance < coord_evaluation_dict[best_coord][1]:
                    best_coord = coord

        return cast(GridCoordType, best_coord)

    @property
    def absolute_data_qubit_coords(self) -> List[GridCoordType]:
        """
        Get the absolute coordinates of the data qubits in the tile.
        The coordinates are represented as a tuple of (x, y).
        """
        absolute_data_qubit_coords = []
        for data_qubit_coord in self.data_qubit_coords:
            absolute_data_qubit_coord = (
                data_qubit_coord[0]
                + self.origin[0],  # origin is given in grid coordinates, so we add it directly
                data_qubit_coord[1] + self.origin[1],
            )
            absolute_data_qubit_coords.append(absolute_data_qubit_coord)

        return absolute_data_qubit_coords

    @property
    def data_qubit_edge_coords(self) -> List[EdgeCoordType]:
        """
        Get the edge coordinates of the data qubits in the tile.
        The edge coordinates are represented as a tuple of (x, y, orientation).
        The orientation is either "h" or "v".
        """
        return self.transform_grid_to_edge_coords(self.data_qubit_coords)

    @property
    def absolute_data_qubit_edge_coords(self) -> List[EdgeCoordType]:
        """
        Get the absolute edge coordinates of the data qubits in the tile.
        The edge coordinates are represented as a tuple of (x, y, orientation).
        The orientation is either "h" or "v".
        """

        absolute_data_qubit_edge_coords = []
        for data_qubit_edge_coord in self.data_qubit_edge_coords:
            absolute_data_qubit_edge_coord = (
                data_qubit_edge_coord[0]
                + self.origin[0] // 2,  # origin is given in grid coordinates, so we divide by 2
                data_qubit_edge_coord[1] + self.origin[1] // 2,
                data_qubit_edge_coord[2],
            )
            absolute_data_qubit_edge_coords.append(absolute_data_qubit_edge_coord)

        return absolute_data_qubit_edge_coords

    def print_self(self) -> None:
        """
        Print the tile in a human-readable format.
        The tile is represented as a grid with data qubits marked by 'X' or 'Z'.
        """
        print(f"Tile: {self.basis} tile, bounding box: {self.bounding_box}, origin: {self.origin}")
        print(f"Relative data qubit coordinates: {self.data_qubit_coords}")
        print(f"Absolute data qubit coordinates: {self.absolute_data_qubit_coords}")
        print(f"Absolute check qubit coordinate: {self.check_qubit.coord}\n")

        self.print_tile(
            bounding_box=self.bounding_box,
            data_qubit_edge_coords=self.data_qubit_edge_coords,
            basis=self.basis,  # type: ignore
            check_qubit_coord=self.check_qubit_coord,
        )

    @staticmethod
    def print_tile(
        bounding_box: Tuple[int, int],
        data_qubit_edge_coords: List[EdgeCoordType],
        basis: BasisType = "X",
        check_qubit_coord: Optional[GridCoordType] = None,
    ) -> None:
        """
        Print the tile in a human-readable format.
        The tile is represented as a grid with data qubits marked by 'X' or 'Z'.

        E.g.:

        bounding_box = (4, 4)
        data_qubit_edge_coords: List[Edge] = [
            (0, 0, "v"),
            (0, 0, "h"),
            (1, 0, "v"),
            (0, 1, "v"),
            (2, 1, "h")
        ]
        print_tile(bounding_box, data_qubit_edge_coords, "Z")
        -----------------------
        |   |   |   |
          _   _   _   _
        |   |   |   |
          _   _   _   _
        Z   |   |   |
          _   _   Z   _
        Z   Z   |   |
        . Z   _   _   _

        """

        width, height = bounding_box

        # Each tile contributes 2 rows and 2 columns, so canvas is (2*height+1) x (2*width+1)
        rows = 2 * height
        cols = 4 * width
        canvas = [[" " for _ in range(cols)] for _ in range(rows)]

        # draw plain canvas
        for x in range(width):
            for y in range(height):
                canvas[2 * y][4 * x + 2] = "_"
                canvas[2 * y + 1][4 * x] = "|"

        # add data qubits
        for x, y, d in data_qubit_edge_coords:
            assert 0 <= x < width, f"x={x} out of bounds for width={width}"
            assert 0 <= y < height, f"y={y} out of bounds for height={height}"
            assert d in ("h", "v"), f"Invalid direction {d}, expected 'h' or 'v'"
            if d == "h":
                # canvas[2 * y][4 * x - 2] = 'X'
                canvas[2 * y][4 * x + 2] = basis
            elif d == "v":
                # canvas[2 * y - 1][4 * x] = 'X'
                canvas[2 * y + 1][4 * x] = basis

        # add check qubit
        if check_qubit_coord is not None:
            x = check_qubit_coord[0]
            y = check_qubit_coord[1]
            assert (
                0 <= x < 2 * width and 0 <= y < 2 * height
            ), f"Check qubit coordinate {check_qubit_coord} out of bounds for tile grid size {(2*width, 2*height)}"
            assert (
                x + y
            ) % 2 == 0, (
                f"Check qubit coordinate {check_qubit_coord} must have even sum of coordinates."
            )
            canvas[y][2 * x] = "c"  # Mark check qubit with 'c''

        canvas[0][0] = "."  # Origin marker

        for row in canvas[::-1]:
            print("".join(row))
        print("\n")

    @staticmethod
    def transform_grid_to_edge_coord(grid_coord: Tuple[int, int]) -> EdgeCoordType:
        """
        Transform a grid coordinate to an edge coordinate.
        The edge coordinate is represented as a tuple of (x, y, orientation).
        The orientation is either "h" or "v".
        """
        if grid_coord[0] % 2 == 0:
            # even x coordinate, vertical edge
            edge_coord = (grid_coord[0] // 2, (grid_coord[1] - 1) // 2)
            orientation = "v"
        else:
            # odd x coordinate, horizontal edge
            edge_coord = ((grid_coord[0] - 1) // 2, grid_coord[1] // 2)
            orientation = "h"

        return edge_coord + (orientation,)

    @staticmethod
    def transform_grid_to_edge_coords(
        grid_coord_list: List[GridCoordType],
    ) -> List[EdgeCoordType]:
        """
        Transform the data qubit coordinates in the tile to edge coordinates.
        The edge coordinates are represented as a tuple of (x, y, orientation).
        The orientation is either "h" or "v".
        """
        return [Tile.transform_grid_to_edge_coord(coord) for coord in grid_coord_list]

    @staticmethod
    def transform_edge_to_grid_coord(edge_coord: EdgeCoordType) -> GridCoordType:
        """
        Transform an edge coordinate to a grid coordinate.
        The edge coordinate is represented as a tuple of (x, y, orientation).
        The orientation is either "h" or "v".
        """
        if edge_coord[2] == "v":
            # vertical edge
            grid_coord = (2 * edge_coord[0], 2 * edge_coord[1] + 1)
        else:
            # horizontal edge
            grid_coord = (2 * edge_coord[0] + 1, 2 * edge_coord[1])

        return grid_coord

    @staticmethod
    def transform_edge_to_grid_coords(
        edge_coord_list: List[EdgeCoordType],
    ) -> List[GridCoordType]:
        """
        Transform a list of edge coordinates to grid coordinates.
        The edge coordinates are represented as a tuple of (x, y, orientation).
        The orientation is either "h" or "v".
        """
        return [Tile.transform_edge_to_grid_coord(edge_coord) for edge_coord in edge_coord_list]

    @staticmethod
    def transpose_tile(
        bounding_box: Tuple[int, int], data_qubit_edge_coords: List[EdgeCoordType]
    ) -> List[EdgeCoordType]:
        """Translate the data qubit coordinates from X tile to Z tile."""

        transformed_data_qubit_edge_coords = []
        for edge_coord in data_qubit_edge_coords:
            assert (
                0 <= edge_coord[0] < bounding_box[0] and 0 <= edge_coord[1] < bounding_box[1]
            ), f"Data qubit coordinate {edge_coord} exceeds tile bounding box {bounding_box}."

            new_orientation = "h" if edge_coord[2] == "v" else "v"
            new_edge_coord = (
                bounding_box[0] - 1 - edge_coord[0],
                bounding_box[1] - 1 - edge_coord[1],
                new_orientation,
            )
            transformed_data_qubit_edge_coords.append(new_edge_coord)

        return transformed_data_qubit_edge_coords


class TileMatrix:
    """Matrix of tiles composing a code layout.

    Attributes
    ----------
    x_tile_params, z_tile_params : dict
        Parameter dictionaries used to instantiate X/Z tiles.
    matrix_size : tuple[int, int]
        Number of tiles in X and Y directions.
    matrix_bounding_box : tuple[int, int]
        Effective bounding box of the matrix in tile units.
    matrix_grid_size : tuple[int, int]
        Grid size in points (2× tile units plus overlap adjustments).
    global_data_qubit_list : list[Qubit]
        All data qubits across tiles (deduplicated by absolute coord).
    global_check_qubit_list : list[Qubit]
        All check qubits across tiles.
    tiles_dict : dict[tuple[int, int], list[Tile]]
        Mapping from matrix coordinates to tiles present there.
    deleted_data_qubit_coords : list[tuple[int, int]]
        Coordinates of pruned data qubits.
    """

    def __init__(
        self,
        x_tile_params: TileParamsType,
        matrix_size: Tuple[int, int],
        x_check_qubit_coord: Optional[GridCoordType] = None,
        z_check_qubit_coord: Optional[GridCoordType] = None,
    ):

        # assemble x tile params
        self.x_tile_params = x_tile_params
        self.tile_bounding_box = x_tile_params["bounding_box"]
        self.tile_grid_size = (
            2 * self.tile_bounding_box[0],
            2 * self.tile_bounding_box[1],
        )
        if x_check_qubit_coord is None:
            x_check_qubit_coord = Tile.get_optimal_check_qubit_coord(
                tile_grid_size=self.tile_grid_size,
                data_qubit_coords=x_tile_params["data_qubit_coords"],
            )
        self.x_tile_params["check_qubit_coord"] = x_check_qubit_coord

        # assemble z tile params
        data_qubit_edge_coords = Tile.transform_grid_to_edge_coords(
            x_tile_params["data_qubit_coords"]
        )
        z_data_qubit_edge_coords = Tile.transpose_tile(
            bounding_box=self.tile_bounding_box,
            data_qubit_edge_coords=data_qubit_edge_coords,
        )
        z_data_qubit_coords = Tile.transform_edge_to_grid_coords(z_data_qubit_edge_coords)
        self.z_tile_params = {
            "bounding_box": self.tile_bounding_box,
            "data_qubit_coords": z_data_qubit_coords,
            "basis": "Z",
            "origin": (0, 0),
        }
        if z_check_qubit_coord is None:
            z_check_qubit_coord = Tile.get_optimal_check_qubit_coord(
                tile_grid_size=self.tile_grid_size,
                data_qubit_coords=z_data_qubit_coords,
                blocked_check_qubit_coord=x_check_qubit_coord,  # block the x check qubit coordinate
            )
        assert (
            z_check_qubit_coord != x_check_qubit_coord
        ), f"Z check qubit coordinate {z_check_qubit_coord} must be different from X check qubit coordinate {x_check_qubit_coord}."
        self.z_tile_params["check_qubit_coord"] = z_check_qubit_coord

        self.matrix_size = matrix_size
        self.matrix_bounding_box = (
            self.matrix_size[0] + self.tile_bounding_box[0] - 1,
            self.matrix_size[1] + self.tile_bounding_box[1] - 1,
        )
        self.matrix_grid_size = (
            2 * self.matrix_bounding_box[0],
            2 * self.matrix_bounding_box[1],
        )

        self.global_data_qubit_list = []
        self.global_check_qubit_list = []

        self.deleted_data_qubit_coords = []  # List of deleted data qubit coordinates

        self.tiles_dict = {}
        for i in range(self.matrix_size[0]):
            for j in range(self.matrix_size[1]):
                # create a unique key for each coordinate in the tile matrix
                key = (i, j)
                self.tiles_dict[key] = []

                x_tile_params = deepcopy(self.x_tile_params)
                x_tile_params["origin"] = (2 * i, 2 * j)
                x_tile = Tile(**x_tile_params, parent_tile_matrix=self)
                self.tiles_dict[key].append(x_tile)

                z_tile_params = deepcopy(self.z_tile_params)
                z_tile_params["origin"] = (2 * i, 2 * j)
                z_tile_seed = Tile(**z_tile_params, parent_tile_matrix=self)  # type: ignore
                self.tiles_dict[key].append(z_tile_seed)

    def get_tiles(self, coord: Tuple[int, int]) -> List[Tile]:
        """
        Get the tiles at the given coordinate.
        """
        if coord in self.tiles_dict:
            return self.tiles_dict[coord]
        raise ValueError(
            f"Tile at coordinate {coord} does not exist in the matrix of size {self.matrix_bounding_box}."
        )

    def add_tile(self, coord: Tuple[int, int], basis: BasisType) -> None:
        """
        Add a tile of the given basis at the specified coordinate.

        """
        assert basis in ("X", "Z"), f"Invalid basis {basis}. Expected 'X' or 'Z'."
        # assert (0 <= coord[0] < self.matrix_bounding_box[0] and
        #         0 <= coord[1] < self.matrix_bounding_box[1]), \
        #     f"Coordinate {coord} exceeds matrix bounding box {self.matrix_bounding_box}."

        if basis == "X":
            tile_params = deepcopy(self.x_tile_params)
        else:
            tile_params = deepcopy(self.z_tile_params)

        # check if the tile already exists at the given coordinate
        if coord in self.tiles_dict:
            if len(self.tiles_dict[coord]) == 1:
                exisiting_tile = self.tiles_dict[coord][0]
                if exisiting_tile.basis == basis:
                    print(
                        f"Tile at {coord} with basis {basis} already exists. Skipping addition.\n"
                    )
                    return
                print(
                    f"Tile at {coord} with basis {exisiting_tile.basis} exists. Adding a tile with basis {basis}.\n"
                )
            if len(self.tiles_dict[coord]) == 2:
                print(f"Tile at {coord} already has two tiles. Skipping addition.\n")
                return

        else:
            # if the tile does not exist, create a new entry in the dictionary
            self.tiles_dict[coord] = []

        tile_params["origin"] = (2 * coord[0], 2 * coord[1])
        tile_params = cast(TileParamsType, tile_params)
        tile = Tile(**tile_params, parent_tile_matrix=self)
        self.tiles_dict[coord].append(tile)

    def add_custom_tile(self, coord: Tuple[int, int], tile_params: TileParamsType) -> None:
        """
        Add a custom tile to the tile matrix.
        """

        assert (
            0 <= coord[0] < self.matrix_bounding_box[0]
            and 0 <= coord[1] < self.matrix_bounding_box[1]
        ), f"Coordinate {coord} exceeds matrix bounding box {self.matrix_bounding_box}."

        if coord in self.tiles_dict:
            print(f"Tile at {coord} already exists. Will not add custom tile.\n")
            return
        # if the tile does not exist, create a new entry in the dictionary
        self.tiles_dict[coord] = []
        tile_params["origin"] = (2 * coord[0], 2 * coord[1])
        tile = Tile(**tile_params, parent_tile_matrix=self)
        self.tiles_dict[coord].append(tile)

    def delete_tile(
        self, coord: Tuple[int, int], basis: BasisType, perform_print: bool = True
    ) -> None:
        """
        Delete a tile of the given basis at the specified coordinate.
        """
        assert basis in ("X", "Z"), f"Invalid basis {basis}. Expected 'X' or 'Z'."

        if coord in self.tiles_dict:
            tile_tuple = self.tiles_dict[coord]
            if len(tile_tuple) == 0:
                print(f"No tile with basis {basis} at {coord} to delete.\n")
                return

            # find the tile with the given basis
            tile_index_to_delete = None
            tile_to_delete = None
            for i, tile in enumerate(tile_tuple):
                if tile.basis == basis:
                    tile_index_to_delete = i
                    tile_to_delete = tile
                    break

            if tile_to_delete is None:
                print(f"No tile with basis {basis} at {coord} to delete.\n")
                return

            # remove the corresponding check qubit from the global check qubit list
            self.global_check_qubit_list.remove(tile_to_delete.check_qubit)
            if perform_print:
                print(
                    f"{tile_to_delete.basis}-check qubit at {tile_to_delete.check_qubit.coord} removed from global check qubit list."
                )

            # remove the check qubit from the connected data qubits' adjacency list
            for data_qubit in tile_to_delete.check_qubit.connected_qubits:
                data_qubit.connected_qubits.remove(tile_to_delete.check_qubit)
                if perform_print:
                    print(
                        f"{tile_to_delete.basis}-check qubit at {tile_to_delete.check_qubit.coord} removed from data qubit at {data_qubit.coord} adjacency list."
                    )
                data_qubit_index = tile_to_delete.check_qubit.connected_qubits.index(data_qubit)
                tile_to_delete.check_qubit.connected_qubits[data_qubit_index] = None

            # remove the entries from the check qubit adjacency list that are None
            tile_to_delete.check_qubit.connected_qubits = [
                q for q in tile_to_delete.check_qubit.connected_qubits if q is not None
            ]
            if perform_print:
                print(
                    f"{tile_to_delete.basis}-check qubit at {tile_to_delete.check_qubit.coord} adjacency list cleaned up."
                )

            # remove the tile from the dictionary
            self.tiles_dict[coord].pop(tile_index_to_delete)
            if perform_print:
                print(f"Tile with basis {basis} at {coord} deleted.")
            if len(self.tiles_dict[coord]) == 0:
                # if there are no tiles left at this coordinate, remove the entry from the dictionary
                del self.tiles_dict[coord]
                if perform_print:
                    print(f"Tile entry at {coord} deleted. No tiles left at this coordinate.")
            if perform_print:
                print("\n")
        else:
            print(f"No tile at {coord} to delete.\n")
            return

    def add_boundary_tiles(self, top_bot_basis: BasisType = "X") -> None:
        """
        Add boundary tiles to the tile matrix. At each edge add B-1 layers of tiles, where B is the bounding box size.
        The top and bottom edges will have tiles with the given basis, while the left and right edges will have tiles with the opposite basis.
        """

        left_right_basis = "Z" if top_bot_basis == "X" else "X"
        num_boundary_layers_left_right = self.tile_bounding_box[0] - 1  # B-1 layers left right
        num_boundary_layers_top_bot = self.tile_bounding_box[1] - 1

        # for now just add top and right boundaries

        # add right boundary tiles
        for i in range(self.matrix_bounding_box[0])[-num_boundary_layers_left_right:]:
            for j in range(self.matrix_size[1]):
                # add right boundary tiles
                self.add_tile((i, j), left_right_basis)

        # add top boundary tiles
        for i in range(self.matrix_size[0]):
            for j in range(self.matrix_bounding_box[1])[-num_boundary_layers_top_bot:]:
                # add top boundary tiles
                self.add_tile((i, j), top_bot_basis)

        # add left boundary tiles
        for i in range(-num_boundary_layers_left_right, 0):
            for j in range(self.matrix_size[1]):
                # add left boundary tiles
                self.add_tile((i, j), left_right_basis)

        # add bottom boundary tiles
        for i in range(self.matrix_size[0]):
            for j in range(-num_boundary_layers_top_bot, 0):
                # add bottom boundary tiles
                self.add_tile((i, j), top_bot_basis)

    def delete_data_qubit(self, coord: GridCoordType, perform_print: bool = True) -> None:
        """
        Delete a data qubit at the given coordinate.
        This will remove the data qubit from the adjacency list of its connected check qubits.
        It will also remove the data qubit from its tile's list of data qubit coordinates, and remove the corresponding bar from the print statement.
        Self.global_data_qubit_list will be updated with None to remove the deleted data qubit.
        """

        assert (
            0 <= coord[0] < self.matrix_grid_size[0] and 0 <= coord[1] < self.matrix_grid_size[1]
        ), f"Coordinate {coord} exceeds matrix grid size {self.matrix_grid_size}."

        # find the data qubit in the global data qubit list
        data_qubit = None
        for i, qubit in enumerate(self.global_data_qubit_list):
            if qubit is None:
                continue
            if qubit.coord == coord:
                data_qubit = qubit
                self.global_data_qubit_list[i] = None  # mark it as deleted
                if perform_print:
                    print(
                        f"TileMatrix.global_data_qubit_list: Data qubit at {coord} marked with None. Remove None values before proceeding."
                    )
                break
        if data_qubit is None:
            print(f"No data qubit at {coord} to delete.\n")
            return

        for check_qubit in data_qubit.connected_qubits:
            # remove the data qubit from the check qubit's adjacency list
            check_qubit.connected_qubits.remove(data_qubit)

            # remove the data qubit coord from the check qubit's tile's data qubit coordinates
            # do this by going through the absolute data qubit coordinates of the check qubit's parent tile
            for i, absolute_data_qubit_coord in enumerate(
                check_qubit.parent_tile.absolute_data_qubit_coords
            ):
                if absolute_data_qubit_coord == data_qubit.coord:
                    check_qubit.parent_tile.data_qubit_coords.pop(i)
                    break

        self.deleted_data_qubit_coords.append(data_qubit.coord)

    def prune_data_qubits(self) -> None:
        """
        Prune all data qubits that are not fully checked.
        """

        for data_qubit in self.global_data_qubit_list:
            if not data_qubit.is_fully_checked:
                self.delete_data_qubit(data_qubit.coord, perform_print=False)

        self.global_data_qubit_list = [q for q in self.global_data_qubit_list if q is not None]

        # go through qubit grid and append coordinates of data qubits that are not in self.global_data_qubit_list
        for i in range(self.matrix_grid_size[0]):
            for j in range(self.matrix_grid_size[1]):
                coord = (i, j)
                # data qubits have odd sum of coordinates
                if (i + j) % 2 == 0:
                    continue
                if coord not in self.deleted_data_qubit_coords:
                    # check if the data qubit at this coordinate is in the global data qubit list
                    data_qubit_exists = any(
                        q.coord == coord for q in self.global_data_qubit_list if q is not None
                    )
                    if not data_qubit_exists:
                        self.deleted_data_qubit_coords.append(coord)

    def prune_tiles(self) -> None:
        """
        Prune all tile that have empty support.
        """
        for tile_key, tile_tuple in list(self.tiles_dict.items()):
            for tile in tile_tuple:
                if tile.check_qubit.has_empty_support:
                    # delete the tile
                    self.delete_tile(tile_key, tile.basis, perform_print=False)

    def print_tiles(
        self,
        tile_locators: Optional[List[Tuple[int, int, BasisType]]] = None,
        print_all: bool = False,
    ) -> None:
        """
        Print a full canvas. Add tiles on top that are provided as an argument.
        """

        width = self.matrix_bounding_box[0]
        height = self.matrix_bounding_box[1]
        lr_bl = self.tile_bounding_box[0] - 1  # B-1 layers for left and right
        tb_bl = self.tile_bounding_box[1] - 1  # B-1 layers for top and bottom

        # Each tile contributes 2 rows and 2 columns, so canvas is (2*height+1) x (2*width+1)
        rows = 2 * (height + 2 * tb_bl)  # we're adding the padding on each side
        cols = 4 * (width + 2 * lr_bl)
        canvas = [[" " for _ in range(cols)] for _ in range(rows)]

        # draw plain canvas
        for x in range(width):
            for y in range(height):
                canvas[2 * y + 2 * tb_bl][4 * x + 2 + 4 * lr_bl] = "_"
                canvas[2 * y + 1 + 2 * tb_bl][4 * x + 4 * lr_bl] = "|"

        if print_all:
            print("Printing all tiles in the matrix. Ignoring tile_locators argument.\n")
            tile_locators = []
            # add all tiles to tile_locators
            for tile_key, tile_tuple in self.tiles_dict.items():
                for tile in tile_tuple:
                    basis = tile.basis
                    tile_locators.append((*tile_key, basis))

            print(f"Number of check qubits: {len(self.global_check_qubit_list)}")
            print(f"Number of data qubits: {len(self.global_data_qubit_list)}\n")

        # print tile
        if tile_locators is not None:
            for i, j, basis in tile_locators:
                tile_tuple = self.get_tiles((i, j))

                tile = None
                for t in tile_tuple:
                    if t.basis == basis:
                        tile = t
                        break
                if tile is None:
                    raise ValueError(
                        f"Tile at ({i}, {j}) with basis {basis} does not exist in the matrix."
                    )

                other_basis = "Z" if basis == "X" else "X"

                # add data qubits
                absolute_data_qubit_edge_coords = tile.absolute_data_qubit_edge_coords
                for x, y, d in absolute_data_qubit_edge_coords:
                    assert (
                        0 <= x < self.matrix_grid_size[0]
                    ), f"x={x} out of bounds for width={self.matrix_grid_size[0]}"
                    assert (
                        0 <= y < self.matrix_grid_size[1]
                    ), f"y={y} out of bounds for height={self.matrix_grid_size[1]}"
                    assert d in (
                        "h",
                        "v",
                    ), f"Invalid direction {d}, expected 'h' or 'v'"
                    if d == "h":
                        existing_string_at_target_loc = canvas[2 * y + 2 * tb_bl][
                            4 * x + 2 + 4 * lr_bl
                        ]
                        if existing_string_at_target_loc == "_":
                            canvas[2 * y + 2 * tb_bl][4 * x + 2 + 4 * lr_bl] = basis
                        elif existing_string_at_target_loc == basis:
                            pass
                        elif existing_string_at_target_loc == other_basis:
                            canvas[2 * y + 2 * tb_bl][4 * x + 2 + 4 * lr_bl] = "B"
                        elif existing_string_at_target_loc == "B":
                            pass
                        else:
                            print(
                                f"Invalid data qubit edge coordinate ({x}, {y}, {d}) at tile ({i}, {j}) with basis {basis}. Not overwriting existing data."
                            )
                    elif d == "v":
                        existing_string_at_target_loc = canvas[2 * y + 1 + 2 * tb_bl][
                            4 * x + 4 * lr_bl
                        ]
                        if existing_string_at_target_loc == "|":
                            canvas[2 * y + 1 + 2 * tb_bl][4 * x + 4 * lr_bl] = basis
                        elif existing_string_at_target_loc == other_basis:
                            canvas[2 * y + 1 + 2 * tb_bl][4 * x + 4 * lr_bl] = "B"
                        elif existing_string_at_target_loc == basis:
                            pass
                        elif existing_string_at_target_loc == "B":
                            pass
                        else:
                            print(
                                f"Invalid data qubit edge coordinate ({x}, {y}, {d}) at tile ({i}, {j}) with basis {basis}. Not overwriting existing data."
                            )

                absolute_origin = tile.origin
                existing_string_at_target_loc = canvas[absolute_origin[1] + 2 * tb_bl][
                    2 * absolute_origin[0] + 4 * lr_bl
                ]
                if existing_string_at_target_loc == " " and not print_all:
                    canvas[absolute_origin[1] + 2 * tb_bl][2 * absolute_origin[0] + 4 * lr_bl] = "."
                elif existing_string_at_target_loc == "c":
                    pass

                # add check qubit
                x = tile.absolute_check_qubit_coord[0]
                y = tile.absolute_check_qubit_coord[1]
                # assert 0 <= x < 2*width and 0 <= y < 2*height, \
                #     f"Check qubit coordinate {tile.absolute_check_qubit_coord} out of bounds for tile grid size {(2*width, 2*height)}"
                assert (
                    x + y
                ) % 2 == 0, f"Check qubit coordinate {tile.absolute_check_qubit_coord} must have even sum of coordinates."
                try:
                    canvas[y + 2 * tb_bl][2 * x + 4 * lr_bl] = "c"  # Mark check qubit with 'c''
                except Exception as exc:
                    raise ValueError(
                        f"Error adding check qubit at ({x}, {y}) for tile ({i}, {j}) with basis {basis}.\n"
                    ) from exc

        # overwrite the deleted data qubit coordinates with ' '
        for coord in self.deleted_data_qubit_coords:
            canvas[coord[1] + 2 * tb_bl][2 * coord[0] + 4 * lr_bl] = " "

        for row in canvas[::-1]:
            print("".join(row))
        print("\n")

    def print_matrix(self) -> None:
        """
        Print all tile origins. A single tile origin is represented by its basis, either "x" or "z".
        If two different bases overlap, it is prented as "b"
        """
        width = self.matrix_bounding_box[0]
        height = self.matrix_bounding_box[1]
        lr_bl = self.tile_bounding_box[0] - 1  # B-1 layers for left and right
        tb_bl = self.tile_bounding_box[1] - 1  # B-1 layers for top and bottom

        # Each tile contributes 2 rows and 2 columns, so canvas is (2*height+1) x (2*width+1)
        rows = 2 * (height + tb_bl)
        cols = 4 * (width + lr_bl)
        canvas = [[" " for _ in range(cols)] for _ in range(rows)]

        print(f"Number of check qubits: {len(self.global_check_qubit_list)}")
        print(f"Number of data qubits: {len(self.global_data_qubit_list)}\n")

        # draw plain canvas
        for x in range(width):
            for y in range(height):
                canvas[2 * y + 2 * tb_bl][4 * x + 2 + 4 * lr_bl] = "_"
                canvas[2 * y + 1 + 2 * tb_bl][4 * x + 4 * lr_bl] = "|"

        # add tile origins
        for tile_key, tile_tuple in self.tiles_dict.items():

            for tile in tile_tuple:
                absolute_origin = tile.origin
                tile_basis = tile.basis.lower()
                other_basis = "z" if tile_basis == "x" else "x"
                absolute_origin = tile.origin

                string_at_target_loc = canvas[absolute_origin[1] + 2 * tb_bl][
                    2 * absolute_origin[0] + 4 * lr_bl
                ]

                if string_at_target_loc == " ":
                    canvas[absolute_origin[1] + 2 * tb_bl][
                        2 * absolute_origin[0] + 4 * lr_bl
                    ] = tile_basis
                elif string_at_target_loc == other_basis:
                    # if its the other basis, mark it with 'b'
                    canvas[absolute_origin[1] + 2 * tb_bl][2 * absolute_origin[0] + 4 * lr_bl] = "b"
                elif string_at_target_loc == tile_basis:
                    # if its the same basis, mark it with 'x' or 'z
                    raise ValueError(
                        f"Tile at ({tile_key[0]}, {tile_key[1]}) with basis {tile_basis} overlaps with another tile at "
                        f"({absolute_origin[0]}, {absolute_origin[1]}) with basis {string_at_target_loc}."
                    )
                elif string_at_target_loc == "b":
                    # if its already marked with 'b', it means that two bases overlap
                    # raise ValueError(
                    #     f"Location ({absolute_origin[0]}, {absolute_origin[1]}) with already marked with 'b'"
                    #     )
                    print(
                        f"Location ({absolute_origin[0]}, {absolute_origin[1]}) with already marked with 'b'"
                    )

        # overwrite the deleted data qubit coordinates with ' '
        for coord in self.deleted_data_qubit_coords:
            canvas[coord[1] + 2 * tb_bl][2 * coord[0] + 4 * lr_bl] = " "

        for row in canvas[::-1]:
            print("".join(row))
        print("\n")

    def get_parity_check_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the parity check matrices for the tile matrix.
        The first matrix is the X parity check matrix, the second is the Z parity check matrix.
        """
        x_check_matrix = []
        z_check_matrix = []

        # assign unique ids to each qubit
        for i, data_qubit in enumerate(self.global_data_qubit_list):
            data_qubit.id = i

        for i, check_qubit in enumerate(self.global_check_qubit_list):
            check_qubit.id = i

        num_x_checks = sum(1 for q in self.global_check_qubit_list if q.qubit_type == "X")
        x_index = 0
        z_index = num_x_checks
        for check_qubit in self.global_check_qubit_list:
            # create a row for the X parity check matrix
            row = [0] * len(self.global_data_qubit_list)
            for data_qubit in check_qubit.connected_qubits:
                assert (
                    data_qubit.qubit_type == "data"
                ), f"Connected qubit {data_qubit.coord} is not a data qubit, but a {data_qubit.type} qubit."
                assert data_qubit.id is not None
                row[data_qubit.id] = 1

            if check_qubit.qubit_type == "X":
                x_check_matrix.append(row)
                check_qubit.id = x_index
                x_index += 1
            elif check_qubit.qubit_type == "Z":
                z_check_matrix.append(row)
                check_qubit.id = z_index
                z_index += 1

        return np.array(x_check_matrix), np.array(z_check_matrix)

    def get_position_dict(self) -> Dict[int, GridCoordType]:
        """
        Get a dictionary that maps the qubit id to its position in the tile matrix.
        The keys are the qubit ids, and the values are the grid coordinates of the qubits.
        """
        assert all(
            q.id is not None for q in self.global_data_qubit_list
        ), "All data qubits must have an id assigned before calling get_position_dict()."
        assert all(
            q.id is not None for q in self.global_check_qubit_list
        ), "All check qubits must have an id assigned before calling get_position_dict()."

        position_dict = {}
        for i, data_qubit in enumerate(self.global_data_qubit_list):
            if data_qubit is not None:
                key = (data_qubit.id, data_qubit.is_data)
                position_dict[key] = data_qubit.coord

        for i, check_qubit in enumerate(self.global_check_qubit_list):
            key = (check_qubit.id, check_qubit.is_data)
            position_dict[key] = check_qubit.coord

        return position_dict


class TileCode(TileMatrix):
    """Tile code composed of a `TileMatrix` with derived properties and graphs.

    Attributes
    ----------
    Hx, Hz : np.ndarray
        X/Z parity-check matrices.
    Hxz : np.ndarray
        vstack of (Hx, Hz).
    tanner_graph : networkx.Graph
        Tanner graph derived from Hx/Hz.
    css_code : CSSCode_w_distance
        Underlying CSS code object used to compute distances.
    n, k, d : int
        Code length, dimension and distance (estimated).
    pos_dict_bare, pos_dict_rich : dict
        Node position maps used for layout and tanner visualization.
    """

    def __init__(
        self,
        x_tile_params: TileParamsType,
        matrix_size: Tuple[int, int],
        x_check_qubit_coord: Optional[GridCoordType] = None,
        z_check_qubit_coord: Optional[GridCoordType] = None,
        top_bot_basis: BasisType = "X",
        timeout: float = 1.0,
    ) -> None:
        super().__init__(x_tile_params, matrix_size, x_check_qubit_coord, z_check_qubit_coord)

        self.add_boundary_tiles(top_bot_basis=top_bot_basis)
        self.prune_data_qubits()
        self.prune_tiles()

        self.update_code_properties(timeout)

    def update_code_properties(self, timeout: float = 1.0) -> None:
        """
        Update the properties of the tile code, including the parity check matrices,
        """
        self.Hx, self.Hz = self.get_parity_check_matrices()
        self.Hxz = np.vstack((self.Hx, self.Hz))
        self.tanner_graph = generate_tanner_graph(self.Hxz)
        self.css_code = CSSCode_w_distance(x_stabilizer_matrix=self.Hx, z_stabilizer_matrix=self.Hz)

        self.n = self.Hxz.shape[1]
        self.k = self.n - np.linalg.matrix_rank(self.Hx) - np.linalg.matrix_rank(self.Hz)
        self.d = self.css_code.estimate_min_distance(timeout_seconds=timeout)

        self.pos_dict_bare = self.get_position_dict()
        self.pos_dict_rich = {node: self.pos_dict_bare[(node.index, node.is_data)] for node in self.tanner_graph.nodes}  # type: ignore

    @classmethod
    def init_from_generalized_toric(
        cls,
        f_ab: Tuple[int, int],
        g_ab: Tuple[int, int],
        matrix_size: Tuple[int, int],
        x_check_qubit_coord: Optional[GridCoordType] = None,
        z_check_qubit_coord: Optional[GridCoordType] = None,
        timeout: float = 1.0,
    ) -> "TileCode":
        """
        Creates a tile code using the degrees shown in https://arxiv.org/pdf/2503.03827.
        """

        f_a, f_b = f_ab[0], f_ab[1]
        g_a, g_b = g_ab[0], g_ab[1]

        f_x = -f_a if f_a < 0 else 0
        f_y = -(f_b + 1) if (f_b + 1) < 0 else 0

        g_x = -(g_a + 1) if (g_a + 1) < 0 else 0
        g_y = -g_b if g_b < 0 else 0

        h_x = max(f_x, g_x)
        h_y = max(f_y, g_y)

        x_tile_edge_coords: List[EdgeCoordType] = [
            (h_x, h_y + 1, "h"),
            (h_x + 1, h_y + 1, "h"),
            (f_a + h_x, f_b + 1 + h_y, "h"),
            (h_x + 1, h_y, "v"),
            (h_x + 1, h_y + 1, "v"),
            (g_a + 1 + h_x, g_b + h_y, "v"),
        ]

        Bx, By = -np.inf, -np.inf
        for edge in x_tile_edge_coords:
            Bx = max(Bx, edge[0])
            By = max(By, edge[1])
        bounding_box = (Bx + 1, By + 1)

        x_tile_grid_coords = Tile.transform_edge_to_grid_coords(x_tile_edge_coords)

        x_tile_params = cast(
            TileParamsType,
            {
                "bounding_box": bounding_box,
                "data_qubit_coords": x_tile_grid_coords,
                "basis": "X",
                "origin": (0, 0),
            },
        )

        return cls(
            x_tile_params,
            matrix_size,
            x_check_qubit_coord=x_check_qubit_coord,
            z_check_qubit_coord=z_check_qubit_coord,
            timeout=timeout,
        )
