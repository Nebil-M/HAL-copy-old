# radial_code.py
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Tuple

import numpy as np
import numpy.typing as npt
from qldpc.codes.common import CSSCode

IntArray = npt.NDArray[np.int64]


# --------------------------------------------------------------------------- #
# Circulant                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Circulant:
    L: int
    shift: int

    # group law ------------------------------------------------------------- #
    def __mul__(self, other: "Circulant | int") -> "Circulant | int":
        if other == 0:
            return 0
        if not isinstance(other, Circulant):
            raise TypeError("Can only multiply by another Circulant or 0")
        if other.L != self.L:
            raise ValueError("Cycle lengths differ")
        return Circulant(self.L, (self.shift + other.shift) % self.L)

    __rmul__ = __mul__

    # helpers ---------------------------------------------------------------- #
    def inverse(self) -> "Circulant":
        return Circulant(self.L, (-self.shift) % self.L)

    def to_binary(self) -> IntArray:
        eye = np.eye(self.L, dtype=np.uint8)
        return np.roll(eye, self.shift, axis=1)


# --------------------------------------------------------------------------- #
# Radial code                                                                 #
# --------------------------------------------------------------------------- #
class RadialCode(CSSCode):
    """
    Radial (QC-LDPC) CSS code constructed via the lifted product.

    Optional arguments `A_proto` / `B_proto` let you reuse or test custom
    base-matrices; if they are omitted they are generated randomly subject to
    the same constraints as in the original script.
    """

    # ------------- public constructor -------------------------------------- #
    def __init__(
        self,
        s: int,
        *,
        r: int | None = None,
        A: IntArray | Iterable[Iterable[int]] | None = None,
        B: IntArray | Iterable[Iterable[int]] | None = None,
        abdiff: bool = False,
        checkdiff: bool = False,
        field: int | None = None,
        max_attempts: int = 1_000,
        max_row_attempts: int = 1_000,
    ) -> None:
        self.s: Final = s
        self.r = r

        # ------------------------------------------------------------------ #
        # build / validate the protographs                                    #
        # ------------------------------------------------------------------ #
        if A is None or B is None:
            A_proto = self._generate_protograph(
                r,
                s,
                stringent=checkdiff,
                max_attempts=max_attempts,
                max_row_attempts=max_row_attempts,
            )
            if abdiff:
                B_proto = self._generate_protograph(
                    r,
                    s,
                    stringent=checkdiff,
                    max_attempts=max_attempts,
                    max_row_attempts=max_row_attempts,
                )
            else:
                B_proto = A_proto
        else:
            r = B.shape[0]
            B_proto = self._validate_proto(B, r, s, name="B_proto")
            r = A.shape[0]
            A_proto = self._validate_proto(A, r, s, name="A_proto")

        # store the integer protographs for later I/O ----------------------- #
        self.A_proto: Final = A_proto.copy()
        self.B_proto: Final = B_proto.copy()

        # map integers → Circulants ----------------------------------------- #
        self.A = np.vectorize(lambda x: Circulant(s, int(x)))(A_proto)
        self.B = np.vectorize(lambda x: Circulant(s, int(x)))(B_proto)

        # parity-check matrices -------------------------------------------- #
        HX, HZ = self._lifted_product(self.A, self.B, s)
        self.HX = HX
        self.HZ = HZ

        super().__init__(HX, HZ, field, promise_equal_distance_xz=False, is_subsystem_code=False)

    # ------------- representation ----------------------------------------- #
    def __str__(self) -> str:
        return f"RadialCode(r={self.r}, s={self.s})"

    # Expose hx/hz properties for linting expectations
    @property
    def hx(self) -> IntArray:
        return self.HX

    @property
    def hz(self) -> IntArray:
        return self.HZ

    # ======================================================================= #
    #                            PUBLIC UTILITIES                             #
    # ======================================================================= #
    def write_pcm_mtx(
        self,
        which: str,
        outfile: str | Path,
    ) -> None:
        """
        Dump `code.hx` or `code.hz` (`which = "X" | "Z"`) in MatrixMarket
        coordinate format, including the protographs in comments.
        """
        M = self.hx if which.upper() == "X" else self.hz
        M = M.astype(np.uint8, copy=False)
        nnz = int(M.sum())

        with open(outfile, "w", encoding="utf8") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("% Field: GF(2)\n")
            f.write(f"% RadialCode(r={self.r}, s={self.s})\n")
            f.write(f"% A = {self.A_proto.flatten().tolist()}\n")
            f.write(f"% B = {self.B_proto.flatten().tolist()}\n")
            f.write(f"{M.shape[0]} {M.shape[1]} {nnz}\n")

            rows, cols = np.nonzero(M)
            for i, j in zip(rows, cols, strict=True):
                f.write(f"{i + 1} {j + 1} 1\n")

    # ======================================================================= #
    #                      PRIVATE / STATIC IMPLEMENTATION                    #
    # ======================================================================= #
    # ---- protograph generation ------------------------------------------- #
    @staticmethod
    def _validate_proto(raw: Iterable[Iterable[int]], r: int, s: int, *, name: str) -> IntArray:
        arr = np.asarray(raw, dtype=np.int64)
        if arr.shape != (r, r):
            raise ValueError(f"{name} must have shape ({r},{r})")
        if np.any(arr < 0) or np.any(arr >= s):
            raise ValueError(f"{name} entries must be in 0,…,{s-1}")
        if np.linalg.matrix_rank(arr) != r:
            raise ValueError(f"{name} is not full rank")
        return arr

    # --------------------------------------------------------------------------- #
    #  SIMPLE 1-TO-1 PORT OF THE ORIGINAL PROTOGRAPH SEARCH                      #
    # --------------------------------------------------------------------------- #
    @classmethod
    def _generate_protograph(
        cls,
        r: int,
        s: int,
        stringent: bool,
        *,
        max_attempts: int,
        max_row_attempts: int,
    ) -> IntArray:
        """
        Direct transcription of the algorithm in the original script:

        – pick the first row at random;
        – keep sampling rows until all three checkers pass
        (difference / rank / 4–cycle);
        – back-off after 1000 row attempts or 1000 global attempts,
        exactly like the original.
        """

        # ------------------------------------------------------------------ #
        # local checker functions – identical to the originals               #
        # ------------------------------------------------------------------ #
        def difference_checker(newrow: IntArray, A: list[IntArray]) -> bool:
            if r == 2:
                return True
            for row in A:  # every existing row
                difference = (newrow - row) % s
                # “make all entries even” (same two-liner as in the script)
                difference = np.array(
                    [x if x % 2 == 0 else s - x for x in difference], dtype=np.int64
                )
                # are all non-zero entries equal?
                non_zero = difference[difference != 0]
                if non_zero.size and np.all(non_zero == non_zero[0]):
                    return False
            return True

        def rank_checker(newrow: IntArray, A: list[IntArray]) -> bool:
            if not A:
                return True
            trial = np.vstack((*A, newrow))
            return np.linalg.matrix_rank(trial) == np.linalg.matrix_rank(A) + 1

        def sum_checker_2(newrow: IntArray, A: list[IntArray]) -> bool:
            """
            4-cycle avoidance (the second of the two `sum_checker_2`
            definitions in the original file).
            """
            trial = np.vstack((*A, newrow))
            for i in range(len(trial) - 1):  # every existing row
                for j in range(r):
                    for k in range(j + 1, r):
                        if (trial[i][j] - trial[i][k] - trial[-1][j] + trial[-1][k]) % s == 0:
                            return False
            return True

        # ------------------------------------------------------------------ #
        # random search                                                      #
        # ------------------------------------------------------------------ #

        for _ in range(max_attempts):  # ≤ 1000 matrix tries
            rows: list[IntArray] = [
                np.array([random.randrange(s) for _ in range(r)], dtype=np.int64)
            ]

            for _ in range(1, r):  # need r rows total
                valid = False
                row_attempts = 0
                while not valid and row_attempts < max_row_attempts:
                    row_attempts += 1
                    candidate = np.array([random.randrange(s) for _ in range(r)], dtype=np.int64)
                    if stringent:
                        valid = (
                            difference_checker(candidate, rows)
                            and rank_checker(candidate, rows)
                            and sum_checker_2(candidate, rows)
                        )
                    else:
                        valid = rank_checker(candidate, rows) and sum_checker_2(candidate, rows)
                if not valid:  # failed for this protograph → restart
                    break
                rows.append(candidate)
            else:  # built r rows successfully
                return np.vstack(rows)

        # all attempts exhausted
        sys.exit("Failed to generate matrix in allotted attempts")

    # ---- lifted product --------------------------------------------------- #
    @staticmethod
    def _identity_protograph(r: int, s: int):
        eye = np.zeros((r, r), dtype=object)
        for i in range(r):
            eye[i, i] = Circulant(s, 0)
        return eye

    @classmethod
    def _lifted_product(cls, A: np.ndarray, B: np.ndarray, s: int) -> Tuple[IntArray, IntArray]:
        I = cls._identity_protograph(len(A), s)

        HX_left = np.kron(A, I)
        HX_right = np.kron(I, B)
        HX_circ = np.hstack((HX_left, HX_right))

        A_star = np.vectorize(lambda c: c.inverse())(A.T)
        B_star = np.vectorize(lambda c: c.inverse())(B.T)
        HZ_left = np.kron(I, B_star)
        HZ_right = np.kron(A_star, I)
        HZ_circ = np.hstack((HZ_left, HZ_right))

        HX = cls._protograph_to_binary(HX_circ, s)
        HZ = cls._protograph_to_binary(HZ_circ, s)
        return HX, HZ

    @staticmethod
    def _protograph_to_binary(proto: np.ndarray, s: int) -> IntArray:
        blocks = [
            np.hstack([np.zeros((s, s), np.uint8) if el == 0 else el.to_binary() for el in row])
            for row in proto
        ]
        return np.vstack(blocks)

    # ---- checker helpers -------------------------------------------------- #
    @staticmethod
    def _rank_increases(new: IntArray, rows: list[IntArray]) -> bool:
        if not rows:
            return True
        trial = np.vstack([*rows, new])
        return np.linalg.matrix_rank(trial) == np.linalg.matrix_rank(rows) + 1

    @staticmethod
    def _sum2_ok(new: IntArray, rows: list[IntArray], s: int) -> bool:
        if not rows:
            return True
        diff = (rows - new) % s  # (|rows|, r)
        col_pairs = (diff[:, :, None] - diff[:, None, :]) % s
        np.fill_diagonal(col_pairs[0], 1)  # just ensure diag non-zero
        return not np.any(col_pairs == 0)

    @staticmethod
    def _difference_ok(new: IntArray, rows: list[IntArray], s: int) -> bool:
        if not rows:
            return True
        d = np.abs((rows - new) % s)
        evenised = np.where(d % 2, s - d, d)
        gcds = np.gcd.reduce(evenised, axis=1)
        return not np.any(gcds == 0)
