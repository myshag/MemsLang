"""Backend-neutral FEM data types shared by mesher, solver and consumers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class FemMesh:
    nodes: List[Tuple[float, float]] = field(default_factory=list)  # um
    cells: List[Tuple[int, ...]] = field(default_factory=list)      # tri verts
    fixed: Set[int] = field(default_factory=set)                    # node idx
    fill: List[float] = field(default_factory=list)                 # per cell
    extra_mass: List[Tuple[int, float]] = field(default_factory=list)  # um^2

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_free_dof(self) -> int:
        return 2 * (len(self.nodes) - len(self.fixed))


@dataclass
class FemResult:
    freqs: List[float] = field(default_factory=list)          # Hz
    vecs: List[List[float]] = field(default_factory=list)     # M-normalised
    dof_of: Dict[int, int] = field(default_factory=dict)      # node -> base
