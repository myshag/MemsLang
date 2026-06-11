"""Finite-element layer for soidlc, built on scikit-fem + gmsh."""
from .result import FemMesh, FemResult
try:
    from .mesh_build import build_mesh, MAX_ELEMENTS
    from .skfem_solve import modal, static_solve
    from .analyze import analyze
except ImportError:
    pass

__all__ = ["FemMesh", "FemResult", "build_mesh", "modal",
           "static_solve", "analyze", "MAX_ELEMENTS"]
