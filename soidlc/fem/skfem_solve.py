"""scikit-fem plane-stress solver: modal (eigsh) and static."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from skfem import (Basis, ElementVector, ElementTriP2, MeshTri,
                   BilinearForm, condense, solve, solver_eigen_scipy_sym)
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity

from .result import FemMesh

UM = 1.0e-6   # mesh coordinates are in micrometres


def _build_basis(mesh: FemMesh) -> Tuple[Basis, Basis]:
    """Build P2 vector and scalar bases from a FemMesh (coords in metres)."""
    p = np.array(mesh.nodes, dtype=float).T * UM        # (2, n_nodes)
    t = np.array(mesh.cells, dtype=np.int64).T          # (3, n_cells)
    sk_mesh = MeshTri(p, t)
    vec_basis = Basis(sk_mesh, ElementVector(ElementTriP2()))
    sca_basis = Basis(sk_mesh, ElementTriP2())
    return vec_basis, sca_basis


def _fill_p2(sca_basis: Basis, fill_per_elem: np.ndarray) -> np.ndarray:
    """Project per-element fill factors onto P2 scalar dofs by averaging.

    Each P2 dof (vertex or edge midpoint) receives the average fill of the
    elements that share that dof.  For all-ones fill this is exactly 1.0.
    """
    nd = sca_basis.nodal_dofs   # (1, n_vertices)
    fd = sca_basis.facet_dofs   # (1, n_facets)
    t = sca_basis.mesh.t        # (3, n_elems)
    t2f = sca_basis.mesh.t2f    # (3, n_elems) — facet indices per element

    fill_p2 = np.zeros(sca_basis.N)
    count_p2 = np.zeros(sca_basis.N)

    for e in range(t.shape[1]):
        fv = float(fill_per_elem[e])
        # vertex dofs
        for vi in range(3):
            dof = int(nd[0, t[vi, e]])
            fill_p2[dof] += fv
            count_p2[dof] += 1
        # edge-midpoint dofs
        for fi in range(3):
            dof = int(fd[0, t2f[fi, e]])
            fill_p2[dof] += fv
            count_p2[dof] += 1

    fill_p2 /= np.maximum(count_p2, 1.0)
    return fill_p2


def _assemble(mesh: FemMesh, E: float, nu: float, rho: float):
    """Return (vec_basis, K, M).  K and M are full (not condensed)."""
    vec_basis, sca_basis = _build_basis(mesh)

    # plane-stress Lame parameters
    mu = E / (2.0 * (1.0 + nu))
    lam_ps = E * nu / (1.0 - nu * nu)
    K = linear_elasticity(lam_ps, mu).assemble(vec_basis)

    # per-element mass fill projected to P2 scalar dofs, then interpolated
    # at vector-P2 quadrature points (6 per element) for the mass form
    fill = (np.array(mesh.fill, dtype=float) if mesh.fill
            else np.ones(len(mesh.cells)))
    fill_dofs = _fill_p2(sca_basis, fill)
    f_interp = sca_basis.interpolate(fill_dofs)

    @BilinearForm
    def mass(u, v, w):
        return rho * w["f"] * dot(u, v)

    M = mass.assemble(vec_basis, f=f_interp)
    return vec_basis, K, M


def _clamped_dof_array(vec_basis: Basis, mesh: FemMesh) -> np.ndarray:
    """Return sorted array of global DOF indices for all fixed vertices."""
    vd = vec_basis.nodal_dofs   # (2, n_vertices)
    out: List[int] = []
    for n in mesh.fixed:
        out.append(int(vd[0, n]))
        out.append(int(vd[1, n]))
    return np.array(sorted(set(out)), dtype=np.int64)


def modal(mesh: FemMesh, E: float, nu: float, rho: float, t: float,
          n_modes: int = 3, return_vectors: bool = True):
    """Compute lowest natural frequencies and mode shapes.

    Parameters
    ----------
    mesh:   FemMesh from mesh_build (nodes in micrometres).
    E:      Young's modulus [Pa].
    nu:     Poisson's ratio [-].
    rho:    density [kg/m³].
    t:      out-of-plane thickness [m] — accepted for signature parity but
            cancels for plane-stress modal frequencies (does not affect result).
    n_modes: number of eigenfrequencies to extract.

    Returns
    -------
    freqs :  list of natural frequencies [Hz], length n_modes.
    vecs  :  list of M-normalised mode vectors over FREE dofs.
             Each vec has length 2 * n_free_vertices; for vertex i with
             dof_of[i] = b, displacement = (vec[b], vec[b+1]).
    dof_of : dict mapping every node index -> base index into each vec.
             Fixed nodes map to -1.
    """
    if not mesh.cells:
        return [], [], {}

    vec_basis, K, M = _assemble(mesh, E, nu, rho)
    D = _clamped_dof_array(vec_basis, mesh)

    n_free = K.shape[0] - len(D)
    k = min(n_modes, n_free - 1)
    if k < 1:
        return [], [], {}

    ls, xs = solve(*condense(K, M, D=D),
                   solver=solver_eigen_scipy_sym(k=k, sigma=0.0))
    # xs is expanded to full DOF size by condense/solve in scikit-fem 12
    freqs = [math.sqrt(abs(float(lv))) / (2.0 * math.pi)
             for lv in np.atleast_1d(ls)]

    # Build dof_of: node index -> base offset in the free-dof vector
    vd = vec_basis.nodal_dofs   # (2, n_vertices)
    n_verts = vd.shape[1]
    dof_of: Dict[int, int] = {}
    free_pairs: List[Tuple[int, int]] = []   # (global_ux, global_uy)
    base = 0
    for n in range(n_verts):
        if n in mesh.fixed:
            dof_of[n] = -1
        else:
            dof_of[n] = base
            free_pairs.append((int(vd[0, n]), int(vd[1, n])))
            base += 2

    # Extract mode vectors: map global dofs through xs
    vecs: List[List[float]] = []
    for col in range(len(freqs)):
        full = xs[:, col] if xs.ndim == 2 else xs
        v: List[float] = []
        for (ix, iy) in free_pairs:
            v.append(float(full[ix]))
            v.append(float(full[iy]))
        vecs.append(v)

    return freqs, vecs, dof_of


def static_solve(mesh: FemMesh, E: float, nu: float, t: float,
                 forces: Dict[int, Tuple[float, float]]):
    """Linear static solve under point forces.

    Parameters
    ----------
    forces : dict node_index -> (Fx, Fy) in SI units [N/m] (per unit thickness).

    Returns
    -------
    disp : dict node_index -> (ux, uy) displacement [m] for free nodes.
    """
    if not mesh.cells:
        return {}

    mu = E / (2.0 * (1.0 + nu))
    lam_ps = E * nu / (1.0 - nu * nu)

    vec_basis, _ = _build_basis(mesh)
    K = linear_elasticity(lam_ps, mu).assemble(vec_basis)
    D = _clamped_dof_array(vec_basis, mesh)

    vd = vec_basis.nodal_dofs
    f_vec = np.zeros(K.shape[0])
    for node, (Fx, Fy) in forces.items():
        f_vec[int(vd[0, node])] += Fx
        f_vec[int(vd[1, node])] += Fy

    u = solve(*condense(K, f_vec, D=D))

    disp: Dict[int, Tuple[float, float]] = {}
    for n in range(vd.shape[1]):
        if n not in mesh.fixed:
            disp[n] = (float(u[int(vd[0, n])]), float(u[int(vd[1, n])]))
    return disp
