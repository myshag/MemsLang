"""scikit-fem plane-stress solver: modal (eigsh) and static."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from skfem import (Basis, ElementVector, ElementTriP1, ElementTriP2, MeshTri,
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


def _fill_p1(sca_p1_basis: Basis, fill_per_elem: np.ndarray) -> np.ndarray:
    """Project per-element fill factors onto P1 scalar dofs by averaging."""
    nd = sca_p1_basis.nodal_dofs   # (1, n_vertices)
    t = sca_p1_basis.mesh.t        # (3, n_elems)

    fill_p1 = np.zeros(sca_p1_basis.N)
    count_p1 = np.zeros(sca_p1_basis.N)

    for e in range(t.shape[1]):
        fv = float(fill_per_elem[e])
        for vi in range(3):
            dof = int(nd[0, t[vi, e]])
            fill_p1[dof] += fv
            count_p1[dof] += 1

    fill_p1 /= np.maximum(count_p1, 1.0)
    return fill_p1


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
    """Return sorted array of global DOF indices for all fixed BCs.

    Clamps:
    - Both dofs (ux, uy) of every vertex in mesh.fixed.
    - All dofs (nodal + edge-midpoint) on every mesh edge (facet) whose
      BOTH endpoint vertices are in mesh.fixed.  This ensures P2 midside
      nodes on fixed edges are also fully constrained.
    """
    fixed_nodes = mesh.fixed
    vd = vec_basis.nodal_dofs   # (2, n_vertices)
    out: List[int] = []

    # Nodal dofs for all fixed vertices
    for n in fixed_nodes:
        out.append(int(vd[0, n]))
        out.append(int(vd[1, n]))

    # Facet (edge) dofs for edges whose both endpoints are fixed
    facets = vec_basis.mesh.facets   # (2, n_facets): each col = [v0, v1]
    both_fixed = np.array([
        facets[0, fi] in fixed_nodes and facets[1, fi] in fixed_nodes
        for fi in range(facets.shape[1])
    ])
    fixed_facet_indices = np.where(both_fixed)[0]
    if fixed_facet_indices.size > 0:
        edge_dofs = vec_basis.get_dofs(facets=fixed_facet_indices).flatten()
        out.extend(int(d) for d in edge_dofs)

    return np.array(sorted(set(out)), dtype=np.int64)


def _build_p1_mass(mesh: FemMesh, rho: float, t: float) -> Tuple[Basis, object]:
    """Build the P1 vector mass matrix for vertex-space M-normalisation.

    The mass matrix includes the out-of-plane thickness ``t`` so that the
    resulting M-normalised mode vectors carry physical units (m / sqrt(kg)).
    This makes the effective-mass formula ``m_eff = 1 / u_drive^2`` in the
    ROM consistent with the physical device mass.

    Returns (p1_vec_basis, M1) where M1 includes lumped extra_mass.
    """
    p = np.array(mesh.nodes, dtype=float).T * UM
    tri = np.array(mesh.cells, dtype=np.int64).T
    sk_mesh = MeshTri(p, tri)

    p1_vec_basis = Basis(sk_mesh, ElementVector(ElementTriP1()))
    p1_sca_basis = Basis(sk_mesh, ElementTriP1())

    fill = (np.array(mesh.fill, dtype=float) if mesh.fill
            else np.ones(len(mesh.cells)))
    fill_dofs = _fill_p1(p1_sca_basis, fill)
    f_interp = p1_sca_basis.interpolate(fill_dofs)

    @BilinearForm
    def mass_form(u, v, w):
        # rho [kg/m^3] * t [m] * fill * N·N gives units kg after 2-D integration
        return rho * t * w["f"] * dot(u, v)

    M1 = mass_form.assemble(p1_vec_basis, f=f_interp)

    # Add lumped extra_mass to diagonal
    if mesh.extra_mass:
        M1 = M1.tolil()
        vd1 = p1_vec_basis.nodal_dofs   # (2, n_vertices)
        for node, area_um2 in mesh.extra_mass:
            m_lump = rho * t * (area_um2 * 1e-12)   # [kg]
            ix = int(vd1[0, node])
            iy = int(vd1[1, node])
            M1[ix, ix] += m_lump
            M1[iy, iy] += m_lump
        M1 = M1.tocsr()

    return p1_vec_basis, M1


def modal(mesh: FemMesh, E: float, nu: float, rho: float, t: float,
          n_modes: int = 3):
    """Compute lowest natural frequencies and mode shapes.

    Parameters
    ----------
    mesh:   FemMesh from mesh_build (nodes in micrometres).
    E:      Young's modulus [Pa].
    nu:     Poisson's ratio [-].
    rho:    density [kg/m³].
    t:      out-of-plane thickness [m] — used ONLY for the kg-metric P1
            normalisation matrix M1 (rho*t*area gives kg there).  The
            eigen mass matrix M is per-unit-thickness (kg/m); its lumped
            finger masses are rho*area_um2*1e-12 [kg/m] — no t factor.
    n_modes: number of eigenfrequencies to extract.

    Returns
    -------
    freqs :  list of natural frequencies [Hz], length n_modes.
    vecs  :  list of M-normalised mode vectors over FREE VERTEX dofs.
             Each vec has length 2 * n_free_vertices; for vertex i with
             dof_of[i] = b, displacement = (vec[b], vec[b+1]).
             Normalised so that full^T M1 full ≈ 1.0 (P1 vertex mass metric).
    dof_of : dict mapping every node index -> base index into each vec.
             Fixed nodes map to -1.
    """
    if not mesh.cells:
        return [], [], {}

    vec_basis, K, M = _assemble(mesh, E, nu, rho)

    # Add lumped extra_mass to the P2 eigen mass matrix.
    # The 2-D plane-stress mass matrix integrates rho * fill * u·v over area
    # (units: kg/m — mass per unit out-of-plane thickness).  The matching lump
    # is therefore rho * area_um2 * 1e-12 [kg/m]; the thickness t cancels
    # exactly as it does everywhere else in the plane-stress eigenproblem.
    # Do NOT include t here — that would underweight the finger mass by the
    # factor t (~25e-6) and make comb fingers numerically invisible.
    # (t IS used in _build_p1_mass for the kg-metric normalisation matrix M1.)
    if mesh.extra_mass:
        M = M.tolil()
        vd = vec_basis.nodal_dofs   # (2, n_vertices)
        for node, area_um2 in mesh.extra_mass:
            m_per_t = rho * (area_um2 * 1e-12)   # [kg/m] — per-unit-thickness
            ix = int(vd[0, node])
            iy = int(vd[1, node])
            M[ix, ix] += m_per_t
            M[iy, iy] += m_per_t
        M = M.tocsr()

    # Fix 1: clamp P2 vertex dofs AND edge-midpoint dofs on fixed edges
    D = _clamped_dof_array(vec_basis, mesh)

    # Fix 1: guard against n_free < 2 or k < 1
    n_free = K.shape[0] - len(D)
    if n_free < 2:
        return [], [], {}
    k = min(n_modes, n_free - 1)
    if k < 1:
        return [], [], {}

    ls, xs = solve(*condense(K, M, D=D),
                   solver=solver_eigen_scipy_sym(k=k, sigma=0.0))
    # xs is expanded to full DOF size by condense/solve in scikit-fem 12
    freqs = [math.sqrt(abs(float(lv))) / (2.0 * math.pi)
             for lv in np.atleast_1d(ls)]

    # Build dof_of: node index -> base offset in the free-vertex-dof vector
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

    # Extract raw vertex-only mode vectors (stripped of P2 midside dofs)
    raw_vecs: List[np.ndarray] = []
    for col in range(len(freqs)):
        full_p2 = xs[:, col] if xs.ndim == 2 else xs
        v = np.empty(len(free_pairs) * 2)
        for idx, (ix, iy) in enumerate(free_pairs):
            v[2 * idx]     = float(full_p2[ix])
            v[2 * idx + 1] = float(full_p2[iy])
        raw_vecs.append(v)

    # Fix 2: M-normalise in P1 vertex-mass metric
    p1_vec_basis, M1 = _build_p1_mass(mesh, rho, t)
    vd1 = p1_vec_basis.nodal_dofs   # (2, n_vertices) — same node ordering
    n_p1_dofs = 2 * n_verts

    vecs: List[List[float]] = []
    for raw in raw_vecs:
        # Lift stripped vector back to full P1 displacement (fixed dofs = 0)
        full_p1 = np.zeros(n_p1_dofs)
        for idx, (n, (ix, iy)) in enumerate(
                [(n, free_pairs[j]) for j, n in enumerate(
                    [ni for ni in range(n_verts) if ni not in mesh.fixed]
                )]):
            p1_ix = int(vd1[0, n])
            p1_iy = int(vd1[1, n])
            full_p1[p1_ix] = raw[2 * idx]
            full_p1[p1_iy] = raw[2 * idx + 1]

        s = math.sqrt(float(full_p1 @ M1 @ full_p1))
        if s > 0.0:
            normalised = raw / s
        else:
            normalised = raw
        vecs.append(normalised.tolist())

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
