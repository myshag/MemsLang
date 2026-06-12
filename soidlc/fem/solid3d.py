"""3-D tetrahedral (TetP2) modal solver for soidlc MEMS compiler.

Meshes a list of G.Shape polygons into a 3-D extruded solid via gmsh, then
solves the 3-D elasticity eigenvalue problem with scikit-fem.

Public API
----------
mesh_island_3d(shapes, thickness_um, h, lump_label) -> Solid3DMesh
modal3d(mesh, E, nu, rho, t, n_modes)               -> (freqs, vecs, dof_of)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .. import geometry as G

# Hard limit on tetrahedral mesh size — raise before entering the solver.
MAX_TETS = 150_000

UM = 1.0e-6   # micrometres → metres conversion factor


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Solid3DMesh:
    """3-D mesh data for one suspended island.

    nodes          (N, 3) float  – node coordinates in micrometres (x, y, z)
    tets           (4, M) int    – vertex indices of each linear tet
    fixed          (N,) bool     – True for nodes whose all dofs are clamped
    extra_mass     list[(node_index, area_um2)]  – lumped finger masses
    anchor_bboxes  list[(x0,y0,x1,y1) um]  – anchored shape bboxes kept for
                   the coordinate-predicate clamp in modal3d
    """
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    tets: np.ndarray = field(default_factory=lambda: np.empty((4, 0), dtype=np.int64))
    fixed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    extra_mass: list = field(default_factory=list)   # [(node_idx, area_um2)]
    anchor_bboxes: list = field(default_factory=list)  # [(x0,y0,x1,y1) um]

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[1]) if self.tets.ndim == 2 else 0


# ---------------------------------------------------------------------------
# Mesher
# ---------------------------------------------------------------------------

def mesh_island_3d(shapes: List[G.Shape],
                   thickness_um: float,
                   h: float = 10.0,
                   lump_label: str = "finger") -> Solid3DMesh:
    """Extrude island polygons into a 3-D tet mesh via gmsh.

    Parameters
    ----------
    shapes       : list of G.Shape (non-finger shapes are meshed; finger shapes
                   are lumped as extra_mass on the nearest node).
    thickness_um : device layer thickness in micrometres.
    h            : target mesh size in micrometres.
    lump_label   : shapes whose label contains this string are lumped.

    Returns
    -------
    Solid3DMesh with node coordinates in micrometres.
    """
    import gmsh

    solids = [s for s in shapes if lump_label not in s.label]
    lumped = [s for s in shapes if lump_label in s.label]

    # Default empty mesh
    empty = Solid3DMesh(
        nodes=np.empty((0, 3)),
        tets=np.empty((4, 0), dtype=np.int64),
        fixed=np.zeros(0, dtype=bool),
        extra_mass=[],
    )
    if not solids:
        return empty

    # interruptible=False: avoids SIGINT conflict in non-main threads (same
    # discipline as mesh_build.py)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("island3d")

        vols = []
        for s in solids:
            poly = s.polygon.normalized()
            pts = [gmsh.model.occ.addPoint(x, y, 0.0)
                   for x, y in poly.exterior]
            lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                     for i in range(len(pts))]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            out = gmsh.model.occ.extrude([(2, surf)], 0, 0, thickness_um)
            vols += [d for d in out if d[0] == 3]

        gmsh.model.occ.synchronize()

        if len(vols) > 1:
            gmsh.model.occ.fuse([vols[0]], vols[1:])
            gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)

        # Extract nodes
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        idx_of = {int(t): i for i, t in enumerate(ntags)}
        nodes_um = np.array(coords, dtype=float).reshape(-1, 3)  # (N,3) in um

        # Extract linear tets (element type 4)
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        tets = None
        for et, conn in zip(etypes, enodes):
            if et == 4:   # 4-node linear tetrahedron
                arr = np.array([idx_of[int(n)] for n in conn], dtype=np.int64)
                tets = arr.reshape(-1, 4).T   # (4, M)
                break

    finally:
        gmsh.finalize()

    if tets is None or tets.shape[1] == 0:
        return empty

    N = nodes_um.shape[0]
    fixed_mask = np.zeros(N, dtype=bool)

    # Mark nodes inside any anchored shape's bbox (exact for rectangular
    # anchors; catches all midside nodes automatically via coordinate test)
    anchored_bboxes = [
        s.polygon.bbox() for s in shapes if s.mech == "anchored"
    ]
    tol = 1e-3  # um tolerance (smaller than any real feature)
    for (x0, y0, x1, y1) in anchored_bboxes:
        in_xy = (
            (nodes_um[:, 0] >= x0 - tol) & (nodes_um[:, 0] <= x1 + tol) &
            (nodes_um[:, 1] >= y0 - tol) & (nodes_um[:, 1] <= y1 + tol)
        )
        fixed_mask |= in_xy   # all z-layers are clamped (full-column clamp)

    # Store the individual anchor bboxes so modal3d can apply them as a
    # coordinate predicate (do NOT merge them into a single hull — that would
    # clamp the entire structure for widely-spaced corner anchors).

    # Lump finger shapes: nearest node by XY centroid
    extra_mass: list = []
    if lumped and N > 0:
        for s in lumped:
            bx0, by0, bx1, by1 = s.polygon.bbox()
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            dist2 = (nodes_um[:, 0] - cx) ** 2 + (nodes_um[:, 1] - cy) ** 2
            best = int(np.argmin(dist2))
            extra_mass.append((best, s.polygon.area()))

    return Solid3DMesh(
        nodes=nodes_um,
        tets=tets,
        fixed=fixed_mask,
        extra_mass=extra_mass,
        anchor_bboxes=anchored_bboxes,
    )


# ---------------------------------------------------------------------------
# Clamped-DOF helper (3D analogue of skfem_solve._clamped_dof_array)
# ---------------------------------------------------------------------------

def _clamped_dof_array_3d(basis, mesh: Solid3DMesh) -> np.ndarray:
    """Return sorted array of clamped global DOF indices for the 3D basis.

    Clamps:
    - All three dofs (ux, uy, uz) of every fixed vertex.
    - All three dofs of every edge-midside node (P2 edge dof) whose BOTH
      endpoint vertices are fixed.

    This mirrors the 2D ``_clamped_dof_array`` in skfem_solve.py and avoids
    over-constraining midside nodes on edges that cross the anchor boundary.
    """
    vd = basis.nodal_dofs   # (3, n_vertices) — three components
    ed = basis.edge_dofs    # (3, n_edges) — three components
    edges = basis.mesh.edges   # (2, n_edges) — vertex pairs

    fixed_set = set(int(i) for i in np.where(mesh.fixed)[0])
    out: List[int] = []

    # Vertex dofs
    for n in fixed_set:
        out.append(int(vd[0, n]))
        out.append(int(vd[1, n]))
        out.append(int(vd[2, n]))

    # Edge-midside dofs: only when both endpoints are fixed
    for ei in range(edges.shape[1]):
        v0, v1 = int(edges[0, ei]), int(edges[1, ei])
        if v0 in fixed_set and v1 in fixed_set:
            out.append(int(ed[0, ei]))
            out.append(int(ed[1, ei]))
            out.append(int(ed[2, ei]))

    return np.array(sorted(set(out)), dtype=np.int64)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def modal3d(mesh: Solid3DMesh,
            E: float,
            nu: float,
            rho: float,
            t: float,
            n_modes: int = 6,
            ) -> Tuple[List[float], List[List[float]], Dict[int, int]]:
    """Compute lowest natural frequencies and M-normalised mode shapes.

    Parameters
    ----------
    mesh    : Solid3DMesh from mesh_island_3d.
    E       : Young's modulus [Pa].
    nu      : Poisson's ratio [-].
    rho     : density [kg/m³].
    t       : thickness [m]  (used for lumped mass: rho * t * area_um2 * 1e-12).
    n_modes : number of modes to extract.

    Returns
    -------
    freqs   : list of natural frequencies [Hz], length ≤ n_modes.
    vecs    : list of M-normalised mode vectors (free vertex dofs only).
              For vertex n with dof_of[n] = b:
                  ux = vec[b], uy = vec[b+1], uz = vec[b+2].
              Normalised so that full^T M1 full ≈ 1 (P1 vertex-mass metric).
    dof_of  : dict  node_index -> base offset (or -1 for fixed nodes).
    """
    from skfem import (MeshTet, Basis, ElementVector, ElementTetP2,
                       ElementTetP1, BilinearForm, condense, solve,
                       solver_eigen_scipy_sym)
    from skfem.helpers import dot
    from skfem.models.elasticity import linear_elasticity

    # Guard: degenerate mesh
    if mesh.n_tets == 0:
        return [], [], {}

    # Guard: mesh size
    if mesh.n_tets > MAX_TETS:
        raise ValueError(
            f"3D mesh too large ({mesh.n_tets} tets > {MAX_TETS}); increase h"
        )

    # Guard: all fixed
    if mesh.fixed.all():
        return [], [], {}

    # Build scikit-fem mesh (coords in metres)
    nodes_m = mesh.nodes * UM          # (N, 3) metres
    sk_mesh = MeshTet(nodes_m.T, mesh.tets)

    # P2 vector basis for stiffness + mass
    basis = Basis(sk_mesh, ElementVector(ElementTetP2()))

    # 3-D Lame parameters (full 3-D, not plane-stress)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))

    K = linear_elasticity(lam, mu).assemble(basis)

    @BilinearForm
    def mass_form(u, v, w):
        return rho * dot(u, v)

    M = mass_form.assemble(basis)

    # Note: unlike the 2D solver, we do NOT add lumped finger masses to the P2
    # eigen mass matrix M here.  In 2D, the mass matrix integrates over area
    # (units kg/m) and the formula rho*t*area is effectively zero relative to the
    # matrix entries.  In 3D, M integrates over volume (units kg), so the same
    # formula would introduce O(1)–O(100)× perturbations to individual diagonal
    # entries, severely distorting eigenfrequencies.  Finger masses ARE included
    # in the P1 normalisation mass M1 below so that M-normalised mode vectors
    # carry the correct effective-mass scaling.

    # Clamp fixed nodes and edge-midside dofs — mirror of 2D _clamped_dof_array.
    # Rule: clamp all 3 dofs of each fixed vertex, plus all 3 dofs of edge
    # midside nodes whose BOTH endpoint vertices are fixed.  This is exact and
    # avoids over-constraining midside nodes on edges between fixed and free
    # vertices (which would happen if we used a bbox coordinate predicate on the
    # solid interior of the anchor region).
    D = _clamped_dof_array_3d(basis, mesh)
    if D.size == 0:
        return [], [], {}

    n_free = K.shape[0] - len(D)
    if n_free < 3:
        return [], [], {}
    k = min(n_modes, n_free - 1)
    if k < 1:
        return [], [], {}

    ls, xs = solve(
        *condense(K, M, D=D),
        solver=solver_eigen_scipy_sym(k=k, sigma=0.0),
    )
    # xs is expanded back to full DOF size by condense/solve in scikit-fem 12
    freqs = [math.sqrt(abs(float(lv))) / (2.0 * math.pi)
             for lv in np.atleast_1d(ls)]

    # Build dof_of: vertex node_index -> base offset in the stripped vector.
    # Three components per free node (ux, uy, uz).
    vd = basis.nodal_dofs   # (3, n_vertices)
    n_verts = vd.shape[1]

    dof_of: Dict[int, int] = {}
    free_triplets: List[Tuple[int, int, int]] = []   # (global_ux, uy, uz)
    base = 0
    for n in range(n_verts):
        if mesh.fixed[n]:
            dof_of[n] = -1
        else:
            dof_of[n] = base
            free_triplets.append((int(vd[0, n]), int(vd[1, n]), int(vd[2, n])))
            base += 3

    # Extract vertex-only stripped mode vectors (drop P2 midside edge dofs)
    n_free_verts = len(free_triplets)
    raw_vecs: List[np.ndarray] = []
    for col in range(len(freqs)):
        full_p2 = xs[:, col] if xs.ndim == 2 else xs
        v = np.empty(n_free_verts * 3)
        for idx, (ix, iy, iz) in enumerate(free_triplets):
            v[3 * idx]     = float(full_p2[ix])
            v[3 * idx + 1] = float(full_p2[iy])
            v[3 * idx + 2] = float(full_p2[iz])
        raw_vecs.append(v)

    # M-normalise using P1 vertex-mass metric (mirror of skfem_solve.py Fix 2)
    p1_basis, M1 = _build_p1_mass_3d(sk_mesh, mesh, rho, t)
    vd1 = p1_basis.nodal_dofs   # (3, n_vertices) — same node ordering

    free_nodes_list = [n for n in range(n_verts) if not mesh.fixed[n]]
    n_p1_dofs = 3 * n_verts

    vecs: List[List[float]] = []
    for raw in raw_vecs:
        full_p1 = np.zeros(n_p1_dofs)
        for idx, n in enumerate(free_nodes_list):
            full_p1[int(vd1[0, n])] = raw[3 * idx]
            full_p1[int(vd1[1, n])] = raw[3 * idx + 1]
            full_p1[int(vd1[2, n])] = raw[3 * idx + 2]

        s = math.sqrt(float(full_p1 @ M1 @ full_p1))
        normalised = raw / s if s > 0.0 else raw
        vecs.append(normalised.tolist())

    return freqs, vecs, dof_of


# ---------------------------------------------------------------------------
# P1 vertex-mass builder (normalisation metric)
# ---------------------------------------------------------------------------

def _build_p1_mass_3d(sk_mesh, mesh: Solid3DMesh,
                      rho: float, t: float):
    """Build the P1 vector mass matrix for M-normalisation.

    Unlike the 2-D solver (where t is the out-of-plane thickness and mass is
    rho*t*area), the 3-D mesh already embeds the full volume so mass is simply
    rho*volume.  We do NOT multiply by t here; the volume integral is exact.

    t is still used for the lumped extra_mass correction:
        m_lump = rho * t * area_um2 * 1e-12  [kg]

    Returns (p1_vec_basis, M1).
    """
    from skfem import Basis, ElementVector, ElementTetP1, BilinearForm
    from skfem.helpers import dot

    p1_basis = Basis(sk_mesh, ElementVector(ElementTetP1()))

    @BilinearForm
    def mass_form(u, v, w):
        return rho * dot(u, v)

    M1 = mass_form.assemble(p1_basis)

    if mesh.extra_mass:
        M1 = M1.tolil()
        vd1 = p1_basis.nodal_dofs   # (3, n_vertices)
        for node_idx, area_um2 in mesh.extra_mass:
            m_lump = rho * t * (area_um2 * 1.0e-12)   # kg
            for comp in range(3):
                d = int(vd1[comp, node_idx])
                M1[d, d] += m_lump
        M1 = M1.tocsr()

    return p1_basis, M1
