/* Bosch DRIE (deep reactive-ion etch) of the structural layer, 2D x-z
 * cross-section.  z increases downward (depth); ions travel +z, so up-facing
 * surfaces (the trench bottom) receive them.
 *
 * Time-multiplexing is modelled with a polymer (passivation) field:
 *   - each cycle deposits polymer on the whole surface (C4F8 step);
 *   - directional ions strip polymer from up-facing surfaces (trench bottom);
 *   - silicon etches only where the polymer is gone.
 * So sidewalls stay protected (no runaway lateral etch) while the bottom
 * drills down and bulges slightly -> one scallop per cycle.  ARDE comes from a
 * visibility factor (fraction of the upper sky a point sees); footing is a
 * lateral boost at the buried-oxide stop layer.
 */
#include "levelset.h"

/* fraction of the upper "sky" cone that a surface point can see (ARDE) */
static double visibility(const double *phi, int nx, int nz, double dx,
                         int i, int j, int nang, double maxlen) {
    int seen = 0;
    double span = 1.39626, step = dx * 0.7;
    for (int a = 0; a < nang; a++) {
        double th = -span + 2.0 * span * a / (nang - 1);
        double ux = sin(th), uz = -cos(th);
        double x = (i + 0.5) * dx, z = (j + 0.5) * dx, t = step * 1.5;
        int blocked = 0;
        while (t < maxlen) {
            double px = x + ux * t, pz = z + uz * t;
            int ci = (int)(px / dx), cj = (int)(pz / dx);
            if (ci < 0 || ci >= nx || cj < 0) break;
            if (cj >= nz) { blocked = 1; break; }
            if (phi[IDX(ci, cj)] < -0.25 * dx) { blocked = 1; break; }
            t += step;
        }
        if (!blocked) seen++;
    }
    return (double)seen / nang;
}

/* deposit polymer on the surface band (C4F8 passivation step) */
static void deposit(const double *phi, double *poly, int nx, int nz,
                    double dx, double p0) {
    double band = 2.0 * dx;
    int n = nx * nz;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++)
        if (fabs(phi[k]) <= band) poly[k] = p0;
}

/* velocity for one sub-phase, also strips polymer (ions, mode 0)
 * mode 0 = anisotropic ion etch, mode 1 = isotropic radical etch */
static void velocity(const double *phi, const unsigned char *mask,
                     const double *vis, double *poly, double *V,
                     int nx, int nz, double dx, int j_box, double R_ion,
                     double R_iso, double sel, double footing, double strip,
                     double dt, int mode, double passiv, double bow,
                     int surf_j) {
    double inv_dx = 1.0 / dx, band = 4.5 * dx, eps = 0.02;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nz; j++)
        for (int i = 0; i < nx; i++) {
            int k = IDX(i, j);
            V[k] = 0.0;
            if (fabs(phi[k]) > band || j >= j_box) continue;
            int ip = i < nx - 1 ? i + 1 : i, im = i > 0 ? i - 1 : i;
            int jp = j < nz - 1 ? j + 1 : j, jm = j > 0 ? j - 1 : j;
            double gx = (phi[IDX(ip, j)] - phi[IDX(im, j)]) * 0.5 * inv_dx;
            double gz = (phi[IDX(i, jp)] - phi[IDX(i, jm)]) * 0.5 * inv_dx;
            double gn = sqrt(gx * gx + gz * gz) + 1e-12;
            double aniso = dmax(0.0, -gz / gn);     /* up-facing => bottom */
            double v = vis[k];
            if (mode == 0) {
                /* ions strip polymer from up-facing surfaces */
                poly[k] -= strip * aniso * (0.3 + 0.7 * v) * dt;
                if (poly[k] < 0.0) poly[k] = 0.0;
                if (poly[k] > eps) continue;        /* still protected */
                double rate = R_ion * aniso * (0.3 + 0.7 * v);
                /* ion-scattering lateral widening at the drilling front,
                 * growing with absolute depth -> re-entrant / negative taper
                 * (applied once per depth level as the front passes) */
                if (bow > 0.0) {
                    double df = (double)(j - surf_j) /
                                (double)(j_box - surf_j + 1);
                    if (df < 0.0) df = 0.0; else if (df > 1.0) df = 1.0;
                    rate += bow * df * R_iso * dmax(0.0, fabs(gx) / gn);
                }
                if (mask[k]) rate /= sel;
                V[k] = rate;
            } else {
                double rate;
                if (poly[k] <= eps) {
                    /* freshly de-passivated bottom: full isotropic etch */
                    rate = R_iso * v;
                    if (j >= j_box - 3 && j < j_box) /* footing at oxide */
                        rate += footing * R_iso * v * dmax(0.0, fabs(gx) / gn);
                } else {
                    /* protected sidewall: residual lateral etch from
                     * imperfect passivation.  The top is exposed for the most
                     * cycles, so it widens most -> positive taper (narrowing
                     * downward); lower passivation = stronger positive taper */
                    double vert = dmax(0.0, fabs(gx) / gn - dmax(0.0, -gz / gn));
                    rate = (1.0 - passiv) * R_iso * v * vert;
                }
                if (mask[k]) rate /= sel;
                V[k] = rate;
            }
        }
}

static void vis_field(const double *phi, double *vis, int nx, int nz,
                      double dx, int nang, double maxlen) {
    double band = 5.0 * dx;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nz; j++)
        for (int i = 0; i < nx; i++) {
            int k = IDX(i, j);
            vis[k] = (fabs(phi[k]) <= band)
                     ? visibility(phi, nx, nz, dx, i, j, nang, maxlen) : 0.0;
        }
}

void bosch_run(double *phi, const unsigned char *mask, int nx, int nz,
               double dx, int j_box, double R_ion, double R_iso, double sel,
               double footing, int ncycles, int n_aniso, int n_iso,
               double dt, int reinit_iters, int vis_angles, double vis_maxlen,
               double passiv, double bow, int surf_j)
{
    int n = nx * nz;
    double *V = malloc(sizeof(double) * n);
    double *vis = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    double *g = malloc(sizeof(double) * n);
    double *poly = calloc(n, sizeof(double));
    double strip = 3.0 / dt;                     /* polymer cleared fast */
    /* reinit only every STRIDE substeps: frequent reinit drifts the
     * interface backward (kills etch rate) and smooths out scalloping */
    int stride = reinit_iters > 0 ? reinit_iters : 6;
    int sc = 0;
    for (int c = 0; c < ncycles; c++) {
        vis_field(phi, vis, nx, nz, dx, vis_angles, vis_maxlen);
        deposit(phi, poly, nx, nz, dx, 1.0);     /* passivation step */
        for (int s = 0; s < n_aniso; s++) {
            velocity(phi, mask, vis, poly, V, nx, nz, dx, j_box,
                     R_ion, R_iso, sel, footing, strip, dt, 0, passiv, bow, surf_j);
            advect(phi, V, g, nx, nz, dx, dt);
            if (++sc % stride == 0) reinit(phi, tmp, nx, nz, dx, 2);
        }
        for (int s = 0; s < n_iso; s++) {
            velocity(phi, mask, vis, poly, V, nx, nz, dx, j_box,
                     R_ion, R_iso, sel, footing, strip, dt, 1, passiv, bow, surf_j);
            advect(phi, V, g, nx, nz, dx, dt);
            if (++sc % stride == 0) reinit(phi, tmp, nx, nz, dx, 2);
        }
    }
    reinit(phi, tmp, nx, nz, dx, 2);             /* final clean-up */
    free(V); free(vis); free(tmp); free(g); free(poly);
}
