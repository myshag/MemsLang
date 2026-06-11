/* Plan-view convex-corner undercut (corner compensation) for anisotropic wet
 * etching.  A top-down (x,y) level-set on the wafer surface: phi < 0 = silicon
 * still present under the mask, phi > 0 = laterally undercut (mask overhangs
 * void).  The front recedes into the masked silicon at a rate set by the
 * in-plane direction of the front normal -- a concave silicon corner is bounded
 * by the two slow {111} faces and stays sharp, while a *convex* corner sees the
 * fast plane and is beveled.  That convex-corner undercut is exactly what
 * compensation structures are designed to defeat.  One run advances the fast
 * corner by r_peak um (normalised unit time), so r_peak is the corner undercut
 * budget (~ undercut_ratio * etch depth).
 */
#include "levelset.h"

/* In-plane lateral undercut rate vs the front-normal angle ``d`` (deg) to the
 * nearest <110> axis, d in [0, 45]:
 *   - d = 0   (<110>) : a {111} sidewall forms, undercut ~ r_slow (~0);
 *   - d ~ peak (<410>): the fastest undercut planes -> r_peak (the budget);
 *   - d = 45  (<100>) : an intermediate {100}-sidewall rate, < r_peak.
 * Lumping <100> in with the corner is wrong and is exactly why a <100> beam
 * compensation works: its sidewalls undercut slower than the convex corner.
 */
static double uc_rate(double d, double r_peak, double r100, double r_slow,
                      double notchw, double dpeak, double peakw) {
    double slowf = exp(-(d / notchw) * (d / notchw));
    double pk = exp(-((d - dpeak) / peakw) * ((d - dpeak) / peakw));
    double base = r100 + (r_peak - r100) * pk;        /* intermediate -> peak */
    return r_slow + (base - r_slow) * (1.0 - slowf);  /* notch down at <110> */
}

void uc_run(double *phi, int nx, int ny, double dx, double r_peak,
            double r100, double r_slow, double notchw, double dpeak,
            double peakw, int nsteps, double dt, int reinit_stride) {
    int n = nx * ny;
    double *V = malloc(sizeof(double) * n);
    double *g = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    double inv_dx = 1.0 / dx, band = 4.5 * dx, d2r = M_PI / 180.0;
    reinit(phi, tmp, nx, ny, dx, 12);            /* clean SDF from a 0/1 init */
    for (int s = 0; s < nsteps; s++) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int k = IDX(i, j);
                V[k] = 0.0;
                if (fabs(phi[k]) > band) continue;
                int ip = i < nx-1 ? i+1 : i, im = i > 0 ? i-1 : i;
                int jp = j < ny-1 ? j+1 : j, jm = j > 0 ? j-1 : j;
                double gx = (phi[IDX(ip,j)] - phi[IDX(im,j)]) * 0.5 * inv_dx;
                double gy = (phi[IDX(i,jp)] - phi[IDX(i,jm)]) * 0.5 * inv_dx;
                double th = atan2(gy, gx) / d2r;          /* normal azimuth */
                double d = fmod(fabs(th), 90.0);
                if (d > 45.0) d = 90.0 - d;               /* dist to <110> */
                V[k] = uc_rate(d, r_peak, r100, r_slow, notchw, dpeak, peakw);
            }
        advect(phi, V, g, nx, ny, dx, dt);
        if ((s + 1) % reinit_stride == 0) reinit(phi, tmp, nx, ny, dx, 2);
    }
    reinit(phi, tmp, nx, ny, dx, 2);
    free(V); free(g); free(tmp);
}
