/* Shared level-set primitives for the feature-scale etch simulators
 * (see levelset.h for the convention).  The per-process velocity fields and
 * entry points live in bosch.c (DRIE), wet.c (KOH/TMAH) and corner.c
 * (plan-view convex-corner undercut); all of them are compiled together into
 * a single etch_core.so.
 */
#include "levelset.h"

/* upwind (Godunov) norm of the gradient for an outward-moving (etching) front */
double grad_norm(const double *phi, int nx, int nz, int i, int j,
                 double inv_dx) {
    int ip = i < nx - 1 ? i + 1 : i, im = i > 0 ? i - 1 : i;
    int jp = j < nz - 1 ? j + 1 : j, jm = j > 0 ? j - 1 : j;
    double dxm = (phi[IDX(i, j)] - phi[IDX(im, j)]) * inv_dx;
    double dxp = (phi[IDX(ip, j)] - phi[IDX(i, j)]) * inv_dx;
    double dzm = (phi[IDX(i, j)] - phi[IDX(i, jm)]) * inv_dx;
    double dzp = (phi[IDX(i, jp)] - phi[IDX(i, j)]) * inv_dx;
    double a = dmin(dxm, 0.0), b = dmax(dxp, 0.0);
    double c = dmin(dzm, 0.0), d = dmax(dzp, 0.0);
    return sqrt(a * a + b * b + c * c + d * d);
}

/* re-distance phi to a signed-distance field (Sussman iteration) */
void reinit(double *phi, double *tmp, int nx, int nz, double dx, int iters) {
    double inv_dx = 1.0 / dx, dt = 0.4 * dx;
    for (int it = 0; it < iters; it++) {
        memcpy(tmp, phi, sizeof(double) * nx * nz);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nz; j++)
            for (int i = 0; i < nx; i++) {
                double p = tmp[IDX(i, j)];
                double s = p / sqrt(p * p + dx * dx);
                int ip = i < nx - 1 ? i + 1 : i, im = i > 0 ? i - 1 : i;
                int jp = j < nz - 1 ? j + 1 : j, jm = j > 0 ? j - 1 : j;
                double dxm = (p - tmp[IDX(im, j)]) * inv_dx;
                double dxp = (tmp[IDX(ip, j)] - p) * inv_dx;
                double dzm = (p - tmp[IDX(i, jm)]) * inv_dx;
                double dzp = (tmp[IDX(i, jp)] - p) * inv_dx;
                double g;
                if (s > 0) {
                    double a = dmax(dxm, 0.0), b = dmin(dxp, 0.0);
                    double c = dmax(dzm, 0.0), d = dmin(dzp, 0.0);
                    g = sqrt(dmax(a * a, b * b) + dmax(c * c, d * d));
                } else {
                    double a = dmin(dxm, 0.0), b = dmax(dxp, 0.0);
                    double c = dmin(dzm, 0.0), d = dmax(dzp, 0.0);
                    g = sqrt(dmax(a * a, b * b) + dmax(c * c, d * d));
                }
                phi[IDX(i, j)] = p - dt * s * (g - 1.0);
            }
    }
}

/* one explicit Euler advection sub-step: phi += dt * V * |grad phi| */
void advect(double *phi, const double *V, double *g,
            int nx, int nz, double dx, double dt) {
    double inv_dx = 1.0 / dx;
    int n = nx * nz;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nz; j++)
        for (int i = 0; i < nx; i++) {
            int k = IDX(i, j);
            g[k] = (V[k] > 0.0) ? grad_norm(phi, nx, nz, i, j, inv_dx) : 0.0;
        }
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) phi[k] += dt * V[k] * g[k];
}

/* upwind gradient norm, 3D (nx, ny, nz) */
double gnorm3(const double *phi, int nx, int ny, int nz,
              int i, int j, int k, double inv_dx) {
    int ip = i < nx - 1 ? i + 1 : i, im = i > 0 ? i - 1 : i;
    int jp = j < ny - 1 ? j + 1 : j, jm = j > 0 ? j - 1 : j;
    int kp = k < nz - 1 ? k + 1 : k, km = k > 0 ? k - 1 : k;
    double dxm = (phi[ID3(i, j, k)] - phi[ID3(im, j, k)]) * inv_dx;
    double dxp = (phi[ID3(ip, j, k)] - phi[ID3(i, j, k)]) * inv_dx;
    double dym = (phi[ID3(i, j, k)] - phi[ID3(i, jm, k)]) * inv_dx;
    double dyp = (phi[ID3(i, jp, k)] - phi[ID3(i, j, k)]) * inv_dx;
    double dzm = (phi[ID3(i, j, k)] - phi[ID3(i, j, km)]) * inv_dx;
    double dzp = (phi[ID3(i, j, kp)] - phi[ID3(i, j, k)]) * inv_dx;
    double a = dmin(dxm, 0.0), b = dmax(dxp, 0.0);
    double c = dmin(dym, 0.0), d = dmax(dyp, 0.0);
    double e = dmin(dzm, 0.0), f = dmax(dzp, 0.0);
    return sqrt(a * a + b * b + c * c + d * d + e * e + f * f);
}

/* re-distance phi to a signed-distance field, 3D */
void reinit3(double *phi, double *tmp, int nx, int ny, int nz,
             double dx, int iters) {
    double inv_dx = 1.0 / dx, dt = 0.3 * dx;
    long n = (long)nx * ny * nz;
    for (int it = 0; it < iters; it++) {
        memcpy(tmp, phi, sizeof(double) * n);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    double p = tmp[ID3(i, j, k)];
                    double s = p / sqrt(p * p + dx * dx);
                    int ip = i < nx-1 ? i+1 : i, im = i > 0 ? i-1 : i;
                    int jp = j < ny-1 ? j+1 : j, jm = j > 0 ? j-1 : j;
                    int kp = k < nz-1 ? k+1 : k, km = k > 0 ? k-1 : k;
                    double dxm=(p-tmp[ID3(im,j,k)])*inv_dx;
                    double dxp=(tmp[ID3(ip,j,k)]-p)*inv_dx;
                    double dym=(p-tmp[ID3(i,jm,k)])*inv_dx;
                    double dyp=(tmp[ID3(i,jp,k)]-p)*inv_dx;
                    double dzm=(p-tmp[ID3(i,j,km)])*inv_dx;
                    double dzp=(tmp[ID3(i,j,kp)]-p)*inv_dx;
                    double g;
                    if (s > 0) {
                        double a=dmax(dxm,0),b=dmin(dxp,0),c=dmax(dym,0),
                               d=dmin(dyp,0),e=dmax(dzm,0),f=dmin(dzp,0);
                        g=sqrt(dmax(a*a,b*b)+dmax(c*c,d*d)+dmax(e*e,f*f));
                    } else {
                        double a=dmin(dxm,0),b=dmax(dxp,0),c=dmin(dym,0),
                               d=dmax(dyp,0),e=dmin(dzm,0),f=dmax(dzp,0);
                        g=sqrt(dmax(a*a,b*b)+dmax(c*c,d*d)+dmax(e*e,f*f));
                    }
                    phi[ID3(i,j,k)] = p - dt * s * (g - 1.0);
                }
    }
}
