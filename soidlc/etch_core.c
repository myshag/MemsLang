/* Level-set feature-scale simulation of the Bosch DRIE process (2D x-z
 * cross-section).  Pure C99 + OpenMP, driven from Python via ctypes.
 *
 * Convention: phi < 0 = solid (silicon/mask), phi > 0 = etched void, the
 * surface is the zero level set.  Etching removes material, so the interface
 * moves into the solid: d(phi)/dt = V|grad phi|, V >= 0, solved with an
 * upwind Godunov Hamiltonian.  z increases downward (depth); ions travel +z
 * (downward), so up-facing surfaces (trench bottom) receive them.
 *
 * Bosch time-multiplexing is modelled with a polymer (passivation) field:
 *   - each cycle deposits polymer on the whole surface (C4F8 step);
 *   - directional ions strip polymer from up-facing surfaces (trench bottom);
 *   - silicon etches only where the polymer is gone.
 * So sidewalls stay protected (no runaway lateral etch) while the bottom
 * drills down and bulges slightly -> one scallop per cycle.  ARDE comes from
 * a visibility factor (fraction of the upper sky a point sees); footing is a
 * lateral boost at the buried-oxide stop layer.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IDX(i, j) ((i) + nx * (j))
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }

static double grad_norm(const double *phi, int nx, int nz, int i, int j,
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

static void reinit(double *phi, double *tmp, int nx, int nz, double dx,
                   int iters) {
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

static void advect(double *phi, const double *V, double *g,
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

/* --- anisotropic wet etch (KOH / TMAH) ----------------------------------
 * Orientation-dependent etch: the rate depends on the angle of the surface
 * normal to the crystal axes.  On a (100) wafer the slow {111} planes meet
 * the surface at 54.74 deg, so the etch self-terminates into facetted pits /
 * V-grooves bounded by {111}.  Same level-set core, different velocity --
 * this is the bulk-micromachining sibling of the Bosch process.
 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static void wet_velocity(const double *phi, const unsigned char *mask,
                         double *V, int nx, int nz, double dx, double r100,
                         double r111, double r110, double ang111,
                         double notchw, double sel) {
    double inv_dx = 1.0 / dx, band = 4.5 * dx, d2r = M_PI / 180.0;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nz; j++)
        for (int i = 0; i < nx; i++) {
            int k = IDX(i, j);
            V[k] = 0.0;
            if (fabs(phi[k]) > band) continue;
            int ip = i < nx - 1 ? i + 1 : i, im = i > 0 ? i - 1 : i;
            int jp = j < nz - 1 ? j + 1 : j, jm = j > 0 ? j - 1 : j;
            double gx = (phi[IDX(ip, j)] - phi[IDX(im, j)]) * 0.5 * inv_dx;
            double gz = (phi[IDX(i, jp)] - phi[IDX(i, jm)]) * 0.5 * inv_dx;
            double gn = sqrt(gx * gx + gz * gz) + 1e-12;
            double nz_up = -gz / gn;                 /* 1 = flat {100} bottom */
            if (nz_up > 1.0) nz_up = 1.0;
            if (nz_up < -1.0) nz_up = -1.0;
            double ang = acos(nz_up) / d2r;          /* deg from {100} normal */
            double far = (ang < ang111) ? r100 : r110;
            double t = (ang - ang111) / notchw;
            double g = 1.0 - exp(-t * t);            /* notch at the {111} angle */
            double rate = r111 + (far - r111) * g;
            if (mask[k]) rate /= sel;
            V[k] = rate;
        }
}

void wet_run(double *phi, const unsigned char *mask, int nx, int nz,
             double dx, double r100, double r111, double r110, double ang111,
             double notchw, double sel, int nsteps, double dt,
             int reinit_stride) {
    int n = nx * nz;
    double *V = malloc(sizeof(double) * n);
    double *g = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    for (int s = 0; s < nsteps; s++) {
        wet_velocity(phi, mask, V, nx, nz, dx, r100, r111, r110,
                     ang111, notchw, sel);
        advect(phi, V, g, nx, nz, dx, dt);
        if ((s + 1) % reinit_stride == 0) reinit(phi, tmp, nx, nz, dx, 2);
    }
    reinit(phi, tmp, nx, nz, dx, 2);
    free(V); free(g); free(tmp);
}

/* === 3D anisotropic wet etch ============================================
 * Full 3D level-set on an (nx,ny,nz) voxel grid.  The slow {111} planes of a
 * (100) wafer have normals at 54.74 deg from [001] at the four <110>
 * azimuths; the rate notches down when the surface normal aligns with any of
 * them, so a square mask opening self-organises into the classic inverted
 * pyramid bounded by four {111} facets.
 */
#define ID3(i, j, k) ((i) + nx * ((j) + ny * (k)))

static double gnorm3(const double *phi, int nx, int ny, int nz,
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

static void reinit3(double *phi, double *tmp, int nx, int ny, int nz,
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

void wet_run_3d(double *phi, const unsigned char *mask, int nx, int ny,
                int nz, double dx, double r100, double r111, double notchw,
                double sel, int nsteps, double dt, int reinit_stride) {
    long n = (long)nx * ny * nz;
    double *V = malloc(sizeof(double) * n);
    double *g = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    double d2r = M_PI / 180.0, band = 4.5 * dx;
    /* four {111} normals: polar 54.74 deg, azimuth 45 + 90k */
    double s111 = sin(54.74 * d2r), c111 = cos(54.74 * d2r), ax[12];
    for (int kk = 0; kk < 4; kk++) {
        double az = (45.0 + 90.0 * kk) * d2r;
        ax[3*kk] = s111 * cos(az);
        ax[3*kk+1] = s111 * sin(az);
        ax[3*kk+2] = c111;
    }
    for (int s = 0; s < nsteps; s++) {
        double inv_dx = 1.0 / dx;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    long id = ID3(i, j, k);
                    V[id] = 0.0;
                    if (fabs(phi[id]) > band) continue;
                    int ip=i<nx-1?i+1:i, im=i>0?i-1:i;
                    int jp=j<ny-1?j+1:j, jm=j>0?j-1:j;
                    int kp=k<nz-1?k+1:k, km=k>0?k-1:k;
                    double gx=(phi[ID3(ip,j,k)]-phi[ID3(im,j,k)])*0.5*inv_dx;
                    double gy=(phi[ID3(i,jp,k)]-phi[ID3(i,jm,k)])*0.5*inv_dx;
                    double gz=(phi[ID3(i,j,kp)]-phi[ID3(i,j,km)])*0.5*inv_dx;
                    double gn=sqrt(gx*gx+gy*gy+gz*gz)+1e-12;
                    double ux=gx/gn, uy=gy/gn, uz=gz/gn;
                    double maxdot=0.0;
                    for (int kk=0; kk<4; kk++) {
                        double d=fabs(ux*ax[3*kk]+uy*ax[3*kk+1]+uz*ax[3*kk+2]);
                        if (d>maxdot) maxdot=d;
                    }
                    if (maxdot>1.0) maxdot=1.0;
                    double ang=acos(maxdot)/d2r;          /* deg to {111} */
                    double gg=1.0-exp(-(ang/notchw)*(ang/notchw));
                    double rate=r111+(r100-r111)*gg;      /* slow on {111} */
                    if (mask[id]) rate/=sel;
                    V[id]=rate;
                }
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    long id=ID3(i,j,k);
                    g[id]=(V[id]>0.0)?gnorm3(phi,nx,ny,nz,i,j,k,inv_dx):0.0;
                }
        #pragma omp parallel for schedule(static)
        for (long id = 0; id < n; id++) phi[id] += dt * V[id] * g[id];
        if ((s + 1) % reinit_stride == 0) reinit3(phi, tmp, nx, ny, nz, dx, 1);
    }
    reinit3(phi, tmp, nx, ny, nz, dx, 2);
    free(V); free(g); free(tmp);
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
