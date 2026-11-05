/* Bosch DRIE of the structural layer driven by the plasma's absolute fluxes,
 * 2D x-z cross-section (z down, ions travel +z).  The flux-driven counterpart
 * of bosch.c: instead of effective knobs, each surface point gets the fluxes
 * that reach it and the surface kinetics turns them into rates.
 *
 * Transport (ballistic, the feature is far below the mean free path):
 *   - neutrals (F in the etch step, the film-forming CFx in the passivation
 *     step) come from an isotropic gas above the feature, `G0` per m^2 and s
 *     onto an open flat surface, and are re-emitted diffusely where they do
 *     not stick: G_s = G0 F_s,sky + sum_t F_st (1 - S_t) G_t, with the view
 *     factors F (cosine law; in the cross-section of a long trench 1/2 cos phi
 *     dphi) from rays cast from every surface point, solved with BiCGSTAB;
 *   - ions come from above with a Gaussian spread of angles `sigma` (no
 *     reflection): flux per unit area sum_a w_a vis_a max(0, -d_a.n) / cos a.
 *
 * Surface kinetics:
 *   - silicon: the adsorbed-F model of Ferma's wafer (Belen et al., J. Vac. Sci.
 *     Technol. A 23, 99 (2005)): F sticks with sF on the free part, the covered
 *     fraction theta is removed thermally (kth F per m^2 and s at full coverage,
 *     as SiF4) and by the ions (Y = A (sqrt(E) - sqrt(Eth)) Si per ion at full
 *     coverage, as SiF2): sF G (1 - theta) = theta (kth + 2 Y Gi), etch rate
 *     theta (kth / 4 + Y Gi) / nSi.  On an open flat wafer this is Ferma's
 *     etch rate;
 *   - the fluorocarbon film (Kokkoris, Tserepi, Boudouvis & Gogolides, J. Vac.
 *     Sci. Technol. A 22, 1896 (2004), as used by Ertl & Selberherr, Microelectron.
 *     Eng. 87, 20 (2010)): deposited by the passivation step, betaDep per CFx and
 *     alphaDep per ion that arrive; removed in the etch step, alphaPoly per ion
 *     and betaPoly per F; silicon and the mask etch only where it is gone.  Its
 *     thickness is sub-cell, carried by the solid cell under each surface
 *     point: a wall that does not move keeps it, freshly etched silicon has
 *     none;
 *   - the mask: alphaMask per ion, betaMask per F.
 *   The ion yields of the film and the mask scale as sqrt(E) - sqrt(EthIon),
 *   equal to the published ones at Eref.  Silicon's yield takes its own energy
 *   (ESi: the yield averaged over the ions' energies, which the plasma's etch
 *   rate implies).
 *
 * Numerics: the surface points are the void cells next to the solid; their
 * fluxes are averaged with their neighbours' along the surface (the rays'
 * sampling noise would roughen the front), recomputed when the front has moved
 * half a cell or the film has cleared from a twentieth of the surface.  The
 * velocity is that of the nearest surface point, with its material, over ten
 * cells each side of the front, and the step keeps every surface point within
 * the CFL limit; a film that clears within a step lets its surface etch for
 * the rest of it.
 *
 * Parameters (p) in the order of the enum below; out: the open flat wafer's
 * etch rate (m/s) and coverage, the film after the first passivation step on
 * the silicon that sees most of the plasma (m), when the first etch step clears
 * the first film (s, -1: never), flux evaluations, time steps.
 */
#include "levelset.h"

enum {
    P_CYCLES, P_TETCH, P_TPASS, P_GF, P_GIE, P_EE, P_GCF, P_GIP,
    P_SF, P_KTH, P_ASI, P_ETHSI, P_APOLY, P_BPOLY, P_AMASK, P_BMASK,
    P_EREF, P_ETHION, P_BDEP, P_ADEP, P_SDEP, P_SPOLY, P_SMASK, P_SIGMA,
    P_NRAYS, P_NIONS, P_NSI, P_CFL, P_REINIT, P_ESI, P_N
};
enum { O_ETCH0, O_THETA0, O_FILM0, O_CLEAR0, O_FLUXCALLS, O_STEPS, O_N };

#define SKY (-1)
#define LOST (-2)

typedef struct {
    int n, cap;
    int *cell, *solid;          /* the void interface cell, its solid neighbour */
    double *x, *z, *nx, *nz;    /* closest point and normal, grid units */
    unsigned char *mat;         /* 0 silicon, 1 mask, 2 stop layer */
} Surf;

static void surf_free(Surf *s) {
    free(s->cell); free(s->solid); free(s->x); free(s->z); free(s->nx);
    free(s->nz); free(s->mat);
}

/* the unit normal into the void (Sobel gradient: smoother than the central
 * difference on a kinked front) */
static void normal_at(const double *phi, int nx, int nz, int i, int j,
                      double *ux, double *uz) {
    int ip = i < nx - 1 ? i + 1 : i, im = i > 0 ? i - 1 : i;
    int jp = j < nz - 1 ? j + 1 : j, jm = j > 0 ? j - 1 : j;
    double gx = 2.0 * (phi[IDX(ip, j)] - phi[IDX(im, j)])
              + phi[IDX(ip, jm)] - phi[IDX(im, jm)]
              + phi[IDX(ip, jp)] - phi[IDX(im, jp)];
    double gz = 2.0 * (phi[IDX(i, jp)] - phi[IDX(i, jm)])
              + phi[IDX(im, jp)] - phi[IDX(im, jm)]
              + phi[IDX(ip, jp)] - phi[IDX(ip, jm)];
    double g = sqrt(gx * gx + gz * gz);
    if (g < 1e-12) { *ux = 0.0; *uz = -1.0; return; }
    *ux = gx / g; *uz = gz / g;
}

/* the interface: void cells with a solid 4-neighbour; cell2sp maps them */
static void find_surface(const double *phi, const unsigned char *mask, int nx,
                         int nz, double dx, int j_box, Surf *s, int *cell2sp) {
    int n = nx * nz;
    for (int k = 0; k < n; k++) cell2sp[k] = -1;
    s->n = 0;
    for (int j = 0; j < nz; j++)
        for (int i = 0; i < nx; i++) {
            int k = IDX(i, j);
            if (phi[k] <= 0.0) continue;
            int solid = (i > 0 && phi[IDX(i - 1, j)] <= 0.0) ||
                        (i < nx - 1 && phi[IDX(i + 1, j)] <= 0.0) ||
                        (j > 0 && phi[IDX(i, j - 1)] <= 0.0) ||
                        (j < nz - 1 && phi[IDX(i, j + 1)] <= 0.0);
            if (!solid) continue;
            if (s->n == s->cap) {
                s->cap = s->cap ? 2 * s->cap : 1024;
                s->cell = realloc(s->cell, sizeof(int) * s->cap);
                s->solid = realloc(s->solid, sizeof(int) * s->cap);
                s->x = realloc(s->x, sizeof(double) * s->cap);
                s->z = realloc(s->z, sizeof(double) * s->cap);
                s->nx = realloc(s->nx, sizeof(double) * s->cap);
                s->nz = realloc(s->nz, sizeof(double) * s->cap);
                s->mat = realloc(s->mat, s->cap);
            }
            int m = s->n++;
            double ux, uz, d = phi[k] / dx;
            normal_at(phi, nx, nz, i, j, &ux, &uz);
            s->cell[m] = k;
            s->nx[m] = ux; s->nz[m] = uz;
            s->x[m] = i + 0.5 - d * ux;
            s->z[m] = j + 0.5 - d * uz;
            /* the solid cell under the closest point, else the nearest solid
             * neighbour */
            int si = (int)floor(s->x[m] - 0.5 * ux), sj = (int)floor(s->z[m] - 0.5 * uz);
            int best = -1;
            double bd = 1e30;
            for (int b = -1; b <= 1; b++)
                for (int a = -1; a <= 1; a++) {
                    int ci = si + a, cj = sj + b;
                    if (ci < 0 || ci >= nx || cj < 0 || cj >= nz) continue;
                    if (phi[IDX(ci, cj)] > 0.0) continue;
                    double ex = ci + 0.5 - s->x[m], ez = cj + 0.5 - s->z[m];
                    double dd = ex * ex + ez * ez;
                    if (dd < bd) { bd = dd; best = IDX(ci, cj); }
                }
            if (best < 0) {          /* the 4-neighbour that made it interface */
                if (i > 0 && phi[IDX(i - 1, j)] <= 0.0) best = IDX(i - 1, j);
                else if (i < nx - 1 && phi[IDX(i + 1, j)] <= 0.0) best = IDX(i + 1, j);
                else if (j < nz - 1 && phi[IDX(i, j + 1)] <= 0.0) best = IDX(i, j + 1);
                else best = IDX(i, j - 1);
            }
            s->solid[m] = best;
            int bj = best / nx;
            s->mat[m] = mask[best] ? 1 : (bj >= j_box ? 2 : 0);
            cell2sp[k] = m;
        }
}

/* the surface point nearest to (x, z) (grid units) within `r` cells */
static int nearest_sp(const Surf *s, const int *cell2sp, int nx, int nz,
                      double x, double z, int r) {
    int ci = (int)floor(x), cj = (int)floor(z), best = -1;
    double bd = 1e30;
    for (int b = -r; b <= r; b++)
        for (int a = -r; a <= r; a++) {
            int i = ci + a, j = cj + b;
            if (i < 0 || i >= nx || j < 0 || j >= nz) continue;
            int m = cell2sp[IDX(i, j)];
            if (m < 0) continue;
            double ex = s->x[m] - x, ez = s->z[m] - z, dd = ex * ex + ez * ez;
            if (dd < bd) { bd = dd; best = m; }
        }
    return best;
}

/* march a ray from (x, z) along (ux, uz) (grid units): SKY when it leaves
 * through the top, else the surface point it hits (LOST if none is found);
 * the side walls of the domain mirror it */
static int march(const double *phi, const Surf *s, const int *cell2sp, int nx,
                 int nz, double x, double z, double ux, double uz) {
    const double step = 0.5;
    int nmax = 8 * (nx + nz);
    double px = x, pz = z;
    for (int it = 0; it < nmax; it++) {
        double qx = px + ux * step, qz = pz + uz * step;
        if (qz < 0.0) return SKY;
        if (qx < 0.0) { qx = -qx; ux = -ux; }
        if (qx >= nx) { qx = 2.0 * nx - qx - 1e-9; ux = -ux; }
        if (qz >= nz) return LOST;
        int ci = (int)qx, cj = (int)qz;
        if (phi[IDX(ci, cj)] <= 0.0) {
            int m = nearest_sp(s, cell2sp, nx, nz, px, pz, 2);
            return m >= 0 ? m : LOST;
        }
        px = qx; pz = qz;
    }
    return LOST;
}

typedef struct {
    int nr;            /* neutral rays per point */
    int *hit;          /* n * nr targets */
    double *sky;       /* fraction of the cosine-weighted hemisphere that is sky */
    double *ion;       /* ion flux per unit area over the source flux */
} Views;

static void views_compute(const double *phi, const Surf *s, const int *cell2sp,
                          int nx, int nz, int nr, int ni, double sigma,
                          Views *v) {
    int n = s->n;
    v->nr = nr;
    v->hit = realloc(v->hit, sizeof(int) * (size_t)n * nr);
    v->sky = realloc(v->sky, sizeof(double) * n);
    v->ion = realloc(v->ion, sizeof(double) * n);
    /* ion directions: Gaussian in angle, ni points over +-3 sigma */
    double ang[64], w[64], wsum = 0.0;
    if (ni > 64) ni = 64;
    if (ni < 1) ni = 1;
    for (int a = 0; a < ni; a++) {
        double t = ni == 1 ? 0.0 : -3.0 + 6.0 * a / (ni - 1);
        ang[a] = t * sigma;
        w[a] = exp(-0.5 * t * t);
        wsum += w[a];
    }
    for (int a = 0; a < ni; a++) w[a] /= wsum;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int m = 0; m < n; m++) {
        double ux = s->nx[m], uz = s->nz[m];
        double x0 = s->x[m] + 0.75 * ux, z0 = s->z[m] + 0.75 * uz;
        int nsky = 0;
        for (int r = 0; r < nr; r++) {
            /* stratified in sin(phi): the cosine law in the cross-section */
            double sp = -1.0 + (2.0 * r + 1.0) / nr, cp = sqrt(1.0 - sp * sp);
            double dx_ = cp * ux - sp * uz, dz_ = sp * ux + cp * uz;
            int t = march(phi, s, cell2sp, nx, nz, x0, z0, dx_, dz_);
            if (t == m) t = LOST;              /* grazing self-hit */
            v->hit[(size_t)m * nr + r] = t;
            if (t == SKY) nsky++;
        }
        v->sky[m] = (double)nsky / nr;
        double f = 0.0;
        for (int a = 0; a < ni; a++) {
            double dxi = sin(ang[a]), dzi = cos(ang[a]);   /* travelling down */
            double c = -(dxi * ux + dzi * uz);
            if (c <= 0.0) continue;
            if (march(phi, s, cell2sp, nx, nz, x0, z0, -dxi, -dzi) == SKY)
                f += w[a] * c / dzi;
        }
        v->ion[m] = f;
    }
}

static void views_free(Views *v) { free(v->hit); free(v->sky); free(v->ion); }

/* y = x - K x, K_st = (1 - S_t) / nr per ray from s that hits t */
static void kmul(const Views *v, int n, const double *keep, const double *x,
                 double *y) {
    int nr = v->nr;
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < n; m++) {
        double acc = 0.0;
        const int *h = v->hit + (size_t)m * nr;
        for (int r = 0; r < nr; r++)
            if (h[r] >= 0) acc += keep[h[r]] * x[h[r]];
        y[m] = x[m] - acc / nr;
    }
}

static double dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* the incident neutral flux G (per G0) from (I - K) G = sky, BiCGSTAB; G holds
 * the first guess when `warm` */
static void radiosity(const Views *v, int n, const double *keep, double *G,
                      int warm) {
    double *r = malloc(sizeof(double) * n), *r0 = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n), *q = malloc(sizeof(double) * n);
    double *t = malloc(sizeof(double) * n), *sv = malloc(sizeof(double) * n);
    if (!warm)
        for (int i = 0; i < n; i++) G[i] = v->sky[i];
    kmul(v, n, keep, G, t);
    for (int i = 0; i < n; i++) { r[i] = v->sky[i] - t[i]; r0[i] = r[i]; p[i] = 0; q[i] = 0; }
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    double bn = sqrt(dot(v->sky, v->sky, n)) + 1e-300;
    for (int it = 0; it < 500; it++) {
        if (sqrt(dot(r, r, n)) < 1e-10 * bn) break;
        double rho1 = dot(r0, r, n);
        if (fabs(rho1) < 1e-300) break;
        double beta = (rho1 / rho) * (alpha / omega);
        for (int i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * q[i]);
        kmul(v, n, keep, p, q);
        double den = dot(r0, q, n);
        if (fabs(den) < 1e-300) break;
        alpha = rho1 / den;
        for (int i = 0; i < n; i++) sv[i] = r[i] - alpha * q[i];
        kmul(v, n, keep, sv, t);
        double tt = dot(t, t, n);
        omega = tt > 0 ? dot(t, sv, n) / tt : 0.0;
        for (int i = 0; i < n; i++) {
            G[i] += alpha * p[i] + omega * sv[i];
            r[i] = sv[i] - omega * t[i];
        }
        rho = rho1;
        if (omega == 0.0) break;
    }
    for (int i = 0; i < n; i++) if (G[i] < 0.0) G[i] = 0.0;
    free(r); free(r0); free(p); free(q); free(t); free(sv);
}

/* a per-surface-point quantity averaged with its neighbours along the surface
 * (the 3 x 3 cells around it, the point itself twice): the rays' sampling
 * noise would otherwise roughen the front */
static void smooth_sp(const Surf *s, const int *cell2sp, int nx, int nz,
                      double *a, double *tmp) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < s->n; m++) {
        int i = s->cell[m] % nx, j = s->cell[m] / nx;
        double acc = 2.0 * a[m], w = 2.0;
        for (int b = -1; b <= 1; b++)
            for (int c = -1; c <= 1; c++) {
                int ci = i + c, cj = j + b;
                if ((b == 0 && c == 0) || ci < 0 || ci >= nx || cj < 0 || cj >= nz) continue;
                int q = cell2sp[IDX(ci, cj)];
                if (q >= 0) { acc += a[q]; w += 1.0; }
            }
        tmp[m] = acc / w;
    }
    memcpy(a, tmp, sizeof(double) * s->n);
}

static double ion_yield(double E, double A, double Eth) {
    return E > Eth ? A * (sqrt(E) - sqrt(Eth)) : 0.0;
}

typedef struct {
    double *G, *Gi, *theta, *keep, *film;   /* per surface point */
} State;

static void state_size(State *st, int n) {
    st->G = realloc(st->G, sizeof(double) * n);
    st->Gi = realloc(st->Gi, sizeof(double) * n);
    st->theta = realloc(st->theta, sizeof(double) * n);
    st->keep = realloc(st->keep, sizeof(double) * n);
    st->film = realloc(st->film, sizeof(double) * n);
}

/* the etch step's fluxes and silicon coverage on the current surface */
static void etch_fluxes(const double *p, const Surf *s, const Views *v,
                        const double *poly, const int *cell2sp, int nx, int nz,
                        State *st) {
    int n = s->n;
    double Ysi = ion_yield(p[P_ESI], p[P_ASI], p[P_ETHSI]);
    for (int m = 0; m < n; m++) {
        st->Gi[m] = p[P_GIE] * v->ion[m];
        st->theta[m] = 1.0;
        st->film[m] = poly[s->solid[m]];
    }
    for (int outer = 0; outer < 40; outer++) {
        for (int m = 0; m < n; m++) {
            double S;
            if (st->film[m] > 0.0) S = p[P_SPOLY];
            else if (s->mat[m] != 0) S = p[P_SMASK];
            else S = p[P_SF] * (1.0 - st->theta[m]);
            st->keep[m] = 1.0 - S;
        }
        radiosity(v, n, st->keep, st->G, outer > 0);
        double change = 0.0;
        for (int m = 0; m < n; m++) {
            double G = p[P_GF] * st->G[m];
            double sG = p[P_SF] * G;
            double loss = p[P_KTH] + 2.0 * Ysi * st->Gi[m];
            double th = sG > 0 ? sG / (sG + loss) : 0.0;
            change = dmax(change, fabs(th - st->theta[m]));
            st->theta[m] = th;
        }
        if (change < 1e-5) break;
    }
    for (int m = 0; m < n; m++) st->G[m] *= p[P_GF];
    smooth_sp(s, cell2sp, nx, nz, st->G, st->keep);
    smooth_sp(s, cell2sp, nx, nz, st->Gi, st->keep);
    for (int m = 0; m < n; m++) {
        double sG = p[P_SF] * st->G[m];
        double loss = p[P_KTH] + 2.0 * Ysi * st->Gi[m];
        st->theta[m] = sG > 0 ? sG / (sG + loss) : 0.0;
    }
}

/* the velocity (m/s) of a bare surface point of material `mat` */
static double bare_rate(const double *p, const State *st, int m, int mat,
                        double Ysi, double Ymask) {
    if (mat == 1) return Ymask * st->Gi[m] + p[P_BMASK] * st->G[m];
    if (mat == 2) return 0.0;
    return st->theta[m] * (0.25 * p[P_KTH] + Ysi * st->Gi[m]) / p[P_NSI];
}

static double flat_etch(const double *p, double *theta) {
    /* the open flat wafer: G = GF, Gi = GIE */
    double Ysi = ion_yield(p[P_ESI], p[P_ASI], p[P_ETHSI]);
    double sG = p[P_SF] * p[P_GF], loss = p[P_KTH] + 2.0 * Ysi * p[P_GIE];
    double th = sG / (sG + loss);
    if (theta) *theta = th;
    return th * (0.25 * p[P_KTH] + Ysi * p[P_GIE]) / p[P_NSI];
}

void drie_run(double *phi, const unsigned char *mask, int nx, int nz,
              double dx, int j_box, const double *p, double *out) {
    int n = nx * nz;
    int cycles = (int)p[P_CYCLES], nr = (int)p[P_NRAYS], ni = (int)p[P_NIONS];
    int stride = (int)p[P_REINIT] > 0 ? (int)p[P_REINIT] : 2;
    double sigma = p[P_SIGMA] * M_PI / 180.0, dxm = dx * 1e-6;
    double kpoly = p[P_APOLY] / dmax(1e-12, sqrt(p[P_EREF]) - sqrt(p[P_ETHION]));
    double kmask = p[P_AMASK] / dmax(1e-12, sqrt(p[P_EREF]) - sqrt(p[P_ETHION]));
    double Ypoly = ion_yield(p[P_EE], kpoly, p[P_ETHION]);
    double Ymask = ion_yield(p[P_EE], kmask, p[P_ETHION]);
    double Ysi = ion_yield(p[P_ESI], p[P_ASI], p[P_ETHSI]);
    double *poly = calloc(n, sizeof(double));      /* film thickness, m */
    double *V = calloc(n, sizeof(double)), *g = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    int *cell2sp = malloc(sizeof(int) * n), *band = malloc(sizeof(int) * n);
    Surf s = {0};
    Views v = {0};
    State st = {0};
    for (int o = 0; o < O_N; o++) out[o] = 0.0;
    out[O_ETCH0] = flat_etch(p, &out[O_THETA0]);
    out[O_CLEAR0] = -1.0;
    int sc = 0;

    for (int c = 0; c < cycles; c++) {
        /* ---- passivation: the film grows where the CFx and ions arrive ---- */
        find_surface(phi, mask, nx, nz, dx, j_box, &s, cell2sp);
        views_compute(phi, &s, cell2sp, nx, nz, nr, ni, sigma, &v);
        state_size(&st, s.n);
        for (int m = 0; m < s.n; m++) st.keep[m] = 1.0 - p[P_SDEP];
        radiosity(&v, s.n, st.keep, st.G, 0);
        out[O_FLUXCALLS] += 1;
        for (int m = 0; m < s.n; m++) st.Gi[m] = v.ion[m];
        smooth_sp(&s, cell2sp, nx, nz, st.G, st.keep);
        smooth_sp(&s, cell2sp, nx, nz, st.Gi, st.keep);
        /* the film lies on the solid cell under each surface point (where
         * two points share one, the thicker deposit) */
        for (int m = 0; m < s.n; m++) tmp[s.solid[m]] = poly[s.solid[m]];
        for (int m = 0; m < s.n; m++) {
            double rate = p[P_BDEP] * p[P_GCF] * st.G[m]
                          + p[P_ADEP] * p[P_GIP] * st.Gi[m];
            int k = s.solid[m];
            poly[k] = dmax(poly[k], tmp[k] + rate * p[P_TPASS]);
        }
        if (c == 0) {
            /* the film after the first step on the silicon that sees the
             * most of the plasma */
            double best = -1.0;
            for (int m = 0; m < s.n; m++)
                if (s.mat[m] == 0 && v.sky[m] > best) {
                    best = v.sky[m];
                    out[O_FILM0] = poly[s.solid[m]];
                }
        }

        /* ---- etch: ions clear the film, F and ions etch what is bare ---- */
        double t = 0.0, moved = 1e30;
        int bare_last = -1;
        while (t < p[P_TETCH] - 1e-12) {
            /* the fluxes again when the front has moved half a cell or the
             * film has cleared from a twentieth of the surface since */
            int bare = 0;
            for (int m = 0; m < s.n && bare_last >= 0; m++)
                bare += poly[s.solid[m]] <= 0.0;
            if (moved > 0.5 * dx || bare_last < 0
                || abs(bare - bare_last) > s.n / 20) {
                find_surface(phi, mask, nx, nz, dx, j_box, &s, cell2sp);
                views_compute(phi, &s, cell2sp, nx, nz, nr, ni, sigma, &v);
                state_size(&st, s.n);
                etch_fluxes(p, &s, &v, poly, cell2sp, nx, nz, &st);
                out[O_FLUXCALLS] += 1;
                moved = 0.0;
                #pragma omp parallel for schedule(static)
                for (int k = 0; k < n; k++) {
                    /* the velocity is extended over ten cells each side:
                     * a narrower band leaves stale level-set values at its
                     * edge that hold the front back (a flat front moved at
                     * three quarters of its speed with three cells) */
                    band[k] = -1;
                    if (fabs(phi[k]) > 10.0 * dx) continue;
                    int i = k % nx, j = k / nx;
                    double ux, uz;
                    normal_at(phi, nx, nz, i, j, &ux, &uz);
                    double d = phi[k] / dx;
                    double px = i + 0.5 - d * ux, pz = j + 0.5 - d * uz;
                    band[k] = nearest_sp(&s, cell2sp, nx, nz, px, pz, 3);
                    if (band[k] < 0)
                        band[k] = nearest_sp(&s, cell2sp, nx, nz, px, pz, 8);
                }
                bare = 0;
                for (int m = 0; m < s.n; m++) bare += poly[s.solid[m]] <= 0.0;
                bare_last = bare;
            }
            /* the time step keeps every surface point, bare or about to be,
             * within the CFL limit; a film that clears inside the step lets
             * its surface etch for the rest of it */
            double vpot = 0.0, tclear = 1e30;
            double *frac = st.film;          /* reused: per surface point */
            for (int m = 0; m < s.n; m++) {
                vpot = dmax(vpot, bare_rate(p, &st, m, s.mat[m], Ysi, Ymask));
                double h = poly[s.solid[m]];
                double strip = Ypoly * st.Gi[m] + p[P_BPOLY] * st.G[m];
                if (h > 0.0 && strip > 0.0) tclear = dmin(tclear, h / strip);
            }
            double dt = p[P_TETCH] - t;
            if (vpot > 0.0) dt = dmin(dt, p[P_CFL] * dxm / vpot);
            for (int m = 0; m < s.n; m++) {
                double h = poly[s.solid[m]];
                double strip = Ypoly * st.Gi[m] + p[P_BPOLY] * st.G[m];
                frac[m] = h <= 0.0 ? 1.0
                        : strip > 0.0 ? dmax(0.0, 1.0 - h / (strip * dt)) : 0.0;
            }
            if (c == 0 && out[O_CLEAR0] < 0.0 && tclear <= dt)
                out[O_CLEAR0] = t + tclear;
            double vapplied = 0.0;
            #pragma omp parallel for schedule(static) reduction(max:vapplied)
            for (int k = 0; k < n; k++) {
                V[k] = 0.0;
                int m = band[k];
                if (m < 0) continue;
                /* extended from the surface point, with its material: a
                 * cell's own material would give the void above the mask
                 * silicon's speed */
                V[k] = frac[m] * bare_rate(p, &st, m, s.mat[m], Ysi, Ymask) * 1e6;  /* um/s */
                if (V[k] > vapplied) vapplied = V[k];
            }
            /* the ions and F wear the film down (where two surface points
             * share its cell, the faster) */
            for (int m = 0; m < s.n; m++) tmp[s.solid[m]] = 0.0;
            for (int m = 0; m < s.n; m++)
                tmp[s.solid[m]] = dmax(tmp[s.solid[m]],
                                       Ypoly * st.Gi[m] + p[P_BPOLY] * st.G[m]);
            for (int m = 0; m < s.n; m++) {
                int k = s.solid[m];
                if (poly[k] > 0.0) poly[k] = dmax(0.0, poly[k] - tmp[k] * dt);
            }
            advect(phi, V, g, nx, nz, dx, dt);
            moved += vapplied * dt;
            if (++sc % stride == 0) reinit(phi, tmp, nx, nz, dx, 2);
            t += dt;
            out[O_STEPS] += 1;
        }
    }
    reinit(phi, tmp, nx, nz, dx, 2);
    free(poly); free(V); free(g); free(tmp); free(cell2sp); free(band);
    surf_free(&s); views_free(&v);
    free(st.G); free(st.Gi); free(st.theta); free(st.keep); free(st.film);
}
