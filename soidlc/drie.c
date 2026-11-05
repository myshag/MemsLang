/* Bosch DRIE of the structural layer driven by the plasma's absolute fluxes,
 * 2D x-z cross-section (z down, ions travel +z).  The flux-driven counterpart
 * of bosch.c: instead of effective knobs, each surface point gets the fluxes
 * that reach it and the surface kinetics turns them into rates.
 *
 * Transport (ballistic, the feature is far below the mean free path):
 *   - neutrals (F in the etch step, each of the film's radicals in the
 *     passivation step) come from an isotropic gas above the feature, `G0` per m^2 and s
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
 *     Eng. 87, 20 (2010)): deposited by the passivation step, each radical
 *     joining it with its own probability (its volume each; soidlc/film.py)
 *     and alphaDep per ion that arrives; removed in the etch step, alphaPoly per ion
 *     and betaPoly per F, and in the passivation step by its ions, with the
 *     film's yield at their energy (EP: a bias on that step; 0, or below the
 *     threshold: none) -- where they remove more than the radicals bring, the
 *     film thins, down to the substrate; silicon and the mask etch only where
 *     it is gone.  It
 *     is a layer with its thickness, as in ViennaPS: a second level set (the
 *     top: the substrate and the film) over the substrate's.  The passivation
 *     step moves the top into the void; the etch step moves it through the
 *     film at the film's rate and, where the film is gone, through the
 *     substrate at its own, and the substrate the top has passed is etched.
 *     The film on a surface point is the top's distance above the substrate
 *     (both are distance functions), so a film thinner than a cell is kept;
 *   - the mask: alphaMask per ion, betaMask per F.
 *   The ion yields of the film and the mask scale as sqrt(E) - sqrt(EthIon),
 *   equal to the published ones at Eref.  Silicon's yield takes its own energy
 *   (ESi: the yield averaged over the ions' energies, which the plasma's etch
 *   rate implies).
 *
 * Numerics: the surface points are the top's void cells next to its solid; their
 * fluxes are averaged with their neighbours' along the surface (the rays'
 * sampling noise would roughen the front), recomputed when the front has moved
 * half a cell or the film has cleared from a twentieth of the surface.  The
 * velocity is that of the nearest surface point, with its material, over the
 * whole grid (label_all), and the step keeps every surface point within
 * the CFL limit; a film that clears within a step lets its surface etch for
 * the rest of it.
 *
 * Parameters (p) in the order of the enum below, then P_NPREC triples of each
 * film radical's flux onto the open flat wafer, its sticking and the film it
 * adds when it sticks (m^3); out: the open flat wafer's
 * etch rate (m/s) and coverage, the film after the first passivation step on
 * the silicon that sees most of the plasma (m), when the first etch step clears
 * the first film (s, -1: never), flux evaluations, time steps.
 */
#include "levelset.h"

enum {
    P_CYCLES, P_TETCH, P_TPASS, P_GF, P_GIE, P_EE, P_GIP,
    P_SF, P_KTH, P_ASI, P_ETHSI, P_APOLY, P_BPOLY, P_AMASK, P_BMASK,
    P_EREF, P_ETHION, P_ADEP, P_SPOLY, P_SMASK, P_SIGMA,
    P_NRAYS, P_NIONS, P_NSI, P_CFL, P_REINIT, P_ESI, P_LSIA, P_LSIB, P_EP,
    P_NPREC, P_N
};
enum { O_ETCH0, O_THETA0, O_FILM0, O_CLEAR0, O_FLUXCALLS, O_STEPS, O_N };

#define SKY (-1)
#define LOST (-2)

typedef struct {
    int n, cap;
    int *cell;                  /* the void interface cell of the top */
    double *x, *z, *nx, *nz;    /* closest point and normal, grid units */
    double *h;                  /* the film on it, um (0: bare) */
    unsigned char *mat;         /* the substrate under it: 0 silicon, 1 mask,
                                   2 stop layer */
} Surf;

static void surf_free(Surf *s) {
    free(s->cell); free(s->x); free(s->z); free(s->nx); free(s->nz);
    free(s->h); free(s->mat);
}

/* each surface point's film: the top's distance above the substrate at its
 * void interface cell (both are distance functions), none below a fiftieth of
 * a cell (where the two level sets coincide up to their redistancing) */
static void film_at(Surf *s, const double *top, const double *sub, double dx) {
    for (int m = 0; m < s->n; m++) {
        int k = s->cell[m];
        double h = sub[k] - top[k];
        s->h[m] = h > 0.02 * dx ? h : 0.0;
    }
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

/* the top's interface: void cells with a solid 4-neighbour (cell2sp maps
 * them), the film on each and the substrate (sub) under it */
static void find_surface(const double *phi, const double *sub,
                         const unsigned char *mask, int nx, int nz, double dx,
                         int j_box, Surf *s, int *cell2sp) {
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
                s->h = realloc(s->h, sizeof(double) * s->cap);
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
            /* the substrate cell nearest to the closest point (under a film
             * thicker than four cells: silicon, until it thins) */
            int si = (int)floor(s->x[m] - 0.5 * ux), sj = (int)floor(s->z[m] - 0.5 * uz);
            int best = -1;
            double bd = 1e30;
            for (int r = 1; r <= 4 && best < 0; r *= 2)
                for (int b = -r; b <= r; b++)
                    for (int a = -r; a <= r; a++) {
                        int ci = si + a, cj = sj + b;
                        if (ci < 0 || ci >= nx || cj < 0 || cj >= nz) continue;
                        if (sub[IDX(ci, cj)] > 0.0) continue;
                        double ex = ci + 0.5 - s->x[m], ez = cj + 0.5 - s->z[m];
                        double dd = ex * ex + ez * ez;
                        if (dd < bd) { bd = dd; best = IDX(ci, cj); }
                    }
            s->mat[m] = best < 0 ? 0 : mask[best] ? 1 : (best / nx >= j_box ? 2 : 0);
            double h = sub[k] - phi[k];
            s->h[m] = h > 0.02 * dx ? h : 0.0;
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

/* every cell's surface point for the velocity: the interface cells' own, the
 * others' through a breadth-first sweep from them (the nearest along the grid).
 * The whole grid gets a velocity: a band leaves stale values at its edge, and
 * on the void side the upwind gradient carries their deficit down to the front,
 * a cell per step (a flat front lost a quarter of its speed on a 10 nm grid) */
static void label_all(const int *cell2sp, int nx, int nz, int *label, int *queue) {
    int n = nx * nz, head = 0, tail = 0;
    for (int k = 0; k < n; k++) {
        label[k] = cell2sp[k];
        if (label[k] >= 0) queue[tail++] = k;
    }
    while (head < tail) {
        int k = queue[head++], i = k % nx, j = k / nx;
        int nb[4] = { i > 0 ? k - 1 : -1, i < nx - 1 ? k + 1 : -1,
                      j > 0 ? k - nx : -1, j < nz - 1 ? k + nx : -1 };
        for (int q = 0; q < 4; q++)
            if (nb[q] >= 0 && label[nb[q]] < 0) {
                label[nb[q]] = label[k];
                queue[tail++] = nb[q];
            }
    }
}

static double ion_yield(double E, double A, double Eth) {
    return E > Eth ? A * (sqrt(E) - sqrt(Eth)) : 0.0;
}

typedef struct {
    double *G, *Gi, *theta, *keep, *v;   /* per surface point */
} State;

static void state_size(State *st, int n) {
    st->G = realloc(st->G, sizeof(double) * n);
    st->Gi = realloc(st->Gi, sizeof(double) * n);
    st->theta = realloc(st->theta, sizeof(double) * n);
    st->keep = realloc(st->keep, sizeof(double) * n);
    st->v = realloc(st->v, sizeof(double) * n);
}

/* the etch step's fluxes and silicon coverage on the current surface */
static void etch_fluxes(const double *p, const Surf *s, const Views *v,
                        const int *cell2sp, int nx, int nz, State *st) {
    int n = s->n;
    double Ysi = ion_yield(p[P_ESI], p[P_ASI], p[P_ETHSI]);
    for (int m = 0; m < n; m++) {
        st->Gi[m] = p[P_GIE] * v->ion[m];
        st->theta[m] = 1.0;
    }
    /* the linear law (P_LSIA, P_LSIB > 0): F sticks on silicon with P_SF,
       no coverage */
    int linear = p[P_LSIA] > 0.0 || p[P_LSIB] > 0.0;
    for (int outer = 0; outer < 40; outer++) {
        for (int m = 0; m < n; m++) {
            double S;
            if (s->h[m] > 0.0) S = p[P_SPOLY];
            else if (s->mat[m] != 0) S = p[P_SMASK];
            else S = p[P_SF] * (linear ? 1.0 : 1.0 - st->theta[m]);
            st->keep[m] = 1.0 - S;
        }
        radiosity(v, n, st->keep, st->G, outer > 0);
        double change = 0.0;
        for (int m = 0; m < n; m++) {
            double G = p[P_GF] * st->G[m];
            double sG = p[P_SF] * G;
            double loss = p[P_KTH] + 2.0 * Ysi * st->Gi[m];
            double th = sG > 0 && !linear ? sG / (sG + loss) : 0.0;
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
        st->theta[m] = sG > 0 && !linear ? sG / (sG + loss) : 0.0;
    }
}

/* the velocity (m/s) of a bare surface point of material `mat` */
static double bare_rate(const double *p, const State *st, int m, int mat,
                        double Ysi, double Ymask) {
    if (mat == 1) return Ymask * st->Gi[m] + p[P_BMASK] * st->G[m];
    if (mat == 2) return 0.0;
    if (p[P_LSIA] > 0.0 || p[P_LSIB] > 0.0)
        return p[P_LSIA] * st->Gi[m] + p[P_LSIB] * st->G[m];
    return st->theta[m] * (0.25 * p[P_KTH] + Ysi * st->Gi[m]) / p[P_NSI];
}

static double flat_etch(const double *p, double *theta) {
    /* the open flat wafer: G = GF, Gi = GIE */
    double Ysi = ion_yield(p[P_ESI], p[P_ASI], p[P_ETHSI]);
    if (p[P_LSIA] > 0.0 || p[P_LSIB] > 0.0) {
        if (theta) *theta = 0.0;
        return p[P_LSIA] * p[P_GIE] + p[P_LSIB] * p[P_GF];
    }
    double sG = p[P_SF] * p[P_GF], loss = p[P_KTH] + 2.0 * Ysi * p[P_GIE];
    double th = sG / (sG + loss);
    if (theta) *theta = th;
    return th * (0.25 * p[P_KTH] + Ysi * p[P_GIE]) / p[P_NSI];
}

/* the film's growth: the top moves into the void at V (um/s) */
static void grow(double *top, const double *V, double *g, int nx, int nz,
                 double dx, double dt) {
    int n = nx * nz;
    for (int k = 0; k < n; k++) top[k] = -top[k];
    advect(top, V, g, nx, nz, dx, dt);
    for (int k = 0; k < n; k++) top[k] = -top[k];
}

/* both level sets back to distance functions, the top over the substrate */
static void redistance(double *top, double *sub, double *tmp, int nx, int nz,
                       double dx) {
    reinit(sub, tmp, nx, nz, dx, 2);
    reinit(top, tmp, nx, nz, dx, 2);
    for (int k = 0; k < nx * nz; k++)
        if (top[k] > sub[k]) top[k] = sub[k];
}

static void drie_core(double *phi, const unsigned char *mask, int nx, int nz,
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
    double Ypass = ion_yield(p[P_EP], kpoly, p[P_ETHION]);   /* the passivation's ions */
    /* phi: the substrate (silicon, mask, stop layer); top: it and its film */
    double *top = malloc(sizeof(double) * n);
    memcpy(top, phi, sizeof(double) * n);
    double *V = calloc(n, sizeof(double)), *g = malloc(sizeof(double) * n);
    double *Vr = calloc(n, sizeof(double));
    double *tmp = malloc(sizeof(double) * n);
    int *cell2sp = malloc(sizeof(int) * n), *band = malloc(sizeof(int) * n);
    int *queue = malloc(sizeof(int) * n);
    Surf s = {0};
    Views v = {0};
    State st = {0};
    for (int o = 0; o < O_N; o++) out[o] = 0.0;
    out[O_ETCH0] = flat_etch(p, &out[O_THETA0]);
    out[O_CLEAR0] = -1.0;
    int sc = 0;

    for (int c = 0; c < cycles; c++) {
        /* ---- passivation: the film grows where its radicals stick and the
         * ions arrive, its surface moving into the void; the ions of a
         * biased step wear it down where they bring more than the radicals,
         * its surface moving back to the substrate ---- */
        double t = 0.0, moved = 1e30, vmax = 0.0;
        while (t < p[P_TPASS] - 1e-12) {
            if (moved > 0.5 * dx) {
                find_surface(top, phi, mask, nx, nz, dx, j_box, &s, cell2sp);
                views_compute(top, &s, cell2sp, nx, nz, nr, ni, sigma, &v);
                state_size(&st, s.n);
                /* the growth (st.v, m/s): the ions', then each radical's with
                 * its own sticking (its transport into the feature too); the
                 * ions' removal (st.Gi) */
                for (int m = 0; m < s.n; m++) {
                    st.v[m] = p[P_ADEP] * p[P_GIP] * v.ion[m];
                    st.Gi[m] = Ypass * p[P_GIP] * v.ion[m];
                }
                for (int q = 0; q < (int)p[P_NPREC]; q++) {
                    const double *r = p + P_N + 3 * q;     /* flux, sticking, volume */
                    if (!(r[0] > 0.0 && r[1] > 0.0)) continue;
                    for (int m = 0; m < s.n; m++) st.keep[m] = 1.0 - r[1];
                    radiosity(&v, s.n, st.keep, st.G, 0);
                    out[O_FLUXCALLS] += 1;
                    for (int m = 0; m < s.n; m++) st.v[m] += r[0] * r[1] * r[2] * st.G[m];
                }
                smooth_sp(&s, cell2sp, nx, nz, st.v, st.keep);
                if (Ypass > 0.0) smooth_sp(&s, cell2sp, nx, nz, st.Gi, st.keep);
                vmax = 0.0;
                for (int m = 0; m < s.n; m++) {
                    st.v[m] *= 1e6, st.Gi[m] *= 1e6;                     /* um/s */
                    vmax = dmax(vmax, dmax(st.v[m], st.Gi[m]));
                }
                label_all(cell2sp, nx, nz, band, queue);
                moved = 0.0;
            }
            if (vmax <= 0.0) break;
            double dt = dmin(p[P_TPASS] - t, p[P_CFL] * dx / vmax);
            if (Ypass > 0.0) film_at(&s, top, phi, dx);
            /* the net growth; where the ions take more, the film thins at the
             * difference, but no further than to the substrate */
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n; k++) {
                int m = band[k];
                double net = m >= 0 ? st.v[m] - st.Gi[m] : 0.0;
                V[k] = dmax(net, 0.0);
                Vr[k] = net < 0.0 ? dmin(-net, s.h[m] / dt) : 0.0;
            }
            grow(top, V, g, nx, nz, dx, dt);
            if (Ypass > 0.0) {
                advect(top, Vr, g, nx, nz, dx, dt);
                #pragma omp parallel for schedule(static)
                for (int k = 0; k < n; k++)
                    if (top[k] > phi[k]) top[k] = phi[k];
            }
            moved += vmax * dt;
            if (++sc % stride == 0) redistance(top, phi, tmp, nx, nz, dx);
            t += dt;
            out[O_STEPS] += 1;
        }

        /* ---- etch: the ions and F wear the film down; where it is gone,
         * they etch the substrate ---- */
        t = 0.0, moved = 1e30;
        int bare_last = -1, kclear = -1;
        while (t < p[P_TETCH] - 1e-12) {
            /* the fluxes again when the front has moved half a cell or the
             * film has cleared from a twentieth of the surface since */
            int bare = 0;
            if (bare_last >= 0) {
                film_at(&s, top, phi, dx);
                for (int m = 0; m < s.n; m++) bare += s.h[m] <= 0.0;
            }
            if (moved > 0.5 * dx || bare_last < 0
                || abs(bare - bare_last) > s.n / 20) {
                find_surface(top, phi, mask, nx, nz, dx, j_box, &s, cell2sp);
                views_compute(top, &s, cell2sp, nx, nz, nr, ni, sigma, &v);
                state_size(&st, s.n);
                etch_fluxes(p, &s, &v, cell2sp, nx, nz, &st);
                out[O_FLUXCALLS] += 1;
                moved = 0.0;
                label_all(cell2sp, nx, nz, band, queue);
                bare = 0;
                for (int m = 0; m < s.n; m++) bare += s.h[m] <= 0.0;
                if (c == 0 && bare_last < 0) {
                    /* the film after the first passivation step on the
                     * silicon that sees the most of the plasma (its gas and
                     * its ions), and where it is (for when it clears) */
                    double best = -1.0;
                    for (int m = 0; m < s.n; m++)
                        if (s.mat[m] == 0 && v.sky[m] + v.ion[m] > best) {
                            best = v.sky[m] + v.ion[m];
                            out[O_FILM0] = s.h[m] * 1e-6;
                            kclear = s.cell[m];
                        }
                }
                bare_last = bare;
            }
            /* the film's removal rate on each point (st.v, m/s); the time
             * step keeps every surface point, bare or about to be, within
             * the CFL limit; a film that clears inside the step lets its
             * surface etch for the rest of it */
            double vpot = 0.0;
            for (int m = 0; m < s.n; m++) {
                st.v[m] = Ypoly * st.Gi[m] + p[P_BPOLY] * st.G[m];
                vpot = dmax(vpot, dmax(st.v[m], bare_rate(p, &st, m, s.mat[m], Ysi, Ymask)));
                vpot = dmax(vpot, bare_rate(p, &st, m, 0, Ysi, Ymask));
            }
            double dt = p[P_TETCH] - t;
            if (vpot > 0.0) dt = dmin(dt, p[P_CFL] * dxm / vpot);
            if (c == 0 && out[O_CLEAR0] < 0.0 && kclear >= 0 && band[kclear] >= 0) {
                double h = (phi[kclear] - top[kclear]) * 1e-6, vp = st.v[band[kclear]];
                if (h <= 0.02 * dxm) out[O_CLEAR0] = t;
                else if (vp * dt >= h) out[O_CLEAR0] = t + h / vp;
            }
            double vapplied = 0.0;
            #pragma omp parallel for schedule(static) reduction(max:vapplied)
            for (int k = 0; k < n; k++) {
                V[k] = 0.0;
                int m = band[k];
                if (m < 0) continue;
                /* the nearest surface point's film and fluxes; the substrate
                 * under it, or a substrate cell's own material (the void
                 * above the mask would otherwise take silicon's speed, or
                 * the mask below silicon its) */
                int mat = phi[k] > 0.0 ? s.mat[m]
                        : mask[k] ? 1 : (k / nx >= j_box ? 2 : 0);
                double vb = bare_rate(p, &st, m, mat, Ysi, Ymask);
                double vp = st.v[m], h = s.h[m] * 1e-6, vel;
                if (h <= 0.0) vel = vb;
                else if (vp * dt > h) vel = (h + vb * (dt - h / vp)) / dt;
                else vel = vp;
                V[k] = vel * 1e6;                                     /* um/s */
                if (V[k] > vapplied) vapplied = V[k];
            }
            advect(top, V, g, nx, nz, dx, dt);
            /* the substrate the top has passed is etched */
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n; k++)
                if (top[k] > phi[k]) phi[k] = top[k];
            moved += vapplied * dt;
            if (++sc % stride == 0) redistance(top, phi, tmp, nx, nz, dx);
            t += dt;
            out[O_STEPS] += 1;
        }
    }
    reinit(phi, tmp, nx, nz, dx, 2);
    free(top); free(V); free(Vr); free(g); free(tmp); free(cell2sp); free(band);
    free(queue);
    surf_free(&s); views_free(&v);
    free(st.G); free(st.Gi); free(st.theta); free(st.keep); free(st.v);
}

/* the grid with rows of void above it for the film on the mask's top (its
 * thickest, on an open flat surface over the cycles, and three cells); the
 * substrate is returned on the caller's grid */
void drie_run(double *phi, const unsigned char *mask, int nx, int nz,
              double dx, int j_box, const double *p, double *out) {
    double kpoly = p[P_APOLY] / dmax(1e-12, sqrt(p[P_EREF]) - sqrt(p[P_ETHION]));
    double dep = p[P_ADEP] * p[P_GIP];
    for (int q = 0; q < (int)p[P_NPREC]; q++)
        dep += p[P_N + 3 * q] * p[P_N + 3 * q + 1] * p[P_N + 3 * q + 2];
    dep = dmax(0.0, dep - ion_yield(p[P_EP], kpoly, p[P_ETHION]) * p[P_GIP]) * p[P_TPASS];
    double strip = (ion_yield(p[P_EE], kpoly, p[P_ETHION]) * p[P_GIE]
                    + p[P_BPOLY] * p[P_GF]) * p[P_TETCH];
    double h = 0.0, peak = 0.0;
    for (int c = 0; c < (int)p[P_CYCLES]; c++) {
        h += dep;
        peak = dmax(peak, h);
        h = dmax(0.0, h - strip);
    }
    int pad = (int)ceil(peak / (dx * 1e-6)) + 3, nz2 = nz + pad;
    double *phi2 = malloc(sizeof(double) * nx * nz2);
    unsigned char *mask2 = calloc((size_t)nx * nz2, 1);
    for (int j = 0; j < pad; j++)
        for (int i = 0; i < nx; i++) phi2[IDX(i, j)] = phi[i] + (pad - j) * dx;
    memcpy(phi2 + (size_t)nx * pad, phi, sizeof(double) * nx * nz);
    memcpy(mask2 + (size_t)nx * pad, mask, (size_t)nx * nz);
    drie_core(phi2, mask2, nx, nz2, dx, j_box + pad, p, out);
    memcpy(phi, phi2 + (size_t)nx * pad, sizeof(double) * nx * nz);
    free(phi2); free(mask2);
}
