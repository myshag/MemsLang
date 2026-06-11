/* Shared level-set core for the feature-scale fabrication simulators.
 *
 * Convention (all modules): phi < 0 = solid (silicon/mask), phi > 0 = etched
 * void, the surface is the zero level set.  Etching moves the interface into
 * the solid, d(phi)/dt = V|grad phi| with V >= 0, solved with an upwind
 * Godunov Hamiltonian.  Each process (Bosch DRIE, KOH wet etch, plan-view
 * corner undercut) differs only in its velocity field V; the grid evolution
 * primitives below are common and live in levelset.c.
 */
#ifndef LEVELSET_H
#define LEVELSET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* row-major flat indexing; the enclosing scope supplies nx (and ny for ID3) */
#define IDX(i, j) ((i) + nx * (j))
#define ID3(i, j, k) ((i) + nx * ((j) + ny * (k)))

static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }

/* --- 2D primitives (nx, nz grid) --- */
double grad_norm(const double *phi, int nx, int nz, int i, int j,
                 double inv_dx);                       /* upwind |grad phi| */
void reinit(double *phi, double *tmp, int nx, int nz, double dx, int iters);
void advect(double *phi, const double *V, double *g,
            int nx, int nz, double dx, double dt);

/* --- 3D primitives (nx, ny, nz grid) --- */
double gnorm3(const double *phi, int nx, int ny, int nz,
              int i, int j, int k, double inv_dx);
void reinit3(double *phi, double *tmp, int nx, int ny, int nz,
             double dx, int iters);

#endif /* LEVELSET_H */
