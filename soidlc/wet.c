/* Anisotropic wet etch (KOH / TMAH) of single-crystal silicon -- the
 * bulk-micromachining sibling of the Bosch process.  Same level-set core,
 * different velocity: the etch rate is orientation dependent (a function of
 * the angle between the surface normal and the crystal axes).  On a (100)
 * wafer the slow {111} planes meet the surface at 54.74 deg, so a mask opening
 * self-terminates into a facetted pit / V-groove / inverted pyramid bounded by
 * {111}.
 */
#include "levelset.h"

/* --- 2D cross-section -------------------------------------------------- */
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

/* --- 3D voxel etch ----------------------------------------------------- *
 * The slow {111} planes of a (100) wafer have normals at 54.74 deg from [001]
 * at the four <110> azimuths; the rate notches down when the surface normal
 * aligns with any of them, so a square mask opening self-organises into the
 * classic inverted pyramid bounded by four {111} facets.
 */
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

/* --- 3D etch driven by a measured rate diagram ------------------------- *
 * Instead of the closed-form {111}-notch above, the per-cell rate is looked up
 * from a calibrated (theta, phi) table sampled (in the wafer frame) from a
 * crystal_rates.RateDiagram: theta in [0, pi] from the surface normal, phi in
 * [0, 2pi).  This decouples the crystallography (data) from the numerics
 * (this kernel) and supports arbitrary wafer orientation and real KOH/TMAH
 * anisotropy maps.  ``tab`` is row-major [ntheta * nphi], normalised so its
 * peak is 1; the caller restores the absolute scale via dt.
 */
void wet_run_3d_tab(double *phi, const unsigned char *mask, int nx, int ny,
                    int nz, double dx, const double *tab, int ntheta, int nphi,
                    double sel, int nsteps, double dt, int reinit_stride) {
    long n = (long)nx * ny * nz;
    double *V = malloc(sizeof(double) * n);
    double *g = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    double band = 4.5 * dx, TWO_PI = 2.0 * M_PI;
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
                    double uz=gz/gn;
                    if (uz>1.0) uz=1.0; else if (uz<-1.0) uz=-1.0;
                    double th=acos(uz), ph=atan2(gy/gn, gx/gn);
                    if (ph<0.0) ph+=TWO_PI;
                    /* bilinear lookup: theta in [0,pi] over ntheta nodes,
                     * phi in [0,2pi) periodic over nphi nodes */
                    double ft=th/M_PI*(ntheta-1);
                    int it=(int)ft; if(it>ntheta-2) it=ntheta-2; if(it<0) it=0;
                    double a=ft-it;
                    double fp=ph/TWO_PI*nphi;
                    int jp0=(int)fp % nphi; if(jp0<0) jp0+=nphi;
                    int jp1=(jp0+1)%nphi; double b=fp-(int)fp;
                    double r00=tab[it*nphi+jp0], r01=tab[it*nphi+jp1];
                    double r10=tab[(it+1)*nphi+jp0], r11=tab[(it+1)*nphi+jp1];
                    double r=(1-a)*((1-b)*r00+b*r01)+a*((1-b)*r10+b*r11);
                    if (mask[id]) r/=sel;
                    V[id]=r;
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

/* --- 3D etch through a thin masking FILM of finite selectivity --------- *
 * The mask is *not* part of phi (which is just the silicon: phi>0 etchant/air
 * above, phi<0 silicon).  Instead ``mfilm[nx*ny]`` holds the remaining film
 * thickness (um) over each column.  While a column still has film it is shaded
 * from *top* attack, but silicon undercut laterally from an open neighbour is
 * always allowed -- so an aligned <110> edge self-limits on {111} while a
 * misaligned edge undercuts and the opening enlarges.  The film thins at the
 * vertical etch rate / selectivity and, once gone, that column's top etches
 * too (mask failure).  Rates come from the same (theta,phi) table.
 */
void wet_run_3d_film(double *phi, int nx, int ny, int nz, double dx,
                     const double *tab, int ntheta, int nphi, double *mfilm,
                     double sel, int nsteps, double dt, int reinit_stride) {
    long n = (long)nx * ny * nz;
    double *V = malloc(sizeof(double) * n);
    double *g = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    double band = 4.5 * dx, TWO_PI = 2.0 * M_PI;
    double r_top = tab[0];                 /* theta=0 pole (vertical) rate */
    for (int s = 0; s < nsteps; s++) {
        double inv_dx = 1.0 / dx;
        /* thin the masking film everywhere it remains (it sits in the bath) */
        #pragma omp parallel for schedule(static)
        for (long c = 0; c < (long)nx * ny; c++)
            if (mfilm[c] > 0.0) {
                mfilm[c] -= r_top / sel * dt;
                if (mfilm[c] < 0.0) mfilm[c] = 0.0;
            }
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
                    double uz=gz/gn;
                    if (uz>1.0) uz=1.0; else if (uz<-1.0) uz=-1.0;
                    double th=acos(uz), ph=atan2(gy/gn, gx/gn);
                    if (ph<0.0) ph+=TWO_PI;
                    double ft=th/M_PI*(ntheta-1);
                    int it=(int)ft; if(it>ntheta-2) it=ntheta-2; if(it<0) it=0;
                    double a=ft-it;
                    double fp=ph/TWO_PI*nphi;
                    int q0=(int)fp % nphi; if(q0<0) q0+=nphi;
                    int q1=(q0+1)%nphi; double b=fp-(int)fp;
                    double r=(1-a)*((1-b)*tab[it*nphi+q0]+b*tab[it*nphi+q1])
                            +a*((1-b)*tab[(it+1)*nphi+q0]+b*tab[(it+1)*nphi+q1]);
                    /* while the film survives, block attack that comes only
                     * from above (top etch); allow lateral / below undercut */
                    if (mfilm[i + nx * j] > 0.0) {
                        int lat = (phi[ID3(im,j,k)]>0)||(phi[ID3(ip,j,k)]>0)||
                                  (phi[ID3(i,jm,k)]>0)||(phi[ID3(i,jp,k)]>0);
                        int below = phi[ID3(i,j,kp)] > 0;
                        if (!lat && !below) continue;     /* protected top */
                    }
                    V[id]=r;
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
