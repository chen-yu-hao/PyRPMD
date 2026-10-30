/* Identical scalar C-compatible Cayley kernel for the C and C++ loop tests.
 * Workspaces and matrix factors are allocated exactly once per trajectory.
 * q,p have PyRPMD C order [bead,atom,xyz], here [P,3,3].
 */
#ifndef PYRPMD_CAYLEY_COMMON_H
#define PYRPMD_CAYLEY_COMMON_H
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int n, nd;
    double dt, a, omega2, m[3], si[4];
    double *d, *l, *z1, *z2, *work;
} loop_ctx;

static void cayley_solve_tridiagonal(const loop_ctx *c, double *x) {
    for (int k = 1; k < c->n; k++) x[k] -= c->l[k] * x[k-1];
    x[c->n - 1] *= c->d[c->n-1];
    for (int k = c->n-2; k >= 0; k--) x[k] = (x[k] + c->a*x[k+1]) * c->d[k];
}

static int cayley_setup(loop_ctx *c, int P, const double *mass, double beta, double dt) {
    memset(c, 0, sizeof(*c));
    if (P < 1 || P > 100000 || !isfinite(beta) || beta <= 0 || !isfinite(dt)) return 1;
    c->n = P; c->nd = 9; c->dt = dt;
    for (int a = 0; a < 3; a++) {
        if (!isfinite(mass[a]) || mass[a] <= 0) return 1;
        c->m[a] = mass[a];
    }
    double omega = P/beta;
    c->omega2 = omega*omega;
    c->a = 0.25 * dt*dt * c->omega2;
    if (!isfinite(c->a) || !isfinite(c->omega2)) return 1;
    c->d= (double *)calloc((size_t)P, sizeof(double));
    c->l= (double *)calloc((size_t)P, sizeof(double));
    c->z1=(double *)calloc((size_t)P, sizeof(double));
    c->z2=(double *)calloc((size_t)P, sizeof(double));
    c->work=(double *)calloc((size_t)P, sizeof(double));
    if (!c->d || !c->l || !c->z1 || !c->z2 || !c->work) return 2;
    if (P < 3) return 0;
    double diag=1+2*c->a;
    c->d[0]=1/diag;
    for (int k=1;k<P;k++) {
        c->l[k]=-c->a*c->d[k-1];
        c->d[k]=1/(diag+c->a*c->l[k]);
    }
    c->z1[0]=-c->a; c->z2[P-1]=-c->a;
    cayley_solve_tridiagonal(c,c->z1); cayley_solve_tridiagonal(c,c->z2);
    double a=1+c->z1[P-1],b=c->z2[P-1],d=c->z1[0],e=1+c->z2[0];
    double det=a*e-b*d;
    if (!(det>0) || !isfinite(det)) return 3;
    c->si[0]=e/det; c->si[1]=-b/det; c->si[2]=-d/det; c->si[3]=a/det;
    return 0;
}
static void cayley_teardown(loop_ctx *c) {
    free(c->d);free(c->l);free(c->z1);free(c->z2);free(c->work);
    memset(c, 0, sizeof(*c));
}
static void cayley_advance(const loop_ctx *c,double *q,double *p){
    const int n=c->n,nd=c->nd;
    for (int j=0;j<nd;j++) {
        const double m=c->m[j/3];
        if (n==1) {q[j]+=c->dt*p[j]/m;continue;}
        double *x=c->work;
        for(int k=0;k<n;k++)x[k]=q[k*nd+j]+.5*c->dt*p[k*nd+j]/m;
        if (n==2) {
            double a=x[0], b=x[1], den=1+4*c->a;
            x[0]=((1+2*c->a)*a+2*c->a*b)/den;
            x[1]=(2*c->a*a+(1+2*c->a)*b)/den;
        } else {
            cayley_solve_tridiagonal(c,x);
            double r=c->si[0]*x[n-1]+c->si[1]*x[0];
            double s=c->si[2]*x[n-1]+c->si[3]*x[0];
            for(int k=0;k<n;k++)x[k]-=c->z1[k]*r+c->z2[k]*s;
        }
        for(int k=0;k<n;k++) {
            q[k*nd+j]=2*x[k]-q[k*nd+j];
            p[k*nd+j]-=c->dt*m*c->omega2*(2*x[k]-x[(k+n-1)%n]-x[(k+1)%n]);
        }
    }
}

/* Independent analytic toy PES, deliberately labelled NOT BKMP2. */
static void harmonic_force(const double *q, double *f, int len) {
    for(int i=0;i<len;i++) f[i] = -0.005 * q[i];
}
static void momentum_kick(double *p,const double *f,int len,double half_dt){
    for(int i=0;i<len;i++)p[i]+=half_dt*f[i];
}
static int check_status(const int32_t *s,int n){for(int i=0;i<n;i++)if(s[i])return i+1;return 0;}
#endif
