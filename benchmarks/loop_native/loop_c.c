#include "cayley_common.h"
#include <stdint.h>
/* Shared author-derived Fortran BKMP2 PES: common to ALL full-PES backends. */
extern void pyrpmd_bkmp2_batch(int32_t n, const double *q,
                                double *energy, double *force, int32_t *status);

/* 0: BKMP2 H3, 1: independent harmonic PES. No thermostat or bias. */
int pyrpmd_loop_c(int32_t P, const double *m, double beta, double dt,
                  int64_t steps, double *q, double *p, int32_t mode) {
    if (steps<0 || (mode!=0 && mode!=1) || !q || !p || !m) return -1;
    loop_ctx c;
    int rc=cayley_setup(&c,P,m,beta,dt);
    if(rc){cayley_teardown(&c);return -2;}
    const int len=P*9;
    double *f=(double *)malloc(sizeof(double)*(size_t)len);
    double *e=(double *)malloc(sizeof(double)*(size_t)P);
    int32_t *status=(int32_t *)malloc(sizeof(int32_t)*(size_t)P);
    if(!f||!e||!status){free(f);free(e);free(status);cayley_teardown(&c);return -3;}
    for(int64_t i=0;i<steps;i++) {
        if(mode==0){pyrpmd_bkmp2_batch(P,q,e,f,status);if(check_status(status,P)){rc=(int)(i+1);break;}}
        else harmonic_force(q,f,len);
        momentum_kick(p,f,len,0.5*dt);
        cayley_advance(&c,q,p);
        if(mode==0){pyrpmd_bkmp2_batch(P,q,e,f,status);if(check_status(status,P)){rc=(int)(i+1);break;}}
        else harmonic_force(q,f,len);
        momentum_kick(p,f,len,0.5*dt);
    }
    free(status);free(e);free(f);cayley_teardown(&c);
    return rc;
}
