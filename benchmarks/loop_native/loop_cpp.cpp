/* C++ compiled loop, exact same C-compatible Cayley math as C benchmark.
 * Compiles independently via g++ -O3, permitting meaningful compiler comparison.
 */
#include "cayley_common.h"
#include <cstdint>
#include <memory>
extern "C" void pyrpmd_bkmp2_batch(int32_t n, const double *q,
                                      double *energy,double *force,int32_t *status);

extern "C" int pyrpmd_loop_cpp(int32_t P,const double *m,double beta,double dt,
                               int64_t steps,double *q,double *p,int32_t mode) {
    if(steps<0 || (mode!=0 && mode!=1) || !q || !p || !m) return -1;
    loop_ctx c;
    int rc=cayley_setup(&c,P,m,beta,dt);
    if(rc){cayley_teardown(&c);return -2;}
    const int len=P*9;
    std::unique_ptr<double[]> f(new double[len]);
    std::unique_ptr<double[]> e(new double[P]);
    std::unique_ptr<int32_t[]> status(new int32_t[P]);
    for(int64_t i=0;i<steps;i++) {
        if(mode==0){pyrpmd_bkmp2_batch(P,q,e.get(),f.get(),status.get());if(check_status(status.get(),P)){rc=(int)(i+1);break;}}
        else harmonic_force(q,f.get(),len);
        momentum_kick(p,f.get(),len,0.5*dt);
        cayley_advance(&c,q,p);
        if(mode==0){pyrpmd_bkmp2_batch(P,q,e.get(),f.get(),status.get());if(check_status(status.get(),P)){rc=(int)(i+1);break;}}
        else harmonic_force(q,f.get(),len);
        momentum_kick(p,f.get(),len,0.5*dt);
    }
    cayley_teardown(&c);
    return rc;
}
