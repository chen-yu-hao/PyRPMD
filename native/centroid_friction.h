/* Constant diagonal centroid OU operator inside the complete native step loop.
 * Included in namespace fused_blocks after Buffer and centroid are defined.
 * NumPy supplies the historical random stream; no Python callback or RNG is
 * used while the propagation loop releases the GIL.
 */
#ifndef PYRPMD_CENTROID_FRICTION_H
#define PYRPMD_CENTROID_FRICTION_H
struct CentroidFriction {
    Vec decay, amplitude, sqrt_mass, noise, center;
    Py_ssize_t nd, beads;
    bool active=false;
    CentroidFriction(PyObject* spec,const Cayley& c,Py_ssize_t steps)
        :nd(c.nd),beads(c.beads) {
        if(!PyDict_Check(spec))throw std::invalid_argument("centroid friction must be a dictionary");
        decay=vector(field(spec,"decay"),nd);
        amplitude=vector(field(spec,"amplitude"),nd);
        sqrt_mass.resize(nd);center.resize(nd);
        for(Py_ssize_t i=0;i<nd;++i){
            if(!std::isfinite(decay[i])||decay[i]<0.||decay[i]>1.
                    ||!std::isfinite(amplitude[i])||amplitude[i]<0.)
                throw std::invalid_argument("invalid centroid OU coefficients");
            sqrt_mass[i]=std::sqrt(c.mass[i/3]);
            active=active || decay[i]!=1. || amplitude[i]!=0.;
        }
        const auto count=product(product(steps,2),nd);
        Buffer source(field(spec,"noise"),product(count,sizeof(double)),"d",false);
        if(!finite_values(source.doubles(),count))throw std::invalid_argument("nonfinite centroid OU noise");
        // Own the noise so caller output buffers cannot alias future draws.
        noise.assign(source.doubles(),source.doubles()+count);
    }
    void apply(double* p,Py_ssize_t step,int half) {
        if(!active)return;
        centroid(p,center.data(),beads,nd);
        const auto* draw=noise.data()+(2*step+half)*nd;
        for(Py_ssize_t i=0;i<nd;++i){
            const double weighted=center[i]/sqrt_mass[i];
            const double changed=decay[i]*weighted+amplitude[i]*draw[i];
            const double increment=sqrt_mass[i]*(changed-weighted);
            for(Py_ssize_t b=0;b<beads;++b)p[b*nd+i]+=increment;
        }
    }
};
#endif
