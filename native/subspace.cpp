// Cubic tensor splines and independent subspace ring polymers. MIT license.
// Python owns random draws, adaptation and statistical analysis; no callbacks
// occur in this module. No fast-math: Metropolis decisions need finite checks.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ring_fft.h"

namespace {
struct Buffer {
    Py_buffer v{};
    explicit Buffer(PyObject* o, bool writable=false) {
        if (PyObject_GetBuffer(o, &v, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS
                              | (writable ? PyBUF_WRITABLE : 0)) < 0)
            throw std::invalid_argument("contiguous buffer required");
    }
    ~Buffer() { if (v.obj) PyBuffer_Release(&v); }
    Buffer(const Buffer&) = delete;
    double* data() const { return static_cast<double*>(v.buf); }
    void check(std::initializer_list<Py_ssize_t> shape, const char* format="d") const {
        if (v.ndim != static_cast<int>(shape.size()) || !v.format || std::strcmp(v.format, format)
            || v.itemsize != (std::strcmp(format, "?") == 0 ? 1 : 8))
            throw std::invalid_argument("subspace buffer dtype or rank mismatch");
        int i=0;
        for (auto n: shape) if (n < 1 || v.shape[i++] != n)
            throw std::invalid_argument("subspace buffer shape mismatch");
    }
};

bool overlaps(const Buffer& a, const Buffer& b) {
    auto x=reinterpret_cast<std::uintptr_t>(a.v.buf), y=reinterpret_cast<std::uintptr_t>(b.v.buf);
    return x<y+static_cast<std::uintptr_t>(b.v.len) && y<x+static_cast<std::uintptr_t>(a.v.len);
}

void disjoint(std::initializer_list<const Buffer*> buffers) {
    for (auto a=buffers.begin(); a!=buffers.end(); ++a)
        for (auto b=a+1; b!=buffers.end(); ++b)
            if (overlaps(**a,**b)) throw std::invalid_argument("native subspace buffers must not alias");
}

// Cox-de Boor basis and its exact first derivative, using the very same
// not-a-knot knots/coefficient tensor as scipy.interpolate.NdBSpline.
void basis(const std::vector<double>& t, int span, double x, int degree, double* out) {
    double left[4]{}, right[4]{};
    out[0]=1.;
    for (int j=1; j<=degree; ++j) {
        left[j]=x-t[span+1-j]; right[j]=t[span+j]-x;
        double saved=0.;
        for (int r=0; r<j; ++r) {
            const double den=right[r+1]+left[j-r];
            const double term=den == 0. ? 0. : out[r]/den;
            out[r]=saved+right[r+1]*term; saved=left[j-r]*term;
        }
        out[j]=saved;
    }
}

struct Spline {
    int d;
    std::array<std::vector<double>,3> knots;
    std::array<Py_ssize_t,3> shape{1,1,1};
    std::vector<double> coefficients;

    bool evaluate(const double* raw, double& energy, double* gradient, long long& outside) const {
        double b[3][4]{}, deriv[3][4]{};
        int first[3]{};
        bool clipped=false;
        for (int a=0; a<d; ++a) {
            if (!std::isfinite(raw[a])) return false;
            const auto& t=knots[a];
            const double x=std::clamp(raw[a],t[3],t[shape[a]]);
            const bool clamp=x != raw[a];
            clipped |= clamp;
            const int span=std::clamp(static_cast<int>(std::upper_bound(t.begin(),t.end(),x)-t.begin())-1,
                                      3,static_cast<int>(shape[a])-1);
            first[a]=span-3;
            basis(t,span,x,3,b[a]);
            double lower[4]{};
            basis(t,span,x,2,lower);
            if (!clamp) for (int j=0; j<4; ++j) {
                const int i=first[a]+j;
                const double l=t[i+3]-t[i], r=t[i+4]-t[i+1];
                deriv[a][j]=(j>0 && l>0 ? 3.*lower[j-1]/l : 0.)
                            -(j<3 && r>0 ? 3.*lower[j]/r : 0.);
            }
        }
        outside += clipped;
        energy=0.;
        std::fill(gradient,gradient+d,0.);
        const int count=1 << (2*d);
        for (int k=0; k<count; ++k) {
            int j[3]{k&3,(k>>2)&3,(k>>4)&3};
            Py_ssize_t index=0;
            double weight=1.;
            for (int a=0; a<d; ++a) { index=index*shape[a]+first[a]+j[a]; weight*=b[a][j[a]]; }
            const double c=coefficients[index];
            energy += c*weight;
            for (int a=0; a<d; ++a) {
                double dw=deriv[a][j[a]];
                for (int other=0; other<d; ++other) if (other!=a) dw*=b[other][j[other]];
                gradient[a]+=c*dw;
            }
        }
        return std::isfinite(energy) && std::all_of(gradient,gradient+d,[](double x){return std::isfinite(x);});
    }
};

struct Ring {
    Spline spline;
    int p, d;
    RingFFT fft;
    explicit Ring(int beads, int dimension): p(beads), d(dimension), fft(beads) {}
    void transform(const double* input, double* output, double* scratch, bool to_beads) const {
        double* real=scratch; double* imag=real+p; double* temp=imag+p;
        const double root=std::sqrt(static_cast<double>(p)), pair=std::sqrt(p/2.);
        for (int a=0; a<d; ++a) {
            std::fill(imag,imag+p,0.);
            if (to_beads) {
                real[0]=root*input[a];
                for (int k=1; 2*k<p; ++k) {
                    real[k]=real[p-k]=pair*input[k*d+a];
                    imag[k]=pair*input[(p-k)*d+a]; imag[p-k]=-imag[k];
                }
                if (p%2==0) real[p/2]=root*input[(p/2)*d+a];
            } else for (int k=0; k<p; ++k) real[k]=input[k*d+a];
            fft.transform(real,imag,temp,to_beads);
            if (to_beads) for (int k=0; k<p; ++k) output[k*d+a]=real[k];
            else {
                output[a]=real[0]/root;
                for (int k=1; k<p; ++k)
                    output[k*d+a]=2*k<p ? real[k]/pair : (2*k==p ? real[k]/root : -imag[k]/pair);
            }
        }
    }
};

constexpr const char* capsule_name="pyrpmd.subspace.Ring";
Ring* ring_from(PyObject* obj) {
    auto* ring=static_cast<Ring*>(PyCapsule_GetPointer(obj,capsule_name));
    if (!ring) throw std::invalid_argument("invalid subspace ring");
    return ring;
}
void destroy(PyObject* obj) { delete static_cast<Ring*>(PyCapsule_GetPointer(obj,capsule_name)); }
int threads_for(int threads, Py_ssize_t batch) {
    if (threads<1) throw std::invalid_argument("threads must be positive");
    #ifndef _OPENMP
    return 1;
    #else
    return static_cast<int>(std::min<Py_ssize_t>(threads,batch));
    #endif
}
PyObject* stats(long long evaluations, long long outside, int threads) {
    return Py_BuildValue("{s:L,s:L,s:i}","evaluations",evaluations,"outside",outside,"threads",threads);
}
PyObject* failure(const std::exception& error) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,error.what());
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject *ko, *co; int p;
    if (!PyArg_ParseTuple(args,"OOi",&ko,&co,&p)) return nullptr;
    try {
        if (!PyTuple_Check(ko) || PyTuple_Size(ko)<1 || PyTuple_Size(ko)>3 || p<1)
            throw std::invalid_argument("one to three knot arrays and positive beads required");
        const int d=static_cast<int>(PyTuple_Size(ko));
        Buffer c(co);
        if (c.v.ndim!=d || !c.v.format || std::strcmp(c.v.format,"d") || c.v.itemsize!=8)
            throw std::invalid_argument("float64 spline coefficient tensor required");
        auto r=std::make_unique<Ring>(p,d);
        r->spline.d=d;
        for (int a=0; a<d; ++a) {
            Buffer t(PyTuple_GetItem(ko,a));
            if (c.v.shape[a]<4) throw std::invalid_argument("at least four spline coefficients per axis required");
            t.check({c.v.shape[a]+4});
            r->spline.shape[a]=c.v.shape[a];
            auto& knots=r->spline.knots[a];
            knots.assign(t.data(),t.data()+t.v.shape[0]);
            if (!std::all_of(knots.begin(),knots.end(),[](double x){return std::isfinite(x);})
                || !std::is_sorted(knots.begin(),knots.end()) || !(knots[3]<knots[c.v.shape[a]]))
                throw std::invalid_argument("invalid cubic spline knots");
        }
        r->spline.coefficients.assign(c.data(),c.data()+c.v.len/8);
        if (!std::all_of(r->spline.coefficients.begin(),r->spline.coefficients.end(),[](double x){return std::isfinite(x);}))
            throw std::invalid_argument("nonfinite spline coefficients");
        PyObject* capsule=PyCapsule_New(r.get(),capsule_name,destroy);
        if (capsule) r.release();
        return capsule;
    } catch (const std::exception& e) { return failure(e); }
}

PyObject* evaluate(PyObject*, PyObject* args) {
    PyObject *ro,*xo,*eo,*go; int nt;
    if (!PyArg_ParseTuple(args,"OOOOi",&ro,&xo,&eo,&go,&nt)) return nullptr;
    try {
        const auto& r=*ring_from(ro);
        Buffer x(xo),e(eo,true),g(go,true);
        if (x.v.ndim!=2) throw std::invalid_argument("points must have shape (n, dimension)");
        const auto n=x.v.shape[0];
        x.check({n,r.d}); e.check({n}); g.check({n,r.d});
        disjoint({&x,&e,&g});
        nt=threads_for(nt,n);
        long long outside=0; int bad=0, used=1;
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel num_threads(nt) reduction(+:outside) reduction(|:bad)
        {
            #ifdef _OPENMP
            #pragma omp single
            used=omp_get_num_threads();
            #endif
            #pragma omp for schedule(static)
            for (Py_ssize_t i=0; i<n; ++i)
                bad |= !r.spline.evaluate(x.data()+i*r.d,e.data()[i],g.data()+i*r.d,outside);
        }
        Py_END_ALLOW_THREADS
        if (bad) { PyErr_SetString(PyExc_FloatingPointError,"nonfinite subspace spline evaluation"); return nullptr; }
        return stats(n,outside,used);
    } catch (const std::exception& e) { return failure(e); }
}

PyObject* advance(PyObject*, PyObject* args) {
    PyObject *ro,*qo,*po,*fo,*eo,*go,*propo,*so; double dt,surface; int steps,constrained,nt;
    if (!PyArg_ParseTuple(args,"OOOOOOOdiiiOd",&ro,&qo,&po,&fo,&eo,&go,&propo,
                          &dt,&steps,&constrained,&nt,&so,&surface)) return nullptr;
    try {
        const auto& r=*ring_from(ro);
        Buffer q(qo,true),p(po,true),f(fo,true),e(eo,true),g(go,true),prop(propo);
        if (q.v.ndim!=3 || steps<1 || !std::isfinite(dt) || dt<=0 || !std::isfinite(surface))
            throw std::invalid_argument("invalid propagation shape, steps or time step");
        const auto batch=q.v.shape[0]; const int width=r.p*r.d;
        q.check({batch,r.p,r.d}); p.check({batch,r.p,r.d}); f.check({batch,r.p,r.d});
        e.check({batch}); g.check({batch,r.p,r.d}); prop.check({3,r.p,r.d});
        std::unique_ptr<Buffer> side;
        if (so!=Py_None) { side=std::make_unique<Buffer>(so,true); side->check({steps+1,batch},"?"); }
        disjoint({&q,&p,&f,&e,&g,&prop});
        if (side) for (auto* buffer: {&q,&p,&f,&e,&g,&prop})
            if (overlaps(*side,*buffer)) throw std::invalid_argument("side buffer must not alias state");
        nt=threads_for(nt,batch);
        const size_t scratch_size=width+4*r.p;
        std::vector<double> work(nt*scratch_size);
        const double* cosine=prop.data(); const double* a=cosine+width; const double* b=a+width;
        const double root=std::sqrt(static_cast<double>(r.p));
        long long outside=0, evaluations=0; int bad=0, used=1;
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel num_threads(nt) reduction(+:outside,evaluations) reduction(|:bad)
        {
            int tid=0;
            #ifdef _OPENMP
            tid=omp_get_thread_num();
            #pragma omp single
            used=omp_get_num_threads();
            #endif
            double* beads=work.data()+tid*scratch_size; double* fft_work=beads+width;
            #pragma omp for schedule(static)
            for (Py_ssize_t i=0; i<batch; ++i) {
                double* qi=q.data()+i*width; double* pi=p.data()+i*width;
                double* fi=f.data()+i*width; double* gi=g.data()+i*width;
                auto* si=side ? static_cast<bool*>(side->v.buf) : nullptr;
                if (si) si[i]=qi[0]/root>surface;
                for (int step=1; step<=steps; ++step) {
                    #pragma omp simd
                    for (int k=0; k<width; ++k) {
                        const double momentum=pi[k]+.5*dt*fi[k];
                        const double position=qi[k];
                        qi[k]=cosine[k]*position+a[k]*momentum;
                        pi[k]=b[k]*position+cosine[k]*momentum;
                    }
                    r.transform(qi,beads,fft_work,true);
                    double energy=0.; bool valid=true;
                    for (int j=0; j<r.p; ++j) {
                        double value=0.;
                        valid &= r.spline.evaluate(beads+j*r.d,value,gi+j*r.d,outside);
                        energy+=value;
                    }
                    evaluations+=r.p;
                    if (!valid) { bad=1; break; }
                    e.data()[i]=energy;
                    r.transform(gi,fi,fft_work,false);
                    #pragma omp simd
                    for (int k=0; k<width; ++k) fi[k]=-fi[k];
                    if (constrained) fi[0]=0.;
                    #pragma omp simd
                    for (int k=0; k<width; ++k) pi[k]+=.5*dt*fi[k];
                    if (si) si[static_cast<Py_ssize_t>(step)*batch+i]=qi[0]/root>surface;
                }
            }
        }
        Py_END_ALLOW_THREADS
        if (bad) { PyErr_SetString(PyExc_FloatingPointError,"nonfinite subspace trajectory"); return nullptr; }
        return stats(evaluations,outside,used);
    } catch (const std::exception& e) { return failure(e); }
}

PyMethodDef methods[]={
    {"create",create,METH_VARARGS,"Create an immutable spline/ring plan."},
    {"evaluate",evaluate,METH_VARARGS,"Evaluate a spline and its analytic gradient."},
    {"advance",advance,METH_VARARGS,"Propagate independent polymers without Python callbacks."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_subspace",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__subspace() {
    PyObject* m=PyModule_Create(&module);
    if (m) {
        #ifdef _OPENMP
        PyModule_AddIntConstant(m,"openmp",1);
        #else
        PyModule_AddIntConstant(m,"openmp",0);
        #endif
    }
    return m;
}
