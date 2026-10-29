/* PyRPMD reentrant Cayley ring-polymer propagation.
 * Cyclic midpoint/Thomas + rank-two correction follows chen-yu-hao/RPMDrate
 * commit 19ba2602bad5b2145b0d0ba0716d7ba3d90a1003, _main.f90, and the
 * thermal-dRPMD implementation at 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78.
 * Original RPMDrate copyright (c) 2012 Joshua W. Allen, William H. Green,
 * Yury V. Suleimanov. Licensed under MIT; see THIRD_PARTY_NOTICES.md.
 * This implementation uses the PyRPMD [bead, atom, xyz] C-contiguous layout.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Cayley {
    Py_ssize_t beads, nd;
    int threads;
    double dt, alpha, omega2;
    std::vector<double> mass, invdiag, lower, z1, z2;
    double si[4];
    Cayley(std::vector<double> m, double beta, Py_ssize_t p, double h, int nt)
      : beads(p), nd(3 * static_cast<Py_ssize_t>(m.size())), threads(nt),
        dt(h), alpha(.25*h*h*(p/beta)*(p/beta)), omega2((p/beta)*(p/beta)),
        mass(std::move(m)), invdiag(p,1.), lower(p,0.), z1(p,0.), z2(p,0.) {
        si[0]=si[3]=1.; si[1]=si[2]=0.;
        if (p < 3) return;
        const double d=1.+2.*alpha;
        invdiag[0]=1./d;
        for (Py_ssize_t k=1;k<p;k++) {
            lower[k]=-alpha*invdiag[k-1];
            invdiag[k]=1./(d+alpha*lower[k]);
        }
        z1[0]=-alpha; z2[p-1]=-alpha;
        solve(z1.data()); solve(z2.data());
        const double a=1.+z1[p-1], b=z2[p-1], c=z1[0], d2=1.+z2[0];
        const double det=a*d2-b*c;
        if (!(det>0.) || !std::isfinite(det)) throw std::runtime_error("Cayley factorization lost precision");
        si[0]=d2/det; si[1]=-b/det; si[2]=-c/det; si[3]=a/det;
    }
    void solve(double* x) const {
        for (Py_ssize_t k=1;k<beads;k++) x[k]-=lower[k]*x[k-1];
        x[beads-1]*=invdiag[beads-1];
        for (Py_ssize_t k=beads-1;k-->0;) x[k]=(x[k]+alpha*x[k+1])*invdiag[k];
    }
    void propagate(double* q,double* p,double* workspace) const {
        const Py_ssize_t n=beads, dims=nd;
        #pragma omp parallel for num_threads(threads) schedule(static) if(threads>1)
        for (Py_ssize_t j=0;j<dims;j++) {
            const double m=mass[j/3];
            if(n==1){ q[j]+=dt*p[j]/m; continue; }
            double* x=workspace+j*n;
            for(Py_ssize_t k=0;k<n;k++) x[k]=q[k*dims+j]+.5*dt*p[k*dims+j]/m;
            if(n==2){
                const double a=x[0],b=x[1],den=1.+4.*alpha;
                x[0]=((1.+2.*alpha)*a+2.*alpha*b)/den;
                x[1]=(2.*alpha*a+(1.+2.*alpha)*b)/den;
            }else{
                solve(x);
                const double c1=si[0]*x[n-1]+si[1]*x[0];
                const double c2=si[2]*x[n-1]+si[3]*x[0];
                for(Py_ssize_t k=0;k<n;k++)x[k]-=z1[k]*c1+z2[k]*c2;
            }
            for(Py_ssize_t k=0;k<n;k++) {
                q[k*dims+j]=2.*x[k]-q[k*dims+j];
                p[k*dims+j]-=dt*m*omega2*(2.*x[k]-x[(k+n-1)%n]-x[(k+1)%n]);
            }
        }
    }
};

static const char* CAPSULE="pyrpmd.Cayley.v1";
static void destroy(PyObject* capsule){
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(capsule,CAPSULE));
    if(c) delete c; else PyErr_Clear();
}
static PyObject* create(PyObject*,PyObject* args){
    PyObject* masses; double beta,dt; Py_ssize_t beads; int threads;
    if(!PyArg_ParseTuple(args,"Odndi",&masses,&beta,&beads,&dt,&threads))return nullptr;
    if(!(beta>0.)||!std::isfinite(beta)||!std::isfinite(dt)||beads<1||threads<1){PyErr_SetString(PyExc_ValueError,"invalid Cayley parameters");return nullptr;}
    #ifndef _OPENMP
    if(threads>1){PyErr_SetString(PyExc_RuntimeError,"native kernel has no OpenMP support");return nullptr;}
    #endif
    PyObject* seq=PySequence_Fast(masses,"masses must be a sequence");
    if(!seq)return nullptr;
    const Py_ssize_t n=PySequence_Fast_GET_SIZE(seq);
    const Py_ssize_t width=3*static_cast<Py_ssize_t>(sizeof(double));
    if(n<1 || n>PY_SSIZE_T_MAX/width || beads>PY_SSIZE_T_MAX/(width*n)){Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"invalid dimensions");return nullptr;}
    std::vector<double> m;
    try { m.resize(n); } catch(const std::bad_alloc&){Py_DECREF(seq);return PyErr_NoMemory();}
    for(Py_ssize_t i=0;i<n;i++){
        m[i]=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq,i));
        if(PyErr_Occurred()){Py_DECREF(seq);return nullptr;}
        if(!(m[i]>0.)||!std::isfinite(m[i])){Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"masses must be finite and positive");return nullptr;}
    }
    Py_DECREF(seq);
    const double omega=beads/beta, alpha=.25*dt*dt*omega*omega;
    if(!std::isfinite(omega*omega)||!std::isfinite(4.*alpha)){PyErr_SetString(PyExc_ValueError,"Cayley coefficients overflow");return nullptr;}
    try {return PyCapsule_New(new Cayley(std::move(m),beta,beads,dt,threads),CAPSULE,destroy);}
    catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
static bool buffer(PyObject* obj,Py_buffer* v,Py_ssize_t bytes,bool writable){
    if(PyObject_GetBuffer(obj,v,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS|(writable?PyBUF_WRITABLE:0))<0)return false;
    if(v->len!=bytes||v->itemsize!=sizeof(double)||!v->format||std::string(v->format)!="d"){
        PyBuffer_Release(v); PyErr_SetString(PyExc_ValueError,"expected a C-contiguous native float64 array of the correct size");return false;
    }
    return true;
}
static PyObject* free_step(PyObject*,PyObject* args){
    PyObject *capsule,*qo,*po;
    if(!PyArg_ParseTuple(args,"OOO",&capsule,&qo,&po))return nullptr;
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(capsule,CAPSULE)); if(!c)return nullptr;
    const Py_ssize_t bytes=c->beads*c->nd*sizeof(double);
    Py_buffer q,p;
    if(!buffer(qo,&q,bytes,true))return nullptr;
    if(!buffer(po,&p,bytes,true)){PyBuffer_Release(&q);return nullptr;}
    if(q.buf==p.buf){PyBuffer_Release(&q);PyBuffer_Release(&p);PyErr_SetString(PyExc_ValueError,"q and p must not alias");return nullptr;}
    try {
        std::vector<double> work(c->beads*c->nd);
        Py_BEGIN_ALLOW_THREADS
        c->propagate(static_cast<double*>(q.buf),static_cast<double*>(p.buf),work.data());
        Py_END_ALLOW_THREADS
    }catch(const std::bad_alloc&){PyBuffer_Release(&q);PyBuffer_Release(&p);return PyErr_NoMemory();}
    PyBuffer_Release(&q);PyBuffer_Release(&p);Py_RETURN_NONE;
}
static PyObject* kick(PyObject*,PyObject* args){
    PyObject *po,*fo;double h;int threads;
    if(!PyArg_ParseTuple(args,"OOdi",&po,&fo,&h,&threads))return nullptr;
    if(!std::isfinite(h)||threads<1){PyErr_SetString(PyExc_ValueError,"invalid kick parameters");return nullptr;}
    Py_buffer p,f;
    if(PyObject_GetBuffer(po,&p,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS|PyBUF_WRITABLE)<0)return nullptr;
    if(p.itemsize!=sizeof(double)||!p.format||std::string(p.format)!="d"){PyBuffer_Release(&p);PyErr_SetString(PyExc_ValueError,"p must be float64");return nullptr;}
    if(!buffer(fo,&f,p.len,false)){PyBuffer_Release(&p);return nullptr;}
    const Py_ssize_t n=p.len/sizeof(double); auto* dst=static_cast<double*>(p.buf); const auto* src=static_cast<const double*>(f.buf);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for num_threads(threads) schedule(static) if(threads>1)
    for(Py_ssize_t i=0;i<n;i++)dst[i]+=h*src[i];
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&p);PyBuffer_Release(&f);Py_RETURN_NONE;
}
static PyObject* has_openmp(PyObject*,PyObject*){
    #ifdef _OPENMP
    Py_RETURN_TRUE;
    #else
    Py_RETURN_FALSE;
    #endif
}
static PyMethodDef methods[]={
 {"create",create,METH_VARARGS,"Create an immutable independent Cayley context."},
 {"free_step",free_step,METH_VARARGS,"Update q,p using the full-ring Cayley map."},
 {"kick",kick,METH_VARARGS,"p += h*physical_force."},
 {"has_openmp",has_openmp,METH_NOARGS,"Whether this build supports OpenMP."},
 {nullptr,nullptr,0,nullptr}
};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_native","PyRPMD C++17/OpenMP kernels",-1,methods};
PyMODINIT_FUNC PyInit__native(){return PyModule_Create(&module);}
