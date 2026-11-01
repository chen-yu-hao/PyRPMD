/* Exact batched mapping flow after Hermitian LAPACK diagonalization.
 * Original PyRPMD implementation; MIT license. Python buffer API only.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Complex=std::complex<double>;
struct Buffer {
    Py_buffer view{}; bool held=false;
    Buffer(PyObject* object, bool writable=false) {
        int flags=PyBUF_FORMAT|PyBUF_ND|PyBUF_C_CONTIGUOUS;
        if(writable) flags|=PyBUF_WRITABLE;
        if(PyObject_GetBuffer(object,&view,flags)<0) throw std::invalid_argument("contiguous buffer required");
        held=true;
    }
    ~Buffer(){if(held)PyBuffer_Release(&view);}
    Buffer(const Buffer&)=delete;
    const double* data()const{return static_cast<const double*>(view.buf);}
    double* output(){return static_cast<double*>(view.buf);}
    void check(const std::vector<Py_ssize_t>& shape,bool complex)const{
        const char* format=complex?"Zd":"d";
        if(view.ndim!=static_cast<int>(shape.size())||!view.format||std::strcmp(view.format,format)
                ||view.itemsize!=(complex?16:8))throw std::invalid_argument("float64/complex128 buffer shape or dtype mismatch");
        for(size_t i=0;i<shape.size();++i)if(view.shape[i]!=shape[i]||shape[i]<1)
            throw std::invalid_argument("electronic buffer shape mismatch");
    }
    bool finite()const{
        for(Py_ssize_t i=0;i<view.len/static_cast<Py_ssize_t>(sizeof(double));++i)
            if(!std::isfinite(data()[i]))return false;
        return true;
    }
};
bool overlaps(const Buffer& a,const Buffer& b){
    const auto x=reinterpret_cast<std::uintptr_t>(a.view.buf), y=reinterpret_cast<std::uintptr_t>(b.view.buf);
    return x<y+static_cast<std::uintptr_t>(b.view.len)&&y<x+static_cast<std::uintptr_t>(a.view.len);
}
Complex get(const double* p,Py_ssize_t i){return Complex(p[2*i],p[2*i+1]);}
void put(double* p,Py_ssize_t i,Complex value){p[2*i]=value.real();p[2*i+1]=value.imag();}
double sinc(double x){return x==0.?1.:std::sin(x)/x;}

PyObject* mapping_flow(PyObject*,PyObject* args){
    PyObject *eo,*uo,*zo,*go,*outzo,*outpo;double duration,gamma;int nt;
    if(!PyArg_ParseTuple(args,"OOOOddOOi",&eo,&uo,&zo,&go,&duration,&gamma,&outzo,&outpo,&nt))return nullptr;
    try{
        if(!std::isfinite(duration)||!std::isfinite(gamma)||nt<1)
            throw std::invalid_argument("finite duration/gamma and positive threads required");
        Buffer e(eo),u(uo),z(zo),g(go),oz(outzo,true),op(outpo,true);
        if(e.view.ndim!=2||g.view.ndim!=4)throw std::invalid_argument("expected batched eigensystem and gradients");
        const Py_ssize_t batch=e.view.shape[0],s=e.view.shape[1],d=g.view.shape[3];
        e.check({batch,s},false);u.check({batch,s,s},true);z.check({batch,s},true);
        g.check({batch,s,s,d},true);oz.check({batch,s},true);op.check({batch,d},false);
        if(!e.finite()||!u.finite()||!z.finite()||!g.finite())throw std::invalid_argument("nonfinite electronic input");
        for(const Buffer* output:{&oz,&op})for(const Buffer* input:{&e,&u,&z,&g})
            if(overlaps(*output,*input))throw std::invalid_argument("electronic input/output buffers must not alias");
        if(overlaps(oz,op))throw std::invalid_argument("electronic output buffers must not alias");
        nt=static_cast<int>(std::min<Py_ssize_t>(nt,batch));
        #ifndef _OPENMP
        nt=1;
        #endif
        // Allocate before releasing the GIL or entering an OpenMP region.
        const size_t width=3*static_cast<size_t>(s)*s+2*static_cast<size_t>(s);
        if(width>std::numeric_limits<size_t>::max()/static_cast<size_t>(nt))throw std::bad_alloc();
        std::vector<Complex> work(width*nt);
        int bad=0;
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel for num_threads(nt) schedule(static) reduction(|:bad) if(nt>1)
        for(Py_ssize_t b=0;b<batch;++b){
            int thread=0;
            #ifdef _OPENMP
            thread=omp_get_thread_num();
            #endif
            Complex* ce=work.data()+width*thread;
            Complex* c=ce+s;Complex* tmp=c+s*s;Complex* density=tmp+s*s;Complex* rotated=density+s*s;
            const double* E=e.data()+b*s;
            const double* U=u.data()+2*b*s*s;
            const double* Z=z.data()+2*b*s;
            const double* G=g.data()+2*b*s*s*d;
            for(Py_ssize_t i=0;i<s;++i){
                ce[i]=0.;
                for(Py_ssize_t a=0;a<s;++a)ce[i]+=std::conj(get(U,a*s+i))*get(Z,a);
            }
            for(Py_ssize_t i=0;i<s;++i)for(Py_ssize_t j=0;j<s;++j){
                const double phase=.5*duration*(E[i]-E[j]);
                c[i*s+j]=std::conj(ce[i])*ce[j]*(duration*sinc(phase))*std::exp(Complex(0.,phase));
            }
            for(Py_ssize_t i=0;i<s;++i)for(Py_ssize_t j=0;j<s;++j){
                Complex value=0.;
                for(Py_ssize_t k=0;k<s;++k)value+=std::conj(get(U,i*s+k))*c[k*s+j];
                tmp[i*s+j]=value;
            }
            for(Py_ssize_t i=0;i<s;++i)for(Py_ssize_t j=0;j<s;++j){
                Complex value=0.;
                for(Py_ssize_t k=0;k<s;++k)value+=tmp[i*s+k]*get(U,j*s+k);
                density[i*s+j]=value;
            }
            for(Py_ssize_t a=0;a<d;++a){
                Complex value=0.;double trace=0.;
                for(Py_ssize_t i=0;i<s;++i){
                    trace+=get(G,(i*s+i)*d+a).real();
                    for(Py_ssize_t j=0;j<s;++j)value+=density[i*s+j]*get(G,(i*s+j)*d+a);
                }
                const double change=-.5*value.real()+.5*gamma*duration*trace;
                op.output()[b*d+a]=change;bad|=!std::isfinite(change);
            }
            for(Py_ssize_t i=0;i<s;++i)rotated[i]=ce[i]*std::exp(Complex(0.,-duration*E[i]));
            for(Py_ssize_t a=0;a<s;++a){
                Complex value=0.;for(Py_ssize_t i=0;i<s;++i)value+=get(U,a*s+i)*rotated[i];
                put(oz.output(),b*s+a,value);
                bad|=!std::isfinite(value.real())||!std::isfinite(value.imag());
            }
        }
        Py_END_ALLOW_THREADS
        if(bad)throw std::runtime_error("electronic flow produced nonfinite output");
        Py_RETURN_NONE;
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& error){if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,error.what());return nullptr;}
}
PyMethodDef methods[]={{"mapping_flow",mapping_flow,METH_VARARGS,"Batched exact mapping flow and nuclear impulse."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_electronic","Exact electronic mapping contractions",-1,methods};
}
PyMODINIT_FUNC PyInit__electronic(){
    PyObject* result=PyModule_Create(&module);
    if(result&&PyModule_AddIntConstant(result,"MAPPING_FLOW_API",1)<0){Py_DECREF(result);return nullptr;}
    return result;
}
