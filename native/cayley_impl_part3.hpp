
static const char* CAPSULE="pyrpmd.Cayley.v1";
static void destroy(PyObject* capsule){
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(capsule,CAPSULE));
    if(c) delete c; else PyErr_Clear();
}
static PyObject* create(PyObject*,PyObject* args){
    PyObject* masses; double beta,dt; Py_ssize_t beads; int threads; const char* scheme="cayley-bcb";
    if(!PyArg_ParseTuple(args,"Odndi|s",&masses,&beta,&beads,&dt,&threads,&scheme))return nullptr;
    const std::string name(scheme);
    int mode=name=="cayley-bcb" ? 0 : name=="normal-mode-bab" ? 1 : name=="velocity-verlet" ? 2 : -1;
    if(mode<0){PyErr_SetString(PyExc_ValueError,"unknown native integration scheme");return nullptr;}
    if(!(beta>0.)||!std::isfinite(beta)||!std::isfinite(dt)||beads<1||threads<1){PyErr_SetString(PyExc_ValueError,"invalid Cayley parameters");return nullptr;}
    #ifndef _OPENMP
    if(threads>1){PyErr_SetString(PyExc_RuntimeError,"native kernel has no OpenMP support");return nullptr;}
    #endif
    PyObject* seq=PySequence_Fast(masses,"masses must be a sequence");
    if(!seq)return nullptr;
    const Py_ssize_t n=PySequence_Fast_GET_SIZE(seq);
    const Py_ssize_t width=3*static_cast<Py_ssize_t>(sizeof(double));
    if(n<1 || n>PY_SSIZE_T_MAX/(6*width) || beads>PY_SSIZE_T_MAX/(6*width*n)){Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"invalid dimensions");return nullptr;}
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
    const double maximum=2.*omega*std::sin(std::acos(-1.)*(beads/2)/beads);
    if(mode==2 && std::abs(dt)*maximum>=2.){PyErr_SetString(PyExc_ValueError,"velocity-Verlet ring stability requires abs(dt)*max(omega_k) < 2");return nullptr;}
    try {return PyCapsule_New(new Cayley(std::move(m),beta,beads,dt,threads,mode),CAPSULE,destroy);}
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
        std::vector<double> work(c->workspace_size());
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
static PyObject* integrator_schemes(PyObject*,PyObject*){
    return Py_BuildValue("(sss)","cayley-bcb","normal-mode-bab","velocity-verlet");
}
#include "fixed_atoms.hpp"
#include "fused.hpp"
#include "propagation.h"

static PyObject* sample_free(PyObject*,PyObject* args){
    PyObject *capsule,*qo,*po,*no,*co,*ao,*fo;
    if(!PyArg_ParseTuple(args,"OOOOOOO",&capsule,&qo,&po,&no,&co,&ao,&fo))return nullptr;
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(capsule,CAPSULE));if(!c)return nullptr;
    if(c->scheme==2){PyErr_SetString(PyExc_ValueError,"PILE-L requires Cayley or exact normal modes");return nullptr;}
    try{
        const Py_ssize_t bytes=c->beads*c->nd*sizeof(double),modes=c->beads*sizeof(double);
        fused_blocks::Buffer q(qo,bytes,"d",true),p(po,bytes,"d",true),noise(no,bytes,"d",false),decay(co,modes,"d",false),amplitude(ao,modes,"d",false);
        Py_buffer fixed;
        if(PyObject_GetBuffer(fo,&fixed,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS)<0)return nullptr;
        struct FixedRelease{Py_buffer* view;~FixedRelease(){PyBuffer_Release(view);}} release_fixed{&fixed};
        if(fixed.len!=c->nd/3 || fixed.itemsize!=1 || !fixed.format || std::string(fixed.format)!="B")
            throw std::invalid_argument("fixed mask must be a uint8 array with one entry per atom");
        const auto overlaps=[](const void* a,Py_ssize_t an,const void* b,Py_ssize_t bn){
            const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
            return x<=y ? y-x<static_cast<std::uintptr_t>(an) : x-y<static_cast<std::uintptr_t>(bn);
        };
        for(const auto* destination:{q.doubles(),p.doubles()}){
            if(overlaps(destination,bytes,noise.doubles(),bytes)||overlaps(destination,bytes,decay.doubles(),modes)
                    ||overlaps(destination,bytes,amplitude.doubles(),modes)||overlaps(destination,bytes,fixed.buf,fixed.len))
                throw std::invalid_argument("PILE-L input and output buffers must not alias");
        }
        if(overlaps(q.doubles(),bytes,p.doubles(),bytes))throw std::invalid_argument("q and p must not alias");
        const auto* mask=static_cast<const unsigned char*>(fixed.buf);
        for(Py_ssize_t j=0;j<c->nd/3;++j)if(mask[j]>1)throw std::invalid_argument("fixed mask entries must be 0 or 1");
        for(Py_ssize_t k=0;k<c->beads;++k){
            if(!std::isfinite(decay.doubles()[k])||decay.doubles()[k]<0.||decay.doubles()[k]>1.
                    ||!std::isfinite(amplitude.doubles()[k])||amplitude.doubles()[k]<0.)
                throw std::invalid_argument("invalid PILE-L damping or noise amplitude");
        }
        std::vector<double> work(6*c->beads*c->nd);
        {
            fused_blocks::ReleaseGIL release;
            c->sample_free(q.doubles(),p.doubles(),noise.doubles(),decay.doubles(),amplitude.doubles(),mask,work.data());
        }
        Py_RETURN_NONE;
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}

static PyMethodDef methods[]={
 {"builtin_pes",builtin_pes::create,METH_VARARGS,"Create an immutable built-in C ABI v1 PES context and export its addresses."},
 {"builtin_lj",builtin_pes::create_lj,METH_VARARGS,"Create a periodic Lennard-Jones C ABI v2 context and export its addresses."},
 {"bind_pes",fused::bind_pes,METH_VARARGS,"Bind a trusted native PES; retain its lifetime owner."},
 {"run_block",fused::run_block,METH_VARARGS,"Transactional fused RPMD with the selected native integrator."},
 {"advance",fused_blocks::advance,METH_VARARGS,"RPMD blocks with the selected integrator, native RATTLE and valid-prefix failure results."},
 {"create",create,METH_VARARGS,"Create an immutable independent ring-propagator context."},
 {"free_step",free_step,METH_VARARGS,"Update q,p using the selected full-ring free step."},
 {"sample_free",sample_free,METH_VARARGS,"Fused half-free/PILE-L/half-free with externally supplied real-mode noise."},
 {"kick",kick,METH_VARARGS,"p += h*physical_force."},
 {"has_openmp",has_openmp,METH_NOARGS,"Whether this build supports OpenMP."},
 {"integrator_schemes",integrator_schemes,METH_NOARGS,"Canonical real-time schemes supported by all native paths."},
 {nullptr,nullptr,0,nullptr}
};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_native","PyRPMD C++17/OpenMP kernels",-1,methods};
PyMODINIT_FUNC PyInit__native(){
    PyObject* result=PyModule_Create(&module);
    if(!result)return nullptr;
    if(PyModule_AddIntConstant(result,"DRPMD_OUTCOMES_API",1)<0
            || PyModule_AddIntConstant(result,"BLOCK_ENERGY_CHECK_API",1)<0
            || PyModule_AddIntConstant(result,"BLOCK_DIAGNOSTICS_API",1)<0
            || PyModule_AddIntConstant(result,"SAMPLING_FREE_API",1)<0
            || PyModule_AddIntConstant(result,"FIXED_ATOMS_API",1)<0){
        Py_DECREF(result);return nullptr;
    }
    return result;
}
