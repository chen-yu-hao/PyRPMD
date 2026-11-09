#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <memory>
#include <mutex>
#include <exception>
#include <cstring>
#include "thermal.hpp"

namespace {
constexpr const char* capsule_name="pyrpmd.NativeEnsemble.v1";
using thermal::Vec;using thermal::Mat;using thermal::require;
struct Handle {std::unique_ptr<thermal::Engine> engine;std::shared_ptr<thermal::State> rollback;PyObject* owner=nullptr;PyObject* callback=nullptr;std::mutex mutex;~Handle(){engine.reset();Py_XDECREF(callback);Py_XDECREF(owner);}};
struct Lock {std::unique_lock<std::mutex> lock;explicit Lock(Handle& h):lock(h.mutex,std::try_to_lock){require(lock.owns_lock(),"native ensemble is busy in another thread");}};
void destroy(PyObject* cap){auto*p=static_cast<Handle*>(PyCapsule_GetPointer(cap,capsule_name));if(p)delete p;else PyErr_Clear();}
Handle& handle(PyObject* obj){auto*p=static_cast<Handle*>(PyCapsule_GetPointer(obj,capsule_name));require(p!=nullptr,"invalid native ensemble handle");return *p;}
PyObject* field(PyObject* dict,const char* key){require(PyDict_Check(dict),"expected a dict");PyObject* p=PyDict_GetItemString(dict,key);require(p!=nullptr,"missing native configuration/state field");return p;}
double number(PyObject*p){double x=PyFloat_AsDouble(p);require(!PyErr_Occurred()&&std::isfinite(x),"expected finite scalar");return x;}
double num(PyObject*d,const char*k){return number(field(d,k));}
uint64_t integer(PyObject* p){require(PyLong_Check(p)&&!PyBool_Check(p),"expected integer");auto n=PyLong_AsUnsignedLongLong(p);require(!PyErr_Occurred(),"invalid unsigned integer");return n;}
uint64_t integer(PyObject*d,const char*k){return integer(field(d,k));}
bool boolean(PyObject*d,const char*k){PyObject* p=field(d,k);require(PyBool_Check(p),"expected boolean");return p==Py_True;}
Vec vector(PyObject*p){
    Py_buffer v{};if(PyObject_GetBuffer(p,&v,PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)<0)throw std::invalid_argument("expected a contiguous float64 array");
    try {require(v.itemsize==8&&v.format&&std::strcmp(v.format,"d")==0&&v.len%8==0,"expected native-endian float64 array");
        Vec out(static_cast<size_t>(v.len/8));if(v.len)std::memcpy(out.data(),v.buf,static_cast<size_t>(v.len));PyBuffer_Release(&v);return out;
    }catch(...){PyBuffer_Release(&v);throw;}
}
Vec vector(PyObject*d,const char*k){return vector(field(d,k));}
Vec raw(PyObject* p){require(PyBytes_Check(p)&&PyBytes_Size(p)%8==0,"invalid checkpoint byte array");Vec v(static_cast<size_t>(PyBytes_Size(p)/8));if(!v.empty())std::memcpy(v.data(),PyBytes_AsString(p),v.size()*8);return v;}
Vec raw(PyObject*d,const char*k){return raw(field(d,k));}
Mat matrix(const Vec&v){require(v.size()==9,"matrix must have nine entries");Mat a;std::copy(v.begin(),v.end(),a.begin());return a;}
void set(PyObject*d,const char*k,PyObject*v){if(!v)throw std::bad_alloc();int rc=PyDict_SetItemString(d,k,v);Py_DECREF(v);if(rc<0)throw std::bad_alloc();}
PyObject* bytes(const double* p,size_t n){require(n<=static_cast<size_t>(PY_SSIZE_T_MAX)/8,"output byte size overflow");return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(p),static_cast<Py_ssize_t>(8*n));}
PyObject* bytes(const Vec&v){return bytes(v.data(),v.size());}
PyObject* error(){
    try{throw;}catch(const std::bad_alloc&){PyErr_Clear();return PyErr_NoMemory();}
    catch(const std::invalid_argument&e){PyErr_Clear();PyErr_SetString(PyExc_ValueError,e.what());}
    catch(const std::exception&e){PyErr_Clear();PyErr_SetString(PyExc_RuntimeError,e.what());}
    catch(...){PyErr_Clear();PyErr_SetString(PyExc_RuntimeError,"unknown native ensemble error");}return nullptr;
}
template<class F> void unlocked(F&& f){
    std::exception_ptr failure;PyThreadState* state=PyEval_SaveThread();
    try{f();}catch(...){failure=std::current_exception();}
    PyEval_RestoreThread(state);if(failure)std::rethrow_exception(failure);
}
std::string python_failure(){
    PyObject *type=nullptr,*value=nullptr,*trace=nullptr;
    PyErr_Fetch(&type,&value,&trace);PyErr_NormalizeException(&type,&value,&trace);
    PyObject* rendered=value?PyObject_Str(value):nullptr;
    const char* message=rendered?PyUnicode_AsUTF8(rendered):nullptr;
    std::string result=message?message:"Python potential evaluation failed";
    Py_XDECREF(rendered);Py_XDECREF(type);Py_XDECREF(value);Py_XDECREF(trace);PyErr_Clear();
    return result;
}
void python_force(PyObject* callback,const Vec& q,const Mat& cell,Vec& energy,Vec& force,Vec& virial){
    PyGILState_STATE gil=PyGILState_Ensure();
    PyObject *qb=nullptr,*hb=nullptr,*result=nullptr;
    try{
        qb=bytes(q);hb=bytes(cell.data(),cell.size());
        if(!qb||!hb)throw std::runtime_error(python_failure());
        result=PyObject_CallFunctionObjArgs(callback,qb,hb,nullptr);
        if(!result)throw std::runtime_error("Python potential: "+python_failure());
        require(PyTuple_Check(result)&&PyTuple_Size(result)==3,"Python force bridge must return three byte arrays");
        Vec e=raw(PyTuple_GET_ITEM(result,0)),f=raw(PyTuple_GET_ITEM(result,1)),v=raw(PyTuple_GET_ITEM(result,2));
        require(e.size()==energy.size()&&f.size()==force.size()&&v.size()==virial.size(),"Python force bridge output size mismatch");
        energy=std::move(e);force=std::move(f);virial=std::move(v);
        Py_DECREF(result);Py_DECREF(qb);Py_DECREF(hb);PyGILState_Release(gil);
    }catch(...){Py_XDECREF(result);Py_XDECREF(qb);Py_XDECREF(hb);PyErr_Clear();PyGILState_Release(gil);throw;}
}
thermal::Config config(PyObject* dict){
    thermal::Config c;
    if(PyObject* threads=PyDict_GetItemString(dict,"threads")){
        const auto count=integer(threads);
        require(count>0&&count<=std::numeric_limits<int>::max(),"threads is out of range");
        c.threads=static_cast<int>(count);
    }
    c.atoms=integer(dict,"atoms");c.beads=integer(dict,"beads");
    require(c.atoms>0&&c.beads>0&&c.atoms<=static_cast<size_t>(PY_SSIZE_T_MAX)/24/c.beads,"invalid/overflowing dimensions");
    require(c.beads<=static_cast<size_t>(PY_SSIZE_T_MAX)/72,"bead count overflow");
    c.beta=num(dict,"beta");c.dt=num(dict,"dt");c.masses=vector(dict,"masses");
    Vec fixed=vector(dict,"fixed");require(fixed.size()==c.atoms,"fixed-atom mask size mismatch");
    for(double x:fixed){require(x==0||x==1,"invalid fixed-atom mask");c.fixed.push_back(static_cast<uint8_t>(x));}
    auto thermostat=integer(dict,"thermostat"),barostat=integer(dict,"barostat");require(thermostat<=12&&barostat<=4,"unknown thermostat/barostat");
    c.thermostat=static_cast<thermal::Thermostat>(thermostat);c.barostat=static_cast<thermal::Barostat>(barostat);
    c.fuse_modes=PyDict_GetItemString(dict,"fuse_modes")?boolean(dict,"fuse_modes"):true;
    c.middle_npt=PyDict_GetItemString(dict,"middle_npt")?boolean(dict,"middle_npt"):false;
    c.cayley=boolean(dict,"cayley");c.initialize_momenta=boolean(dict,"initialize_momenta");c.has_cell=boolean(dict,"has_cell");
    c.gamma=num(dict,"gamma");c.gamma0=num(dict,"gamma0");c.pile_scale=num(dict,"pile_scale");c.tau=num(dict,"tau");
    c.interval=integer(dict,"interval");c.seed=integer(dict,"seed");c.gle_size=integer(dict,"gle_size");require(c.gle_size<=64,"GLE dimension exceeds 64");
    if(PyObject* rng=PyDict_GetItemString(dict,"rng")){
        require(PyUnicode_Check(rng),"rng must be a string");
        const char* name=PyUnicode_AsUTF8(rng);require(name!=nullptr,"invalid rng string");
        require(std::strcmp(name,"philox-v1")==0,"unsupported native rng");c.philox=true;
    }
    if(c.thermostat==thermal::Thermostat::NHC){c.nhc_length=integer(dict,"nhc_length");c.nhc_substeps=integer(dict,"nhc_substeps");}
    c.gle_T=vector(dict,"gle_T");c.gle_S=vector(dict,"gle_S");c.gle_Croot=vector(dict,"gle_Croot");
    c.pressure=num(dict,"pressure");c.baro_tau=num(dict,"baro_tau");c.baro_gamma=num(dict,"baro_gamma");c.compressibility=num(dict,"compressibility");
    Vec pbc=vector(dict,"pbc");require(pbc.size()==3,"PBC needs three flags");for(int i=0;i<3;++i){require(pbc[i]==0||pbc[i]==1,"invalid PBC flag");c.pbc[i]=static_cast<int32_t>(pbc[i]);}
    Vec mask=vector(dict,"cell_mask");require(mask.size()==9,"cell mask needs nine flags");for(int i=0;i<9;++i){require(mask[i]==0||mask[i]==1,"invalid cell mask");c.cell_mask[i]=static_cast<uint8_t>(mask[i]);}
    c.reference_cell=matrix(vector(dict,"reference_cell"));c.external_stress=matrix(vector(dict,"external_stress"));
    c.potential=static_cast<int>(integer(dict,"potential"));c.address=static_cast<uintptr_t>(integer(dict,"address"));c.context=static_cast<uintptr_t>(integer(dict,"context"));c.parameters=vector(dict,"parameters");
    if(c.potential==6){
        PyObject* callback=field(dict,"python_callback");require(PyCallable_Check(callback),"Python force bridge is not callable");
        c.python_force=[callback](const Vec& q,const Mat& cell,Vec& e,Vec& f,Vec& v){python_force(callback,q,cell,e,f,v);};
    }
    return c;
}
PyObject* create(PyObject*,PyObject*args){
    PyObject *cfg,*q,*p,*cell,*owner;if(!PyArg_ParseTuple(args,"OOOOO",&cfg,&q,&p,&cell,&owner))return nullptr;
    try{
        auto c=config(cfg);Vec qv=vector(q),pv=vector(p);Mat h=matrix(vector(cell));
        auto result=std::make_unique<Handle>();Py_INCREF(owner);result->owner=owner;
        if(c.potential==6){result->callback=field(cfg,"python_callback");Py_INCREF(result->callback);}
        unlocked([&]{result->engine=std::make_unique<thermal::Engine>(std::move(c),std::move(qv),std::move(pv),h);});
        PyObject* cap=PyCapsule_New(result.get(),capsule_name,destroy);if(!cap)return nullptr;result.release();return cap;
    }catch(...){return error();}
}
PyObject* run_impl(PyObject*args,bool keep_prefix){
    PyObject *cap,*steps,*stride,*options=Py_None;int centroids,moments=0;unsigned long long sample_start=0;const char* path="";if(!PyArg_ParseTuple(args,"OOOp|spKO",&cap,&steps,&stride,&centroids,&path,&moments,&sample_start,&options))return nullptr;
    PyObject* dict=nullptr;
    try{auto& h=handle(cap);Lock lock(h);uint64_t n=integer(steps),save=integer(stride);thermal::Output out;
        thermal::CommonObservations common;
        if(options!=Py_None){
            PyObject* mask=field(options,"mask");require(PyBytes_Check(mask),"observation mask must contain binary bytes");
            const auto* data=reinterpret_cast<const uint8_t*>(PyBytes_AsString(mask));
            common.mask.assign(data,data+PyBytes_Size(mask));
            common.record_energy=boolean(options,"record_energy");common.check_energy=boolean(options,"check_energy");
            common.energy_parts=boolean(options,"energy_parts");common.ring_radii=boolean(options,"ring_radii");
            if(PyDict_GetItemString(options,"positions_only"))common.positions_only=boolean(options,"positions_only");
        }
        unlocked([&]{out=h.engine->run(n,save,centroids!=0,path,moments!=0,sample_start,keep_prefix,options==Py_None?nullptr:&common);});
        dict=PyDict_New();if(!dict)throw std::bad_alloc();
        set(dict,"samples",bytes(out.samples));set(dict,"centroids",bytes(out.centroids));set(dict,"frames",PyLong_FromSize_t(out.frames));
        if(options!=Py_None)set(dict,"energy_parts",bytes(out.energy_parts));
        set(dict,"step",PyLong_FromUnsignedLongLong(h.engine->s.step));set(dict,"force_evaluations",PyLong_FromUnsignedLongLong(h.engine->s.evaluations));
        if(keep_prefix){set(dict,"completed",PyLong_FromUnsignedLongLong(out.completed));set(dict,"error",PyUnicode_FromString(out.error.c_str()));}
        return dict;
    }catch(...){Py_XDECREF(dict);return error();}
}
PyObject* run(PyObject*,PyObject*args){return run_impl(args,false);}
PyObject* run_prefix(PyObject*,PyObject*args){return run_impl(args,true);}
PyObject* set_state(PyObject*,PyObject*args){
    PyObject *cap,*qo,*po,*step;if(!PyArg_ParseTuple(args,"OOOO",&cap,&qo,&po,&step))return nullptr;
    try{auto& h=handle(cap);Lock lock(h);Vec q=vector(qo),p=vector(po);auto clock=integer(step);
        unlocked([&]{h.engine->set_state(std::move(q),std::move(p),clock);});Py_RETURN_NONE;
    }catch(...){return error();}
}
PyObject* copy_thermostat(PyObject*,PyObject*args){
    PyObject *cap,*source,*step=Py_None;
    if(!PyArg_ParseTuple(args,"OO|O",&cap,&source,&step))return nullptr;
    try{auto& h=handle(cap);auto& other=handle(source);bool has_step=step!=Py_None;
        uint64_t clock=has_step?integer(step):0;
        Lock target_lock(h);
        if(&h==&other){h.engine->copy_thermostat_from(*other.engine,has_step,clock);}
        else{Lock source_lock(other);h.engine->copy_thermostat_from(*other.engine,has_step,clock);}
        Py_RETURN_NONE;
    }catch(...){return error();}
}
PyObject* apply_thermostat(PyObject*,PyObject*args){
    PyObject *cap,*qo,*po,*step=Py_None;double dt;
    if(!PyArg_ParseTuple(args,"OOOd|O",&cap,&qo,&po,&dt,&step))return nullptr;
    try{auto& h=handle(cap);Lock lock(h);Vec q=vector(qo),p=vector(po),out;
        bool has_step=step!=Py_None;uint64_t clock=has_step?integer(step):0;
        unlocked([&]{out=h.engine->apply_thermostat(std::move(q),std::move(p),dt,has_step,clock);});
        return bytes(out);
    }catch(...){return error();}
}
PyObject* state(PyObject*,PyObject*args){
    PyObject* cap;if(!PyArg_ParseTuple(args,"O",&cap))return nullptr;PyObject* dict=nullptr;
    try{auto&h=handle(cap);Lock lock(h);const auto&s=h.engine->s;dict=PyDict_New();if(!dict)throw std::bad_alloc();
        set(dict,"q",bytes(s.q));set(dict,"p",bytes(s.p));set(dict,"aux",bytes(s.aux));set(dict,"energy",bytes(s.energy));set(dict,"force",bytes(s.force));set(dict,"virial",bytes(s.virial));
        set(dict,"cell",bytes(s.cell.data(),9));set(dict,"cell_p",bytes(s.cell_p.data(),9));
        set(dict,"step",PyLong_FromUnsignedLongLong(s.step));set(dict,"force_evaluations",PyLong_FromUnsignedLongLong(s.evaluations));
        std::string rng=h.engine->rng_state();set(dict,"rng",PyUnicode_FromStringAndSize(rng.data(),static_cast<Py_ssize_t>(rng.size())));return dict;
    }catch(...){Py_XDECREF(dict);return error();}
}
constexpr const char* rollback_name="pyrpmd.NativeEnsembleRollback.v1";
struct Rollback {
    std::shared_ptr<const thermal::State> state;PyObject* owner;
    Rollback(std::shared_ptr<const thermal::State> value,PyObject* cap):state(std::move(value)),owner(cap){Py_INCREF(owner);}
    ~Rollback(){Py_DECREF(owner);}
};
void destroy_rollback(PyObject* cap){
    auto* token=static_cast<Rollback*>(PyCapsule_GetPointer(cap,rollback_name));
    if(token)delete token;else PyErr_Clear();
}
PyObject* snapshot(PyObject*,PyObject*args){
    PyObject* cap;if(!PyArg_ParseTuple(args,"O",&cap))return nullptr;
    try{auto& h=handle(cap);Lock lock(h);
        // Reuse vector capacity only when no outstanding token owns it.
        if(!h.rollback||!h.rollback.unique())h.rollback=std::make_shared<thermal::State>();
        *h.rollback=h.engine->s;
        auto token=std::make_unique<Rollback>(h.rollback,cap);
        PyObject* result=PyCapsule_New(token.get(),rollback_name,destroy_rollback);
        if(!result)return nullptr;token.release();return result;
    }catch(...){return error();}
}
PyObject* restore_snapshot(PyObject*,PyObject*args){
    PyObject *cap,*saved;if(!PyArg_ParseTuple(args,"OO",&cap,&saved))return nullptr;
    try{auto& h=handle(cap);Lock lock(h);
        auto* token=static_cast<Rollback*>(PyCapsule_GetPointer(saved,rollback_name));
        require(token&&token->owner==cap,"rollback token belongs to another native ensemble");
        h.engine->restore(*token->state);Py_RETURN_NONE;
    }catch(...){return error();}
}
PyObject* physical_state(PyObject*,PyObject*args){
    PyObject *cap,*q=Py_None,*p=Py_None;if(!PyArg_ParseTuple(args,"O|OO",&cap,&q,&p))return nullptr;
    PyObject* dict=nullptr;Py_buffer qview{},pview{};
    try{auto&h=handle(cap);Lock lock(h);const auto&s=h.engine->s;
        if(q!=Py_None||p!=Py_None){
            require(PyObject_GetBuffer(q,&qview,PyBUF_C_CONTIGUOUS|PyBUF_FORMAT|PyBUF_WRITABLE)==0,"expected writable q output");
            require(PyObject_GetBuffer(p,&pview,PyBUF_C_CONTIGUOUS|PyBUF_FORMAT|PyBUF_WRITABLE)==0,"expected writable p output");
            for(auto* view:{&qview,&pview})require(view->itemsize==8&&view->format&&std::strcmp(view->format,"d")==0&&view->len==Py_ssize_t(s.q.size()*8),"invalid physical output buffer");
        }
        dict=PyDict_New();if(!dict)throw std::bad_alloc();
        set(dict,"energy",bytes(s.energy));set(dict,"cell",bytes(s.cell.data(),9));
        set(dict,"step",PyLong_FromUnsignedLongLong(s.step));
        if(qview.obj){std::memcpy(qview.buf,s.q.data(),s.q.size()*8);std::memcpy(pview.buf,s.p.data(),s.p.size()*8);}
        if(qview.obj)PyBuffer_Release(&qview);if(pview.obj)PyBuffer_Release(&pview);
        return dict;
    }catch(...){if(qview.obj)PyBuffer_Release(&qview);if(pview.obj)PyBuffer_Release(&pview);Py_XDECREF(dict);return error();}
}
PyObject* initial_energy_terms(PyObject*,PyObject*args){
    PyObject *cap,*objects[5];
    if(!PyArg_ParseTuple(args,"OOOOOO",&cap,&objects[0],&objects[1],&objects[2],&objects[3],&objects[4]))return nullptr;
    Py_buffer views[5]{};
    try{auto& h=handle(cap);Lock lock(h);const auto& c=h.engine->c;
        const size_t d=3*c.atoms,l=d*c.beads;
        for(size_t i=0;i<5;++i){
            const int flags=PyBUF_C_CONTIGUOUS|PyBUF_FORMAT|(i>=3?PyBUF_WRITABLE:0);
            require(PyObject_GetBuffer(objects[i],&views[i],flags)==0,"expected contiguous initial-energy buffers");
            require(views[i].itemsize==8&&views[i].format&&std::strcmp(views[i].format,"d")==0
                    &&views[i].len==Py_ssize_t((i==2?c.atoms:l)*8),"invalid initial-energy buffer");
        }
        const auto* q=static_cast<const double*>(views[0].buf);
        const auto* p=static_cast<const double*>(views[1].buf);
        const auto* masses=static_cast<const double*>(views[2].buf);
        auto* kinetic=static_cast<double*>(views[3].buf);
        auto* spring=static_cast<double*>(views[4].buf);
        // Materialize the original scalar terms in one pass. NumPy still owns
        // the reduction order; no dot-product contraction or reassociation.
        unlocked([&]{
            for(size_t k=0;k<c.beads;++k){const size_t next=(k+1)%c.beads;
#ifdef _OPENMP
#pragma omp simd
#endif
                for(size_t atom=0;atom<c.atoms;++atom)for(size_t axis=0;axis<3;++axis){
                    const size_t j=3*atom+axis,i=k*d+j;
                    const double delta=q[i]-q[next*d+j];
                    kinetic[i]=p[i]*p[i]/(2*masses[atom]);
                    spring[i]=(masses[atom]*delta)*delta;
                }
            }
        });
        for(auto& view:views)PyBuffer_Release(&view);
        return bytes(h.engine->s.energy);
    }catch(...){for(auto& view:views)if(view.obj)PyBuffer_Release(&view);return error();}
}
PyObject* restore(PyObject*,PyObject*args){
    PyObject *cap,*dict;if(!PyArg_ParseTuple(args,"OO",&cap,&dict))return nullptr;
    try{auto&h=handle(cap);Lock lock(h);thermal::State s;
        s.q=raw(dict,"q");s.p=raw(dict,"p");s.aux=raw(dict,"aux");s.energy=raw(dict,"energy");s.force=raw(dict,"force");s.virial=raw(dict,"virial");
        s.cell=matrix(raw(dict,"cell"));s.cell_p=matrix(raw(dict,"cell_p"));s.step=integer(dict,"step");s.evaluations=integer(dict,"force_evaluations");
        PyObject* obj=field(dict,"rng");require(PyUnicode_Check(obj),"RNG state must be text");const char* text=PyUnicode_AsUTF8(obj);require(text!=nullptr,"invalid RNG text");h.engine->set_rng(text,s);
        h.engine->restore(std::move(s));Py_RETURN_NONE;
    }catch(...){return error();}
}
PyObject* has_openmp(PyObject*,PyObject*){
#ifdef _OPENMP
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}
PyMethodDef methods[]={
    {"has_openmp",has_openmp,METH_NOARGS,"Whether this thermal extension supports OpenMP."},
    {"create",create,METH_VARARGS,"Create a persistent native ensemble and evaluate the initial PES."},
    {"run",run,METH_VARARGS,"Advance an entire operator block; native PES modes need no Python callbacks."},
    {"run_prefix",run_prefix,METH_VARARGS,"Advance an operator block retaining only fully accepted steps and their observations."},
    {"copy_thermostat",copy_thermostat,METH_VARARGS,"Copy compatible bath/RNG state between executors without changing physical state."},
    {"set_state",set_state,METH_VARARGS,"Import external q/p/step and transactionally refresh forces, retaining bath state."},
    {"apply_thermostat",apply_thermostat,METH_VARARGS,"Apply only the bath to external q/p, retaining transactional RNG and auxiliary state."},
    {"snapshot",snapshot,METH_VARARGS,"Capture a private binary rollback token, reusing unowned storage."},
    {"restore_snapshot",restore_snapshot,METH_VARARGS,"Restore a token owned by this executor."},
    {"initial_energy_terms",initial_energy_terms,METH_VARARGS,"Prepare unchanged energy terms for NumPy reduction."},
    {"physical_state",physical_state,METH_VARARGS,"Copy physical arrays without serializing a checkpoint."},
    {"state",state,METH_VARARGS,"Copy all state, including cached forces, auxiliary momenta and Gaussian RNG cache."},
    {"restore",restore,METH_VARARGS,"Validate and atomically restore a complete native state."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_thermal","C++17 persistent NVT/NPT trajectories.",-1,methods,nullptr,nullptr,nullptr,nullptr};
}
#define THERMAL_STRING_INNER(x) #x
#define THERMAL_STRING(x) THERMAL_STRING_INNER(x)
PyMODINIT_FUNC PyInit__thermal(){
    PyObject* m=PyModule_Create(&module);if(!m)return nullptr;
#if defined(__GLIBCXX__)
    const char* abi="mt19937_64-normal-v1-libstdc++-" THERMAL_STRING(__GLIBCXX__);
#elif defined(_LIBCPP_VERSION)
    const char* abi="mt19937_64-normal-v1-libc++-" THERMAL_STRING(_LIBCPP_VERSION);
#elif defined(_MSVC_STL_VERSION)
    const char* abi="mt19937_64-normal-v1-msvc-" THERMAL_STRING(_MSVC_STL_VERSION);
#else
    const char* abi="mt19937_64-normal-v1-unknown-standard-library";
#endif
    if(PyModule_AddStringConstant(m,"rng_abi",abi)<0
            ||PyModule_AddStringConstant(m,"philox_abi","philox-v1-salmon-sc11")<0
            ||PyModule_AddIntConstant(m,"THERMAL_KERNEL_API",3)<0
            ||PyModule_AddIntConstant(m,"COMMON_POSITION_OBSERVATIONS_API",1)<0
            ||PyModule_AddIntConstant(m,"COMPOSABLE_OPERATORS_API",1)<0
            ||PyModule_AddIntConstant(m,"MIDDLE_BZP_API",1)<0){Py_DECREF(m);return nullptr;}return m;
}
