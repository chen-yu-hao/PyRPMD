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
struct Handle {std::unique_ptr<thermal::Engine> engine;PyObject* owner=nullptr;std::mutex mutex;~Handle(){engine.reset();Py_XDECREF(owner);}};
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
thermal::Config config(PyObject* dict){
    thermal::Config c;
    c.atoms=integer(dict,"atoms");c.beads=integer(dict,"beads");
    require(c.atoms>0&&c.beads>0&&c.atoms<=static_cast<size_t>(PY_SSIZE_T_MAX)/24/c.beads,"invalid/overflowing dimensions");
    require(c.beads<=static_cast<size_t>(PY_SSIZE_T_MAX)/72,"bead count overflow");
    c.beta=num(dict,"beta");c.dt=num(dict,"dt");c.masses=vector(dict,"masses");
    Vec fixed=vector(dict,"fixed");require(fixed.size()==c.atoms,"fixed-atom mask size mismatch");
    for(double x:fixed){require(x==0||x==1,"invalid fixed-atom mask");c.fixed.push_back(static_cast<uint8_t>(x));}
    auto thermostat=integer(dict,"thermostat"),barostat=integer(dict,"barostat");require(thermostat<=11&&barostat<=4,"unknown thermostat/barostat");
    c.thermostat=static_cast<thermal::Thermostat>(thermostat);c.barostat=static_cast<thermal::Barostat>(barostat);
    c.cayley=boolean(dict,"cayley");c.initialize_momenta=boolean(dict,"initialize_momenta");c.has_cell=boolean(dict,"has_cell");
    c.gamma=num(dict,"gamma");c.gamma0=num(dict,"gamma0");c.pile_scale=num(dict,"pile_scale");c.tau=num(dict,"tau");
    c.interval=integer(dict,"interval");c.seed=integer(dict,"seed");c.gle_size=integer(dict,"gle_size");require(c.gle_size<=64,"GLE dimension exceeds 64");
    c.gle_T=vector(dict,"gle_T");c.gle_S=vector(dict,"gle_S");c.gle_Croot=vector(dict,"gle_Croot");
    c.pressure=num(dict,"pressure");c.baro_tau=num(dict,"baro_tau");c.baro_gamma=num(dict,"baro_gamma");c.compressibility=num(dict,"compressibility");
    Vec pbc=vector(dict,"pbc");require(pbc.size()==3,"PBC needs three flags");for(int i=0;i<3;++i){require(pbc[i]==0||pbc[i]==1,"invalid PBC flag");c.pbc[i]=static_cast<int32_t>(pbc[i]);}
    Vec mask=vector(dict,"cell_mask");require(mask.size()==9,"cell mask needs nine flags");for(int i=0;i<9;++i){require(mask[i]==0||mask[i]==1,"invalid cell mask");c.cell_mask[i]=static_cast<uint8_t>(mask[i]);}
    c.reference_cell=matrix(vector(dict,"reference_cell"));c.external_stress=matrix(vector(dict,"external_stress"));
    c.potential=static_cast<int>(integer(dict,"potential"));c.address=static_cast<uintptr_t>(integer(dict,"address"));c.context=static_cast<uintptr_t>(integer(dict,"context"));c.parameters=vector(dict,"parameters");return c;
}
PyObject* create(PyObject*,PyObject*args){
    PyObject *cfg,*q,*p,*cell,*owner;if(!PyArg_ParseTuple(args,"OOOOO",&cfg,&q,&p,&cell,&owner))return nullptr;
    try{
        auto c=config(cfg);Vec qv=vector(q),pv=vector(p);Mat h=matrix(vector(cell));
        auto result=std::make_unique<Handle>();Py_INCREF(owner);result->owner=owner;
        unlocked([&]{result->engine=std::make_unique<thermal::Engine>(std::move(c),std::move(qv),std::move(pv),h);});
        PyObject* cap=PyCapsule_New(result.get(),capsule_name,destroy);if(!cap)return nullptr;result.release();return cap;
    }catch(...){return error();}
}
PyObject* run(PyObject*,PyObject*args){
    PyObject *cap,*steps,*stride;int centroids;const char* path="";if(!PyArg_ParseTuple(args,"OOOp|s",&cap,&steps,&stride,&centroids,&path))return nullptr;
    PyObject* dict=nullptr;
    try{auto& h=handle(cap);Lock lock(h);uint64_t n=integer(steps),save=integer(stride);thermal::Output out;
        unlocked([&]{out=h.engine->run(n,save,centroids!=0,path);});
        dict=PyDict_New();if(!dict)throw std::bad_alloc();
        set(dict,"samples",bytes(out.samples));set(dict,"centroids",bytes(out.centroids));set(dict,"frames",PyLong_FromSize_t(out.frames));
        set(dict,"step",PyLong_FromUnsignedLongLong(h.engine->s.step));set(dict,"force_evaluations",PyLong_FromUnsignedLongLong(h.engine->s.evaluations));return dict;
    }catch(...){Py_XDECREF(dict);return error();}
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
PyObject* restore(PyObject*,PyObject*args){
    PyObject *cap,*dict;if(!PyArg_ParseTuple(args,"OO",&cap,&dict))return nullptr;
    try{auto&h=handle(cap);Lock lock(h);thermal::State s;
        s.q=raw(dict,"q");s.p=raw(dict,"p");s.aux=raw(dict,"aux");s.energy=raw(dict,"energy");s.force=raw(dict,"force");s.virial=raw(dict,"virial");
        s.cell=matrix(raw(dict,"cell"));s.cell_p=matrix(raw(dict,"cell_p"));s.step=integer(dict,"step");s.evaluations=integer(dict,"force_evaluations");
        PyObject* obj=field(dict,"rng");require(PyUnicode_Check(obj),"RNG state must be text");const char* text=PyUnicode_AsUTF8(obj);require(text!=nullptr,"invalid RNG text");h.engine->set_rng(text,s);
        h.engine->restore(std::move(s));Py_RETURN_NONE;
    }catch(...){return error();}
}
PyMethodDef methods[]={
    {"create",create,METH_VARARGS,"Create a persistent native ensemble and evaluate the initial PES."},
    {"run",run,METH_VARARGS,"Advance an entire trajectory block without Python callbacks or GIL acquisition."},
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
    if(PyModule_AddStringConstant(m,"rng_abi",abi)<0){Py_DECREF(m);return nullptr;}return m;
}
