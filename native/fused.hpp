/* Fused BCB propagation. Included by cayley.cpp, reusing its validated Cayley
 * factorization. No Python calls, allocation or GIL acquisition in step loops.
 * CV/flux conventions follow cv.py / bias.py and their retained MIT notices. */
#include "include/pyrpmd/pes_api_v1.h"
#include "cv_core.hpp"
#include <array>
#include <memory>
#include <sstream>
#include <exception>

namespace fused {
using native_cv::numeric::Failure;
using native_cv::numeric::ensure;
using native_cv::numeric::field;
using native_cv::numeric::number;
using native_cv::numeric::numbers;
using CV=native_cv::Coordinate<true>;
using Bias=native_cv::Bias<true>;

static constexpr const char* PES_CAPSULE="pyrpmd.NativePES.v1";
using CABIFn=pyrpmd_rc_v1(*)(pyrpmd_pes_context_v1*,const pyrpmd_pes_input_v1*,pyrpmd_pes_output_v1*,pyrpmd_error_v1*);
using FortranBatchFn=void(*)(int32_t,const double*,double*,double*,int32_t*);
struct Binding {
    int kind; uintptr_t address,context; Py_ssize_t natoms; PyObject* owner;
    ~Binding(){Py_XDECREF(owner);} // Destructor invoked with GIL held.
};
static void destroy_binding(PyObject* capsule){
    auto* b=static_cast<Binding*>(PyCapsule_GetPointer(capsule,PES_CAPSULE));if(b)delete b;else PyErr_Clear();
}
static PyObject* bind_pes(PyObject*,PyObject* args){
    int kind; unsigned long long addr,context; Py_ssize_t n; PyObject* owner;
    if(!PyArg_ParseTuple(args,"iKKnO",&kind,&addr,&context,&n,&owner))return nullptr;
    if((kind!=0&&kind!=1)||!addr||n<1||n>PY_SSIZE_T_MAX/3){
        PyErr_SetString(PyExc_ValueError,"invalid native PES binding");return nullptr;}
    auto* b=new(std::nothrow)Binding{kind,static_cast<uintptr_t>(addr),static_cast<uintptr_t>(context),n,nullptr};
    if(!b)return PyErr_NoMemory();
    Py_INCREF(owner);b->owner=owner;
    PyObject* cap=PyCapsule_New(b,PES_CAPSULE,destroy_binding);if(!cap)delete b;return cap;
}
struct PES {
    Binding& binding; Py_ssize_t P,d;
    std::vector<double> energy,force; std::vector<int32_t> status;
    PES(Binding& b,Py_ssize_t p,Py_ssize_t dims):binding(b),P(p),d(dims),energy(p),force(p*dims),status(p) {
        ensure(binding.natoms*3==d,"PES atom count differs from integrator");
        ensure(binding.kind!=1||P<=INT32_MAX,"Fortran batch exceeds int32");
    }
    void evaluate(const double* q){
        std::fill(energy.begin(),energy.end(),std::numeric_limits<double>::quiet_NaN());
        std::fill(force.begin(),force.end(),std::numeric_limits<double>::quiet_NaN());
        std::fill(status.begin(),status.end(),-1);
        if(binding.kind==1)reinterpret_cast<FortranBatchFn>(binding.address)(static_cast<int32_t>(P),q,energy.data(),force.data(),status.data());
        else{
            pyrpmd_pes_input_v1 in{1,sizeof(pyrpmd_pes_input_v1),1,1,P,d/3,q,P*d};
            pyrpmd_pes_output_v1 out{1,sizeof(pyrpmd_pes_output_v1),energy.data(),P,force.data(),P*d,status.data(),P,nullptr,0,0};
            char msg[512]={0};pyrpmd_error_v1 err{msg,512,0};
            int rc=reinterpret_cast<CABIFn>(binding.address)(reinterpret_cast<pyrpmd_pes_context_v1*>(binding.context),&in,&out,&err);
            msg[511]=0;
            if(rc)throw Failure(std::string("native PES global error ")+std::to_string(rc)+": "+msg);
        }
        for(Py_ssize_t b=0;b<P;++b)if(status[b]!=0||!std::isfinite(energy[b]))throw Failure("native PES frame "+std::to_string(b)+" failed, status="+std::to_string(status[b]));
        for(double f:force)ensure(std::isfinite(f),"native PES returned nonfinite force");
    }
};
struct Buffer {
    Py_buffer view{};bool acquired=false;
    Buffer(PyObject* obj,Py_ssize_t count){ensure(count>=0&&count<=PY_SSIZE_T_MAX/static_cast<Py_ssize_t>(sizeof(double)),"buffer size overflow");
        if(!buffer(obj,&view,count*sizeof(double),true))throw Failure("invalid native output/state buffer");
        acquired=true;}
    ~Buffer(){if(acquired)PyBuffer_Release(&view);}
    double* data(){return static_cast<double*>(view.buf);}
};
inline void centroid(const double* q,double* out,Py_ssize_t P,Py_ssize_t d){
    std::fill(out,out+d,0.);for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t j=0;j<d;++j)out[j]+=q[b*d+j];
    for(Py_ssize_t j=0;j<d;++j)out[j]/=static_cast<double>(P);
}
struct Fragment { std::vector<size_t> atoms; std::array<double,3> ref; double mass=0.; };
static PyObject* run_block(PyObject*,PyObject* args){
    PyObject *context,*binding,*qo,*po,*eo,*co,*vo,*spec;
    Py_ssize_t steps,stride;
    if(!PyArg_ParseTuple(args,"OOOOnnOOOO",&context,&binding,&qo,&po,&steps,&stride,&eo,&co,&vo,&spec))return nullptr;
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(context,CAPSULE));if(!c)return nullptr;
    auto* binding_ptr=static_cast<Binding*>(PyCapsule_GetPointer(binding,PES_CAPSULE));if(!binding_ptr)return nullptr;
    try{
        ensure(steps>0&&stride>0,"steps and stride must be positive");
        const Py_ssize_t P=c->beads,d=c->nd, size=P*d, rows=1+(steps-1)/stride;
        NativeFixedAtoms fixed(spec,d/3);
        ensure(rows<=PY_SSIZE_T_MAX/d,"observation size overflow");
        PyObject* obs=field(spec,"observers");ensure(PyList_Check(obs),"observers must be a list");
        const Py_ssize_t ncv=PyList_Size(obs);ensure(ncv==0||rows<=PY_SSIZE_T_MAX/ncv,"CV output size overflow");
        std::vector<CV> cvs;for(Py_ssize_t i=0;i<ncv;++i)cvs.emplace_back(PyList_GetItem(obs,i),d/3);
        std::vector<Bias> biases;PyObject* bs=field(spec,"biases");ensure(PyList_Check(bs),"biases must be a list");
        for(Py_ssize_t i=0;i<PyList_Size(bs);++i)biases.emplace_back(PyList_GetItem(bs,i),d/3);
        std::vector<Fragment> fragments;PyObject* fs=field(spec,"fragments");ensure(PyList_Check(fs),"fragments must be a list");
        for(Py_ssize_t i=0;i<PyList_Size(fs);++i){PyObject* f=PyList_GetItem(fs,i);Fragment item;
            PyObject* indices=field(f,"atoms");ensure(PyList_Check(indices)&&PyList_Size(indices)>0,"empty fragment");
            for(Py_ssize_t j=0;j<PyList_Size(indices);++j){long a=PyLong_AsLong(PyList_GetItem(indices,j));
                ensure(!PyErr_Occurred()&&a>=0&&a<d/3,"fragment index out of range");
                ensure(!fixed.mask[a],"fragment COM constraints cannot include fixed atoms");
                item.atoms.push_back(a);item.mass+=c->mass[a];}
            auto r=numbers(field(f,"reference"),3);std::copy(r.begin(),r.end(),item.ref.begin());fragments.push_back(std::move(item));}
        Buffer qb(qo,size),pb(po,size),eb(eo,rows);
        std::unique_ptr<Buffer> cb,vb;
        if(co!=Py_None)cb=std::make_unique<Buffer>(co,rows*d);
        if(vo!=Py_None)vb=std::make_unique<Buffer>(vo,rows*ncv);
        ensure(ncv==0||vb!=nullptr,"CV output required");
        std::vector<Buffer*> arrays{&qb,&pb,&eb};if(cb)arrays.push_back(cb.get());if(vb)arrays.push_back(vb.get());
        for(size_t i=0;i<arrays.size();++i)for(size_t j=i+1;j<arrays.size();++j){
            uintptr_t a=reinterpret_cast<uintptr_t>(arrays[i]->view.buf),b=reinterpret_cast<uintptr_t>(arrays[j]->view.buf);
            size_t an=arrays[i]->view.len,bn=arrays[j]->view.len;
            ensure(!an||!bn||a+an<=b||b+bn<=a,"native state and output buffers must not alias");
        }
        // Work buffers are private to this call; no caller state is mutated until
        // all PES calls and observations succeed. Reused for every timestep.
        std::vector<double> q(qb.data(),qb.data()+size),p(pb.data(),pb.data()+size),work(c->workspace_size()),center(d);
        for(double v:q)ensure(std::isfinite(v),"nonfinite q");
        for(double v:p)ensure(std::isfinite(v),"nonfinite p");
        fixed.validate(q.data(),p.data(),P);
        PES pes(*binding_ptr,P,d);
        double initial=0.,minimum=std::numeric_limits<double>::infinity(),maximum=-minimum;
        double clock=number(field(spec,"time"));
        // Optional block-local endpoint-force reuse, as in advance(): each step
        // starts at exactly the coordinates of the previous endpoint call.
        // Fragment projection moves q after that call and always re-evaluates.
        PyObject* reuse_obj=PyDict_GetItemString(spec,"reuse_forces");
        const int reuse_flag=reuse_obj ? PyObject_IsTrue(reuse_obj) : 0;
        ensure(reuse_flag>=0,"invalid reuse_forces option");
        const bool reuse=reuse_flag && fragments.empty();
        Py_ssize_t step=0,sample=0;std::exception_ptr failure;
        auto project=[&](){for(const auto& f:fragments){double com[3]={0,0,0},vel[3]={0,0,0};
            for(Py_ssize_t b=0;b<P;++b)for(size_t a:f.atoms)for(int k=0;k<3;++k){com[k]+=q[b*d+3*a+k]*c->mass[a];vel[k]+=p[b*d+3*a+k];}
            for(int k=0;k<3;++k){com[k]=com[k]/(P*f.mass)-f.ref[k];vel[k]/=P*f.mass;}
            for(Py_ssize_t b=0;b<P;++b)for(size_t a:f.atoms)for(int k=0;k<3;++k){q[b*d+3*a+k]-=com[k];p[b*d+3*a+k]-=c->mass[a]*vel[k];}}
            fixed.project(q.data(),p.data(),P);};
        auto force=[&](){pes.evaluate(q.data());double bias_energy=0.;
            if(!biases.empty()){centroid(q.data(),center.data(),P,d);for(auto& b:biases)bias_energy+=b.apply(center.data(),pes.force.data(),P,c->mass);}
            for(double v:pes.force)ensure(std::isfinite(v),"nonfinite biased force");
            fixed.clear_force(pes.force.data(),P);
            return bias_energy;};
        auto energy=[&](double bias_energy){double e=bias_energy;for(double v:pes.energy)e+=v;
            for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t j=0;j<d;++j){double dq=q[b*d+j]-q[((b+1)%P)*d+j];
                e+=p[b*d+j]*p[b*d+j]/(2*c->mass[j/3])+.5*c->omega2*c->mass[j/3]*dq*dq;}
            ensure(std::isfinite(e),"nonfinite ring-polymer energy");return e;};
        Py_BEGIN_ALLOW_THREADS
        try{
            bool force_valid=false;double bias_energy=0.;
            for(step=0;step<steps;++step){
                if(!force_valid)bias_energy=force();
                if(step==0)initial=energy(bias_energy);
                for(Py_ssize_t j=0;j<size;++j)p[j]+=.5*c->dt*pes.force[j];
                project();
                c->propagate(q.data(),p.data(),work.data());project();
                bias_energy=force();for(Py_ssize_t j=0;j<size;++j)p[j]+=.5*c->dt*pes.force[j];
                project();
                for(double v:p)ensure(std::isfinite(v),"nonfinite momentum");
                // For projected fragment COMs, final projection changes position
                // only at roundoff, just as the reference constrained BCB path.
                clock+=c->dt;ensure(std::isfinite(clock),"state clock overflow");
                const double e=energy(bias_energy);minimum=std::min(minimum,e);maximum=std::max(maximum,e);
                force_valid=reuse;
                if((step+1)%stride==0||step+1==steps){eb.data()[sample]=e;
                    if(cb||ncv){centroid(q.data(),center.data(),P,d);
                        if(cb)std::copy(center.begin(),center.end(),cb->data()+sample*d);
                        if(vb)for(Py_ssize_t k=0;k<ncv;++k){double v=cvs[k].eval(center.data());ensure(std::isfinite(v),"nonfinite CV observation");vb->data()[sample*ncv+k]=v;}}
                    ++sample;
                }
            }
        }catch(...){failure=std::current_exception();}
        Py_END_ALLOW_THREADS
        if(failure){
            try {std::rethrow_exception(failure);}
            catch(const std::exception& e){throw Failure("fused propagation failed at block step "+std::to_string(step+1)+": "+e.what());}
            catch(...){throw Failure("unknown native exception");}
        }
        if(PyErr_CheckSignals()<0)return nullptr;
        PyObject* result=Py_BuildValue("(dddd)",initial,minimum,maximum,clock);if(!result)return nullptr;
        std::copy(q.begin(),q.end(),qb.data());std::copy(p.begin(),p.end(),pb.data());
        return result;
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
     catch(const std::exception& e){if(!PyErr_Occurred())PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
}
} // namespace fused
