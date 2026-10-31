/* Fused BCB propagation, native PES calls and synchronous observations.
 * MIT; CV/flux/RATTLE definitions follow the Python reference implementation.
 * Included after Cayley in cayley.cpp. No Python API is used in the hot loop.
 */
#ifndef PYRPMD_PROPAGATION_H
#define PYRPMD_PROPAGATION_H
#include "include/pyrpmd/pes_api_v1.h"
#include <array>
#include <cstring>
#include <memory>
#include <numeric>
#include <map>

namespace fused_blocks {
using Vec = std::vector<double>;
constexpr double pi = 3.1415926535897932384626433832795;

struct PyRef {
    PyObject* p;
    explicit PyRef(PyObject* x): p(x) { if (!p) throw std::runtime_error("invalid native descriptor"); }
    ~PyRef() { Py_DECREF(p); }
    PyRef(const PyRef&) = delete;
};
static PyObject* field(PyObject* d, const char* name) {
    if (!PyDict_Check(d)) throw std::invalid_argument("native descriptor must be a dictionary");
    auto* v=PyDict_GetItemString(d,name);
    if (!v) throw std::invalid_argument(std::string("missing native field: ")+name);
    return v;
}
static double number(PyObject* v) {
    double x=PyFloat_AsDouble(v);
    if (PyErr_Occurred() || !std::isfinite(x)) throw std::invalid_argument("native parameter must be finite");
    return x;
}
static std::string text(PyObject* v) {
    if (!PyUnicode_Check(v)) throw std::invalid_argument("native tag must be a string");
    Py_ssize_t size=0;
    const char* s=PyUnicode_AsUTF8AndSize(v,&size);
    if (!s) throw std::invalid_argument("invalid native string");
    return std::string(s,static_cast<size_t>(size));
}
static Vec vector(PyObject* v, Py_ssize_t size=-1) {
    PyRef seq(PySequence_Fast(v,"expected a numeric sequence"));
    auto n=PySequence_Fast_GET_SIZE(seq.p);
    if (size>=0 && n!=size) throw std::invalid_argument("native parameter length mismatch");
    Vec out(n);
    for (Py_ssize_t i=0;i<n;++i) out[i]=number(PySequence_Fast_GET_ITEM(seq.p,i));
    return out;
}
static bool finite(const Vec& v) {
    return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);});
}
struct Diff {
    double v=0;
    Vec g,h;
    Diff(Py_ssize_t n,bool second): g(n,0),h(second?n*n:0,0) {}
    void add(const Diff& b,double scale) {
        v+=scale*b.v;
        for (size_t i=0;i<g.size();++i) g[i]+=scale*b.g[i];
        for (size_t i=0;i<h.size();++i) h[i]+=scale*b.h[i];
    }
    void scale(double x) { v*=x;for(auto& y:g)y*=x;for(auto& y:h)y*=x; }
};
static double norm_value(const Vec& w,const double* q) {
    double d[3]={0,0,0};
    for (size_t a=0;a<w.size();++a) for(int c=0;c<3;++c) d[c]+=w[a]*q[3*a+c];
    return std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
}
static Diff norm(const Vec& w,const double* q,bool second) {
    const auto n=3*w.size();
    Diff out(n,second);
    double u[3]={0,0,0};
    for(size_t a=0;a<w.size();++a) for(int c=0;c<3;++c) u[c]+=w[a]*q[3*a+c];
    out.v=std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    if (!(out.v>=1e-12)) throw std::runtime_error("CV derivative undefined at zero distance");
    for(auto& c:u)c/=out.v;
    for(size_t i=0;i<n;++i) out.g[i]=w[i/3]*u[i%3];
    if(second) for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j)
        out.h[i*n+j]=w[i/3]*w[j/3]*((i%3==j%3?1.:0.)-u[i%3]*u[j%3])/out.v;
    return out;
}
struct Pair { Vec forming,breaking;double reference; };
struct CV {
    std::string kind;
    bool bead_average=false;
    Vec weights,other;
    double rinf=0,xi=0;
    std::vector<std::vector<Pair>> paths;
    std::unique_ptr<CV> coordinate;
    Py_ssize_t nd;
    CV(PyObject* d,Py_ssize_t n): nd(n) {
        kind=text(field(d,"kind"));
        auto estimator=text(field(d,"estimator"));
        if(estimator!="centroid" && estimator!="bead_average") throw std::invalid_argument("unknown native CV estimator");
        bead_average=estimator=="bead_average";
        if(kind=="norm" || kind=="difference" || kind=="rate") weights=vector(field(d,"weights"),nd/3);
        if(kind=="linear") weights=vector(field(d,"weights"),nd);
        if(kind=="difference") other=vector(field(d,"other"),nd/3);
        if(kind=="rate") {
            if(bead_average) throw std::invalid_argument("rate CV must use centroid");
            rinf=number(field(d,"rinf"));
            PyRef rows(PySequence_Fast(field(d,"paths"),"paths must be sequences"));
            for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(rows.p);++i) {
                PyRef row(PySequence_Fast(PySequence_Fast_GET_ITEM(rows.p,i),"pairs must be sequences"));
                std::vector<Pair> pairs;
                for(Py_ssize_t j=0;j<PySequence_Fast_GET_SIZE(row.p);++j) {
                    auto* p=PySequence_Fast_GET_ITEM(row.p,j);
                    pairs.push_back({vector(field(p,"forming"),nd/3),vector(field(p,"breaking"),nd/3),number(field(p,"reference"))});
                }
                if(pairs.empty() || (!paths.empty() && pairs.size()!=paths[0].size())) throw std::invalid_argument("invalid native rate paths");
                paths.push_back(std::move(pairs));
            }
            if(paths.empty()) throw std::invalid_argument("empty native rate paths");
        } else if(kind=="surface") {
            if(bead_average) throw std::invalid_argument("surface must use centroid");
            coordinate=std::make_unique<CV>(field(d,"coordinate"),nd);
            if(coordinate->kind!="rate") throw std::invalid_argument("surface requires rate CV");
            xi=number(field(d,"xi"));
        } else if(kind!="linear" && kind!="norm" && kind!="difference") throw std::invalid_argument("unknown native CV kind");
    }
    std::pair<Diff,Diff> surfaces(const double* q,bool second) const {
        auto s0=norm(weights,q,second);s0.scale(-1);s0.v+=rinf;
        size_t selected_path=0,selected_bond=0;
        double smallest=std::numeric_limits<double>::infinity();
        for(size_t b=0;b<paths[0].size();++b) {
            double largest=-std::numeric_limits<double>::infinity();size_t path=0;
            for(size_t p=0;p<paths.size();++p) {
                const auto& x=paths[p][b];
                double v=x.reference-norm_value(x.forming,q)+norm_value(x.breaking,q);
                if(v>largest){largest=v;path=p;}
            }
            if(largest<smallest){smallest=largest;selected_path=path;selected_bond=b;}
        }
        const auto& x=paths[selected_path][selected_bond];
        auto s1=norm(x.forming,q,second);s1.scale(-1);s1.v+=x.reference;
        s1.add(norm(x.breaking,q,second),1);
        return {std::move(s0),std::move(s1)};
    }
    Diff eval(const double* q,bool second=false) const {
        if(kind=="norm") return norm(weights,q,second);
        if(kind=="difference") {auto d=norm(weights,q,second);d.add(norm(other,q,second),-1);return d;}
        if(kind=="linear") {Diff d(nd,second);d.g=weights;for(Py_ssize_t i=0;i<nd;++i)d.v+=weights[i]*q[i];return d;}
        if(kind=="surface") {auto s=coordinate->surfaces(q,second);s.first.scale(1-xi);s.first.add(s.second,xi);return std::move(s.first);}
        auto s=surfaces(q,second);
        const Diff& a=s.first;Diff b=a;b.add(s.second,-1);
        if(std::abs(b.v)<1e-10) throw std::runtime_error("RPMDrate xi is singular where s0=s1");
        Diff out(nd,second);out.v=a.v/b.v;
        for(Py_ssize_t i=0;i<nd;++i) out.g[i]=a.g[i]/b.v-a.v*b.g[i]/(b.v*b.v);
        if(second) for(Py_ssize_t i=0;i<nd;++i) for(Py_ssize_t j=0;j<nd;++j)
            out.h[i*nd+j]=a.h[i*nd+j]/b.v-(a.g[i]*b.g[j]+b.g[i]*a.g[j]+a.v*b.h[i*nd+j])/(b.v*b.v)
                +2*a.v*b.g[i]*b.g[j]/(b.v*b.v*b.v);
        return out;
    }
    Diff ring(const Vec& q,const Vec& centroid,Py_ssize_t beads) const {
        Diff out(q.size(),false);
        if(!bead_average) {
            auto d=eval(centroid.data());out.v=d.v;
            for(Py_ssize_t b=0;b<beads;++b) for(Py_ssize_t i=0;i<nd;++i) out.g[b*nd+i]=d.g[i]/beads;
        } else {
            for(Py_ssize_t b=0;b<beads;++b){auto d=eval(q.data()+b*nd);out.v+=d.v/beads;for(Py_ssize_t i=0;i<nd;++i)out.g[b*nd+i]=d.g[i]/beads;}
        }
        return out;
    }
};
static std::vector<CV> cvs(PyObject* obj,Py_ssize_t nd) {
    PyRef seq(PySequence_Fast(obj,"CVs must be a sequence"));std::vector<CV> out;
    for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq.p);++i)out.emplace_back(PySequence_Fast_GET_ITEM(seq.p,i),nd);
    return out;
}
static void centroid(const Vec& q,Vec& c,Py_ssize_t beads,Py_ssize_t nd) {
    std::fill(c.begin(),c.end(),0.);
    for(Py_ssize_t b=0;b<beads;++b) for(Py_ssize_t i=0;i<nd;++i)c[i]+=q[b*nd+i];
    for(auto& x:c)x/=beads;
}
struct Bias {
    std::string kind;std::vector<CV> cv;Vec center,k,periods;
    Bias(PyObject* d,Py_ssize_t nd): kind(text(field(d,"kind"))),cv(cvs(field(d,"cvs"),nd)) {
        if(kind=="umbrella") {
            center=vector(field(d,"center"),cv.size());k=vector(field(d,"stiffness"),cv.size()*cv.size());
            periods=vector(field(d,"periods"),cv.size());
        } else if(kind!="flux" || cv.size()!=1 || cv[0].bead_average) throw std::invalid_argument("invalid native bias");
    }
    double add(const Vec& q,const Vec& c,Vec& f,const Cayley& integrator,double beta) const {
        const auto P=integrator.beads,nd=integrator.nd;
        if(kind=="flux") {
            auto d=cv[0].eval(c.data(),true);double z=0;
            for(Py_ssize_t i=0;i<nd;++i)z+=d.g[i]*d.g[i]/integrator.mass[i/3];
            if(!(z>1e-30) || !std::isfinite(z))throw std::runtime_error("rate coordinate has zero/invalid mass metric");
            for(Py_ssize_t i=0;i<nd;++i){double v=0;for(Py_ssize_t j=0;j<nd;++j)v+=d.h[i*nd+j]*d.g[j]/integrator.mass[j/3];v/=beta*z;for(Py_ssize_t b=0;b<P;++b)f[b*nd+i]+=v;}
            return -P*std::log(std::sqrt(z/(2*pi*beta)))/beta;
        }
        std::vector<Diff> values;Vec delta(cv.size()),derivative(cv.size(),0.);
        for(size_t i=0;i<cv.size();++i){values.push_back(cv[i].ring(q,c,P));delta[i]=values[i].v-center[i];if(periods[i]>0)delta[i]-=periods[i]*std::floor((delta[i]+.5*periods[i])/periods[i]);}
        for(size_t i=0;i<cv.size();++i)for(size_t j=0;j<cv.size();++j)derivative[i]+=k[i*cv.size()+j]*delta[j];
        double e=0;
        for(size_t i=0;i<cv.size();++i){e+=P*.5*delta[i]*derivative[i];for(size_t j=0;j<f.size();++j)f[j]-=P*derivative[i]*values[i].g[j];}
        return e;
    }
};
struct Fragment {Vec weights,reference;};
static void project(Vec& q,Vec& p,const std::vector<Fragment>& fragments,const Cayley& c) {
    for(const auto& f:fragments){double total=std::accumulate(f.weights.begin(),f.weights.end(),0.);double center[3]={0,0,0},velocity[3]={0,0,0};
        for(Py_ssize_t b=0;b<c.beads;++b)for(size_t a=0;a<f.weights.size();++a)if(f.weights[a]>0)for(int k=0;k<3;++k){center[k]+=f.weights[a]*q[b*c.nd+3*a+k];velocity[k]+=p[b*c.nd+3*a+k];}
        for(int k=0;k<3;++k){center[k]/=c.beads*total;velocity[k]/=c.beads*total;}
        for(Py_ssize_t b=0;b<c.beads;++b)for(size_t a=0;a<f.weights.size();++a)if(f.weights[a]>0)for(int k=0;k<3;++k){q[b*c.nd+3*a+k]-=center[k]-f.reference[k];p[b*c.nd+3*a+k]-=f.weights[a]*velocity[k];}
    }
}
using CABI=decltype(&pyrpmd_pes_evaluate_v1);
using FortranBatch=void(*)(int32_t,const double*,double*,double*,int32_t*);
struct PES {
    void* address;void* context;std::string kind;
    Vec energies,force;std::vector<int32_t> status;Py_ssize_t natoms,beads;
    PES(PyObject* obj,const Cayley& c):energies(c.beads),force(c.beads*c.nd),status(c.beads),natoms(c.nd/3),beads(c.beads) {
        PyRef seq(PySequence_Fast(obj,"invalid native PES binding"));
        if(PySequence_Fast_GET_SIZE(seq.p)!=4)throw std::invalid_argument("native binding must have four items");
        kind=text(PySequence_Fast_GET_ITEM(seq.p,0));
        address=PyLong_AsVoidPtr(PySequence_Fast_GET_ITEM(seq.p,1));context=PyLong_AsVoidPtr(PySequence_Fast_GET_ITEM(seq.p,2));
        auto n=PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(seq.p,3));
        if(PyErr_Occurred() || !address || n!=natoms || (kind!="cabi-v1" && kind!="fortran-batch-v1") || (kind=="fortran-batch-v1" && beads>INT32_MAX))throw std::invalid_argument("invalid native PES address, kind, or atom count");
    }
    void evaluate(const Vec& q) {
        if(!finite(q))throw std::runtime_error("PES received nonfinite coordinates");
        const double nan=std::numeric_limits<double>::quiet_NaN();
        std::fill(energies.begin(),energies.end(),nan);std::fill(force.begin(),force.end(),nan);std::fill(status.begin(),status.end(),-1);
        if(kind=="fortran-batch-v1")reinterpret_cast<FortranBatch>(address)(static_cast<int32_t>(beads),q.data(),energies.data(),force.data(),status.data());
        else {
            pyrpmd_pes_input_v1 in{1,sizeof(pyrpmd_pes_input_v1),1,1,beads,natoms,q.data(),static_cast<int64_t>(q.size())};
            pyrpmd_pes_output_v1 out{1,sizeof(pyrpmd_pes_output_v1),energies.data(),beads,force.data(),static_cast<int64_t>(force.size()),status.data(),beads,nullptr,0,0};
            char message[4096]={0};pyrpmd_error_v1 error{message,sizeof(message),0};
            int32_t rc=reinterpret_cast<CABI>(address)(static_cast<pyrpmd_pes_context_v1*>(context),&in,&out,&error);
            message[sizeof(message)-1]=0;
            if(rc)throw std::runtime_error("native PES global status "+std::to_string(rc)+": "+message);
        }
        for(Py_ssize_t b=0;b<beads;++b)if(status[b])throw std::runtime_error("native PES frame "+std::to_string(b)+" status "+std::to_string(status[b]));
        if(!finite(energies)||!finite(force))throw std::runtime_error("native PES returned nonfinite energy/forces");
    }
};
struct Buffer {
    Py_buffer b{};
    Buffer(PyObject* obj,Py_ssize_t bytes,const char* format,bool writable) {
        if(PyObject_GetBuffer(obj,&b,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS|(writable?PyBUF_WRITABLE:0))<0)throw std::invalid_argument("expected contiguous native output/input buffer");
        if(b.len!=bytes || !b.format || std::string(b.format)!=format) {PyBuffer_Release(&b);b.obj=nullptr;throw std::invalid_argument("native buffer size/dtype mismatch");}
    }
    ~Buffer(){if(b.obj)PyBuffer_Release(&b);}
    Buffer(const Buffer&)=delete;
    double* doubles(){return static_cast<double*>(b.buf);}
};
static bool overlap(const Buffer& a,const Buffer& b) {
    auto x=reinterpret_cast<uintptr_t>(a.b.buf),y=reinterpret_cast<uintptr_t>(b.b.buf);
    return a.b.len && b.b.len && x<y+b.b.len && y<x+a.b.len;
}
static Py_ssize_t product(Py_ssize_t a,Py_ssize_t b) {
    if(a<0||b<0||(a && b>PY_SSIZE_T_MAX/a))throw std::invalid_argument("native array size overflow");
    return a*b;
}
struct ReleaseGIL {PyThreadState* state;ReleaseGIL():state(PyEval_SaveThread()){}~ReleaseGIL(){PyEval_RestoreThread(state);}};
static void kick(Vec& p,const Vec& f,const Cayley& c) {
    auto n=static_cast<Py_ssize_t>(p.size());double h=.5*c.dt;
    #pragma omp parallel for num_threads(c.threads) schedule(static) if(c.threads>1)
    for(Py_ssize_t i=0;i<n;++i)p[i]+=h*f[i];
}
static double metric(const Diff& d,const Cayley& c) {
    double z=0;for(Py_ssize_t i=0;i<c.nd;++i)z+=d.g[i]*d.g[i]/c.mass[i/3];
    if(!std::isfinite(z)||z<=1e-28)throw std::runtime_error("singular constrained dividing surface");
    return z;
}

// Numeric labels cross the Python boundary only once per block: -1 is
// unresolved, 0 is nonreactive, and 1..N index descriptor.channel_labels.
// Only the exact Python classifiers which export these descriptors use this
// path. User callbacks and pilot candidate discovery remain Python operations.
static long long integer(PyObject* value,const char* name,long long minimum) {
    if(PyBool_Check(value) || !PyLong_Check(value))
        throw std::invalid_argument(std::string(name)+" must be an integer");
    long long result=PyLong_AsLongLong(value);
    if(PyErr_Occurred() || result<minimum)
        throw std::invalid_argument(std::string("invalid ")+name);
    return result;
}
static std::vector<std::string> strings(PyObject* value) {
    PyRef seq(PySequence_Fast(value,"expected a string sequence"));
    std::vector<std::string> result;
    for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq.p);++i)
        result.push_back(text(PySequence_Fast_GET_ITEM(seq.p,i)));
    return result;
}
struct Outcome {
    std::string kind;
    Vec masses,cutoffs;
    std::vector<std::string> symbols,reactants,labels;
    std::vector<std::vector<std::string>> channels;
    double exit_distance,bond_cutoff=0;
    long long start,interval,persistence,candidate_since,checks=0;
    int candidate,resolved=-1;
    bool pending_check=false;
    Py_ssize_t natoms;

    Outcome(PyObject* spec,Py_ssize_t n,Py_ssize_t steps):natoms(n) {
        auto* d=field(spec,"descriptor");
        kind=text(field(d,"kind"));
        if(kind!="connectivity" && kind!="h3_exchange")
            throw std::invalid_argument("unsupported native outcome classifier");
        masses=vector(field(d,"masses"),n);
        for(double mass:masses)if(!(mass>0))
            throw std::invalid_argument("outcome masses must be positive");
        exit_distance=number(field(d,"exit_distance"));
        if(!(exit_distance>0))throw std::invalid_argument("outcome exit distance must be positive");
        labels=strings(field(d,"channel_labels"));
        if(labels.size()>static_cast<size_t>(INT32_MAX))
            throw std::invalid_argument("too many outcome channels");
        for(size_t i=0;i<labels.size();++i){
            if(labels[i].empty())throw std::invalid_argument("empty outcome channel label");
            for(size_t j=0;j<i;++j)if(labels[i]==labels[j])
                throw std::invalid_argument("duplicate outcome channel label");
        }
        if(kind=="connectivity") {
            symbols=strings(field(d,"symbols"));
            if(symbols.size()!=static_cast<size_t>(n))
                throw std::invalid_argument("outcome symbol count mismatch");
            cutoffs=vector(field(d,"pair_cutoffs"),product(n,n-1)/2);
            for(double cutoff:cutoffs)if(!(cutoff>0 && cutoff<exit_distance))
                throw std::invalid_argument("outcome cutoffs must lie below exit distance");
            reactants=strings(field(d,"reactants"));
            if(reactants.size()!=2 || !std::is_sorted(reactants.begin(),reactants.end()))
                throw std::invalid_argument("outcome reactants must be two sorted formulas");
            PyRef seq(PySequence_Fast(field(d,"channels"),"outcome channels must be sequences"));
            for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq.p);++i){
                auto signature=strings(PySequence_Fast_GET_ITEM(seq.p,i));
                if(signature.size()<2 || !std::is_sorted(signature.begin(),signature.end()) || signature==reactants)
                    throw std::invalid_argument("invalid outcome channel signature");
                for(const auto& known:channels)if(signature==known)
                    throw std::invalid_argument("duplicate outcome channel signature");
                channels.push_back(std::move(signature));
            }
            if(channels.size()!=labels.size())
                throw std::invalid_argument("outcome channel label count mismatch");
        } else {
            bond_cutoff=number(field(d,"bond_cutoff"));
            if(n!=3 || !(bond_cutoff>0 && bond_cutoff<exit_distance) ||
               labels!=std::vector<std::string>{"H2[0,1]+H","H2[0,2]+H"})
                throw std::invalid_argument("invalid H3 outcome descriptor");
        }
        start=integer(field(spec,"start_step"),"outcome start_step",0);
        if(start>std::numeric_limits<long long>::max()-steps)
            throw std::invalid_argument("outcome step counter overflow");
        interval=integer(field(spec,"check_interval"),"outcome check_interval",1);
        persistence=integer(field(spec,"persistence_steps"),"outcome persistence_steps",0);
        auto code=integer(field(spec,"candidate_code"),"outcome candidate_code",-1);
        if(code>static_cast<long long>(labels.size()))
            throw std::invalid_argument("unknown outcome candidate code");
        candidate=static_cast<int>(code);
        candidate_since=integer(field(spec,"candidate_since"),"outcome candidate_since",-1);
        if((candidate<0)!=(candidate_since<0) || candidate_since>start)
            throw std::invalid_argument("inconsistent outcome persistence state");
    }

    static double distance(const Vec& q,Py_ssize_t a,Py_ssize_t b) {
        double square=0;
        for(int k=0;k<3;++k){double delta=q[3*a+k]-q[3*b+k];square+=delta*delta;}
        return std::sqrt(square);
    }

    int classify(const Vec& q,const Vec& p,double roundoff) const {
        // NumPy/BLAS reductions and the scalar native reductions may differ by
        // a few ulps. A near-boundary value is an execution-routing decision,
        // never a tolerance on the chemical criterion: Python makes the exact
        // original comparison at this scheduled step. Scaling by absolute
        // coordinates/velocities also covers cancellation in fragment COMs.
        if(!finite(q)||!finite(p))return -2;
        double coordinate_scale=0,velocity_scale=0;
        for(Py_ssize_t atom=0;atom<natoms;++atom)for(int k=0;k<3;++k){
            coordinate_scale=std::max(coordinate_scale,std::abs(q[3*atom+k]));
            velocity_scale=std::max(velocity_scale,std::abs(p[3*atom+k]/masses[atom]));
        }
        auto uncertain=[&](double value,double threshold,double scale){
            if(!std::isfinite(value)||!std::isfinite(scale))return true;
            double tolerance=roundoff*std::max({std::abs(value),std::abs(threshold),scale});
            return std::abs(value-threshold)<=tolerance;
        };
        if(kind=="h3_exchange") {
            const int pairs[3][2]={{0,1},{0,2},{1,2}};
            int selected=-1;
            for(int pair=0;pair<3;++pair){
                double length=distance(q,pairs[pair][0],pairs[pair][1]);
                if(uncertain(length,bond_cutoff,coordinate_scale))return -2;
                if(length<bond_cutoff){
                    if(selected>=0)return -1;
                    selected=pair;
                }
            }
            if(selected<0)return -1;
            int a=pairs[selected][0],b=pairs[selected][1],atom=3-a-b;
            double total=masses[a]+masses[b],square=0,outward=0;
            for(int k=0;k<3;++k){
                double r=q[3*atom+k]-(q[3*a+k]*masses[a]+q[3*b+k]*masses[b])/total;
                double v=p[3*atom+k]/masses[atom]-(p[3*a+k]+p[3*b+k])/total;
                square+=r*r;outward+=r*v;
            }
            double separation=std::sqrt(square);
            if(uncertain(separation,exit_distance,coordinate_scale))return -2;
            if(separation<exit_distance)return -1;
            if(uncertain(outward,0.,coordinate_scale*velocity_scale))return -2;
            if(outward/separation<=0)return -1;
            return selected==2 ? 0 : selected+1;
        }

        // Union-find supplies connected components with increasing atom order,
        // matching the Python classifier's sorted membership lists.
        std::vector<Py_ssize_t> parent(natoms);
        std::iota(parent.begin(),parent.end(),0);
        auto root=[&](Py_ssize_t atom){
            while(parent[atom]!=atom){parent[atom]=parent[parent[atom]];atom=parent[atom];}
            return atom;
        };
        size_t pair=0;
        for(Py_ssize_t a=0;a<natoms;++a)for(Py_ssize_t b=a+1;b<natoms;++b,++pair){
            double length=distance(q,a,b);
            if(uncertain(length,cutoffs[pair],coordinate_scale))return -2;
            if(length<cutoffs[pair])parent[root(b)]=root(a);
        }
        std::vector<std::vector<Py_ssize_t>> groups;
        std::vector<Py_ssize_t> roots;
        for(Py_ssize_t atom=0;atom<natoms;++atom){
            auto r=root(atom);auto known=std::find(roots.begin(),roots.end(),r);
            if(known==roots.end()){roots.push_back(r);groups.push_back({atom});}
            else groups[known-roots.begin()].push_back(atom);
        }
        if(groups.size()<2)return -1;
        std::vector<std::array<double,3>> centers(groups.size()),velocities(groups.size());
        std::vector<std::string> signature;
        for(size_t i=0;i<groups.size();++i){
            double total=0;std::map<std::string,Py_ssize_t> counts;
            for(auto atom:groups[i]){
                total+=masses[atom];++counts[symbols[atom]];
                for(int k=0;k<3;++k){centers[i][k]+=masses[atom]*q[3*atom+k];velocities[i][k]+=p[3*atom+k];}
            }
            for(int k=0;k<3;++k){centers[i][k]/=total;velocities[i][k]/=total;}
            std::string formula;
            for(const auto& entry:counts){formula+=entry.first;if(entry.second!=1)formula+=std::to_string(entry.second);}
            signature.push_back(std::move(formula));
        }
        for(size_t i=0;i<groups.size();++i)for(size_t j=i+1;j<groups.size();++j){
            double square=0,outward=0;
            for(int k=0;k<3;++k){double r=centers[j][k]-centers[i][k],v=velocities[j][k]-velocities[i][k];square+=r*r;outward+=r*v;}
            double separation=std::sqrt(square);
            if(uncertain(separation,exit_distance,coordinate_scale))return -2;
            if(separation<exit_distance)return -1;
            if(uncertain(outward,0.,coordinate_scale*velocity_scale))return -2;
            if(outward<=0)return -1;
        }
        std::sort(signature.begin(),signature.end());
        if(signature==reactants)return 0;
        for(size_t i=0;i<channels.size();++i)if(signature==channels[i])return static_cast<int>(i+1);
        return -1;
    }

    bool check(const Vec& q,const Vec& p,Vec& center,Vec& momentum,
               Py_ssize_t beads,Py_ssize_t nd,Py_ssize_t step) {
        const long long now=start+step+1;
        if(now%interval)return false;
        centroid(q,center,beads,nd);centroid(p,momentum,beads,nd);
        int label=classify(center,momentum,64*std::numeric_limits<double>::epsilon()*(beads+natoms+1));
        if(label==-2){pending_check=true;return true;}
        if(label!=candidate){candidate=label;candidate_since=label<0 ? -1 : now;}
        ++checks;
        if(label>=0 && now-candidate_since>=persistence){resolved=label;return true;}
        return false;
    }

    PyObject* result() const {
        return Py_BuildValue("{s:i,s:L,s:i,s:L,s:O}","candidate_code",candidate,
                             "candidate_since",candidate_since,"resolved_code",resolved,
                             "checks",checks,"pending_check",pending_check ? Py_True : Py_False);
    }
};

// Optional observations share the accepted-step transaction and the caller's
// sparse observation mask. They never call a PES or a Python observable.
struct NuclearDiagnostics {
    std::unique_ptr<Buffer> energy_parts,ring_radii;

    NuclearDiagnostics(PyObject* spec,Py_ssize_t rows,Py_ssize_t natoms) {
        if(!spec || spec==Py_None)return;
        if(!PyDict_Check(spec))throw std::invalid_argument("diagnostics must be a buffer dictionary");
        Py_ssize_t position=0;PyObject *key,*value;
        while(PyDict_Next(spec,&position,&key,&value)){
            const auto name=text(key);
            if(name=="energy_parts")energy_parts=std::make_unique<Buffer>(value,product(product(rows,4),sizeof(double)),"d",true);
            else if(name=="ring_radii")ring_radii=std::make_unique<Buffer>(value,product(product(rows,natoms),sizeof(double)),"d",true);
            else throw std::invalid_argument("unsupported nuclear diagnostic");
        }
    }

    void write(Py_ssize_t row,const Vec& q,const Vec& p,const Vec& center,
               const Vec& momentum,const PES& pes,const Cayley& c) const {
        const auto P=c.beads,nd=c.nd;
        if(energy_parts){
            // H_P/P = V/P + K_centroid + K_noncentroid/P + S/P.
            // Noncentroid refers to bead momenta, not molecular normal modes.
            double centroid_energy=0,internal_energy=0,spring_energy=0;
            for(Py_ssize_t i=0;i<nd;++i)centroid_energy+=momentum[i]*momentum[i]/(2*c.mass[i/3]);
            for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t i=0;i<nd;++i){
                const auto j=b*nd+i;
                const double dp=p[j]-momentum[i],dq=q[j]-q[((b+1)%P)*nd+i];
                internal_energy+=dp*dp/(2*c.mass[i/3]);
                spring_energy+=.5*c.omega2*c.mass[i/3]*dq*dq;
            }
            auto* values=energy_parts->doubles()+4*row;
            values[0]=std::accumulate(pes.energies.begin(),pes.energies.end(),0.)/P;
            values[1]=centroid_energy;values[2]=internal_energy/P;values[3]=spring_energy/P;
            for(int i=0;i<4;++i)if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite nuclear energy diagnostics");
        }
        if(ring_radii){
            auto* values=ring_radii->doubles()+row*(nd/3);
            for(Py_ssize_t a=0;a<nd/3;++a){
                double square=0;
                for(Py_ssize_t b=0;b<P;++b)for(int k=0;k<3;++k){
                    const double dq=q[b*nd+3*a+k]-center[3*a+k];square+=dq*dq;
                }
                values[a]=std::sqrt(square/P);
                if(!std::isfinite(values[a]))throw std::runtime_error("nonfinite ring radius diagnostics");
            }
        }
    }
};

static PyObject* advance(PyObject*,PyObject* args) {
    PyObject *capsule,*binding,*qo,*po,*options,*mo,*ro,*so;
    if(!PyArg_ParseTuple(args,"OOOOOOOO",&capsule,&binding,&qo,&po,&options,&mo,&ro,&so))return nullptr;
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(capsule,CAPSULE));if(!c)return nullptr;
    try {
        const auto P=c->beads,nd=c->nd,S=product(P,nd);
        double beta=number(field(options,"beta")),clock=number(field(options,"time"));
        if(!(beta>0) || std::abs((P/beta)*(P/beta)-c->omega2)>1e-14*std::max(1.,c->omega2))throw std::invalid_argument("native beta mismatch");
        auto steps=PyObject_Length(mo);
        if(steps<0||steps>4096)throw std::invalid_argument("native blocks are limited to 4096 steps");
        Buffer q(qo,product(S,sizeof(double)),"d",true),p(po,product(S,sizeof(double)),"d",true),mask(mo,steps,"B",false);
        const auto* observed=static_cast<const unsigned char*>(mask.b.buf);Py_ssize_t rows=0;
        for(Py_ssize_t i=0;i<steps;++i){if(observed[i]>1)throw std::invalid_argument("observation mask must be binary");rows+=observed[i];}
        auto observations=cvs(field(options,"cvs"),nd);
        const Py_ssize_t width=2+2*nd+static_cast<Py_ssize_t>(observations.size());
        Buffer records(ro,product(product(rows,width),sizeof(double)),"d",true);
        std::unique_ptr<Buffer> states,previous;
        if(so!=Py_None)states=std::make_unique<Buffer>(so,product(product(product(rows,2),S),sizeof(double)),"d",true);
        auto* previous_obj=PyDict_GetItemString(options,"previous");
        if(previous_obj && previous_obj!=Py_None)
            previous=std::make_unique<Buffer>(previous_obj,product(product(2,S),sizeof(double)),"d",true);
        NuclearDiagnostics diagnostics(PyDict_GetItemString(options,"diagnostics"),rows,nd/3);
        const Buffer* buffers[]={&q,&p,&mask,&records,states.get(),previous.get(),
                                 diagnostics.energy_parts.get(),diagnostics.ring_radii.get()};
        for(int i=0;i<8;++i)for(int j=i+1;j<8;++j)if(buffers[i]&&buffers[j]&&overlap(*buffers[i],*buffers[j]))throw std::invalid_argument("native buffers must not overlap");
        std::unique_ptr<Outcome> outcome;
        auto* outcome_obj=PyDict_GetItemString(options,"outcome");
        if(outcome_obj && outcome_obj!=Py_None)outcome=std::make_unique<Outcome>(outcome_obj,nd/3,steps);
        PES pes(binding,*c);
        PyRef bs(PySequence_Fast(field(options,"biases"),"biases must be a sequence"));std::vector<Bias> biases;
        for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(bs.p);++i)biases.emplace_back(PySequence_Fast_GET_ITEM(bs.p,i),nd);
        std::unique_ptr<CV> constraint;
        if(field(options,"constraint")!=Py_None)constraint=std::make_unique<CV>(field(options,"constraint"),nd);
        if(constraint && (constraint->bead_average || c->dt==0 || !biases.empty()))throw std::invalid_argument("native constraints require centroid, nonzero dt and no bias");
        double tolerance=number(field(options,"constraint_tolerance"));
        if(!(tolerance>0))throw std::invalid_argument("invalid constraint tolerance");
        int record_energy=PyObject_IsTrue(field(options,"record_energy"));if(record_energy<0)throw std::invalid_argument("invalid energy option");
        auto* check_obj=PyDict_GetItemString(options,"check_energy");
        int check_energy=check_obj ? PyObject_IsTrue(check_obj) : 0;
        if(check_energy<0)throw std::invalid_argument("invalid energy checking option");
        std::vector<Fragment> fragments;PyRef fs(PySequence_Fast(field(options,"fragments"),"fragments must be a sequence"));Vec used(nd/3,0);
        for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(fs.p);++i){auto* f=PySequence_Fast_GET_ITEM(fs.p,i);Fragment frag{vector(field(f,"weights"),nd/3),vector(field(f,"reference"),3)};double total=0;
            for(Py_ssize_t a=0;a<nd/3;++a){double w=frag.weights[a];if(w<0||(w>0&&(used[a]||w!=c->mass[a])))throw std::invalid_argument("invalid fragment mass/overlap");if(w>0){used[a]=1;total+=w;}}
            if(!(total>0))throw std::invalid_argument("empty native fragment");
            fragments.push_back(std::move(frag));}
        if(constraint && !fragments.empty())throw std::invalid_argument("combined centroid constraints are unsupported");
        auto* reuse_obj=PyDict_GetItemString(options,"reuse_forces");
        int reuse=reuse_obj ? PyObject_IsTrue(reuse_obj) : 0;
        if(reuse<0)throw std::invalid_argument("invalid reuse_forces option");
        // The opt-in cache is local to this call and restricted to ordinary
        // stateless PES dynamics. Projection, constraint and bias semantics
        // retain their reference evaluation schedule.
        const bool reuse_forces=reuse && !constraint && fragments.empty() && biases.empty();
        bool force_valid=false;
        Vec current_q(q.doubles(),q.doubles()+S),current_p(p.doubles(),p.doubles()+S),trial_q(S),trial_p(S),work(c->workspace_size()),center(nd),pc(nd),forces(S),free_q(S);
        if(!finite(current_q)||!finite(current_p))throw std::invalid_argument("nonfinite initial state");
        Py_ssize_t completed=0,written=0;std::string failure;
        double previous_clock=clock;
        auto evaluate=[&](const Vec& x){pes.evaluate(x);forces=pes.force;
            if(!biases.empty())centroid(x,center,P,nd);
            double eb=0;for(const auto& b:biases)eb+=b.add(x,center,forces,*c,beta);
            if(!std::isfinite(eb)||!finite(forces))throw std::runtime_error("nonfinite biased forces/energy");
            return eb;
        };
        auto total_energy=[&](const Vec& x,const Vec& momentum,double eb){
            double energy=std::accumulate(pes.energies.begin(),pes.energies.end(),0.)+eb;
            for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t i=0;i<nd;++i){
                const auto j=b*nd+i;const double dq=x[j]-x[((b+1)%P)*nd+i];
                energy+=momentum[j]*momentum[j]/(2*c->mass[i/3])+.5*c->omega2*c->mass[i/3]*dq*dq;
            }
            if(!std::isfinite(energy))throw std::runtime_error("nonfinite total ring energy");
            return energy;
        };
        {
            ReleaseGIL release;
            for(Py_ssize_t step=0;step<steps;++step) {
                try {
                    trial_q=current_q;trial_p=current_p;
                    std::unique_ptr<Diff> old_normal;
                    if(constraint){centroid(trial_q,center,P,nd);old_normal=std::make_unique<Diff>(constraint->eval(center.data()));metric(*old_normal,*c);if(std::abs(old_normal->v)>std::max(1e-8,10*tolerance))throw std::runtime_error("parent state is not on dividing surface");}
                    if(!force_valid){
                        const double eb=evaluate(trial_q);
                        if(check_energy && step==0)total_energy(trial_q,trial_p,eb);
                    }
                    kick(trial_p,forces,*c);project(trial_q,trial_p,fragments,*c);
                    c->propagate(trial_q.data(),trial_p.data(),work.data());project(trial_q,trial_p,fragments,*c);
                    if(constraint){free_q=trial_q;double lambda=0;bool converged=false;
                        for(int iteration=0;iteration<50;++iteration){centroid(trial_q,center,P,nd);auto d=constraint->eval(center.data());metric(d,*c);if(std::abs(d.v)<tolerance){converged=true;break;}
                            double derivative=0;for(Py_ssize_t i=0;i<nd;++i)derivative+=d.g[i]*old_normal->g[i]/c->mass[i/3];
                            if(std::abs(derivative)<1e-25)throw std::runtime_error("RATTLE normals became orthogonal");
                            lambda-=d.v/derivative;
                            for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t i=0;i<nd;++i)trial_q[b*nd+i]=free_q[b*nd+i]+lambda*old_normal->g[i]/c->mass[i/3];}
                        if(!converged)throw std::runtime_error("centroid RATTLE constraint failed to converge");
                        for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t i=0;i<nd;++i)trial_p[b*nd+i]+=lambda/c->dt*old_normal->g[i];}
                    double eb=evaluate(trial_q);kick(trial_p,forces,*c);project(trial_q,trial_p,fragments,*c);
                    if(constraint){centroid(trial_q,center,P,nd);auto d=constraint->eval(center.data());double z=metric(d,*c);centroid(trial_p,pc,P,nd);double v=0;for(Py_ssize_t i=0;i<nd;++i)v+=d.g[i]*pc[i]/c->mass[i/3];for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t i=0;i<nd;++i)trial_p[b*nd+i]-=v/z*d.g[i];}
                    if(!finite(trial_q)||!finite(trial_p))throw std::runtime_error("nonfinite integration state");
                    // Fragment projection can change last-bit coordinates after the
                    // final force. Re-evaluate only when a physical energy is requested.
                    double energy=std::numeric_limits<double>::quiet_NaN();
                    if(check_energy || ((record_energy || diagnostics.energy_parts) && observed[step])){
                        if(!fragments.empty())eb=evaluate(trial_q);
                        energy=total_energy(trial_q,trial_p,eb);
                    }
                    if(observed[step]) {
                        centroid(trial_q,center,P,nd);centroid(trial_p,pc,P,nd);
                        auto* row=records.doubles()+written*width;row[0]=clock+c->dt;row[1]=std::numeric_limits<double>::quiet_NaN();
                        if(record_energy)row[1]=energy;
                        std::copy(center.begin(),center.end(),row+2);std::copy(pc.begin(),pc.end(),row+2+nd);
                        diagnostics.write(written,trial_q,trial_p,center,pc,pes,*c);
                        for(size_t i=0;i<observations.size();++i){auto d=observations[i].ring(trial_q,center,P);if(!std::isfinite(d.v))throw std::runtime_error("nonfinite CV observation");row[2+2*nd+i]=d.v;}
                        if(states){auto* dest=states->doubles()+written*2*S;std::copy(trial_q.begin(),trial_q.end(),dest);std::copy(trial_p.begin(),trial_p.end(),dest+S);}
                    }
                    if(!std::isfinite(clock+c->dt))throw std::runtime_error("nonfinite integration clock");
                    bool resolved=outcome && outcome->check(trial_q,trial_p,center,pc,P,nd,step);
                    previous_clock=clock;
                    current_q.swap(trial_q);current_p.swap(trial_p);++completed;if(observed[step])++written;clock+=c->dt;
                    force_valid=reuse_forces;
                    if(resolved)break;
                } catch(const std::exception& e){failure=e.what();break;}
            }
            // After a successful final swap, trial_q/p still hold the state
            // immediately before that accepted step. One copy is sufficient
            // for a Python classifier's final-step rollback.
            if(previous && completed && failure.empty()){
                std::copy(trial_q.begin(),trial_q.end(),previous->doubles());
                std::copy(trial_p.begin(),trial_p.end(),previous->doubles()+S);
            }
            std::copy(current_q.begin(),current_q.end(),q.doubles());std::copy(current_p.begin(),current_p.end(),p.doubles());
        }
        if(outcome || previous){
            auto* result=outcome ? outcome->result() : PyDict_New();if(!result)return nullptr;
            if(previous && completed && failure.empty()){
                auto* value=PyFloat_FromDouble(previous_clock);
                if(!value || PyDict_SetItemString(result,"previous_time",value)<0){Py_XDECREF(value);Py_DECREF(result);return nullptr;}
                Py_DECREF(value);
            }
            return Py_BuildValue("nndsN",completed,written,clock,failure.c_str(),result);
        }
        return Py_BuildValue("nnds",completed,written,clock,failure.c_str());
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
      catch(const std::exception& e) {if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
} // namespace fused_blocks
#endif
