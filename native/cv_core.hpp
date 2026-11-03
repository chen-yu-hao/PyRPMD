/* Shared native CV, bias and mass-metric mathematics. MIT; definitions follow
 * cv.py / bias.py and their retained notices. Both descriptor ABIs are lowered
 * once at construction; no Python calls occur in numerical evaluation.
 */
#ifndef PYRPMD_CV_CORE_HPP
#define PYRPMD_CV_CORE_HPP
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace native_cv {
using Vec=std::vector<double>;
namespace numeric {
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };
inline void ensure(bool ok, const char* text) { if (!ok) throw Failure(text); }
inline PyObject* field(PyObject* d, const char* k) {
    ensure(PyDict_Check(d), "native descriptor must be a dict");
    PyObject* x=PyDict_GetItemString(d,k);
    ensure(x!=nullptr, "incomplete native descriptor"); return x;
}
inline double number(PyObject* p) {
    double v=PyFloat_AsDouble(p); ensure(!PyErr_Occurred() && std::isfinite(v),"invalid native scalar"); return v;
}
inline std::vector<double> numbers(PyObject* p, size_t count) {
    ensure(PyList_Check(p)||PyTuple_Check(p), "native vector must be a list or tuple");
    ensure(static_cast<size_t>(PySequence_Size(p))==count,"native vector dimension mismatch");
    std::vector<double> v(count);
    for(size_t i=0;i<count;++i){ PyObject* x=PySequence_GetItem(p,i); if(!x)throw Failure("invalid vector");
        try {v[i]=number(x);} catch(...){Py_DECREF(x);throw;} Py_DECREF(x); }
    return v;
}
} // namespace numeric
namespace tagged {
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
} // namespace tagged
struct Diff { double v=0.;Vec g;explicit Diff(size_t n):g(n,0.){} };
struct Norm { double factor; std::vector<double> weights; };
struct NormHVP { std::array<double,3> unit; double radius; };
struct Primitive {
    double constant; std::vector<double> linear; std::vector<Norm> norms; size_t n, d;
    explicit Primitive(size_t nat,double c=0.):constant(c),linear(3*nat,0.),n(nat),d(3*nat) {}
    Primitive(PyObject* obj,size_t nat):n(nat),d(3*nat) {
        using namespace numeric;
        constant=number(field(obj,"constant")); linear=numbers(field(obj,"linear"),d);
        PyObject* terms=field(obj,"norms"); ensure(PyList_Check(terms),"norms must be a list");
        for(Py_ssize_t i=0;i<PyList_Size(terms);++i){PyObject* t=PyList_GetItem(terms,i);
            norms.push_back({number(field(t,"factor")),numbers(field(t,"weights"),n)});}
    }
    template<bool Numeric> double eval(const double* x,double* g=nullptr,std::vector<NormHVP>* prepared=nullptr) const {
        double value=constant;
        if(g)std::copy(linear.begin(),linear.end(),g);
        for(size_t j=0;j<d;++j)value+=linear[j]*x[j];
        for(size_t j=0;j<norms.size();++j){
            const auto& term=norms[j];
            double v[3]={0,0,0};
            for(size_t a=0;a<n;++a)for(size_t c=0;c<3;++c)v[c]+=term.weights[a]*x[3*a+c];
            const double r=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
            // The numeric ABI validates inactive candidates too; the tagged ABI
            // historically validates only the selected derivative's distances.
            if constexpr(Numeric) numeric::ensure(r>=1.e-12 && std::isfinite(r),"CV distance undefined at coincident atoms");
            else if(g || prepared) numeric::ensure(r>=1.e-12,"CV derivative undefined at zero distance");
            value+=term.factor*r;
            double u[3]={v[0]/r,v[1]/r,v[2]/r};
            if(g)for(size_t a=0;a<n;++a)for(size_t c=0;c<3;++c)g[3*a+c]+=term.factor*term.weights[a]*u[c];
            if(prepared)(*prepared)[j]={{u[0],u[1],u[2]},r};
        }
        return value;
    }
    // Apply the curvature prepared by eval once the metric direction is known.
    void hvp(const double* direction,double* hv,const std::vector<NormHVP>& prepared) const {
        std::fill(hv,hv+d,0.);
        for(size_t j=0;j<norms.size();++j){
            const auto& term=norms[j];const auto& h=prepared[j];double tangent[3]={0,0,0};
            for(size_t a=0;a<n;++a)for(size_t c=0;c<3;++c)tangent[c]+=term.weights[a]*direction[3*a+c];
            const double radial=h.unit[0]*tangent[0]+h.unit[1]*tangent[1]+h.unit[2]*tangent[2];
            for(size_t c=0;c<3;++c)tangent[c]=(tangent[c]-h.unit[c]*radial)/h.radius;
            for(size_t a=0;a<n;++a)for(size_t c=0;c<3;++c)hv[3*a+c]+=term.factor*term.weights[a]*tangent[c];
        }
    }
};
template<bool Numeric> struct Coordinate {
    int kind; size_t d, npairs=0; double xi=0;
    std::unique_ptr<Primitive> simple, entry;
    std::vector<Primitive> candidates;
    bool bead_average=false;
    mutable std::vector<double> g0,g1,hv0,hv1,ring_grad;
    mutable std::vector<NormHVP> geometry0,geometry1;
    struct MetricState { size_t which=0;double a=0,b=0; };
    Coordinate(PyObject* obj,size_t size):d(Numeric?3*size:size) {
        const size_t n=d/3;
        if constexpr(Numeric) {
            using namespace numeric;
            kind=static_cast<int>(number(field(obj,"kind")));
            ensure(kind>=0&&kind<=2,"unsupported native CV kind");
            if(kind==0)simple=std::make_unique<Primitive>(field(obj,"primitive"),n);
            else {
                entry=std::make_unique<Primitive>(field(obj,"entry"),n);
                PyObject* paths=field(obj,"paths");ensure(PyList_Check(paths)&&PyList_Size(paths)>0,"empty rate paths");
                for(Py_ssize_t p=0;p<PyList_Size(paths);++p){PyObject* row=PyList_GetItem(paths,p);
                    ensure(PyList_Check(row)&&PyList_Size(row)>0,"empty rate bond pairs");
                    if(p==0)npairs=PyList_Size(row);
                    ensure(static_cast<size_t>(PyList_Size(row))==npairs,"unequal rate bond rows");
                    for(size_t j=0;j<npairs;++j)candidates.emplace_back(PyList_GetItem(row,j),n);
                }
                if(kind==2)xi=number(field(obj,"xi"));
            }
        } else {
            using namespace tagged;
            auto tag=text(field(obj,"kind")),estimator=text(field(obj,"estimator"));
            if(estimator!="centroid" && estimator!="bead_average")throw std::invalid_argument("unknown native CV estimator");
            bead_average=estimator=="bead_average";
            kind=tag=="surface"?2:(tag=="rate"?1:0);
            if(kind) {
                if(bead_average)throw std::invalid_argument(tag=="rate"?"rate CV must use centroid":"surface must use centroid");
                if(kind==2) {
                    Coordinate coordinate(field(obj,"coordinate"),d);
                    if(coordinate.kind!=1)throw std::invalid_argument("surface requires rate CV");
                    entry=std::move(coordinate.entry);candidates=std::move(coordinate.candidates);npairs=coordinate.npairs;
                    xi=number(field(obj,"xi"));
                } else {
                    auto weights=vector(field(obj,"weights"),n);
                    entry=std::make_unique<Primitive>(n,number(field(obj,"rinf")));
                    entry->norms.push_back({-1.,std::move(weights)});
                    PyRef rows(PySequence_Fast(field(obj,"paths"),"paths must be sequences"));
                    for(Py_ssize_t p=0;p<PySequence_Fast_GET_SIZE(rows.p);++p) {
                        PyRef row(PySequence_Fast(PySequence_Fast_GET_ITEM(rows.p,p),"pairs must be sequences"));
                        auto count=PySequence_Fast_GET_SIZE(row.p);
                        if(!count || (p && static_cast<size_t>(count)!=npairs))throw std::invalid_argument("invalid native rate paths");
                        npairs=count;
                        for(Py_ssize_t j=0;j<count;++j){auto* pair=PySequence_Fast_GET_ITEM(row.p,j);
                            auto forming=vector(field(pair,"forming"),n),breaking=vector(field(pair,"breaking"),n);
                            candidates.emplace_back(n,number(field(pair,"reference")));
                            candidates.back().norms={{-1.,std::move(forming)},{1.,std::move(breaking)}};
                        }
                    }
                    if(candidates.empty())throw std::invalid_argument("empty native rate paths");
                }
            } else {
                simple=std::make_unique<Primitive>(n);
                if(tag=="linear")simple->linear=vector(field(obj,"weights"),d);
                else if(tag=="norm" || tag=="difference") {
                    simple->norms.push_back({1.,vector(field(obj,"weights"),n)});
                    if(tag=="difference")simple->norms.push_back({-1.,vector(field(obj,"other"),n)});
                } else throw std::invalid_argument("unknown native CV kind");
            }
            ring_grad.resize(d);
        }
        if(kind==0)geometry0.resize(simple->norms.size());
        else {
            g0.resize(d);g1.resize(d);hv0.resize(d);hv1.resize(d);geometry0.resize(entry->norms.size());
            for(const auto& candidate:candidates)geometry1.resize(std::max(geometry1.size(),candidate.norms.size()));
        }
    }
    double eval(const double* x,double* g=nullptr,MetricState* prepared=nullptr) const {
        if(kind==0)return simple->eval<Numeric>(x,g,prepared?&geometry0:nullptr);
        double selected=std::numeric_limits<double>::infinity(); size_t which=0;
        // min over paired bonds of max over equivalent TSs; ties choose first.
        for(size_t j=0;j<npairs;++j){double largest=-std::numeric_limits<double>::infinity();size_t arg=j;
            for(size_t k=j;k<candidates.size();k+=npairs){double s=candidates[k].template eval<Numeric>(x);
                if(s>largest){largest=s;arg=k;}}
            if(largest<selected){selected=largest;which=arg;}}
        const bool derivative=g || prepared;
        const double a=entry->eval<Numeric>(x,derivative?g0.data():nullptr,prepared?&geometry0:nullptr);
        const double s=candidates[which].template eval<Numeric>(x,derivative?g1.data():nullptr,prepared?&geometry1:nullptr);
        if(prepared)*prepared={which,a,a-s};
        if(kind==2){
            if(g)for(size_t j=0;j<d;++j)g[j]=(1.-xi)*g0[j]+xi*g1[j];
            return (1.-xi)*a+xi*s;
        }
        const double b=a-s;
        numeric::ensure(Numeric?std::abs(b)>=1.e-10:!(std::abs(b)<1.e-10),"RPMDrate xi is singular where s0=s1");
        if(g)for(size_t j=0;j<d;++j)g[j]=g0[j]/b-a*(g0[j]-g1[j])/(b*b);
        return a/b;
    }
    // Ring Jacobians retain division by P before the tagged umbrella applies P.
    // Scratch belongs to this descriptor, never to a global or checkpoint state.
    double ring_into(const double* q,const double* centroid,size_t beads,double* g) const {
        double value=0.;
        if(!bead_average){value=eval(centroid,ring_grad.data());
            for(size_t b=0;b<beads;++b)for(size_t i=0;i<d;++i)g[b*d+i]=ring_grad[i]/beads;
        } else for(size_t b=0;b<beads;++b){value+=eval(q+b*d,ring_grad.data())/beads;
            for(size_t i=0;i<d;++i)g[b*d+i]=ring_grad[i]/beads;}
        return value;
    }
    Diff ring(const Vec& q,const Vec& centroid,size_t beads) const {
        Diff out(q.size());out.v=ring_into(q.data(),centroid.data(),beads,out.g.data());return out;
    }
    double metric(const double* x,double* g,double* direction,double* hv,const std::vector<double>& mass) const {
        MetricState prepared;eval(x,g,&prepared);
        double z=0.;for(size_t j=0;j<d;++j){z+=g[j]*g[j]/mass[j/3];direction[j]=g[j]/mass[j/3];}
        numeric::ensure(z>1.e-30&&std::isfinite(z),Numeric?"rate coordinate has invalid mass metric":"rate coordinate has zero/invalid mass metric");
        if(kind==0){simple->hvp(direction,hv,geometry0);return z;}
        entry->hvp(direction,hv0.data(),geometry0);
        candidates[prepared.which].hvp(direction,hv1.data(),geometry1);
        if(kind==2){for(size_t j=0;j<d;++j)hv[j]=(1.-xi)*hv0[j]+xi*hv1[j];return z;}
        const double a=prepared.a,b=prepared.b;double da=0.,db=0.;
        for(size_t j=0;j<d;++j){da+=g0[j]*direction[j];db+=(g0[j]-g1[j])*direction[j];}
        for(size_t i=0;i<d;++i){
            const double gi=g0[i]-g1[i];
            hv[i]=hv0[i]/b-(g0[i]*db+gi*da+a*(hv0[i]-hv1[i]))/(b*b)+2.*a*gi*db/(b*b*b);
        }
        return z;
    }
};

// Only the legacy return/argument shapes differ at the integration boundary.
struct BlockCV : Coordinate<false> {
    using Coordinate::Coordinate;
    Diff eval(const double* q) const {Diff out(d);out.v=Coordinate::eval(q,out.g.data());return out;}
};

template<bool Numeric> struct Bias {
    int kind;size_t d;double beta=1.;
    std::vector<Coordinate<Numeric>> cv;
    Vec center,K,periods;
    mutable Vec grad,hvp,direction,delta;
    Bias(PyObject* obj,size_t size):d(Numeric?3*size:size) {
        if constexpr(Numeric) {
            using namespace numeric;
            kind=static_cast<int>(number(field(obj,"kind")));ensure(kind==0||kind==1,"unsupported bias");
            auto* specs=field(obj,"cvs");ensure(PyList_Check(specs)&&PyList_Size(specs)>0,"empty bias CVs");
            for(Py_ssize_t i=0;i<PyList_Size(specs);++i)cv.emplace_back(PyList_GetItem(specs,i),size);
            if(kind==0){center=numbers(field(obj,"center"),cv.size());K=numbers(field(obj,"stiffness"),cv.size()*cv.size());}
            else {ensure(cv.size()==1,"metric needs one CV");beta=number(field(obj,"beta"));ensure(beta>0,"invalid metric beta");}
        } else {
            using namespace tagged;
            auto tag=text(field(obj,"kind"));kind=tag=="umbrella"?0:1;
            PyRef specs(PySequence_Fast(field(obj,"cvs"),"CVs must be a sequence"));
            for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(specs.p);++i)cv.emplace_back(PySequence_Fast_GET_ITEM(specs.p,i),size);
            if(kind==0){center=vector(field(obj,"center"),cv.size());K=vector(field(obj,"stiffness"),cv.size()*cv.size());periods=vector(field(obj,"periods"),cv.size());}
            else if(tag!="flux" || cv.size()!=1 || cv[0].bead_average)throw std::invalid_argument("invalid native bias");
        }
        grad.resize(cv.size()*d);delta.resize(cv.size());
        if(kind==1){hvp.resize(d);direction.resize(d);}
    }
    double apply(const double* q,const double* centroid,double* force,size_t P,const Vec& mass,double thermal_beta) const {
        if(kind==1){
            const double z=cv[0].metric(centroid,grad.data(),direction.data(),hvp.data(),mass);
            for(size_t i=0;i<d;++i)for(size_t b=0;b<P;++b)force[b*d+i]+=hvp[i]/(thermal_beta*z);
            return -static_cast<double>(P)*std::log(std::sqrt(z/(2.*3.14159265358979323846*thermal_beta)))/thermal_beta;
        }
        const size_t k=cv.size(),width=Numeric?d:P*d;double e=0.;
        if constexpr(!Numeric)grad.resize(k*width);
        for(size_t i=0;i<k;++i){
            if constexpr(Numeric)delta[i]=cv[i].eval(centroid,grad.data()+i*width)-center[i];
            else {delta[i]=cv[i].ring_into(q,centroid,P,grad.data()+i*width)-center[i];
                if(periods[i]>0)delta[i]-=periods[i]*std::floor((delta[i]+.5*periods[i])/periods[i]);}
        }
        for(size_t i=0;i<k;++i){double kd=0.;for(size_t j=0;j<k;++j)kd+=K[i*k+j]*delta[j];
            // Keep each ABI's original bead-scaling order, including under FMA.
            if constexpr(Numeric){e+=.5*delta[i]*kd;
                for(size_t j=0;j<d;++j)for(size_t b=0;b<P;++b)force[b*d+j]-=kd*grad[i*d+j];}
            else {e+=P*.5*delta[i]*kd;for(size_t j=0;j<P*d;++j)force[j]-=P*kd*grad[i*width+j];}
        }
        return Numeric?static_cast<double>(P)*e:e;
    }
    double apply(const double* centroid,double* force,size_t P,const Vec& mass) const {
        return apply(nullptr,centroid,force,P,mass,beta);
    }
    template<class Integrator> double add(const Vec& q,const Vec& c,Vec& f,const Integrator& integrator,double thermal_beta) const {
        return apply(q.data(),c.data(),f.data(),integrator.beads,integrator.mass,thermal_beta);
    }
};
} // namespace native_cv
#endif
