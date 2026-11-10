// Streaming MBAR reductions. No NumPy C ABI, no global OpenMP configuration.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
struct Buffer {
    Py_buffer b{};
    ~Buffer() { if (b.obj) PyBuffer_Release(&b); }
    void get(PyObject* object, int dimensions, bool integer=false) {
        if (PyObject_GetBuffer(object, &b, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS))
            throw std::invalid_argument("MBAR requires contiguous arrays");
        const bool format = b.format && (integer ?
            (!std::strcmp(b.format,"q") || !std::strcmp(b.format,"l")) : !std::strcmp(b.format,"d"));
        if (b.ndim!=dimensions || b.itemsize!=8 || !format)
            throw std::invalid_argument("MBAR requires native float64/int64 arrays with the documented ranks");
    }
    const double* data() const { return static_cast<const double*>(b.buf); }
    const int64_t* ints() const { return static_cast<const int64_t*>(b.buf); }
    Py_ssize_t size() const { return b.len / 8; }
};

struct Accumulator {
    std::vector<double> potentials, probabilities, maximum, scaled, totals, compensation, matrix;
    double objective=0, objective_compensation=0;
    Accumulator(size_t k, bool gram): potentials(k), probabilities(k),
        maximum(k,-std::numeric_limits<double>::infinity()), scaled(k), totals(k), compensation(k),
        matrix(gram ? k*k : 0) {}
};

void add(double value, double& sum, double& compensation) {
    const double corrected=value-compensation, next=sum+corrected;
    compensation=(next-sum)-corrected;sum=next;
}

void log_add(double value, double& maximum, double& sum) {
    if (value>maximum) { sum=sum*std::exp(maximum-value)+1; maximum=value; }
    else sum+=std::exp(value-maximum);
}

PyObject* bytes(const std::vector<double>& value) {
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(value.data()),
                                   static_cast<Py_ssize_t>(value.size()*sizeof(double)));
}

PyObject* reduce(PyObject*, PyObject* args) {
    PyObject *av,*ac,*ak,*ap,*ar,*ai,*af,*an,*at;
    double beta; int mode, threads;
    if (!PyArg_ParseTuple(args,"OOOOOOOOOdii",&av,&ac,&ak,&ap,&ar,&ai,&af,&an,&at,&beta,&mode,&threads)) return nullptr;
    try {
        Buffer values,centers,stiffness,periods,rows,indices,free,counts,target;
        values.get(av,2); centers.get(ac,2); stiffness.get(ak,3); periods.get(ap,1);
        rows.get(ar,1,true); indices.get(ai,1,true); free.get(af,1); counts.get(an,1); target.get(at,1);
        const bool umbrella=centers.size()!=0;
        const Py_ssize_t k=rows.size(), d=umbrella ? values.b.shape[1] : 0;
        const Py_ssize_t source_n=values.b.shape[umbrella ? 0 : 1];
        const Py_ssize_t n=indices.size() ? indices.size() : source_n;
        const Py_ssize_t source_k=umbrella ? centers.b.shape[0] : values.b.shape[0];
        if (k<1 || n<1 || mode<0 || mode>3 || threads<1 || threads>1024 ||
            free.size()!=k || counts.size()!=k || target.size()!=n ||
            !std::isfinite(beta) || beta<=0)
            throw std::invalid_argument("invalid MBAR reduction dimensions/options");
        if (k>PY_SSIZE_T_MAX/k/8 || (mode==3 && n>PY_SSIZE_T_MAX/k/8))
            throw std::overflow_error("MBAR output size overflow");
        if (umbrella && (d<1 || d>3 || centers.b.shape[1]!=d || periods.size()!=d ||
            stiffness.b.shape[0]!=source_k || stiffness.b.shape[1]!=d || stiffness.b.shape[2]!=d))
            throw std::invalid_argument("invalid harmonic umbrella dimensions");
        for (Py_ssize_t j=0;j<k;++j) {
            if (rows.ints()[j]<0 || rows.ints()[j]>=source_k || counts.data()[j]<=0 ||
                !std::isfinite(counts.data()[j]) || !std::isfinite(free.data()[j]))
                throw std::invalid_argument("invalid MBAR state index/count/free energy");
        }
        for (Py_ssize_t j=0;j<indices.size();++j)
            if (indices.ints()[j]<0 || indices.ints()[j]>=source_n)
                throw std::invalid_argument("MBAR sample index out of range");
        for (Py_ssize_t j=0;j<periods.size();++j)
            if (!std::isfinite(periods.data()[j]) || periods.data()[j]<0)
                throw std::invalid_argument("invalid MBAR period");
        // Keep small fits serial. Each active thread owns only O(K^2) scratch.
        threads=std::min<int>(threads,static_cast<int>(std::min<Py_ssize_t>(n,1024)));
#ifndef _OPENMP
        threads=1;
#endif
        std::vector<Accumulator> local;
        for (int i=0;i<threads;++i) local.emplace_back(k,mode>0);
        std::vector<double> logn(k),update(k),gradient(k),matrix(mode>0 ? k*k : 0),
            logweights(mode>=2 ? n : 0),state_weights(mode==3 ? n*k : 0);
        for (Py_ssize_t i=0;i<k;++i) logn[i]=std::log(counts.data()[i]);
        std::atomic<bool> invalid{false}; int actual_threads=1;
        double objective=0;
        Py_BEGIN_ALLOW_THREADS
#pragma omp parallel num_threads(threads) if(threads>1)
        {
            int thread=0;
#ifdef _OPENMP
            thread=omp_get_thread_num();
#pragma omp single
            actual_threads=omp_get_num_threads();
#endif
            auto& acc=local[thread];
#pragma omp for schedule(static)
            for (Py_ssize_t sample=0;sample<n;++sample) {
                const Py_ssize_t index=indices.size() ? indices.ints()[sample] : sample;
                double common=std::numeric_limits<double>::infinity();
                for (Py_ssize_t i=0;i<k;++i) {
                    const auto state=rows.ints()[i]; double energy=0;
                    if (umbrella) {
                        double delta[3];
                        for (Py_ssize_t axis=0;axis<d;++axis) {
                            double v=values.data()[index*d+axis]-centers.data()[state*d+axis];
                            const double period=periods.data()[axis];
                            if (period>0) v-=period*std::floor(v/period+.5);
                            delta[axis]=v;
                        }
                        for (Py_ssize_t a=0;a<d;++a) for (Py_ssize_t b=0;b<d;++b)
                            energy+=delta[a]*stiffness.data()[(state*d+a)*d+b]*delta[b];
                        energy*=.5*beta;
                    } else energy=values.data()[state*source_n+index];
                    if (!std::isfinite(energy)) invalid=true;
                    acc.potentials[i]=energy; common=std::min(common,energy);
                }
                double maximum=-std::numeric_limits<double>::infinity();
                for (Py_ssize_t i=0;i<k;++i)
                    maximum=std::max(maximum,logn[i]+free.data()[i]-(acc.potentials[i]-common));
                double denominator=0;
                for (Py_ssize_t i=0;i<k;++i) {
                    acc.probabilities[i]=std::exp(logn[i]+free.data()[i]-(acc.potentials[i]-common)-maximum);
                    denominator+=acc.probabilities[i];
                }
                const double logd=maximum+std::log(denominator);
                add(logd,acc.objective,acc.objective_compensation);
                for (Py_ssize_t i=0;i<k;++i) {
                    log_add(-(acc.potentials[i]-common)-logd,acc.maximum[i],acc.scaled[i]);
                    const double p=acc.probabilities[i]/denominator;
                    add(p,acc.totals[i],acc.compensation[i]);
                    acc.probabilities[i]=mode>=2 ? p/counts.data()[i] : p;
                    if (mode==3) state_weights[sample*k+i]=acc.probabilities[i];
                }
                if (mode>0) for (Py_ssize_t i=0;i<k;++i) {
                    const double p=acc.probabilities[i];
                    // Exact zeros need no outer-product work; no probability cutoff.
                    if (p==0) continue;
                    double* row=acc.matrix.data()+i*k;
#pragma omp simd
                    for (Py_ssize_t j=0;j<=i;++j) row[j]+=p*acc.probabilities[j];
                }
                if (mode>=2) {
                    if (!std::isfinite(target.data()[sample])) invalid=true;
                    logweights[sample]=-target.data()[sample]+common-logd;
                }
            }
        }
        for (Py_ssize_t i=0;i<k;++i) {
            double maximum=-std::numeric_limits<double>::infinity(),sum=0,total=0;
            for (int t=0;t<actual_threads;++t) {
                log_add(local[t].maximum[i]+std::log(local[t].scaled[i]),maximum,sum);
                total+=local[t].totals[i];
            }
            update[i]=-maximum-std::log(sum);gradient[i]=total-counts.data()[i];
            if (mode>0) for (Py_ssize_t j=0;j<=i;++j) {
                double value=0;
                for (int t=0;t<actual_threads;++t) value+=local[t].matrix[i*k+j];
                if (mode==1) value=(i==j ? total : 0)-value;
                matrix[i*k+j]=matrix[j*k+i]=value;
            }
        }
        for (int t=0;t<actual_threads;++t) objective+=local[t].objective;
        for (Py_ssize_t i=0;i<k;++i) objective-=counts.data()[i]*free.data()[i];
        const double gauge=update[0];for (auto& value:update) value-=gauge;
        Py_END_ALLOW_THREADS
        if (invalid || !std::isfinite(objective))
            throw std::invalid_argument("nonfinite MBAR reduced potentials or reduction");
        PyObject* result=PyTuple_New(7);
        if (!result) return nullptr;
        const std::vector<double>* arrays[]={&update,&gradient,&matrix,&logweights,&state_weights};
        for (int i=0;i<5;++i) {
            PyObject* value=bytes(*arrays[i]);
            if (!value) {Py_DECREF(result);return nullptr;}
            PyTuple_SET_ITEM(result,i,value);
        }
        PyTuple_SET_ITEM(result,5,PyFloat_FromDouble(objective));
        PyTuple_SET_ITEM(result,6,PyLong_FromLong(actual_threads));
        return result;
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
      catch(const std::exception& error) {if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,error.what());return nullptr;}
}
PyMethodDef methods[]={{"reduce",reduce,METH_VARARGS,"Bounded MBAR fixed-point/Newton/overlap reduction; releases the GIL."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_mbar",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__mbar() {
    PyObject* result=PyModule_Create(&module);
    if (!result)return nullptr;
#ifdef _OPENMP
    PyModule_AddIntConstant(result,"OPENMP",1);
#else
    PyModule_AddIntConstant(result,"OPENMP",0);
#endif
    return result;
}
