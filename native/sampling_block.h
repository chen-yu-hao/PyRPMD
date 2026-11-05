/*
 * Native PILE-L/TRPMD sampling blocks.
 *
 * The Python layer owns the random stream and supplies one already-scaled
 * real-mode noise frame per step.  This entry point deliberately reuses the
 * validated Cayley::sample_free implementation instead of duplicating the
 * normal-mode/free-map code.  All descriptors are parsed before the GIL is
 * released; the hot loop only calls the native PES ABI and C++ descriptor
 * implementations.
 */
#ifndef PYRPMD_SAMPLING_BLOCK_H
#define PYRPMD_SAMPLING_BLOCK_H

#include "propagation.h"

namespace fused_blocks {

/* Match NumPy's float64 reduction for a contiguous one-dimensional view.
 * Dynamics.energy() reduces each flattened energy component with NumPy's
 * pairwise summation kernel; a scalar C++ accumulator otherwise changes the
 * last bit for larger rings. */
static double numpy_sum(const double* values, Py_ssize_t n) {
    if(n<=0)return 0.;
    if(n<=128){
        double lanes[8]={0.,0.,0.,0.,0.,0.,0.,0.};
        Py_ssize_t i=0;
        for(;i+8<=n;i+=8){
            lanes[0]+=values[i];lanes[1]+=values[i+1];
            lanes[2]+=values[i+2];lanes[3]+=values[i+3];
            lanes[4]+=values[i+4];lanes[5]+=values[i+5];
            lanes[6]+=values[i+6];lanes[7]+=values[i+7];
        }
        double result=((lanes[0]+lanes[1])+(lanes[2]+lanes[3]))
                     +((lanes[4]+lanes[5])+(lanes[6]+lanes[7]));
        for(;i<n;++i)result+=values[i];
        return result;
    }
    Py_ssize_t split=n/2;split-=split%8;
    return numpy_sum(values,split)+numpy_sum(values+split,n-split);
}

/*
 * advance_sampling(context, binding, q, p, noise, decay, amplitude, fixed,
 *                  observation_mask, records, spec)
 *
 * `noise` is a flat float64 [steps, P, N, 3] array.  `records` has one row
 * per selected observation and columns
 *
 *   time, total_ring_energy, centroid_q[3N], centroid_p[3N], CV values.
 *
 * `spec` contains the scalar/descriptor values used by the deterministic
 * fused block (`beta`, `time`, `cvs`, `biases`, `record_energy`,
 * `check_energy`, `reuse_forces`) and may contain float64 `step_energies` and
 * accepted `energy_states` output buffers.  The latter is private wrapper
 * storage used to preserve the Python energy reduction order. The return
 * tuple is `(completed, written, clock, error)`.
 * A failed step is never committed; q/p and output arrays contain the valid
 * prefix exactly as in fused_blocks::advance.
 */
static PyObject* advance_sampling(PyObject*,PyObject* args) {
    PyObject *capsule,*binding,*qo,*po,*no,*decay_o,*amplitude_o,*fixed_o,
             *mask_o,*records_o,*spec;
    if(!PyArg_ParseTuple(args,"OOOOOOOOOOO",&capsule,&binding,&qo,&po,&no,
                         &decay_o,&amplitude_o,&fixed_o,&mask_o,&records_o,
                         &spec))return nullptr;
    auto* c=static_cast<Cayley*>(PyCapsule_GetPointer(capsule,CAPSULE));
    if(!c)return nullptr;
    try {
        if(!PyDict_Check(spec))throw std::invalid_argument("sampling spec must be a dictionary");
        const auto P=c->beads,nd=c->nd,S=product(P,nd),natoms=nd/3;
        auto steps=PyObject_Length(mask_o);
        if(steps<0||steps>4096)throw std::invalid_argument("native sampling blocks are limited to 4096 steps");
        Buffer q(qo,product(S,sizeof(double)),"d",true);
        Buffer p(po,product(S,sizeof(double)),"d",true);
        Buffer noise(no,product(product(steps,S),sizeof(double)),"d",false);
        Buffer decay(decay_o,product(P,sizeof(double)),"d",false);
        Buffer amplitude(amplitude_o,product(P,sizeof(double)),"d",false);
        Buffer mask(mask_o,steps,"B",false);
        Py_buffer fixed_view{};
        if(PyObject_GetBuffer(fixed_o,&fixed_view,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS)<0)
            throw std::invalid_argument("fixed mask must be a contiguous uint8 array");
        struct FixedRelease { Py_buffer* view; ~FixedRelease(){PyBuffer_Release(view);} } fixed_release{&fixed_view};
        if(fixed_view.len!=natoms || fixed_view.itemsize!=1 || !fixed_view.format
                || std::string(fixed_view.format)!="B")
            throw std::invalid_argument("fixed mask must have one uint8 entry per atom");
        const auto* fixed=static_cast<const unsigned char*>(fixed_view.buf);
        for(Py_ssize_t atom=0;atom<natoms;++atom)
            if(fixed[atom]>1)throw std::invalid_argument("fixed mask entries must be 0 or 1");
        const auto* observed=static_cast<const unsigned char*>(mask.b.buf);
        Py_ssize_t rows=0;
        for(Py_ssize_t i=0;i<steps;++i){
            if(observed[i]>1)throw std::invalid_argument("observation mask must be binary");
            rows+=observed[i];
        }
        auto observations=cvs(field(spec,"cvs"),nd);
        const Py_ssize_t width=2+2*nd+static_cast<Py_ssize_t>(observations.size());
        Buffer records(records_o,product(product(rows,width),sizeof(double)),"d",true);
        std::unique_ptr<Buffer> step_energies;
        auto* step_obj=PyDict_GetItemString(spec,"step_energies");
        if(step_obj && step_obj!=Py_None)
            step_energies=std::make_unique<Buffer>(step_obj,product(steps,sizeof(double)),"d",true);
        std::unique_ptr<Buffer> energy_states;
        auto* states_obj=PyDict_GetItemString(spec,"energy_states");
        if(states_obj && states_obj!=Py_None)
            energy_states=std::make_unique<Buffer>(states_obj,product(product(steps,2*S),sizeof(double)),"d",true);
        const Buffer* buffers[]={&q,&p,&noise,&decay,&amplitude,&mask,&records,
                                 step_energies.get(),energy_states.get()};
        for(int i=0;i<9;++i)for(int j=i+1;j<9;++j)
            if(buffers[i]&&buffers[j]&&overlap(*buffers[i],*buffers[j]))
                throw std::invalid_argument("native sampling buffers must not overlap");

        const double beta=number(field(spec,"beta")),clock0=number(field(spec,"time"));
        if(!(beta>0.)||!std::isfinite(beta)
                ||std::abs((P/beta)*(P/beta)-c->omega2)>1e-14*std::max(1.,c->omega2))
            throw std::invalid_argument("native beta mismatch");
        if(!std::isfinite(clock0))throw std::invalid_argument("invalid sampling clock");
        for(Py_ssize_t k=0;k<P;++k){
            if(!std::isfinite(decay.doubles()[k])||decay.doubles()[k]<0.||decay.doubles()[k]>1.
                    ||!std::isfinite(amplitude.doubles()[k])||amplitude.doubles()[k]<0.)
                throw std::invalid_argument("invalid PILE-L damping or noise amplitude");
        }
        int record_energy=PyObject_IsTrue(field(spec,"record_energy"));
        if(record_energy<0)throw std::invalid_argument("invalid energy option");
        auto* check_obj=PyDict_GetItemString(spec,"check_energy");
        int check_energy=check_obj ? PyObject_IsTrue(check_obj) : 0;
        if(check_energy<0)throw std::invalid_argument("invalid energy checking option");
        auto* reuse_obj=PyDict_GetItemString(spec,"reuse_forces");
        int reuse_forces=reuse_obj ? PyObject_IsTrue(reuse_obj) : 0;
        if(reuse_forces<0)throw std::invalid_argument("invalid reuse_forces option");

        PES pes(binding,*c);
        PyRef bs(PySequence_Fast(field(spec,"biases"),"biases must be a sequence"));
        std::vector<Bias> biases;
        for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(bs.p);++i)
            biases.emplace_back(PySequence_Fast_GET_ITEM(bs.p,i),nd);

        Vec current_q(q.doubles(),q.doubles()+S), current_p(p.doubles(),p.doubles()+S);
        Vec trial_q(S),trial_p(S),sample_work(6*S);
        Vec center(nd),centroid_p(nd);
        bool force_valid=false;
        double clock=clock0;
        std::string failure;
        Py_ssize_t completed=0,written=0;
        double bias_energy=0.;
        auto evaluate=[&](const Vec& x){
            pes.evaluate(x);
            if(!biases.empty())centroid(x,center,P,nd);
            double eb=0.;
            for(const auto& b:biases)eb+=b.add(x,center,pes.force,*c,beta);
            if(!std::isfinite(eb)||!finite(pes.force))
                throw std::runtime_error("nonfinite biased forces/energy");
            for(Py_ssize_t b=0;b<P;++b)
                for(Py_ssize_t i=0;i<nd;++i)
                    if(fixed[i/3])pes.force[b*nd+i]=0.;
            return eb;
        };
        auto total_energy=[&](const Vec& x,const Vec& momentum,double eb){
            Vec kinetic(S),spring_products(S);
            for(Py_ssize_t b=0;b<P;++b)for(Py_ssize_t i=0;i<nd;++i){
                const auto j=b*nd+i;
                const double dq=x[j]-x[((b+1)%P)*nd+i];
                kinetic[j]=momentum[j]*momentum[j]/(2*c->mass[i/3]);
                // NumPy forms ``masses * dq * dq`` before the scalar spring
                // prefactor; retaining that order is required for bitwise
                // agreement with Dynamics.energy().
                spring_products[j]=c->mass[i/3]*dq*dq;
            }
            // Keep the component order and reduction order identical to
            // Dynamics.energy(): potential, kinetic, spring, then bias.
            double energy=numpy_sum(pes.energies.data(),static_cast<Py_ssize_t>(pes.energies.size()));
            energy+=numpy_sum(kinetic.data(),S);
            energy+=.5*c->omega2*numpy_sum(spring_products.data(),S);
            energy+=eb;
            if(!std::isfinite(energy))throw std::runtime_error("nonfinite total ring energy");
            return energy;
        };
        if(!finite(current_q)||!finite(current_p))throw std::invalid_argument("nonfinite initial state");
        for(Py_ssize_t atom=0;atom<natoms;++atom)if(fixed[atom])
            for(Py_ssize_t b=0;b<P;++b)for(int k=0;k<3;++k)
                if(current_p[b*nd+3*atom+k]!=0.)throw std::invalid_argument("fixed atoms must have zero momenta");
        {
            ReleaseGIL release;
            for(Py_ssize_t step=0;step<steps;++step){
                try {
                    if(!force_valid)bias_energy=evaluate(current_q);
                    if(check_energy && step==0)total_energy(current_q,current_p,bias_energy);
                    trial_q=current_q;trial_p=current_p;
                    kick(trial_p,pes.force,*c);
                    c->sample_free(trial_q.data(),trial_p.data(),
                                   noise.doubles()+step*S,decay.doubles(),amplitude.doubles(),
                                   fixed,sample_work.data());
                    bias_energy=evaluate(trial_q);
                    kick(trial_p,pes.force,*c);
                    if(!finite(trial_q)||!finite(trial_p))
                        throw std::runtime_error("nonfinite integration state");
                    double energy=std::numeric_limits<double>::quiet_NaN();
                    if(check_energy || step_energies || (record_energy && observed[step]))
                        energy=total_energy(trial_q,trial_p,bias_energy);
                    const double next_clock=clock+c->dt;
                    if(!std::isfinite(next_clock))throw std::runtime_error("nonfinite integration clock");
                    Vec observed_values(observations.size());
                    if(observed[step]){
                        centroid(trial_q,center,P,nd);centroid(trial_p,centroid_p,P,nd);
                        for(size_t i=0;i<observations.size();++i){
                            auto d=observations[i].ring(trial_q,center,P);
                            if(!std::isfinite(d.v))throw std::runtime_error("nonfinite CV observation");
                            observed_values[i]=d.v;
                        }
                    }
                    // All checks, including CV evaluation and clock overflow,
                    // precede output writes so a failed step leaves only the
                    // valid accepted prefix in every caller-owned buffer.
                    if(step_energies)step_energies->doubles()[step]=energy;
                    if(observed[step]){
                        auto* row=records.doubles()+written*width;
                        row[0]=next_clock;
                        row[1]=record_energy ? energy : std::numeric_limits<double>::quiet_NaN();
                        std::copy(center.begin(),center.end(),row+2);
                        std::copy(centroid_p.begin(),centroid_p.end(),row+2+nd);
                        std::copy(observed_values.begin(),observed_values.end(),row+2+2*nd);
                    }
                    if(energy_states){
                        auto* snapshot=energy_states->doubles()+step*2*S;
                        std::copy(trial_q.begin(),trial_q.end(),snapshot);
                        std::copy(trial_p.begin(),trial_p.end(),snapshot+S);
                    }
                    current_q.swap(trial_q);current_p.swap(trial_p);
                    ++completed;if(observed[step])++written;clock=next_clock;
                    force_valid=bool(reuse_forces);
                } catch(const std::exception& e){failure=e.what();break;}
            }
            std::copy(current_q.begin(),current_q.end(),q.doubles());
            std::copy(current_p.begin(),current_p.end(),p.doubles());
        }
        return Py_BuildValue("nnds",completed,written,clock,failure.c_str());
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
      catch(const std::exception& e) {if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}

} // namespace fused_blocks

#endif
