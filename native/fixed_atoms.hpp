/* Shared static-atom projection for both native block lowerings. */
#ifndef PYRPMD_FIXED_ATOMS_HPP
#define PYRPMD_FIXED_ATOMS_HPP
struct NativeFixedAtoms {
    std::vector<unsigned char> mask;
    std::vector<double> positions;
    size_t atoms=0;
    bool any=false;
    NativeFixedAtoms(PyObject* spec,size_t n):mask(n,0),positions(3*n,0.),atoms(n){
        if(!PyDict_Check(spec))throw std::invalid_argument("fixed-atom options must be a dictionary");
        PyObject* indices=PyDict_GetItemString(spec,"fixed_atoms");
        if(!indices)return;
        PyObject* seq=PySequence_Fast(indices,"fixed_atoms must be an index sequence");
        if(!seq)throw std::invalid_argument("invalid fixed atom indices");
        try{
            for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq);++i){
                PyObject* item=PySequence_Fast_GET_ITEM(seq,i);
                if(!PyLong_Check(item)||PyBool_Check(item))throw std::invalid_argument("fixed atom index must be an integer");
                auto index=PyLong_AsSsize_t(item);
                if(PyErr_Occurred()||index<0||static_cast<size_t>(index)>=n||mask[index])throw std::invalid_argument("fixed atom is repeated or out of range");
                mask[index]=1;any=true;
            }
        }catch(...){Py_DECREF(seq);throw;}
        Py_DECREF(seq);
        if(!any)return;
        PyObject* ref=PyDict_GetItemString(spec,"fixed_positions");
        if(!ref)throw std::invalid_argument("fixed_positions is required for fixed atoms");
        seq=PySequence_Fast(ref,"fixed_positions must be a flat coordinate sequence");
        if(!seq)throw std::invalid_argument("invalid fixed positions");
        try{
            if(PySequence_Fast_GET_SIZE(seq)!=static_cast<Py_ssize_t>(3*n))throw std::invalid_argument("fixed_positions dimension mismatch");
            for(size_t j=0;j<3*n;++j){
                positions[j]=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq,j));
                if(PyErr_Occurred()||!std::isfinite(positions[j]))throw std::invalid_argument("fixed_positions must be finite");
            }
        }catch(...){Py_DECREF(seq);throw;}
        Py_DECREF(seq);
    }
    void validate(const double* q,const double* p,size_t beads) const {
        if(!any)return;
        for(size_t b=0;b<beads;++b)for(size_t a=0;a<atoms;++a)if(mask[a])for(size_t k=0;k<3;++k){
            size_t j=3*a+k,i=b*3*atoms+j;
            if(q[i]!=positions[j]||p[i]!=0.)throw std::invalid_argument("fixed atoms must have reference positions and zero momenta");
        }
    }
    void project(double* q,double* p,size_t beads) const {
        if(!any)return;
        for(size_t b=0;b<beads;++b)for(size_t a=0;a<atoms;++a)if(mask[a])for(size_t k=0;k<3;++k){
            size_t j=3*a+k,i=b*3*atoms+j;q[i]=positions[j];p[i]=0.;
        }
    }
    // Momentum half of project() for a state whose fixed positions already hold.
    void project_momenta(double* p,size_t beads) const {
        if(!any)return;
        for(size_t b=0;b<beads;++b)for(size_t a=0;a<atoms;++a)if(mask[a])for(size_t k=0;k<3;++k)p[b*3*atoms+3*a+k]=0.;
    }
    void clear_force(double* f,size_t beads) const {
        if(!any)return;
        for(size_t b=0;b<beads;++b)for(size_t a=0;a<atoms;++a)if(mask[a])for(size_t k=0;k<3;++k)f[b*3*atoms+3*a+k]=0.;
    }
};
#endif
