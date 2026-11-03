            }
            energy = .5 * cosine_energy + .5 * context.confinement * confinement_norm2;
            break;
        }
        }
        if (!status) {
            if (!std::isfinite(energy)) status = PYRPMD_STRUCTURE_NONFINITE_V1;
            for (int64_t j = 0; j < dims; ++j)
                if (!std::isfinite(f[j])) status = PYRPMD_STRUCTURE_NONFINITE_V1;
            if (status) diagnostic = "nonfinite built-in PES energy or forces";
        }
        if (status) {
            energy = nan;
            std::fill(f, f + dims, nan);
        }
        output->energies[b] = energy;
        output->structure_status[b] = status;
        if (output->structure_messages)
            std::snprintf(output->structure_messages + b * output->structure_message_stride,
                          static_cast<size_t>(output->structure_message_stride), "%s", diagnostic);
    }
    message(error, "");
    return PYRPMD_OK_V1;
}

// Only construction/destruction hold the GIL. The callable address and context
// pointer are leased by NativePESBinding together with their owning capsule.
constexpr const char* capsule_name = "pyrpmd.BuiltinPES.v1";

struct OwnedObject {
    PyObject* value;
    explicit OwnedObject(PyObject* value_) : value(value_) {}
    ~OwnedObject() { Py_XDECREF(value); }
    OwnedObject(const OwnedObject&) = delete;
    OwnedObject& operator=(const OwnedObject&) = delete;
};

inline PyObject* parameter(PyObject* spec, const char* name) {
    PyObject* value = PyDict_GetItemString(spec, name);
    if (!value) throw std::invalid_argument(std::string("missing built-in PES parameter: ") + name);
    return value;
}

inline double real(PyObject* value) {
    const double result = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) throw std::invalid_argument("built-in PES parameters must be real numbers");
    if (!std::isfinite(result)) throw std::invalid_argument("built-in PES parameters must be finite");
    return result;
}

inline double scalar(PyObject* spec, const char* name, bool positive = false) {
    const double result = real(parameter(spec, name));
    if (positive ? result <= 0. : result < 0.)
        throw std::invalid_argument(std::string("invalid built-in PES parameter: ") + name);
    return result;
}

inline int64_t integer(PyObject* value) {
    if (PyBool_Check(value)) throw std::invalid_argument("built-in PES indices and counts must be integers");
    OwnedObject indexed(PyNumber_Index(value));
    if (!indexed.value) throw std::invalid_argument("built-in PES indices and counts must be integers");
    const long long result = PyLong_AsLongLong(indexed.value);
    if (PyErr_Occurred()) throw std::invalid_argument("built-in PES integer is out of range");
    return static_cast<int64_t>(result);
}

inline std::vector<double> vector(PyObject* spec, const char* name, int64_t size) {
    OwnedObject sequence(PySequence_Fast(parameter(spec, name), "built-in PES parameter must be a sequence"));
    if (!sequence.value) throw std::invalid_argument("built-in PES parameter must be a sequence");
    if (PySequence_Fast_GET_SIZE(sequence.value) != size)
        throw std::invalid_argument(std::string("invalid built-in PES parameter size: ") + name);
    std::vector<double> result(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i)
        result[i] = real(PySequence_Fast_GET_ITEM(sequence.value, i));
    return result;
}

inline std::vector<int64_t> atom_indices(PyObject* value, int64_t natoms) {
    if (value == Py_None) return {};
    OwnedObject sequence(PySequence_Fast(value, "built-in PES atoms must be a sequence"));
    if (!sequence.value) throw std::invalid_argument("built-in PES atoms must be a sequence");
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.value);
    if (size < 1) throw std::invalid_argument("built-in PES atom selection must be nonempty");
    std::vector<int64_t> result(static_cast<size_t>(size));
    for (Py_ssize_t i = 0; i < size; ++i) {
        result[i] = integer(PySequence_Fast_GET_ITEM(sequence.value, i));
        if (result[i] < 0 || result[i] >= natoms)
            throw std::invalid_argument("built-in PES atom index out of range");
    }
    std::vector<int64_t> sorted(result);
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::invalid_argument("built-in PES atoms must be distinct");
    return result;
}

inline void destroy(PyObject* capsule) {
    auto* context = static_cast<Context*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (context) delete context;
    else PyErr_Clear();
}

inline PyObject* create(PyObject*, PyObject* args) {
    const char* name;
    PyObject *atom_count, *spec;
    if (!PyArg_ParseTuple(args, "sOO", &name, &atom_count, &spec)) return nullptr;
    try {
        const int64_t natoms = integer(atom_count);
        if (!PyDict_Check(spec) || natoms < 1 ||
            natoms > PY_SSIZE_T_MAX / (3 * static_cast<Py_ssize_t>(sizeof(double))))
            throw std::invalid_argument("invalid built-in PES atom count or parameter dictionary");
        const std::string model(name);
        Kind kind;
        if (model == "free") kind = Kind::free;
        else if (model == "harmonic") kind = Kind::harmonic;
        else if (model == "morse_pair") kind = Kind::morse_pair;
        else if (model == "double_well") kind = Kind::double_well;
        else if (model == "periodic_cosine") kind = Kind::periodic_cosine;
        else throw std::invalid_argument("unknown built-in PES kind");
        auto context = std::make_unique<Context>(kind, natoms);
        if (kind == Kind::harmonic) {
            context->k = vector(spec, "k", 3 * natoms);
            context->center = vector(spec, "center", 3 * natoms);
            for (double k : context->k)
                if (k < 0.) throw std::invalid_argument("harmonic force constants must be nonnegative");
        } else if (kind == Kind::morse_pair) {
            context->atoms = atom_indices(parameter(spec, "atoms"), natoms);
            if (context->atoms.size() != 2)
                throw std::invalid_argument("Morse potential requires two atoms");
            context->depth = scalar(spec, "depth", true);
            context->alpha = scalar(spec, "alpha", true);
            context->equilibrium = scalar(spec, "equilibrium", true);
        } else if (kind == Kind::double_well) {
            context->height = scalar(spec, "height", true);
            context->width = scalar(spec, "width", true);
            context->confinement = scalar(spec, "confinement");
            context->atom = integer(parameter(spec, "atom"));
            const int64_t axis = integer(parameter(spec, "axis"));
            if (context->atom < 0 || context->atom >= natoms || axis < 0 || axis > 2)
                throw std::invalid_argument("double-well atom or axis out of range");
            context->axis = static_cast<int>(axis);
        } else if (kind == Kind::periodic_cosine) {
            const auto inverse = vector(spec, "inverse", 9);
            const auto heights = vector(spec, "height", 3);
            const auto repeats = vector(spec, "repeats", 3);
            const auto perpendicular = vector(spec, "perpendicular", 9);
            for (int a = 0; a < 3; ++a) {
                if (heights[a] < 0. || repeats[a] < 1. || std::floor(repeats[a]) != repeats[a])
                    throw std::invalid_argument("invalid periodic-cosine heights or repeats");
            }
            std::copy(inverse.begin(), inverse.end(), context->inverse.begin());
            std::copy(heights.begin(), heights.end(), context->heights.begin());
            std::copy(repeats.begin(), repeats.end(), context->repeats.begin());
            std::copy(perpendicular.begin(), perpendicular.end(), context->perpendicular.begin());
            context->atoms = atom_indices(parameter(spec, "atoms"), natoms);
            context->confinement = scalar(spec, "confinement");
        }
        PyObject* capsule = PyCapsule_New(context.get(), capsule_name, destroy);
        if (!capsule) return nullptr;
        Context* pointer = context.release();
        PyObject* result = Py_BuildValue("OKK", capsule,
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(&pyrpmd_builtin_evaluate_v1)),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pointer)));
        Py_DECREF(capsule);
        return result;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& exception) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }
}

// ---- Periodic shifted Lennard-Jones through C ABI v2 ---------------------

struct LJFrameCache {
    bool valid = false;
    int64_t natoms = 0;
    std::array<double, 9> cell{};
    std::vector<double> reference;
