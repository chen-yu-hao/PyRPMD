/* Analytic built-in potentials through the physical PES C ABI v1.
 * The capsule owns an immutable parameter snapshot. Evaluation performs no
 * Python calls, allocation, cell updates, unit conversion, or bead scaling.
 */
#ifndef PYRPMD_BUILTIN_PES_HPP
#define PYRPMD_BUILTIN_PES_HPP

#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "include/pyrpmd/pes_api_v1.h"

namespace builtin_pes {

enum class Kind { free, harmonic, morse_pair, double_well, periodic_cosine };

struct Context {
    Kind kind;
    int64_t natoms;
    std::vector<double> k, center;
    std::vector<int64_t> atoms;
    int64_t atom = 0;
    int axis = 0;
    double depth = 0., alpha = 0., equilibrium = 0.;
    double height = 0., width = 0., confinement = 0.;
    std::array<double, 3> heights{}, repeats{};
    std::array<double, 9> inverse{}, perpendicular{};

    Context(Kind kind_, int64_t natoms_) : kind(kind_), natoms(natoms_) {}
};

inline void message(pyrpmd_error_v1* error, const char* text) noexcept {
    if (!error) return;
    error->required_capacity = static_cast<int64_t>(std::strlen(text)) + 1;
    if (error->message && error->capacity > 0)
        std::snprintf(error->message, static_cast<size_t>(error->capacity), "%s", text);
}

inline pyrpmd_rc_v1 reject(pyrpmd_error_v1* error, const char* text,
                          pyrpmd_rc_v1 code = PYRPMD_INVALID_ARGUMENT_V1) noexcept {
    message(error, text);
    return code;
}

// C ABI v1 reports per-frame failures with NaN outputs. Neither a C++ exception
// nor a failed frame may cross this boundary as a nominally successful result.
extern "C" inline pyrpmd_rc_v1 pyrpmd_builtin_evaluate_v1(
    pyrpmd_pes_context_v1* opaque, const pyrpmd_pes_input_v1* input,
    pyrpmd_pes_output_v1* output, pyrpmd_error_v1* error) noexcept {
    if (error && (error->capacity < 0 || (error->capacity > 0 && !error->message)))
        return reject(error, "invalid built-in PES error buffer");
    if (!opaque || !input || !output)
        return reject(error, "built-in PES requires context, input and output");
    if (input->abi_version != PYRPMD_PES_ABI_VERSION_V1 ||
        output->abi_version != PYRPMD_PES_ABI_VERSION_V1 ||
        input->struct_size != static_cast<int32_t>(sizeof(*input)) ||
        output->struct_size != static_cast<int32_t>(sizeof(*output)))
        return reject(error, "built-in PES ABI version or structure size mismatch",
                      PYRPMD_ABI_MISMATCH_V1);
    const auto& context = *reinterpret_cast<const Context*>(opaque);
    const int64_t batches = input->nstructures, natoms = input->natoms;
    if (batches < 1 || natoms < 1 || natoms != context.natoms ||
        natoms > INT64_MAX / 3 || batches > INT64_MAX / (3 * natoms) ||
        input->layout != PYRPMD_LAYOUT_BNC_V1 ||
        input->scalar_type != PYRPMD_SCALAR_FLOAT64_V1)
        return reject(error, "invalid built-in PES dimensions or layout");
    const int64_t dims = 3 * natoms, count = batches * dims;
    if (static_cast<uint64_t>(count) >
            static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(double) ||
        input->positions_count != count || output->forces_count != count ||
        output->energies_count != batches || output->structure_status_count != batches ||
        !input->positions || !output->energies || !output->forces || !output->structure_status)
        return reject(error, "invalid built-in PES buffers");
    if (output->structure_messages) {
        const int64_t stride = output->structure_message_stride;
        if (stride < 1 || batches > INT64_MAX / stride ||
            output->structure_messages_capacity < batches * stride)
            return reject(error, "invalid built-in PES message buffer");
    } else if (output->structure_message_stride || output->structure_messages_capacity) {
        return reject(error, "omitted built-in PES messages require zero sizes");
    }
    const double pi = std::acos(-1.);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int64_t b = 0; b < batches; ++b) {
        const double* q = input->positions + b * dims;
        double* f = output->forces + b * dims;
        double energy = 0.;
        int32_t status = PYRPMD_STRUCTURE_OK_V1;
        const char* diagnostic = "";
        for (int64_t j = 0; j < dims; ++j) {
            f[j] = 0.;
            if (!std::isfinite(q[j])) status = PYRPMD_STRUCTURE_NONFINITE_V1;
        }
        if (status) diagnostic = "built-in PES coordinates contain NaN/Inf";
        if (!status) switch (context.kind) {
        case Kind::free:
            break;
        case Kind::harmonic:
            for (int64_t j = 0; j < dims; ++j) {
                const double d = q[j] - context.center[j];
                energy += context.k[j] * d * d;
                f[j] = -context.k[j] * d;
            }
            energy *= .5;
            break;
        case Kind::morse_pair: {
            const int64_t a = context.atoms[0], c = context.atoms[1];
            double d[3], norm2 = 0.;
            for (int j = 0; j < 3; ++j) {
                d[j] = q[3*a+j] - q[3*c+j];
                norm2 += d[j] * d[j];
            }
            const double r = std::sqrt(norm2);
            if (r <= 1e-14) {
                status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V1;
                diagnostic = "coincident Morse-pair atoms";
                break;
            }
            const double e = std::exp(-context.alpha * (r - context.equilibrium));
            energy = context.depth * ((1. - e) * (1. - e));
            const double dv = 2. * context.depth * context.alpha * e * (1. - e);
            for (int j = 0; j < 3; ++j) {
                f[3*a+j] = -dv * d[j] / r;
                f[3*c+j] = -f[3*a+j];
            }
            break;
        }
        case Kind::double_well: {
            const int64_t selected = 3 * context.atom + context.axis;
            const double y = q[selected] / context.width, y2m1 = y*y - 1.;
            double norm2 = 0.;
            for (int64_t j = 0; j < dims; ++j) {
                norm2 += q[j] * q[j];
                f[j] = -context.confinement * q[j];
            }
            energy = context.height * (y2m1 * y2m1) +
                (.5 * context.confinement * norm2 -
                 .5 * context.confinement * q[selected] * q[selected]);
            f[selected] = -4. * context.height * y * y2m1 / context.width;
            break;
        }
        case Kind::periodic_cosine: {
            const int64_t selected = context.atoms.empty() ? natoms :
                static_cast<int64_t>(context.atoms.size());
            double cosine_energy = 0., confinement_norm2 = 0.;
            for (int64_t i = 0; i < selected; ++i) {
                const int64_t atom = context.atoms.empty() ? i : context.atoms[i];
                const double* position = q + 3 * atom;
                double gradient[3], perpendicular[3];
                for (int a = 0; a < 3; ++a) {
                    double fractional = 0.;
                    perpendicular[a] = 0.;
                    for (int c = 0; c < 3; ++c) {
                        fractional += position[c] * context.inverse[3*c+a];
                        perpendicular[a] += position[c] * context.perpendicular[3*c+a];
                    }
                    const double phase = (2. * pi) * fractional * context.repeats[a];
                    cosine_energy += context.heights[a] * (1. - std::cos(phase));
                    gradient[a] = pi * context.heights[a] * context.repeats[a] * std::sin(phase);
                    confinement_norm2 += perpendicular[a] * perpendicular[a];
                }
                for (int c = 0; c < 3; ++c) {
                    double force = 0.;
                    for (int a = 0; a < 3; ++a)
                        force -= gradient[a] * context.inverse[3*c+a];
                    f[3*atom+c] = force - context.confinement * perpendicular[c];
                }
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

} // namespace builtin_pes
#endif // PYRPMD_BUILTIN_PES_HPP
