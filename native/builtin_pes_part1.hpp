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
#include <utility>
#include <vector>
#include "include/pyrpmd/pes_api_v1.h"
#include "include/pyrpmd/pes_api_v2.h"

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
    // Each frame owns disjoint outputs and follows the same serial arithmetic.
    // This preserves bitwise frame results while scaling ring-polymer batches.
    #pragma omp parallel for schedule(static) if(count >= 4096)
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
            double perpendicular2 = 0.;
            for (int64_t j = 0; j < dims; ++j) {
                if (j != selected) perpendicular2 += q[j] * q[j];
                f[j] = -context.confinement * q[j];
            }
            energy = context.height * (y2m1 * y2m1) + .5 * context.confinement * perpendicular2;
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
