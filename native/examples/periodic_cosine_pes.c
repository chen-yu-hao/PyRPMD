/* ABI v2 teaching potential, one hydrogen, arbitrary triclinic/partial PBC.
 * With a cell H (row vectors), s = q H^-1 and
 * E = sum_a |H_a|^2 g_a(s_a),
 * g_a(s) = (1-cos(2*pi*s))/(4*pi^2) for periodic directions, s^2/2 otherwise.
 * Without a cell: E = |q|^2/2. This is an interface test, not a chemistry PES.
 * Analytic forces and the FULL affine-strain derivative are returned.
 */
#include "pyrpmd/pes_api_v2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct pyrpmd_pes_context_v2 { int initialized; };
static const double TAU = 6.283185307179586476925286766559;
static const char META[] =
    "{\"schema\":\"pyrpmd.pes/2\",\"abi_version\":2,"
    "\"provider_name\":\"periodic-cosine-example\",\"provider_version\":\"1\","
    "\"natoms\":1,\"atomic_numbers\":[1],"
    "\"state\":{\"charge\":0,\"multiplicity\":1},"
    "\"layout\":\"B,N,3:C\",\"dtype\":\"float64\","
    "\"position_unit\":\"bohr\",\"energy_unit\":\"hartree\","
    "\"force_unit\":\"hartree/bohr\",\"boundary_conditions\":\"periodic\","
    "\"cell_layout\":\"B,3,3:C:rows\",\"pbc_layout\":\"B,3:C\","
    "\"virial_unit\":\"hartree\",\"virial_layout\":\"B,3,3:C\","
    "\"virial_convention\":\"minus_dE_dstrain\","
    "\"capabilities\":{\"native_batch\":true,\"virial\":true,"
    "\"stateful\":false,\"thread_safety\":\"reentrant\","
    "\"requires_bead_affinity\":false,\"execution_device\":\"cpu\"}}";

static void message(pyrpmd_error_v2 *error, const char *text) {
    if (!error) return;
    error->required_capacity = (int64_t)strlen(text) + 1;
    if (error->message && error->capacity > 0)
        snprintf(error->message, (size_t)error->capacity, "%s", text);
}

int32_t pyrpmd_pes_abi_version_v2(void) { return 2; }

pyrpmd_rc_v2 pyrpmd_pes_create_v2(
    const pyrpmd_create_options_v2 *options,
    pyrpmd_pes_context_v2 **out, pyrpmd_error_v2 *error) {
    if (!out) return PYRPMD_INVALID_ARGUMENT_V2;
    *out = NULL;
    if (!options || options->abi_version != 2 ||
        options->struct_size != (int32_t)sizeof(*options))
        return PYRPMD_ABI_MISMATCH_V2;
    if (options->config_json_bytes < 0 ||
        (options->config_json_bytes && !options->config_json))
        return PYRPMD_INVALID_ARGUMENT_V2;
    *out = malloc(sizeof(**out));
    if (!*out) return PYRPMD_INTERNAL_ERROR_V2;
    (*out)->initialized = 1;
    message(error, "");
    return PYRPMD_OK_V2;
}

pyrpmd_rc_v2 pyrpmd_pes_metadata_v2(
    const pyrpmd_pes_context_v2 *context, char *out, int64_t capacity,
    int64_t *required, pyrpmd_error_v2 *error) {
    if (!context || !required || capacity < 0)
        return PYRPMD_INVALID_ARGUMENT_V2;
    *required = (int64_t)sizeof(META);
    if (capacity < *required) {
        if (out && capacity) out[0] = '\0';
        return PYRPMD_BUFFER_TOO_SMALL_V2;
    }
    if (!out) return PYRPMD_INVALID_ARGUMENT_V2;
    memcpy(out, META, sizeof(META));
    message(error, "");
    return PYRPMD_OK_V2;
}

static int inverse(const double *h, double *inv) {
    double det = h[0]*(h[4]*h[8]-h[5]*h[7])
               - h[1]*(h[3]*h[8]-h[5]*h[6])
               + h[2]*(h[3]*h[7]-h[4]*h[6]);
    if (!isfinite(det) || det == 0.) return 0;
    inv[0]=(h[4]*h[8]-h[5]*h[7])/det;
    inv[1]=(h[2]*h[7]-h[1]*h[8])/det;
    inv[2]=(h[1]*h[5]-h[2]*h[4])/det;
    inv[3]=(h[5]*h[6]-h[3]*h[8])/det;
    inv[4]=(h[0]*h[8]-h[2]*h[6])/det;
    inv[5]=(h[2]*h[3]-h[0]*h[5])/det;
    inv[6]=(h[3]*h[7]-h[4]*h[6])/det;
    inv[7]=(h[1]*h[6]-h[0]*h[7])/det;
    inv[8]=(h[0]*h[4]-h[1]*h[3])/det;
    for (int j=0; j<9; ++j) if (!isfinite(h[j]) || !isfinite(inv[j])) return 0;
    return 1;
}

pyrpmd_rc_v2 pyrpmd_pes_evaluate_v2(
    pyrpmd_pes_context_v2 *context, const pyrpmd_pes_input_v2 *x,
    pyrpmd_pes_output_v2 *y, pyrpmd_error_v2 *error) {
    if (!context || !x || !y) return PYRPMD_INVALID_ARGUMENT_V2;
    if (x->abi_version != 2 || y->abi_version != 2 ||
        x->struct_size != (int32_t)sizeof(*x) ||
        y->struct_size != (int32_t)sizeof(*y)) return PYRPMD_ABI_MISMATCH_V2;
    int64_t b = x->nstructures;
    if (b < 1 || b > INT64_MAX/9 || x->natoms != 1 ||
        x->layout != 1 || x->scalar_type != 1 ||
        x->positions_count != 3*b || x->pbc_count != 3*b ||
        y->energies_count != b || y->forces_count != 3*b ||
        y->structure_status_count != b ||
        !x->positions || !x->pbc || !y->energies ||
        !y->forces || !y->structure_status) return PYRPMD_INVALID_ARGUMENT_V2;
    if ((x->cells && x->cells_count != 9*b) ||
        (!x->cells && x->cells_count != 0) ||
        (y->virials && y->virials_count != 9*b) ||
        (!y->virials && y->virials_count != 0)) return PYRPMD_INVALID_ARGUMENT_V2;
    if (y->structure_messages) {
        if (y->structure_message_stride < 1 ||
            b > INT64_MAX/y->structure_message_stride ||
            y->structure_messages_capacity < b*y->structure_message_stride)
            return PYRPMD_INVALID_ARGUMENT_V2;
    } else if (y->structure_message_stride || y->structure_messages_capacity) {
        return PYRPMD_INVALID_ARGUMENT_V2;
    }
    for (int64_t i=0; i<3*b; ++i)
        if (x->pbc[i] < 0 || x->pbc[i] > 1 || (!x->cells && x->pbc[i]))
            return PYRPMD_INVALID_ARGUMENT_V2;
    for (int64_t i=0; i<b; ++i) {
        const double *q = x->positions + 3*i;
        double *f = y->forces + 3*i;
        double *w = y->virials ? y->virials + 9*i : NULL;
        double energy = 0., inverse_cell[9];
        int valid = 1;
        for (int j=0; j<3; ++j) { f[j] = 0.; if (!isfinite(q[j])) valid = 0; }
        if (w) for (int j=0; j<9; ++j) w[j] = 0.;
        if (!x->cells) {
            for (int j=0; j<3; ++j) {
                energy += .5*q[j]*q[j]; f[j] = -q[j];
                if (w) for (int k=0; k<3; ++k) w[3*j+k] = -q[j]*q[k];
            }
        } else {
            const double *h = x->cells + 9*i;
            valid = valid && inverse(h, inverse_cell);
            if (valid) for (int a=0; a<3; ++a) {
                double s=0., norm2=0.;
                for (int j=0; j<3; ++j) {
                    s += q[j]*inverse_cell[3*j+a];
                    norm2 += h[3*a+j]*h[3*a+j];
                }
                double g = x->pbc[3*i+a] ? (1.-cos(TAU*s))/(TAU*TAU) : .5*s*s;
                double derivative = x->pbc[3*i+a] ? sin(TAU*s)/TAU : s;
                energy += norm2*g;
                for (int j=0; j<3; ++j) {
                    f[j] -= norm2*derivative*inverse_cell[3*j+a];
                    if (w) for (int k=0; k<3; ++k)
                        w[3*j+k] -= 2.*g*h[3*a+j]*h[3*a+k];
                }
            }
        }
        valid = valid && isfinite(energy);
        for (int j=0; j<3; ++j) valid = valid && isfinite(f[j]);
        if (w) for (int j=0; j<9; ++j) valid = valid && isfinite(w[j]);
        y->structure_status[i] = valid ? PYRPMD_STRUCTURE_OK_V2 : PYRPMD_STRUCTURE_NONFINITE_V2;
        y->energies[i] = valid ? energy : NAN;
        if (!valid) {
            for (int j=0; j<3; ++j) f[j] = NAN;
            if (w) for (int j=0; j<9; ++j) w[j] = NAN;
        }
        if (y->structure_messages)
            snprintf(y->structure_messages+i*y->structure_message_stride,
                     (size_t)y->structure_message_stride, "%s", valid ? "" : "nonfinite geometry or result");
    }
    message(error, "");
    return PYRPMD_OK_V2;
}

void pyrpmd_pes_destroy_v2(pyrpmd_pes_context_v2 *context) { free(context); }
