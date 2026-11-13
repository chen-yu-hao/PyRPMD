/* Analytic teaching A+BC <-> AB+C potential, not a fitted H3 surface.
 * The function is examples/rate/toy_exchange.py with D=.15, A=.36 Eh.
 * Fixed atom order D,H,H; physical atomic-unit energies and analytic forces.
 */
#include "pyrpmd/pes_api_v1.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct pyrpmd_pes_context_v1 { int unused; };
static const char metadata[] =
"{\"schema\":\"pyrpmd.pes/1\",\"abi_version\":1,"
"\"provider_name\":\"teaching-exchange-D015-A036\",\"provider_version\":\"1\","
"\"artifact_sha256\":{},\"natoms\":3,\"atom_ids\":[\"D0\",\"H1\",\"H2\"],"
"\"atomic_numbers\":[1,1,1],\"state\":{\"label\":\"teaching\",\"charge\":0,\"multiplicity\":2},"
"\"available_states\":[{\"label\":\"teaching\",\"charge\":0,\"multiplicity\":2}],"
"\"layout\":\"B,N,3:C\",\"dtype\":\"float64\",\"position_unit\":\"bohr\","
"\"energy_unit\":\"hartree\",\"force_unit\":\"hartree/bohr\","
"\"boundary_conditions\":\"nonperiodic\",\"capabilities\":{\"native_batch\":true,"
"\"force_kind\":\"analytic\",\"thread_safety\":\"reentrant\",\"stateful\":false,"
"\"requires_bead_affinity\":false,\"execution_device\":\"cpu\"}}";

int32_t pyrpmd_pes_abi_version_v1(void) { return 1; }
pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1 *options,
        pyrpmd_pes_context_v1 **out, pyrpmd_error_v1 *error) {
    (void)error;
    if (!out) return PYRPMD_INVALID_ARGUMENT_V1;
    *out = NULL;
    if (!options || options->abi_version != 1 || options->struct_size != sizeof(*options))
        return PYRPMD_ABI_MISMATCH_V1;
    *out = calloc(1, sizeof(**out));
    return *out ? PYRPMD_OK_V1 : PYRPMD_INTERNAL_ERROR_V1;
}
pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1 *context,
        char *out, int64_t capacity, int64_t *required, pyrpmd_error_v1 *error) {
    (void)error;
    if (!context || !required || capacity < 0) return PYRPMD_INVALID_ARGUMENT_V1;
    *required = sizeof(metadata);
    if (capacity < *required) {
        if (out && capacity) out[0] = 0;
        return PYRPMD_BUFFER_TOO_SMALL_V1;
    }
    if (!out) return PYRPMD_INVALID_ARGUMENT_V1;
    memcpy(out, metadata, sizeof(metadata));
    return PYRPMD_OK_V1;
}
pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1 *context,
        const pyrpmd_pes_input_v1 *input, pyrpmd_pes_output_v1 *output,
        pyrpmd_error_v1 *error) {
    (void)error;
    if (!context || !input || !output) return PYRPMD_INVALID_ARGUMENT_V1;
    if (input->abi_version != 1 || output->abi_version != 1 ||
            input->struct_size != sizeof(*input) || output->struct_size != sizeof(*output))
        return PYRPMD_ABI_MISMATCH_V1;
    const int64_t count = input->nstructures;
    if (count < 1 || count > INT64_MAX/9 || input->natoms != 3 || input->layout != 1 ||
            input->scalar_type != 1 || input->positions_count != 9*count ||
            output->energies_count != count || output->forces_count != 9*count ||
            output->structure_status_count != count || !input->positions ||
            !output->energies || !output->forces || !output->structure_status)
        return PYRPMD_INVALID_ARGUMENT_V1;
    for (int64_t frame=0; frame<count; ++frame) {
        const double *q = input->positions + 9*frame;
        double *force = output->forces + 9*frame;
        double a[3], b[3], ra2=0, rb2=0, dot=0;
        for (int k=0; k<3; ++k) {
            a[k]=q[k]-q[3+k]; b[k]=q[6+k]-q[3+k];
            ra2+=a[k]*a[k]; rb2+=b[k]*b[k]; dot+=a[k]*b[k];
        }
        const double ra=sqrt(ra2), rb=sqrt(rb2);
        if (!isfinite(ra+rb) || ra<1e-10 || rb<1e-10) {
            output->energies[frame]=NAN;
            for (int k=0; k<9; ++k) force[k]=NAN;
            output->structure_status[frame]=PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V1;
            continue;
        }
        const double depth=.15, alpha=1.2, bending=.1;
        const double cosine=dot/(ra*rb), ea=exp(-alpha*(ra-1.2)), eb=exp(-alpha*(rb-1.2));
        const double pair=ea*eb, coupling=.36+bending*(1+cosine);
        output->energies[frame]=depth*(ea*ea-2*ea+eb*eb-2*eb)+pair*coupling;
        const double dra=2*depth*alpha*ea*(1-ea)-alpha*pair*coupling;
        const double drb=2*depth*alpha*eb*(1-eb)-alpha*pair*coupling;
        for (int k=0; k<3; ++k) {
            const double ua=a[k]/ra, ub=b[k]/rb;
            const double ga=dra*ua+bending*pair*(ub-cosine*ua)/ra;
            const double gb=drb*ub+bending*pair*(ua-cosine*ub)/rb;
            force[k]=-ga; force[6+k]=-gb; force[3+k]=ga+gb;
        }
        output->structure_status[frame]=PYRPMD_STRUCTURE_OK_V1;
        if (output->structure_messages && output->structure_message_stride>0 &&
                frame < output->structure_messages_capacity/output->structure_message_stride)
            output->structure_messages[frame*output->structure_message_stride]=0;
    }
    return PYRPMD_OK_V1;
}
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1 *context) { free(context); }
