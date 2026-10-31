/*
 * PyRPMD physical PES ABI v2: batched positions, periodic cells and virials.
 * C11/C++17, host-memory IEEE binary64, synchronous calls, atomic units.
 * Normative semantics, including strain convention, are in native/ABI.md.
 * This is a separate ABI: never pass v2 structures to v1 function pointers.
 */
#ifndef PYRPMD_PES_API_V2_H
#define PYRPMD_PES_API_V2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYRPMD_PES_ABI_VERSION_V2 INT32_C(2)
#define PYRPMD_LAYOUT_BNC_V2 INT32_C(1)
#define PYRPMD_SCALAR_FLOAT64_V2 INT32_C(1)

typedef int32_t pyrpmd_rc_v2;
#define PYRPMD_OK_V2 INT32_C(0)
#define PYRPMD_INVALID_ARGUMENT_V2 INT32_C(1)
#define PYRPMD_ABI_MISMATCH_V2 INT32_C(2)
#define PYRPMD_INVALID_CONFIG_V2 INT32_C(3)
#define PYRPMD_BUFFER_TOO_SMALL_V2 INT32_C(4)
#define PYRPMD_BACKEND_FAILURE_V2 INT32_C(5)
#define PYRPMD_INTERNAL_ERROR_V2 INT32_C(6)

typedef int32_t pyrpmd_structure_status_v2;
#define PYRPMD_STRUCTURE_OK_V2 INT32_C(0)
#define PYRPMD_STRUCTURE_FAILED_V2 INT32_C(1)
#define PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2 INT32_C(2)
#define PYRPMD_STRUCTURE_NONFINITE_V2 INT32_C(3)

typedef struct pyrpmd_pes_context_v2 pyrpmd_pes_context_v2;

/* Optional diagnostic sink. required_capacity includes NUL; truncated messages
 * must remain valid NUL-terminated UTF-8. A zero capacity allows NULL message. */
typedef struct pyrpmd_error_v2 {
    char *message;
    int64_t capacity;
    int64_t required_capacity;
} pyrpmd_error_v2;

typedef struct pyrpmd_create_options_v2 {
    int32_t abi_version;
    int32_t struct_size;
    const char *config_json; /* UTF-8 JSON, no trailing NUL required */
    int64_t config_json_bytes;
} pyrpmd_create_options_v2;

typedef struct pyrpmd_pes_input_v2 {
    int32_t abi_version;
    int32_t struct_size;
    int32_t layout;
    int32_t scalar_type;
    int64_t nstructures; /* B >= 1 */
    int64_t natoms;      /* N >= 1, same composition in every frame */
    const double *positions; /* unwrapped Cartesian bohr, [B,N,3], C order */
    int64_t positions_count;  /* exactly B*N*3 */
    const double *cells; /* bohr, [B,3,3], C order, lattice vectors as ROWS */
    int64_t cells_count; /* exactly B*9; or 0 and NULL for no cell */
    const int32_t *pbc;  /* [B,3], C order, exactly 0 or 1 in each entry */
    int64_t pbc_count;   /* exactly B*3, including nonperiodic frames */
} pyrpmd_pes_input_v2;

typedef struct pyrpmd_pes_output_v2 {
    int32_t abi_version;
    int32_t struct_size;
    double *energies; /* hartree, [B] */
    int64_t energies_count;
    double *forces; /* hartree/bohr, [B,N,3], F = -dE/dq */
    int64_t forces_count;
    pyrpmd_structure_status_v2 *structure_status; /* [B], every entry written */
    int64_t structure_status_count;
    char *structure_messages; /* optional per-frame NUL-terminated UTF-8 */
    int64_t structure_message_stride; /* bytes per frame, including NUL */
    int64_t structure_messages_capacity; /* >= B*stride; guard overflow */
    double *virials; /* hartree, [B,3,3], C order; W = -dE/dstrain */
    int64_t virials_count; /* exactly B*9; or 0 and NULL to omit */
} pyrpmd_pes_output_v2;

/* All arrays are borrowed only for this call. Check ABI version/struct_size
 * before inspecting other members; reject negative sizes and overflowing
 * products before pointer arithmetic. Cells must be finite and nonsingular.
 * A frame without a cell must have all PBC flags zero. Supplied cells may
 * have full, partial or no periodicity; never infer flags from the cell.
 *
 * Virial request is valid only with metadata capabilities.virial=true. W has
 * no kinetic, spring, bias or 1/P term. It includes the full explicit cell
 * dependence at fixed fractional coordinates, not merely sum(q outer F).
 * Missing/unsupported virial must never be represented by a fabricated zero.
 *
 * A nonzero global return invalidates all outputs. Global OK means every
 * status is defined, not that every frame succeeded. Failed frames have NaN
 * energies, forces and requested virials, and a nonzero per-frame status.
 * Message fields are all NULL/zero when omitted; otherwise stride >= 1.
 */
int32_t pyrpmd_pes_abi_version_v2(void);
pyrpmd_rc_v2 pyrpmd_pes_create_v2(
    const pyrpmd_create_options_v2 *options,
    pyrpmd_pes_context_v2 **out_context,
    pyrpmd_error_v2 *error);
pyrpmd_rc_v2 pyrpmd_pes_metadata_v2(
    const pyrpmd_pes_context_v2 *context,
    char *json_buffer, int64_t capacity, int64_t *required_capacity,
    pyrpmd_error_v2 *error);
pyrpmd_rc_v2 pyrpmd_pes_evaluate_v2(
    pyrpmd_pes_context_v2 *context,
    const pyrpmd_pes_input_v2 *input,
    pyrpmd_pes_output_v2 *output,
    pyrpmd_error_v2 *error);
/* Must accept NULL, release all resources and never throw or abort. */
void pyrpmd_pes_destroy_v2(pyrpmd_pes_context_v2 *context);

#ifdef __cplusplus
}
#endif
#endif
