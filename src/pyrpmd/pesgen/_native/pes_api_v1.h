/*
 * PyRPMD implemented, versioned PES contract, version 1.
 * Versioned physical PES plugin ABI. Implemented by the PyRPMD ctypes loader.
 * See ABI.md for normative units, ownership, concurrency, and error semantics.
 *
 * C11 and C++17 compatible. All indices are zero based. Arrays are host-memory
 * IEEE-754 binary64, C contiguous. No device pointers or asynchronous lifetime.
 * Integer ABI fields are signed for standard Fortran ISO_C_BINDING interop.
 * Versions/sizes/counts/lengths/capacities must be nonnegative; B,N must be > 0.
 * Reject negative values and check products against INT64_MAX before multiplying.
 */
#ifndef PYRPMD_PES_API_V1_H
#define PYRPMD_PES_API_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYRPMD_PES_ABI_VERSION_V1 INT32_C(1)
#define PYRPMD_LAYOUT_BNC_V1 INT32_C(1) /* [structure][atom][xyz] */
#define PYRPMD_SCALAR_FLOAT64_V1 INT32_C(1)

/* Fixed-width integer return values, not implementation-sized C enums. */
typedef int32_t pyrpmd_rc_v1;
#define PYRPMD_OK_V1 INT32_C(0)
#define PYRPMD_INVALID_ARGUMENT_V1 INT32_C(1)
#define PYRPMD_ABI_MISMATCH_V1 INT32_C(2)
#define PYRPMD_INVALID_CONFIG_V1 INT32_C(3)
#define PYRPMD_BUFFER_TOO_SMALL_V1 INT32_C(4)
#define PYRPMD_BACKEND_FAILURE_V1 INT32_C(5)
#define PYRPMD_INTERNAL_ERROR_V1 INT32_C(6)

/* Used only after evaluate returns PYRPMD_OK_V1. */
typedef int32_t pyrpmd_structure_status_v1;
#define PYRPMD_STRUCTURE_OK_V1 INT32_C(0)
#define PYRPMD_STRUCTURE_FAILED_V1 INT32_C(1)
#define PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V1 INT32_C(2)
#define PYRPMD_STRUCTURE_NONFINITE_V1 INT32_C(3)

/* The provider defines the struct body privately; callers never dereference it. */
typedef struct pyrpmd_pes_context_v1 pyrpmd_pes_context_v1;

/*
 * Optional diagnostic sink, initialized by the caller before each API call.
 * If capacity > 0, message points to capacity writable bytes. If capacity == 0,
 * message may be NULL. required_capacity is written by the callee and includes
 * the final NUL. Truncated messages remain NUL terminated and valid UTF-8.
 * The entire error pointer may be NULL to suppress diagnostics.
 */
typedef struct pyrpmd_error_v1 {
    char *message;
    int64_t capacity;
    int64_t required_capacity;
} pyrpmd_error_v1;

typedef struct pyrpmd_create_options_v1 {
    int32_t abi_version; /* PYRPMD_PES_ABI_VERSION_V1 */
    int32_t struct_size; /* (int32_t)sizeof(pyrpmd_create_options_v1) */
    /* UTF-8 JSON object; exact byte length, no terminating NUL required. */
    const char *config_json;
    int64_t config_json_bytes;
} pyrpmd_create_options_v1;

typedef struct pyrpmd_pes_input_v1 {
    int32_t abi_version;
    int32_t struct_size; /* (int32_t)sizeof(pyrpmd_pes_input_v1) */
    int32_t layout;      /* PYRPMD_LAYOUT_BNC_V1 */
    int32_t scalar_type; /* PYRPMD_SCALAR_FLOAT64_V1 */
    int64_t nstructures; /* B >= 1 */
    int64_t natoms;      /* N >= 1; must match the context metadata */
    const double *positions; /* bohr, [B,N,3], read only */
    int64_t positions_count; /* exactly B*N*3 elements, NOT bytes */
} pyrpmd_pes_input_v1;

typedef struct pyrpmd_pes_output_v1 {
    int32_t abi_version;
    int32_t struct_size; /* (int32_t)sizeof(pyrpmd_pes_output_v1) */
    double *energies;     /* hartree, [B] */
    int64_t energies_count; /* exactly B */
    double *forces;       /* hartree/bohr, [B,N,3], F = -dV/dq */
    int64_t forces_count; /* exactly B*N*3 */
    pyrpmd_structure_status_v1 *structure_status; /* [B] */
    int64_t structure_status_count; /* exactly B */
    /*
     * Optional per-structure UTF-8 messages. Frame b starts at
     * structure_messages + b*structure_message_stride. The stride is in bytes
     * and includes room for the NUL. All three members must be zero/NULL when
     * messages are omitted. Otherwise stride >= 1 and capacity >= B*stride.
     * Failed-frame messages may be truncated; status codes are authoritative.
     */
    char *structure_messages;
    int64_t structure_message_stride;
    int64_t structure_messages_capacity;
} pyrpmd_pes_output_v1;

/* Required exported symbols. No C++ exception may escape any of these calls. */
int32_t pyrpmd_pes_abi_version_v1(void);

/*
 * On failure *out_context is NULL and all partial resources are released.
 * On success the caller owns one opaque context until destroy.
 */
pyrpmd_rc_v1 pyrpmd_pes_create_v1(
    const pyrpmd_create_options_v1 *options,
    pyrpmd_pes_context_v1 **out_context,
    pyrpmd_error_v1 *error);

/*
 * Return immutable resolved metadata as UTF-8 JSON (schema in ABI.md).
 * required_capacity is mandatory and includes the final NUL. A sizing query
 * with json_buffer == NULL and capacity == 0 returns BUFFER_TOO_SMALL.
 * Insufficient capacity never returns partial JSON: if capacity > 0, the first
 * byte is set to NUL. On success all JSON bytes and the NUL are written.
 */
pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(
    const pyrpmd_pes_context_v1 *context,
    char *json_buffer,
    int64_t capacity,
    int64_t *required_capacity,
    pyrpmd_error_v1 *error);

/*
 * Synchronous batch evaluation. OK means that every frame has a valid status,
 * not that every frame succeeded. Failed frames have NaN energy and forces.
 * A non-OK global return invalidates the entire batch, including frame status.
 * This interface returns physical PES energy/force only: no springs, bias,
 * constraint force, or 1/P scaling. All output memory belongs to the caller.
 */
pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(
    pyrpmd_pes_context_v1 *context,
    const pyrpmd_pes_input_v1 *input,
    pyrpmd_pes_output_v1 *output,
    pyrpmd_error_v1 *error);

/*
 * Accepts NULL as a no-op. Must not throw, abort, or leave a live context.
 * No concurrent call using this context is permitted during destruction.
 */
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1 *context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PYRPMD_PES_API_V1_H */
