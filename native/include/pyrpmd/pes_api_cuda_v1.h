/*
 * PyRPMD physical PES CUDA device-memory extension contract, version 1.
 * Optional companion symbol table alongside PES ABI v1 (pes_api_v1.h) or
 * PES ABI v2 (pes_api_v2.h) for stream-ordered, zero-copy GPU evaluation.
 * See native/ABI.md for normative units, stream ordering, and error semantics.
 *
 * C11 and C++17 compatible. Header consumers do NOT require CUDA Toolkit
 * headers: cudaStream_t is passed as an opaque void* handle.
 *
 * All device arrays are IEEE-754 binary64 (or int32 for PBC/status), C-order
 * contiguous allocations residing on device_id. Counts are numbers of elements,
 * NOT bytes. All GPU reads and writes performed by an evaluation call MUST be
 * stream-ordered with respect to cuda_stream before the function returns.
 */
#ifndef PYRPMD_PES_API_CUDA_V1_H
#define PYRPMD_PES_API_CUDA_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYRPMD_PES_CUDA_ABI_VERSION_V1 INT32_C(1)
#define PYRPMD_LAYOUT_BNC_CUDA_V1 INT32_C(1)     /* [structure][atom][xyz], C order */
#define PYRPMD_SCALAR_FLOAT64_CUDA_V1 INT32_C(1) /* IEEE-754 binary64 */

/* Fixed-width integer return values matching PES ABI v1/v2 conventions. */
typedef int32_t pyrpmd_cuda_rc_v1;
#define PYRPMD_CUDA_OK_V1 INT32_C(0)
#define PYRPMD_CUDA_INVALID_ARGUMENT_V1 INT32_C(1)
#define PYRPMD_CUDA_ABI_MISMATCH_V1 INT32_C(2)
#define PYRPMD_CUDA_INVALID_CONFIG_V1 INT32_C(3)
#define PYRPMD_CUDA_BUFFER_TOO_SMALL_V1 INT32_C(4)
#define PYRPMD_CUDA_BACKEND_FAILURE_V1 INT32_C(5)
#define PYRPMD_CUDA_INTERNAL_ERROR_V1 INT32_C(6)

/* Per-structure status values written on device when d_structure_status != NULL. */
typedef int32_t pyrpmd_cuda_structure_status_v1;
#define PYRPMD_CUDA_STRUCTURE_OK_V1 INT32_C(0)
#define PYRPMD_CUDA_STRUCTURE_FAILED_V1 INT32_C(1)
#define PYRPMD_CUDA_STRUCTURE_OUT_OF_DOMAIN_V1 INT32_C(2)
#define PYRPMD_CUDA_STRUCTURE_NONFINITE_V1 INT32_C(3)

/*
 * Optional host-memory diagnostic sink for synchronous launch/validation errors.
 * Same layout and UTF-8 NUL-termination rules as pyrpmd_error_v1 / v2.
 */
typedef struct pyrpmd_cuda_error_v1 {
    char *message;
    int64_t capacity;
    int64_t required_capacity;
} pyrpmd_cuda_error_v1;

typedef struct pyrpmd_pes_cuda_input_v1 {
    int32_t cuda_abi_version;  /* PYRPMD_PES_CUDA_ABI_VERSION_V1 */
    int32_t struct_size;       /* (int32_t)sizeof(pyrpmd_pes_cuda_input_v1) */
    int32_t layout;            /* PYRPMD_LAYOUT_BNC_CUDA_V1 */
    int32_t scalar_type;       /* PYRPMD_SCALAR_FLOAT64_CUDA_V1 */
    int32_t device_id;         /* CUDA device ordinal >= 0; must match context */
    int32_t flags;             /* Reserved bitmask; must be 0 in v1 */
    void *cuda_stream;         /* Borrowed cudaStream_t on device_id (NULL = default stream 0) */
    int64_t nstructures;       /* B >= 1 */
    int64_t natoms;            /* N >= 1; must match context metadata */
    const double *d_positions; /* Device pointer: unwrapped bohr, [B,N,3], C order, read-only */
    int64_t positions_count;   /* Exactly B*N*3 elements, NOT bytes */
    const double *d_cells;     /* Optional device pointer: bohr, [B,3,3] rows; NULL when omitted */
    int64_t cells_count;       /* Exactly B*9, or 0 when d_cells == NULL */
    const int32_t *d_pbc;      /* Optional device pointer: int32 {0,1}, [B,3]; NULL when omitted */
    int64_t pbc_count;         /* Exactly B*3, or 0 when d_pbc == NULL */
} pyrpmd_pes_cuda_input_v1;

typedef struct pyrpmd_pes_cuda_output_v1 {
    int32_t cuda_abi_version;  /* PYRPMD_PES_CUDA_ABI_VERSION_V1 */
    int32_t struct_size;       /* (int32_t)sizeof(pyrpmd_pes_cuda_output_v1) */
    double *d_energies;        /* Device pointer: hartree, [B] */
    int64_t energies_count;    /* Exactly B */
    double *d_forces;          /* Device pointer: hartree/bohr, [B,N,3], F = -dE/dq */
    int64_t forces_count;      /* Exactly B*N*3 */
    double *d_virials;         /* Optional device pointer: hartree, [B,3,3]; NULL to omit */
    int64_t virials_count;     /* Exactly B*9, or 0 when d_virials == NULL */
    /*
     * Optional device pointer [B] for per-frame status codes. Callers such as
     * TorchDynamics that accumulate finiteness checks on device across multi-step
     * integration blocks may pass NULL and structure_status_count == 0 to avoid
     * extra status buffer writes and any per-step host synchronization. Even when
     * omitted, failed/nonfinite frames MUST write NaN into d_energies, d_forces,
     * and requested d_virials on cuda_stream.
     */
    pyrpmd_cuda_structure_status_v1 *d_structure_status;
    int64_t structure_status_count; /* Exactly B, or 0 when d_structure_status == NULL */
} pyrpmd_pes_cuda_output_v1;

/*
 * Optional exported symbols alongside pyrpmd_pes_*_v1 or pyrpmd_pes_*_v2.
 * Context creation, metadata query, and destruction reuse the base v1/v2 ABI,
 * and context metadata must declare:
 *   "execution_device": "cuda:<device_id>"
 *   "capabilities": { "cuda_device_memory": true, ... }
 *
 * No C++ exception may cross these entrypoints. Device pointers are borrowed
 * only for work enqueued onto input->cuda_stream (or internally bridged to and
 * from input->cuda_stream via CUDA events before returning). Providers must NOT
 * synchronize the host thread per step unless required to report a launch error,
 * and must NOT implicitly use the legacy default stream when a non-NULL
 * cuda_stream is supplied.
 */
int32_t pyrpmd_pes_cuda_abi_version_v1(void);

pyrpmd_cuda_rc_v1 pyrpmd_pes_evaluate_cuda_v1(
    void *context,
    const pyrpmd_pes_cuda_input_v1 *input,
    pyrpmd_pes_cuda_output_v1 *output,
    pyrpmd_cuda_error_v1 *error);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PYRPMD_PES_API_CUDA_V1_H */
