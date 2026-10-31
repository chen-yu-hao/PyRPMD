/* Optional synchronous native thermal-engine callback, version 1.
 * This is NOT a replacement/version bump for the public PES plugin ABI.
 * Units: bohr, hartree, atomic time; physical (not bead-scaled) E and F.
 * cell[9] contains ROW lattice vectors, q[B,N,3] is unwrapped, pbc[3] is 0/1.
 * virial[B,3,3] = -dU/d epsilon for r'=(I+epsilon)r and
 * cell'=cell*(I+epsilon)^T. Includes explicit cell dependence, excludes
 * kinetic, ring springs and bias. Missing virial is not equivalent to zero.
 * Fill every output; return zero on success. On error return nonzero and
 * write a NUL-terminated diagnostic (capacity includes NUL). No C++ exception,
 * Python callback, asynchronous use, or retained pointer may cross this ABI.
 * The caller owns the buffers, pins the library and serializes the context.
 */
#ifndef PYRPMD_THERMAL_PES_H
#define PYRPMD_THERMAL_PES_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int32_t (*pyrpmd_thermal_evaluate_v1)(
    void *context, int64_t beads, int64_t natoms, const double *q,
    const double *cell, const int32_t *pbc,
    double *energy, double *force, double *virial,
    char *error, int64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
