#!/usr/bin/env bash
# Run from the repository root. Native binaries are rebuilt in place.
set -euo pipefail
thermal_python=${PYRPMD_THERMAL_PYTHON:-python}
thermal_output=${PYRPMD_THERMAL_VALIDATION:-docs/validation/thermal-kernel-performance-2026-11-04}
mkdir -p "$thermal_output"
for thermal_variant in no-openmp fma openmp; do
    thermal_no_openmp=0
    thermal_flags='-march=native -ffp-contract=off'
    if [[ $thermal_variant == no-openmp ]]; then thermal_no_openmp=1; fi
    if [[ $thermal_variant == fma ]]; then thermal_flags='-march=native -mfma -ffp-contract=fast'; fi
    (
        set -x
        env PYRPMD_NO_OPENMP="$thermal_no_openmp" CXXFLAGS="$thermal_flags" \
            "$thermal_python" setup.py build_ext --inplace --force
    ) > "$thermal_output/build-$thermal_variant.log" 2>&1
    (
        set -x
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$thermal_python" -m pytest -q \
            tests/test_native_thermal_performance.py tests/test_native_ensembles.py \
            tests/test_native_nhc.py tests/test_native_prefix.py \
            tests/test_native_periodic_sync.py tests/test_native_pimd_followup.py \
            tests/test_python_nhc_fallback.py tests/test_composed_native_observations.py \
            tests/test_propagation_composition.py tests/test_builtin_native_pes.py
    ) > "$thermal_output/tests-$thermal_variant.log" 2>&1
    tail -1 "$thermal_output/tests-$thermal_variant.log"
done
