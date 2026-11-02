"""Collect the frozen HCH4 checks through the standard ``testpaths=tests``.

The original executed functions and fixture stay in the benchmark module;
imports preserve their source hashes, parameter marks and library lookup.
No optional external PES is fetched by collection or execution.
"""

from benchmarks.hch4.test_native import (
    library,
    test_native_binding_does_not_bypass_overridden_evaluate,
    test_native_binding_shares_lock_and_retains_owner,
    test_paper_mapping_is_explicit_and_uses_all_four_paths,
    test_reference_numbers_have_separate_units_and_rounding,
)

__all__ = [
    "library",
    "test_native_binding_does_not_bypass_overridden_evaluate",
    "test_native_binding_shares_lock_and_retains_owner",
    "test_paper_mapping_is_explicit_and_uses_all_four_paths",
    "test_reference_numbers_have_separate_units_and_rounding",
]
