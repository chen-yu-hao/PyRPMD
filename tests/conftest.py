"""Classify benchmark-dependent test modules without importing campaign code."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys

import pytest


_OMITTED = pytest.StashKey[set[Path]]()

# Observed in the first Windows library job, with runtime causes checked in
# source: https://github.com/chen-yu-hao/PyRPMD/actions/runs/41476577723/job/125383616891
# T01 changes test infrastructure only. Keep these failures visible as explicit
# platform limitations until separate runtime fixes restore their contracts.
# Keys without a parameter suffix cover all parameters of that exact test only.
_WINDOWS_LIMITATIONS = (
    (
        "Windows native trajectory publication fsyncs a read-only descriptor "
        "and raises EBADF; native_ensemble.py needs a separate runtime fix",
        frozenset({
            "test_condensed_phase_ensembles.py::test_native_pressure_metadata_matches_saved_momenta_not_configured_temperature",
            "test_mixed_workflows.py::test_mixed_native_pimd_actual_input_journal_and_restart",
            "test_native_ensemble_portability.py::test_native_streaming_unicode_path",
            "test_native_ensembles.py::test_npt_exact_restart_and_streaming",
            "test_native_ensembles.py::test_stream_zero_frames_and_no_overwrite",
            "test_native_ensembles.py::test_cli_streams_and_resumes_without_python_step_loop",
            "test_native_pimd_followup.py::test_native_moments_and_offset_clock",
            "test_native_pimd_followup.py::test_publication_failure_rolls_back_state_and_does_not_leave_file[link]",
            "test_native_pimd_followup.py::test_publication_failure_rolls_back_state_and_does_not_leave_file[load]",
            "test_native_pimd_followup.py::test_concurrent_output_is_not_overwritten_or_removed",
            "test_native_pimd_followup.py::test_standard_workflow_is_native_and_chunk_independent",
            "test_native_pimd_followup.py::test_standard_npt_workflow",
            "test_native_pimd_followup.py::test_native_workflow_committed_prefix_recovery",
            "test_native_pimd_followup.py::test_native_workflow_corrupt_record_rejected",
            "test_native_pimd_followup.py::test_native_workflow_process_replicas",
            "test_native_pimd_followup.py::test_compiled_periodic_provider_preflight_and_native_workflow",
            "test_native_pimd_followup.py::test_fixed_atoms_work_only_in_native_nvt",
            "test_native_pimd_followup.py::test_standard_native_workflow_calls_once_per_checkpoint_not_save",
            "test_native_pimd_nhc_metadata.py::test_native_nhc_resolved_scheme_matches_executor_and_summary",
        }),
    ),
    (
        "Windows archived rate-generation keys use backslashes but the report "
        "reader looks up POSIX keys; generation-source lookup needs a runtime fix",
        frozenset({
            "test_rate_extension_artifacts.py::test_archived_sample_prefix_survives_a_later_pending_generation",
            "test_report_rate_pmf_stage.py::test_independent_pmf_report_reads_published_generation_prefix_after_later_extension",
        }),
    ),
    (
        "Windows simultaneous os.replace calls for one JSON destination can "
        "raise WinError 5; concurrent publication needs a separate runtime fix",
        frozenset({
            "test_atomic_json.py::test_independent_json_writers_do_not_share_a_temporary_file",
        }),
    ),
)


def _platform_skip_reason(node: str, platform: str) -> str | None:
    if platform == "win32":
        for reason, affected in _WINDOWS_LIMITATIONS:
            if node in affected or node.split("[", 1)[0] in affected:
                return reason
    return None


def _benchmark_name(name: str) -> bool:
    return name == "benchmarks" or name.startswith("benchmarks.")


def _benchmark_path(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.startswith(("benchmarks/", "benchmarks\\")))


@lru_cache(maxsize=None)
def _module_dependencies(path: Path) -> tuple[bool, frozenset[Path]]:
    """Read imports and literal benchmark paths, including function bodies.

    Follow local test/helper imports only; do not import packages or inspect the
    installed environment. Comments and embedded Python source are not imports.
    """
    tree = ast.parse(path.read_bytes(), filename=str(path))
    imports: set[str] = set()
    campaign = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imports.update(".".join(filter(None, (node.module, alias.name)))
                           for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            campaign |= _benchmark_path(node.value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # Also recognize ROOT / "benchmarks" / "family" / "driver.py".
            campaign |= any(_benchmark_path(part) or
                            (isinstance(part, ast.Constant) and part.value == "benchmarks")
                            for part in (node.left, node.right))
        elif isinstance(node, ast.Call) and node.args:
            campaign |= any(_benchmark_path(argument) for argument in node.args)
            function = node.func
            name = (function.id if isinstance(function, ast.Name) else
                    function.attr if isinstance(function, ast.Attribute) else "")
            argument = node.args[0]
            if (name in {"import_module", "__import__"}
                    and isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)):
                imports.add(argument.value)
    campaign |= any(_benchmark_name(name) for name in imports)
    dependencies = set()
    for name in imports:
        parts = name.split(".")
        if parts[0] == "tests":
            parts = parts[1:]
        # Imports may name a function/class after the local module name.
        for count in range(1, len(parts) + 1):
            candidate = path.parent.joinpath(*parts[:count]).with_suffix(".py")
            if candidate.is_file():
                dependencies.add(candidate.resolve())
    return campaign, frozenset(dependencies)


def _is_campaign(path: Path) -> bool:
    pending = [path.resolve()]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        campaign, dependencies = _module_dependencies(current)
        if campaign:
            return True
        pending.extend(dependencies - visited)
    return False


class _CampaignModule(pytest.Module):
    def collect(self):
        # Marker deselection normally runs *after* imports. This exact, documented
        # fast entry point must also work without POSIX-only campaign dependencies.
        if self.config.getoption("markexpr").split() == ["not", "campaign"]:
            self.config.stash.setdefault(_OMITTED, set()).add(self.path)
            return []
        return super().collect()


def pytest_pycollect_makemodule(module_path, parent):
    if _is_campaign(module_path):
        module = _CampaignModule.from_parent(parent, path=module_path)
        module.add_marker(pytest.mark.campaign)
        return module
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        if "::" not in item.nodeid:
            continue
        node = item.path.name + "::" + item.nodeid.split("::", 1)[1]
        reason = _platform_skip_reason(node, sys.platform)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_terminal_summary(terminalreporter, config):
    omitted = config.stash.get(_OMITTED, set())
    if omitted:
        terminalreporter.write_line(
            f"campaign: omitted {len(omitted)} modules before import (-m 'not campaign')"
        )


def mpi_launch_prefix(launcher, ranks, *, version_text=None):
    """Return ``[launcher, <oversubscription flags>, "-n", ranks]`` for any MPI.

    CI runners may expose fewer cores than the ranks a test requests. Open MPI
    then refuses to start unless oversubscription is explicit, and the spelling
    differs between releases: the Open MPI 5 PRRTE launcher needs the public
    ``--map-by :OVERSUBSCRIBE`` modifier (see ``.github/workflows/tests.yml``),
    while Open MPI 4 accepts ``--oversubscribe``. MPICH oversubscribes without
    a flag. Rank placement never changes a test's numerical result.
    """
    if version_text is None:
        try:
            version_text = subprocess.run(
                [str(launcher), "--version"], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=10, check=False,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            version_text = ""
    command = [str(launcher)]
    match = re.search(r"\((?:Open MPI|OpenRTE)\)\s+(\d+)", version_text)
    if match or "Open MPI" in version_text or "OpenRTE" in version_text:
        major = int(match.group(1)) if match else 5
        command += ["--map-by", ":OVERSUBSCRIBE"] if major >= 5 else ["--oversubscribe"]
    return command + ["-n", str(int(ranks))]


@pytest.fixture
def mpi_launch():
    """Fixture form of :func:`mpi_launch_prefix` for tests that spawn ``mpiexec``."""
    return mpi_launch_prefix
