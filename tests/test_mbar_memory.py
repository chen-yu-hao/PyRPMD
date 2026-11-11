"""MBAR admission uses reclaimable RAM and respects finite job limits."""

from io import StringIO

import pytest

from pyrpmd.analysis import rate_pmf


GiB = 1024**3


def memory_view(monkeypatch, available, files, free=GiB):
    def meminfo(*args, **kwargs):
        if available is None:
            raise OSError("no proc meminfo")
        return StringIO(f"MemFree: {free // 1024} kB\nMemAvailable: {available // 1024} kB\n")

    def read(path, *args, **kwargs):
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(str(path)) from None

    monkeypatch.setattr("builtins.open", meminfo)
    monkeypatch.setattr(rate_pmf.Path, "read_text", read)
    monkeypatch.setattr(rate_pmf.os, "sysconf", lambda key: 4096 if key == "SC_PAGE_SIZE" else free // 4096)


def test_reclaimable_cache_does_not_needlessly_serialize_bootstrap(monkeypatch):
    memory_view(monkeypatch, 8 * GiB, {})
    available = rate_pmf._available_memory_bytes()
    assert available == 8 * GiB
    assert rate_pmf._cap_bootstrap_workers(4, 200, GiB // 8, available) == 4


@pytest.mark.parametrize("host_available,expected", [(8 * GiB, 2 * GiB), (GiB, GiB)])
def test_v2_parent_limit_accounts_for_usage_and_siblings(monkeypatch, host_available, expected):
    memory_view(monkeypatch, host_available, {
        "/proc/self/cgroup": "0::/jobs/run\n",
        "/sys/fs/cgroup/jobs/run/memory.max": "max\n",
        "/sys/fs/cgroup/jobs/memory.max": str(3 * GiB),
        "/sys/fs/cgroup/jobs/memory.current": str(GiB),
    })
    assert rate_pmf._available_memory_bytes() == expected


def test_v1_limit_and_unlimited_parent(monkeypatch):
    memory_view(monkeypatch, 8 * GiB, {
        "/proc/self/cgroup": "4:cpu:/jobs/run\n5:memory:/jobs/run\n",
        "/sys/fs/cgroup/memory/jobs/run/memory.limit_in_bytes": str(4 * GiB),
        "/sys/fs/cgroup/memory/jobs/run/memory.usage_in_bytes": str(3 * GiB),
        "/sys/fs/cgroup/memory/jobs/memory.limit_in_bytes": str(2**63 - 4096),
        "/sys/fs/cgroup/memory/jobs/memory.usage_in_bytes": str(GiB),
    })
    assert rate_pmf._available_memory_bytes() == GiB


def test_exhausted_cgroup_cannot_be_hidden_by_free_host_ram(monkeypatch):
    memory_view(monkeypatch, 8 * GiB, {
        "/proc/self/cgroup": "0::/job\n",
        "/sys/fs/cgroup/job/memory.max": str(GiB),
        "/sys/fs/cgroup/job/memory.current": str(GiB + 1),
    })
    assert rate_pmf._available_memory_bytes() == 0


def test_portable_fallback_without_proc(monkeypatch):
    memory_view(monkeypatch, None, {})
    assert rate_pmf._available_memory_bytes() == GiB
    monkeypatch.setattr(rate_pmf.os, "sysconf", lambda key: -1)
    assert rate_pmf._available_memory_bytes() is None
