"""Portable oversubscription flags for tests that spawn mpiexec."""

import pytest


@pytest.mark.parametrize("version, flags", [
    ("mpiexec (Open MPI) 5.0.11\n\nReport bugs to https://www.open-mpi.org/community/help/", ["--map-by", ":OVERSUBSCRIBE"]),
    ("mpiexec (Open MPI) 6.0.0", ["--map-by", ":OVERSUBSCRIBE"]),
    ("mpiexec (OpenRTE) 4.1.6\n", ["--oversubscribe"]),
    ("mpirun (Open MPI) 4.1.2", ["--oversubscribe"]),
    ("HYDRA build details:\n    Version: 4.2.0\n    Release Date: unreleased", []),
    ("", []),
])
def test_launch_prefix_matches_launcher_family(mpi_launch, version, flags):
    assert mpi_launch("/usr/bin/mpiexec", 4, version_text=version) == ["/usr/bin/mpiexec", *flags, "-n", "4"]


def test_unparsed_open_mpi_version_uses_the_current_spelling(mpi_launch):
    assert mpi_launch("mpiexec", 2, version_text="Open MPI development build") == [
        "mpiexec", "--map-by", ":OVERSUBSCRIBE", "-n", "2"]
