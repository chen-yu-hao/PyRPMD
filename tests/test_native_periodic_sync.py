"""Compiled periodic providers expose the accepted native cell to Python readers."""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble, NativePeriodicPES


@pytest.fixture(scope="module")
def cell_library(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required")
    directory = tmp_path_factory.mktemp("periodic-cell-sync")
    source, library = directory/"volume.c", directory/"volume.so"
    source.write_text(r'''
#include <stdint.h>
#include <stdio.h>
static int calls=0, failure=-1;
void set_failure(int n) { calls=0; failure=n; }
int32_t evaluate(void* context, int64_t beads, int64_t atoms,
        const double* q, const double* h, const int32_t* pbc,
        double* e, double* f, double* virial, char* error, int64_t capacity) {
    ++calls;
    if(calls==failure) { snprintf(error, (size_t)capacity, "controlled cell PES failure"); return 1; }
    double volume=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    for(int64_t b=0; b<beads; ++b) {
        e[b]=.2*volume;
        for(int64_t j=0; j<3*atoms; ++j) f[b*3*atoms+j]=0.;
        for(int j=0; j<9; ++j) virial[9*b+j]=(j%4==0 ? -.2*volume : 0.);
    }
    return 0;
}
''')
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", str(source), "-o", str(library)], check=True)
    return library


def engine(library):
    provider = NativePeriodicPES(library, "evaluate", 2)
    provider.lib.set_failure(-1)
    e = NativeEnsemble(np.ones((3, 2, 3))*.2, [1., 2.], 2., .01,
                       p=np.ones((3, 2, 3))*.1, cell=np.eye(3)*2,
                       thermostat="pile-l", barostat={"mode": "mttk", "pressure": .1, "tau": 1.},
                       potential=provider, seed=13)
    return e, provider


def check_provider(e, provider):
    state = e.state()
    np.testing.assert_array_equal(provider.cell, state["cell"])
    np.testing.assert_array_equal(provider.pbc, [True, True, True])
    evaluated = provider.evaluate(state["q"])
    np.testing.assert_array_equal(evaluated.energy, state["energy"])
    np.testing.assert_array_equal(evaluated.virial, state["virial"])


def test_periodic_provider_tracks_run_restore_and_explicit_state_import(cell_library):
    e, provider = engine(cell_library)
    check_provider(e, provider)  # Constructor also binds previously unbound providers.
    e.run(4)
    check_provider(e, provider)
    saved = e.state()
    e.run(6)
    assert not np.array_equal(saved["cell"], e.state()["cell"])
    check_provider(e, provider)
    e.restore(saved)
    check_provider(e, provider)
    e.set_state(saved["q"]+.1, saved["p"], 11)
    check_provider(e, provider)


def test_periodic_provider_tracks_accepted_prefix_and_failed_block(cell_library):
    e, provider = engine(cell_library)
    e.run(2)
    before = e.state()
    provider.lib.set_failure(3)
    with pytest.raises(RuntimeError, match="controlled cell PES failure"):
        e.run(6)
    assert e.state()["digest"] == before["digest"]
    check_provider(e, provider)
    provider.lib.set_failure(3)
    prefix = e.run_prefix(6)
    assert prefix["completed"] == 2 and prefix["error"]
    check_provider(e, provider)


def test_binding_publication_failure_restores_native_and_provider_state(cell_library, monkeypatch):
    e, provider = engine(cell_library)
    e.run(2)
    before = e.state()
    original = provider.bind_system

    def bind_then_fail(system):
        original(system)
        raise RuntimeError("controlled cell publication failure")

    monkeypatch.setattr(provider, "bind_system", bind_then_fail)
    with pytest.raises(RuntimeError, match="cell publication failure"):
        e.run(4)
    assert e.state()["digest"] == before["digest"]
    np.testing.assert_array_equal(provider.cell, before["cell"])


def test_common_dynamics_energy_reads_current_native_pressure_cell(cell_library):
    provider = NativePeriodicPES(cell_library, "evaluate", 2)
    provider.lib.set_failure(-1)
    dyn = Dynamics(provider, [1., 2.], 2., 3, .01, seed=13,
                   thermostat="pile-l", barostat={"mode": "mttk", "pressure": .1, "tau": 1.},
                   cell=np.eye(3)*2, pbc=[True]*3, potential_identity="volume-cell-control")
    state = State(np.ones((3, 2, 3))*.2, np.ones((3, 2, 3))*.1)
    result = dyn.propagate(state, 6, backend="native")
    np.testing.assert_array_equal(provider.cell, dyn.cell)
    assert dyn.energy(state) == pytest.approx(result.energies[-1], abs=1e-13)
