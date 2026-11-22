"""Coupled electronic states along the 1D path of the subspace pipeline (conical-intersection model)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.subspace_complex.models import abstract_rate_run  # noqa: E402
from benchmarks.subspace_rph.reference import QUANTUM, build_path  # noqa: E402

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.multistate_subspace import MultistateSubspaceModel, nonadiabatic_subspace_factor
from pyrpmd.analysis.subspace_electronic import electronic_options, path_geometries
from pyrpmd.input_schema import InputError
from pyrpmd.input_subspace import rate_subspace_options

CM = 1.0 / 219474.6313632
RPMD = {"nbeads": 16, "windows": 12, "walkers": 8, "burn_in": 20, "sweeps": 40, "trajectories": 48, "seed": 7}


@dataclass(frozen=True)
class CrossingCI2D:
    """Diabats crossing at x = 0, coupled by lam (y - y_c): a conical intersection at (0, y_c).

    Mass-weighted coordinates.  The ground-state reaction path runs near y = 0
    and passes the intersection 2.5 transverse widths away, through an
    avoided crossing of gap ~ 2 lam y_c.
    """

    D: float = 0.04
    width: float = 20.0
    omega: float = 1500.0 * CM
    lam: float = 1.3333e-4
    y_c: float = 30.0
    c: float = 40.0

    @property
    def dimension(self):
        return 2

    def diabatic(self, q):
        q = np.asarray(q, dtype=float)
        x, y = q[..., 0], q[..., 1]
        rise = np.tanh(x / self.width)
        harmonic = 0.5 * self.omega**2 * y**2
        out = np.zeros(x.shape + (2, 2))
        out[..., 0, 0] = 0.5 * self.D * (1 + rise) + harmonic
        out[..., 1, 1] = 0.5 * self.D * (1 - rise) + harmonic
        out[..., 0, 1] = out[..., 1, 0] = self.lam * (y - self.y_c) * np.exp(-((x / self.c) ** 2))
        return out

    def _parts(self, q):
        q = np.asarray(q, dtype=float)
        x, y = q[..., 0], q[..., 1]
        rise = np.tanh(x / self.width)
        envelope = np.exp(-((x / self.c) ** 2))
        half, dhalf = 0.5 * self.D * rise, 0.5 * self.D * (1 - rise**2) / self.width
        v12 = self.lam * (y - self.y_c) * envelope
        dv12x = v12 * (-2.0 * x / self.c**2)
        root = np.sqrt(half**2 + v12**2)
        energy = 0.5 * self.D + 0.5 * self.omega**2 * y**2 - root
        gradient = np.stack([-(half * dhalf + v12 * dv12x) / root,
                             self.omega**2 * y - v12 * self.lam * envelope / root], axis=-1)
        return energy, gradient

    def energy(self, q):
        return self._parts(q)[0]

    def gradient(self, q):
        return self._parts(q)[1]


class Source:
    """Adiabatic data with scrambled signs and state order, counting calls."""

    def __init__(self, model, seed=3, norm=1.0):
        self.model, self.rng, self.norm, self.calls = model, np.random.default_rng(seed), norm, 0

    def __call__(self, geometry):
        self.calls += 1
        energies, vectors = np.linalg.eigh(self.model.diabatic(np.asarray(geometry, dtype=float)))
        order = self.rng.permutation(2)
        signs = self.rng.choice([-1.0, 1.0], size=2)
        return {"energies": energies[order], "vectors": self.norm * (vectors * signs)[:, order]}


def specification(directory, source, **electronic):
    return {"output": {"directory": str(directory), "reuse": electronic.pop("reuse", False)},
            "path": {"source": "mep", "step": 4.0}, "subspace": {"dimension": 1, "environment": "adiabatic"},
            "rpmd": RPMD, "electronic": {"source": "python", "object": source, "nstates": 2, **electronic}}


def reference_factor(model, temperature):
    """eta_NA from the exact diabatic matrix at the 1D model's own grid geometries."""
    run = abstract_rate_run(model, temperature, 16)
    path, info = build_path(run)
    single, quantum, _ = pipeline.path_model(run, path, {**QUANTUM, "environment": "adiabatic", "path_info": info,
                                                         "validity": {}, "hessian_displacement": 5e-3})
    diabatic = model.diabatic(path_geometries(path, run.cartesian_masses, single.axes[0]))
    shift = single.values - np.linalg.eigvalsh(diabatic)[:, 0]
    coupled = MultistateSubspaceModel(single.axes, diabatic + shift[:, None, None] * np.eye(2), single.masses,
                                      single.dividing_surface, single.reactant, single.reactant_reference)
    factor = nonadiabatic_subspace_factor(coupled, run.beta, cap_width=quantum["cap_width"],
                                          energy_floor=quantum["energy_floor"])["factor"]
    return factor, single, path, run


def test_path_geometries_are_the_geometries_of_the_path_model():
    model = CrossingCI2D()
    run = abstract_rate_run(model, 300.0, 16)
    path, info = build_path(run)
    frozen, _, _ = pipeline.path_model(run, path, {**QUANTUM, "environment": "frozen", "path_info": info,
                                                   "validity": {}, "hessian_displacement": 5e-3})
    geometries = path_geometries(path, run.cartesian_masses, frozen.axes[0])
    assert np.allclose(model.energy(geometries), frozen.values, rtol=0, atol=1e-14)


def test_pipeline_applies_the_nonadiabatic_factor_of_a_conical_intersection(tmp_path):
    model = CrossingCI2D()
    exact, _, _, run = reference_factor(model, 300.0)
    report = pipeline.run_subspace_pipeline(specification(tmp_path / "grid", Source(model), points="grid"), run=run)
    record = report["nonadiabatic"]["1"]
    assert report["status"] == "complete" and report["applied_dimension"] == 1
    assert record["valid"] and record["status"] == "complete"
    # Diabatization of scrambled adiabatic data on the model grid is exact.
    assert record["factor"] == pytest.approx(exact, rel=1e-9)
    # Adiabat 0 of the coupled model is the single-surface model of C_S.
    assert abs(record["born_oppenheimer_consistency"]) < 1e-10
    factors = report["applied_factors"]
    assert factors["single_surface"] == pytest.approx(report["corrections"]["1"]["factor"])
    assert factors["nonadiabatic"] == pytest.approx(record["factor"])
    assert factors["total"] == pytest.approx(factors["single_surface"] * factors["nonadiabatic"])
    assert 0.1 < record["factor"] < 0.4
    assert [g["gate"] for g in record["validity"]["gates"]] == [
        "electronic_leakage", "electronic_resolution", "ground_state_mismatch"]
    assert record["electronic"]["ground_state_mismatch_kt"] < 1e-8
    assert record["branching"]["adiabatic0"] > 0.999
    saved = json.loads((tmp_path / "grid" / "subspace_pipeline.json").read_text())
    assert saved["options"]["electronic"]["object"].endswith("Source")

    # 64 electronic points splined onto the grid.
    coarse = pipeline.run_subspace_pipeline(specification(tmp_path / "coarse", Source(model), points=64), run=run)
    assert coarse["nonadiabatic"]["1"]["factor"] == pytest.approx(exact, rel=5e-4)
    assert coarse["nonadiabatic"]["1"]["electronic"]["interpolation_error_kt"] < 0.1


def test_nonadiabatic_factor_tends_to_one_as_the_gap_opens():
    weak, *_ = reference_factor(CrossingCI2D(lam=0.6667e-4), 300.0)
    medium, *_ = reference_factor(CrossingCI2D(), 300.0)
    strong, *_ = reference_factor(CrossingCI2D(lam=5.3332e-4), 300.0)
    assert weak < medium < strong < 1.0
    assert strong > 0.8 and weak < 0.1


def test_cached_electronic_data_are_reused_with_a_reused_single_surface_factor(tmp_path):
    model = CrossingCI2D()
    run = abstract_rate_run(model, 300.0, 16)
    source = Source(model)
    first = pipeline.run_subspace_pipeline(specification(tmp_path, source, points=64, reuse=True), run=run)
    assert source.calls == 64 and first["nonadiabatic"]["1"]["electronic"]["cached"] is False
    second = pipeline.run_subspace_pipeline(specification(tmp_path, source, points=64, reuse=True), run=run)
    assert source.calls == 64, "the cached electronic data must be reused"
    assert second["nonadiabatic"]["1"]["electronic"]["cached"] is True
    assert second["models"]["1"] == first["models"]["1"]  # the 1D factor itself was reused
    assert second["applied_factors"]["total"] == pytest.approx(first["applied_factors"]["total"], rel=1e-12)


def test_file_source_requests_the_geometries_then_uses_external_data(tmp_path):
    model = CrossingCI2D()
    run = abstract_rate_run(model, 300.0, 16)
    data = tmp_path / "electronic.npz"
    spec = specification(tmp_path / "out", None, points=64)
    spec["electronic"] = {"source": "file", "file": str(data), "nstates": 2, "points": 64}
    pending = pipeline.run_subspace_pipeline(spec, run=run)
    assert pending["status"] == "electronic_data_required" and pending["applied_factors"] is None
    request = np.load(pending["nonadiabatic"]["1"]["request"])
    geometries = request["geometries"]
    energies, vectors = np.linalg.eigh(model.diabatic(geometries))
    overlaps = np.einsum("nia,nib->nab", vectors[:-1], vectors[1:])
    np.savez(data, geometries=geometries, energies=energies, overlaps=overlaps)
    done = pipeline.run_subspace_pipeline(spec, run=run)
    python = pipeline.run_subspace_pipeline(specification(tmp_path / "python", Source(model), points=64), run=run)
    assert done["status"] == "complete"
    assert done["nonadiabatic"]["1"]["factor"] == pytest.approx(python["nonadiabatic"]["1"]["factor"], rel=1e-12)


def test_leaking_overlaps_reject_the_factor_unless_allowed(tmp_path):
    model = CrossingCI2D()
    run = abstract_rate_run(model, 300.0, 16)
    leaking = Source(model, norm=0.9)  # singular values 0.81 < 0.9
    report = pipeline.run_subspace_pipeline(specification(tmp_path / "reject", leaking), run=run)
    record = report["nonadiabatic"]["1"]
    assert record["status"] == "invalid" and not record["valid"] and record["factor"] is None
    assert record["diagnostic_factor"] > 0
    assert report["status"] == "invalid_model" and report["applied_dimension"] is None
    assert "nonadiabatic factor" in report["applied_dimension_reason"]
    assert any("dropped 1D" in w for w in report["warnings"])
    allowed = pipeline.run_subspace_pipeline(
        specification(tmp_path / "allow", Source(model, norm=0.9), leakage="allow"), run=run)
    assert allowed["status"] == "complete"
    assert allowed["nonadiabatic"]["1"]["validity"]["overrides"] == ["electronic_leakage"]


def test_electronic_options_and_native_input_are_checked():
    assert electronic_options(None) is None
    assert electronic_options({"enabled": False}) is None
    resolved = electronic_options({"source": "python", "object": "m:f", "nstates": 3, "points": "grid"})
    assert resolved["state"] == 0 and resolved["leakage"] == "reject" and resolved["points"] == "grid"
    for bad, message in (({"source": "python", "nstates": 2}, "needs electronic.object"),
                         ({"source": "file", "nstates": 2}, "needs electronic.file"),
                         ({"source": "python", "object": "m:f"}, "nstates"),
                         ({"source": "python", "object": "m:f", "nstates": 2, "state": 2}, "state"),
                         ({"source": "python", "object": "m:f", "nstates": 2, "spin": 1}, "unknown")):
        with pytest.raises(ValueError, match=message):
            electronic_options(bad)
    config = {"task": "rate", "rate": {"subspace_correction": {
        "electronic": {"source": "backend", "object": "plugin.py:make_backend", "nstates": 2,
                       "arguments": {"basis": "sto-3g"}}}}}
    options = rate_subspace_options(config)
    assert options["electronic"]["arguments"] == {"basis": "sto-3g"} and options["electronic"]["points"] == 64
    config["rate"]["subspace_correction"]["electronic"]["points"] = 4
    with pytest.raises(InputError, match="electronic.points"):
        rate_subspace_options(config)


def test_backend_source_diabatizes_a_toy_avoided_crossing_with_phase_flips(tmp_path):
    from pyrpmd.analysis.multistate_subspace import propagation_diabatization
    from pyrpmd.analysis.subspace_electronic import electronic_line

    options = electronic_options({
        "source": "backend", "object": "pyrpmd.potentials.excited_state.toy:AvoidedCrossingBackend",
        "arguments": {"natoms": 1, "slope": 0.02, "coupling": 0.005, "phase_flip_threshold": 0.2}, "nstates": 2})
    x = np.linspace(-1.5, 1.5, 7)
    path = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    line = electronic_line(options, path, np.ones(3), np.linspace(0.0, 2.999, 40), tmp_path)
    assert line.overlaps.shape == (39, 2, 2)
    assert line.identity["backend"]["backend"] == "toy-avoided-crossing"
    assert (tmp_path / "electronic_1d.npz").is_file()
    position = line.geometries[:, 0]
    common = 0.5 * 0.001 * np.sum(line.geometries**2, axis=1)
    exact = np.zeros((len(position), 2, 2))
    exact[:, 0, 0], exact[:, 1, 1] = 0.02 * position, -0.02 * position
    exact[:, 0, 1] = exact[:, 1, 0] = 0.005
    exact += common[:, None, None] * np.eye(2)
    frame = np.linalg.eigh(exact[0] - common[0] * np.eye(2))[1]  # the backend's states at the anchor
    recovered = propagation_diabatization(line.energies, line.overlaps).diabatic
    assert np.max(np.abs(recovered - np.einsum("ia,nij,jb->nab", frame, exact, frame))) < 1e-12


class CrossingCIDPEM:
    """CrossingCI2D as a one-atom Cartesian DPEM (x, y as above, z a spectator mode)."""

    nstates, natoms = 2, 1
    metadata = {"provider": "test_dpem", "nstates": 2, "natoms": 1, "thread_safety": "reentrant"}

    def __init__(self, base=CrossingCI2D(), omega_z=1200.0 * CM):
        self.base, self.omega_z = base, omega_z

    def evaluate(self, q):
        from pyrpmd.nonadiabatic import DiabaticResult

        q = np.asarray(q, dtype=float)
        x, y, z = q[:, 0, 0], q[:, 0, 1], q[:, 0, 2]
        b = self.base
        matrix = b.diabatic(q[:, 0, :2]) + (0.5 * self.omega_z**2 * z**2)[:, None, None] * np.eye(2)
        rise = np.tanh(x / b.width)
        envelope = np.exp(-((x / b.c) ** 2))
        gradients = np.zeros((len(q), 2, 2, 1, 3))
        slope = 0.5 * b.D * (1 - rise**2) / b.width
        gradients[:, 0, 0, 0, 0], gradients[:, 1, 1, 0, 0] = slope, -slope
        for i in range(2):
            gradients[:, i, i, 0, 1] = b.omega**2 * y
            gradients[:, i, i, 0, 2] = self.omega_z**2 * z
        dx = b.lam * (y - b.y_c) * envelope * (-2.0 * x / b.c**2)
        dy = b.lam * envelope
        for i, j in ((0, 1), (1, 0)):
            gradients[:, i, j, 0, 0], gradients[:, i, j, 0, 1] = dx, dy
        return DiabaticResult(matrix, gradients)


class Adiabat:
    """Lowest adiabat of a one-atom DPEM through the diabatic_adiabat provider, on flat coordinates."""

    dimension = 3

    def __init__(self, dpem):
        from pyrpmd.potentials.diabatic import DiabaticAdiabatPotential

        self.pes = DiabaticAdiabatPotential(dpem)

    def energy(self, x):
        return float(self.pes.evaluate(np.asarray(x, dtype=float).reshape(1, 1, 3)).energy[0])

    def gradient(self, x):
        return -self.pes.evaluate(np.asarray(x, dtype=float).reshape(1, 1, 3)).forces.ravel()


def test_diabatic_adiabat_provider_on_the_published_nah2_dpem():
    from pyrpmd.potentials import make_potential
    from pyrpmd.potentials.base import check_potential
    from pyrpmd.potentials.nah2_leps import NaH2LEPS5G

    q = np.array([[[0.1, 0.2, 4.0], [0.0, 0.3, -0.7], [0.05, 0.0, 0.75]],
                  [[0.0, 0.1, 3.0], [0.1, 0.0, -0.8], [0.0, 0.2, 0.7]]])
    matrices = NaH2LEPS5G().evaluate(q).matrix
    for state in (0, 1):
        pes = make_potential({"provider": "diabatic_adiabat", "diabatic": {"model": "nah2_leps_5g_1992"},
                              "state": state}, {"symbols": ["Na", "H", "H"]})
        assert np.allclose(pes.evaluate(q).energy, np.linalg.eigvalsh(matrices)[:, state], rtol=0, atol=1e-14)
        assert check_potential(pes, q, 1e-5)["max_absolute_force_error"] < 1e-8
        assert pes.metadata["diabatic"]["provider"] == "molecular_diabatic"
    from pyrpmd.potentials.base import PotentialError

    with pytest.raises(PotentialError, match="states"):
        make_potential({"provider": "diabatic_adiabat", "diabatic": {"model": "tully1"}, "states": 1})
    with pytest.raises(ValueError, match="needs a diabatic"):
        make_potential({"provider": "diabatic_adiabat"})


def test_named_and_plugin_dpems_are_built_from_tables():
    from pyrpmd.potentials.diabatic import make_diabatic_potential
    from pyrpmd.potentials.nonadiabatic_models import TullyModelOne

    model = make_diabatic_potential({"model": "tully1", "params": {"natoms": 1}})
    assert isinstance(model, TullyModelOne)
    live = CrossingCIDPEM()
    assert make_diabatic_potential(live) is live
    with pytest.raises(ValueError):
        make_diabatic_potential({"model": "no_such_surface"})
    with pytest.raises(ValueError, match="model = "):
        make_diabatic_potential({"provider": "python", "function": "m:f"})


def test_pipeline_calls_a_diabatic_potential_for_both_the_rate_pes_and_eta(tmp_path):
    dpem = CrossingCIDPEM()
    run = abstract_rate_run(Adiabat(dpem), 300.0, 16)
    spec = specification(tmp_path / "dpem", None)
    spec["electronic"] = {"source": "diabatic", "potential": dpem, "nstates": 2}
    report = pipeline.run_subspace_pipeline(spec, run=run)
    record = report["nonadiabatic"]["1"]
    assert report["status"] == "complete" and record["valid"]
    assert record["electronic"]["diabatization"] == "supplied"
    assert record["options"]["points"] == "grid" and record["electronic"]["interpolation_error_kt"] == 0.0
    # The rate PES is the DPEM's own lowest adiabat.
    assert record["electronic"]["ground_state_mismatch_kt"] < 1e-8
    assert abs(record["born_oppenheimer_consistency"]) < 1e-10

    class Vectors:
        def __call__(self, geometry):
            energies, vectors = np.linalg.eigh(dpem.evaluate(np.asarray(geometry).reshape(1, 1, 3)).matrix[0])
            if geometry[0] > 0:  # reorder and flip the states past the crossing, as a code may
                energies, vectors = energies[::-1], -vectors[:, ::-1]
            return {"energies": energies, "vectors": vectors}

    adiabatic = pipeline.run_subspace_pipeline(
        specification(tmp_path / "vectors", Vectors(), points="grid"), run=run)
    assert record["factor"] == pytest.approx(adiabatic["nonadiabatic"]["1"]["factor"], rel=1e-9)
    assert 0.1 < record["factor"] < 0.5
    saved = json.loads((tmp_path / "dpem" / "subspace_pipeline.json").read_text())
    assert saved["options"]["electronic"]["potential"].endswith("CrossingCIDPEM")


def test_diabatic_electronic_tables_are_checked():
    resolved = electronic_options({"source": "diabatic", "potential": {"model": "nah2_leps_5g_1992"}, "nstates": 2})
    assert resolved["points"] == "grid"
    assert electronic_options({"source": "diabatic", "potential": {"model": "tully1"}, "nstates": 2,
                               "points": 80})["points"] == 80
    with pytest.raises(ValueError, match="exactly one of electronic.potential"):
        electronic_options({"source": "diabatic", "nstates": 2})
    with pytest.raises(ValueError, match="exactly one of electronic.potential"):
        electronic_options({"source": "diabatic", "potential": {"model": "tully1"}, "object": "m:f", "nstates": 2})
    config = {"task": "rate", "rate": {"subspace_correction": {"electronic": {
        "source": "diabatic", "potential": {"provider": "diabatic_python", "factory": "/abs/dpem.py:make"},
        "nstates": 2}}}}
    assert rate_subspace_options(config)["electronic"]["potential"]["factory"] == "/abs/dpem.py:make"
