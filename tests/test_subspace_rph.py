"""Reaction-path Hamiltonian 2D subspace on the MEP (``subspace.model_2d = "rph"``).

The ribbon ``x(s, q) = w(s) + q n(s)`` around the MEP, with a Bishop normal,
has the metric ``diag((1 - kappa q)^2, 1)`` and is unrolled into a plane in
which the reaction-path kinetic energy is flat.  For a two-dimensional system
the development is congruent to the system, so ``V_S`` must reproduce the
potential itself; in three dimensions the normal carries only part of the
curvature and the rest of the bath enters adiabatically.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_rph import develop_path, rph_bath
from pyrpmd.input_schema import InputError
from pyrpmd.input_subspace import rate_subspace_options

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.subspace_rph.models import CM, Curved, Embedded, model_run  # noqa: E402
from benchmarks.subspace_rph.reference import QUANTUM, build_path  # noqa: E402

GATING = {"validity": {}, "hessian_displacement": 5e-3}


def rph(run, path, info, **options):
    return pipeline.rph_model(run, path, {**QUANTUM, "environment": "adiabatic", **GATING, "path_info": info,
                                          **options})


# --- the development ------------------------------------------------------------

def test_a_planar_path_unrolls_onto_itself():
    x = np.linspace(-8, 8, 161)
    path = np.column_stack([x, 0.8 * np.exp(-x**2 / 4)])
    development = develop_path(path, [1.0, 1.0], rigid_body=False, extend=3.0)
    assert development.captured_curvature == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    s, q = rng.uniform(3.0, development.s[-1] - 3.0, 300), rng.uniform(-1.0, 1.0, 300)
    c, _, normal = development.point(s)
    points = c + q[:, None] * normal          # around the developed curve, inside its curvature radius
    configurations = development.configuration(*development.locate(points))
    developed = np.linalg.norm(points[:, None] - points[None], axis=2)
    original = np.linalg.norm(configurations[:, None] - configurations[None], axis=2)
    assert np.abs(developed - original).max() < 2e-3


def test_the_bishop_ribbon_has_the_rph_metric_on_a_twisted_path():
    # A helix-like path in 3D: curvature and torsion everywhere.
    t = np.linspace(0, 3 * np.pi, 400)
    path = np.column_stack([4 * np.cos(t), 4 * np.sin(t), 1.5 * t])
    development = develop_path(path, [1.0, 1.0, 1.0], rigid_body=False)
    k = len(development.s) // 2
    s, ds = development.s[k], 1e-3
    for q in (-1.0, 0.5, 1.5):
        a, b = development.configuration([s - ds, s + ds], q)
        speed = np.linalg.norm(b - a) / (2 * ds)
        assert speed == pytest.approx(1.0 - development.kappa[k] * q, rel=2e-3)
    # The normal stays a unit normal.  A helix twists: its curvature direction
    # turns about the tangent while a Bishop normal does not, so one normal
    # coordinate carries only part of the curvature (the rest stays adiabatic).
    assert np.allclose(np.linalg.norm(development.n, axis=1), 1.0)
    assert np.abs(np.einsum("ij,ij->i", development.n, development.t)).max() < 1e-8
    assert 0.3 < development.captured_curvature < 0.8


# --- the model on model potentials --------------------------------------------------

@pytest.fixture(scope="module")
def curved_2d():
    potential = Curved(2)
    run = model_run(potential, 500.0, 16)
    path, info = build_path(run)
    return potential, run, path, info


def test_rph_v_s_of_a_two_dimensional_system_is_its_potential(curved_2d):
    potential, run, path, info = curved_2d
    model, quantum, model_info, rpmd_model = rph(run, path, info, spacing_2d=[4.0, 3.0])
    assert model_info["captured_curvature"] == pytest.approx(1.0, abs=1e-6)
    assert model.provenance["model_2d"] == "rph" and model.provenance["bath_stable"]
    # Developed coordinates differ from (x, y) by a congruence that fixes the
    # saddle (0, 10) at the origin; the normal's sign may mirror the plane.
    X, Y = np.meshgrid(*rpmd_model.axes, indexing="ij")
    keep = rpmd_model.values < model_info["truncation_energy"]
    points = np.column_stack([X[keep], Y[keep]])
    best = np.inf
    for mirror in (1.0, -1.0):
        for a in np.radians(np.linspace(-3, 3, 61)):
            c, s_ = np.cos(a), np.sin(a)
            xy = (points * [1.0, mirror]) @ np.array([[c, s_], [-s_, c]]) + np.array([0.0, 10.0])
            best = min(best, float(np.abs(potential.energy(xy) - rpmd_model.values[keep]).max()))
    # Against values up to the truncation energy (~0.04 Eh): the residual is
    # the linear interpolation of the developed curve.
    assert best < 2e-4
    assert model.reactant_reference in model.axes[0]     # snapped onto a DVR node
    assert model.dividing_surface == pytest.approx(0.0, abs=1e-6)


def test_rph_on_a_non_planar_path_keeps_the_third_mode_adiabatic():
    potential = Curved(3, omega_z=1200.0 * CM)
    run = model_run(potential, 500.0, 16)
    path, info = build_path(run)
    model, quantum, model_info, rpmd_model = rph(run, path, info, spacing_2d=[5.0, 4.0])
    assert 0.5 < model_info["captured_curvature"] < 0.999
    # The bath is the z mode: half its frequency, softened at the barrier.
    development = develop_path(path, run.cartesian_masses, rigid_body=False)
    k = int(np.argmax([run.energy(p) for p in path]))
    arcs = [float(np.interp(k, np.arange(len(path)), pipeline._arc(path, run.cartesian_masses)))]
    bath = rph_bath(run.gradient, development, arcs, run.cartesian_masses, rigid_body=False)
    softened = potential.omega_z * (1 - potential.soften_z)
    assert bath["energy"][0] == pytest.approx(0.5 * softened, rel=0.05)
    assert bath["bath_stable"]


def test_rph_is_invariant_under_embedding_and_rotation():
    base = Curved(2)
    embedded = Embedded(base, (900 * CM, 2500 * CM))
    values = []
    for potential in (base, embedded):
        run = model_run(potential, 500.0, 16)
        path, info = build_path(run)
        model, _, model_info, _ = rph(run, path, info, spacing_2d=[4.0, 3.0])
        values.append((model.values - model.values.min(), model_info["captured_curvature"]))
    (a, ca), (b, cb) = values
    assert ca == pytest.approx(cb, abs=1e-3)
    shape = tuple(min(i, j) for i, j in zip(a.shape, b.shape))
    low = (a[: shape[0], : shape[1]] < 0.03) & (b[: shape[0], : shape[1]] < 0.03)
    assert np.abs(a[: shape[0], : shape[1]] - b[: shape[0], : shape[1]])[low].max() < 2e-4


# --- wiring -------------------------------------------------------------------------

def test_reference_slices_are_snapped_onto_dvr_nodes():
    axis = np.linspace(-10, 10, 81)                       # spacing 0.25
    assert pipeline._grid_reference(axis, -7.26) == pytest.approx(-7.25)
    assert pipeline._grid_reference(axis, -7.1) == pytest.approx(-7.0)


def test_model_2d_switch_defaults_to_rph():
    config = {"task": "rate", "rate": {"subspace_correction": {"enabled": False}}}
    assert rate_subspace_options(config)["model_2d"] == "rph"
    config["rate"]["subspace_correction"]["model_2d"] = "plane"
    assert rate_subspace_options(config)["model_2d"] == "plane"
    config["rate"]["subspace_correction"]["model_2d"] = "irc"
    with pytest.raises(InputError, match="model_2d"):
        rate_subspace_options(config)
    assert pipeline.DEFAULTS["subspace"]["model_2d"] == "rph"
    assert pipeline.DEFAULTS["path"]["source"] == "mep"


def test_saved_two_dimensional_factors_are_reused_only_for_the_same_model():
    saved = SimpleNamespace(dimension=2, model={"construction": {"environment": "adiabatic", "model_2d": "plane"}})
    assert not pipeline._saved_model_matches_options(saved, 2, {"environment": "adiabatic"})
    assert pipeline._saved_model_matches_options(saved, 2, {"environment": "adiabatic", "model_2d": "plane"})
    saved.model["construction"]["model_2d"] = "rph"
    assert pipeline._saved_model_matches_options(saved, 2, {"environment": "adiabatic"})


def test_the_channel_stays_open_beyond_the_box(curved_2d):
    """No wall across the channel ends: ring polymers that leave must not bounce back.

    The quantum rate absorbs products before the box ends, so a wall there was
    invisible to it but sent recrossing ring polymers back across the
    dividing surface (kappa 0.84 instead of 0.98 on this model).
    """
    potential, run, path, info = curved_2d
    model, quantum, model_info, rpmd_model = rph(run, path, info, spacing_2d=[4.0, 3.0])
    wall = rpmd_model.values.max()
    # Along the path, from one end of the grid to the other, nothing reaches the wall.
    X, Y = np.meshgrid(*rpmd_model.axes, indexing="ij")
    floor_line = rpmd_model.values.min(axis=1)
    assert floor_line.max() < 0.5 * wall
    # Both grid ends are flat channels near the asymptote.
    assert floor_line[0] == pytest.approx(floor_line[-1], abs=2e-3)
    assert floor_line[0] < model_info["energy_floor"] + 0.01


def test_the_reference_slice_lies_where_quantum_and_centroid_densities_agree():
    """On a slope the quantum and the centroid reactant density differ at O(hbar^2).

    The 0.3 kT energy rule alone put the 300 K reference slice at x = -71 on
    the curved-valley model, where beta^2 |V''| / 12 is ~0.15: C_S came out
    ~5% high for the RPH model, which is exact for a 2D system.
    """
    potential = Curved(2)
    run = model_run(potential, 300.0, 16)
    path, info = build_path(run)
    arc = pipeline._arc(path, run.cartesian_masses)
    energies = np.array([run.energy(p) for p in path])
    mismatch = pipeline._density_mismatch(arc, energies, run.beta)
    loose = pipeline._free_channel(energies, np.zeros_like(mismatch), run.beta, 0.3, np.inf)
    strict = pipeline._free_channel(energies, mismatch, run.beta, 0.3, 1e-2)
    edge = lambda free: path[int(np.argmin(free[: int(np.argmax(energies))])) - 1, 0]
    assert edge(loose) == pytest.approx(-71.0, abs=4.0)
    assert edge(strict) < -95.0
    # The mismatch of the analytic tail at x: beta^2 V''/12 + beta^3 V'^2/24.
    a, V0 = potential.a, potential.V0
    x = -80.0
    sech2, tanh = 1 / np.cosh(x / a) ** 2, np.tanh(x / a)
    d1 = -2 * V0 / a * sech2 * tanh
    d2 = 2 * V0 / a**2 * sech2 * (3 * tanh**2 - 1)
    exact = run.beta**2 * abs(d2) / 12 + run.beta**3 * d1**2 / 24
    k = int(np.argmin(abs(path[:, 0] - x)))
    assert mismatch[k] == pytest.approx(exact, rel=0.2)


def test_the_plane_absorber_runs_past_the_path_ends(curved_2d):
    """The reactant absorber may start beyond the path; its ramp must not be cut there.

    The arc length of grid points beyond the path start used to stop at the
    start point, so the absorber never reached its strength and reactant
    waves reflected off the grid edge (k_QM 22% low on this model).
    """
    potential, run, path, info = curved_2d
    direction = np.asarray(pipeline._static_curvature_width(run, path, 0.5)["candidate_direction"])
    options = {**QUANTUM, "environment": "adiabatic", **GATING, "path_info": info, "spacing_2d": [4.0, 4.0]}
    model, quantum, model_info, rpmd_model = pipeline.plane_model(run, path, direction, options)
    box_lo, box_hi = model_info["box_arc"]
    arc = pipeline._arc(path, run.cartesian_masses)
    assert box_lo < arc[0]                          # the box starts beyond the path
    X, Y = np.meshgrid(*model.axes, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])
    eps = quantum["absorber"](points).reshape(X.shape)
    row = int(np.argmin(np.abs(model.axes[1])))     # through the reactant channel
    # The reactant ramp reaches its full strength 30 v / width at the grid edge.
    wavelength = 2.0 * np.pi / np.sqrt(2.0 / run.beta)
    full = 30.0 * np.sqrt(2.0 / run.beta) / (2.0 * wavelength)
    assert eps[0, row] >= 0.9 * full
    assert eps[np.argmin(np.abs(model.axes[0] - model.dividing_surface)), row] == 0.0


def test_rph_orients_symmetric_120_degree_bend_and_rejects_hairpin():
    """Bisecting reactant reference and product absorber handles 120 deg bends and rejects 180 deg hairpins."""
    from pyrpmd.constants import KB_AU

    r0 = 8.0
    v0 = 0.015
    a = 18.0
    omega_y = 0.008

    class Bent120Potential:
        dimension = 2

        @staticmethod
        def _parts(q):
            q = np.asarray(q, dtype=float)
            x, y = q[..., 0], q[..., 1]
            root = np.sqrt(x**2 + r0**2)
            y_ref = -np.sqrt(3.0) * root
            dy_ref = -np.sqrt(3.0) * x / root
            sech2 = 1.0 / np.cosh(x / a) ** 2
            tanh = np.tanh(x / a)
            dy = y - y_ref
            val = v0 * sech2 + 0.5 * omega_y**2 * dy**2
            gx = -2.0 * v0 / a * sech2 * tanh - omega_y**2 * dy * dy_ref
            gy = omega_y**2 * dy
            return val, np.stack([gx, gy], axis=-1)

        def energy(self, q):
            return self._parts(q)[0]

        def gradient(self, q):
            return self._parts(q)[1]

    pot = Bent120Potential()
    ts = np.array([0.0, -np.sqrt(3.0) * r0])
    direction = np.array([1.0, 0.0])

    def xi_vg(x):
        vec = np.asarray(x, dtype=float).ravel()
        return 1.0 + float((vec - ts) @ direction) / 100.0, direction / 100.0

    run = SimpleNamespace(
        root=None, beta=1.0 / (KB_AU * 300.0), nbeads=16, rigid_body=False, xi_star=1.0,
        cartesian_masses=np.ones(2), system=SimpleNamespace(geometry=ts + [0.2, 0.2], masses=np.ones(2)),
        energy=lambda x: float(pot.energy(x)), gradient=lambda x: np.asarray(pot.gradient(x), dtype=float),
        xi_value_gradient=xi_vg, xi=lambda x: xi_vg(x)[0], xi_gradient=lambda x: xi_vg(x)[1],
    )
    path, info = build_path(run, step=3.0)
    model, _, model_info, _ = rph(run, path, info, spacing_2d=[4.0, 4.0])
    assert abs(model_info["development"]["turn_angle_degrees"]) == pytest.approx(120.0, abs=3.0)
    assert model.dividing_surface == pytest.approx(0.0, abs=1e-6)

    # 180-degree U-turn: reactant and product tangents cancel, falling back to barrier tangent and rejecting cleanly.
    radius = 25.0
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 81)
    arm = np.linspace(1.0, 120.0, 40)
    p_left = np.column_stack([-arm[::-1], np.full_like(arm, -radius)])
    p_arc = np.column_stack([radius * np.cos(theta), -radius * np.sin(theta)])
    p_right = np.column_stack([-arm, np.full_like(arm, radius)])
    u_path = np.vstack([p_left, p_arc, p_right])
    u_arc = pipeline._arc(u_path, np.ones(2))
    s_rel = u_arc - u_arc[len(u_arc) // 2]

    def hairpin_energy(x):
        pts = np.asarray(x, dtype=float)
        idx = int(np.argmin(np.linalg.norm(u_path - pts, axis=1)))
        d = np.linalg.norm(pts - u_path[idx])
        return float(v0 / np.cosh(s_rel[idx] / 14.0) ** 2 + 0.5 * (0.01**2) * d**2)

    def hairpin_gradient(x):
        pts = np.asarray(x, dtype=float)
        idx = int(np.argmin(np.linalg.norm(u_path - pts, axis=1)))
        return (0.01**2) * (pts - u_path[idx])

    hairpin_run = SimpleNamespace(
        root=None, beta=1.0 / (KB_AU * 300.0), nbeads=16, rigid_body=False, xi_star=1.0,
        cartesian_masses=np.ones(2), system=SimpleNamespace(geometry=u_path[len(u_path) // 2], masses=np.ones(2)),
        energy=hairpin_energy,
        gradient=hairpin_gradient,
        xi=lambda x: 1.0,
    )
    with pytest.raises(ValueError, match="does not advance along axis 0"):
        rph(hairpin_run, u_path, {"reactant_termination": {"converged": True}, "product_termination": {"converged": True}})


def test_rph_suppresses_asymptotic_soft_mode_half_width_without_grid_exit():
    """Asymptotic near-zero transverse curvature must not blow up half_width or leak walkers outside grid."""
    from pyrpmd.analysis.subspace_correction import validity_gates
    from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate
    from pyrpmd.constants import KB_AU

    v0 = 0.016
    a = 14.0
    b = 10.0
    c = 20.0
    w_bar = 2000.0 * CM
    w_asym = 50.0 * CM  # 2.3e-4 Eh; unclipped 6 * 1/sqrt(2*w_asym) ~ 280 a.u.

    class SoftAsymptoteValley:
        dimension = 2

        @staticmethod
        def _parts(q):
            q = np.asarray(q, dtype=float)
            x, y = q[..., 0], q[..., 1]
            sech2 = 1.0 / np.cosh(x / a) ** 2
            tanh = np.tanh(x / a)
            g = b * np.exp(-x**2 / c**2)
            dg = -2.0 * x / c**2 * g
            env = np.exp(-x**2 / 24.0**2)
            denv = -2.0 * x / 24.0**2 * env
            w = w_asym + (w_bar - w_asym) * env
            dw = (w_bar - w_asym) * denv
            dy = y - g
            val = v0 * sech2 + 0.5 * w**2 * dy**2
            gx = -2.0 * v0 / a * sech2 * tanh + w * dw * dy**2 - w**2 * dy * dg
            gy = w**2 * dy
            return val, np.stack([gx, gy], axis=-1)

        def energy(self, q):
            return self._parts(q)[0]

        def gradient(self, q):
            return self._parts(q)[1]

    pot = SoftAsymptoteValley()
    run = model_run(pot, 300.0, 16)
    path, info = build_path(run, step=4.0)
    # Also ensure path reaches asymptotic flat soft-mode channel cleanly.
    opts = {**QUANTUM, "environment": "adiabatic", **GATING, "path_info": info}
    model, _, model_info, rpmd_model = pipeline.rph_model(run, path, opts)
    assert model_info["half_width"] < 65.0
    rpmd_res = ring_polymer_rate(
        rpmd_model, 1.0 / (KB_AU * 300.0), 16,
        windows=12, walkers=16, burn_in=40, sweeps=80, trajectories=64, seed=3,
    )
    gates = validity_gates(model.provenance, rpmd_res.diagnostics)
    assert gates["valid"], gates
    grid_exit = rpmd_res.diagnostics["grid_exit"]
    assert grid_exit["ti_sample_outside_fraction"] == 0.0
    assert grid_exit["surface_sample_outside_fraction"] == 0.0


def test_project_rph_values_batch_parity_and_failure_fallback():
    """RateRun.energies and project_rph_values batch path match scalar path to machine precision with safe fallback."""
    from pyrpmd.analysis.subspace_rph import project_rph_values
    from pyrpmd.potentials.base import PESResult

    call_batches = []

    class MockProvider:
        def evaluate(self, q):
            arr = np.asarray(q, dtype=float)
            assert arr.ndim == 3 and arr.shape[1:] == (3, 3)
            call_batches.append(len(arr))
            r01 = np.linalg.norm(arr[:, 0] - arr[:, 1], axis=1)
            r12 = np.linalg.norm(arr[:, 1] - arr[:, 2], axis=1)
            energy = 0.01 * (r01 - 1.8) ** 2 + 0.02 * (r12 - 1.8) ** 2
            forces = np.zeros_like(arr)
            status = np.where((r01 < 0.25) | (r12 < 0.25), 1, 0).astype(np.int32)
            energy = np.where(status == 0, energy, np.nan)
            return PESResult(energy=energy, forces=forces, status=status)

    provider = MockProvider()
    run = pipeline.RateRun(
        root=Path("."),
        config={},
        system=SimpleNamespace(masses=np.ones(3)),
        reaction={},
        coordinate=None,
        provider=provider,
        beta=1000.0,
        nbeads=16,
        xi_star=1.0,
    )
    configs = np.array([
        [[-1.8, 0.0, 0.0], [0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],  # r01 < 0.25 -> status 1
        [[-2.0, 0.1, 0.0], [0.0, 0.0, 0.0], [1.9, -0.1, 0.0]],
    ]).reshape(3, 9)
    batched_e = run.energies(configs)
    assert call_batches == [3]
    assert np.isfinite(batched_e[0]) and np.isnan(batched_e[1]) and np.isfinite(batched_e[2])
    assert batched_e[0] == pytest.approx(run.energy(configs[0]), abs=1e-15)
    assert np.isnan(run.energy(configs[1]))
    assert pipeline._run_batch_energy(run) is not None

    # Monkeypatching run.energy on the instance must disable stale batch bypass.
    run.energy = lambda x: 42.0
    assert pipeline._run_batch_energy(run) is None

    # Now test project_rph_values on a 9D path with finite, NaN, and exception-raising batches.
    ts = np.array([-1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0])
    path = np.array([ts + k * np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0]) for k in np.linspace(-3, 3, 13)])
    dev = develop_path(path, np.ones(9), rigid_body=True, extend=2.0)
    dev.orient(float(dev.s[len(dev.s) // 2]), [1.0, 0.0])
    bath = {"arc": dev.s, "energy": 0.003 * np.exp(-0.05 * (dev.s - dev.s[len(dev.s) // 2]) ** 2)}
    gx = np.linspace(-4.0, 4.0, 21)
    gy = np.linspace(-3.0, 3.0, 17)

    def scalar_pes(x):
        p = np.asarray(x, dtype=float).reshape(3, 3)
        r01 = float(np.linalg.norm(p[0] - p[1]))
        r12 = float(np.linalg.norm(p[1] - p[2]))
        if r01 < 0.9 or r12 < 0.9:
            raise ValueError("overlap")
        return 0.015 * (r01 - 1.8) ** 2 + 0.025 * (r12 - 1.8) ** 2

    def batch_raise(q):
        arr = np.asarray(q, dtype=float).reshape(-1, 3, 3)
        r01 = np.linalg.norm(arr[:, 0] - arr[:, 1], axis=1)
        r12 = np.linalg.norm(arr[:, 1] - arr[:, 2], axis=1)
        if np.any((r01 < 0.9) | (r12 < 0.9)):
            raise RuntimeError("batch failure in overlap region")
        return 0.015 * (r01 - 1.8) ** 2 + 0.025 * (r12 - 1.8) ** 2

    def batch_nan(q):
        arr = np.asarray(q, dtype=float).reshape(-1, 3, 3)
        r01 = np.linalg.norm(arr[:, 0] - arr[:, 1], axis=1)
        r12 = np.linalg.norm(arr[:, 1] - arr[:, 2], axis=1)
        out = 0.015 * (r01 - 1.8) ** 2 + 0.025 * (r12 - 1.8) ** 2
        return np.where((r01 < 0.9) | (r12 < 0.9), np.nan, out)

    def batch_clean(q):
        arr = np.asarray(q, dtype=float).reshape(-1, 3, 3)
        r01 = np.linalg.norm(arr[:, 0] - arr[:, 1], axis=1)
        r12 = np.linalg.norm(arr[:, 1] - arr[:, 2], axis=1)
        return 0.015 * (r01 - 1.8) ** 2 + 0.025 * (r12 - 1.8) ** 2

    v_ref, bare_ref = project_rph_values(
        scalar_pes, dev, (gx, gy), bath, arc_range=(-4.0, 4.0), half_width=2.5, wall_energy=0.5, return_bare=True
    )
    v_raise, bare_raise = project_rph_values(
        scalar_pes, dev, (gx, gy), bath, arc_range=(-4.0, 4.0), half_width=2.5, wall_energy=0.5,
        return_bare=True, batch_potential=batch_raise,
    )
    v_nan, bare_nan = project_rph_values(
        scalar_pes, dev, (gx, gy), bath, arc_range=(-4.0, 4.0), half_width=2.5, wall_energy=0.5,
        return_bare=True, batch_potential=batch_nan,
    )
    assert np.max(np.abs(v_raise - v_ref)) < 1e-14
    assert np.max(np.abs(bare_raise - bare_ref)) < 1e-14
    assert np.max(np.abs(v_nan - v_ref)) < 1e-14
    assert np.max(np.abs(bare_nan - bare_ref)) < 1e-14

    # All-finite parity on clean domain.
    clean_scalar = lambda x: float(batch_clean(np.asarray(x)[None])[0])
    v_cs = project_rph_values(clean_scalar, dev, (gx, gy), bath, arc_range=(-4.0, 4.0), half_width=2.5, wall_energy=0.5)
    v_cb = project_rph_values(
        clean_scalar, dev, (gx, gy), bath, arc_range=(-4.0, 4.0), half_width=2.5, wall_energy=0.5,
        batch_potential=batch_clean,
    )
    assert np.max(np.abs(v_cb - v_cs)) < 1e-14


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "benchmarks" / "h3" / "build" / "libbkmp2.so").is_file(),
    reason="build BKMP2 with benchmarks/h3/build_pes.py",
)
def test_real_bkmp2_3d_h3_rph_and_dvr_models():
    """Full 9D Cartesian 3D H + H2 (BKMP2, 200 K) builds 2D RPH and 1D DVR models cleanly."""
    from benchmarks.h3.pes import build as build_bkmp2
    from pyrpmd.analysis.subspace_paths import mep_core
    from pyrpmd.analysis.subspace_rph import develop_path, project_rph_values, rph_bath
    from pyrpmd.constants import AMU_TO_AU, KB_AU
    from pyrpmd.cv import RPMDrateCoordinate

    provider = build_bkmp2()
    masses_au = np.full(3, 1.00782503223 * AMU_TO_AU)
    ts_geom = np.array([[-1.7570000251, 0.0, 0.0], [0.0, 0.0, 0.0], [1.7570000251, 0.0, 0.0]])
    coordinate = RPMDrateCoordinate([[0], [1, 2]], masses_au, ts_geom, [[0, 1]], [[1, 2]], 10.0,
                                  equivalent_paths=[[0, 2, 1]])
    run = pipeline.RateRun(
        root=Path("."),
        config={},
        system=SimpleNamespace(masses=masses_au, geometry=ts_geom),
        reaction={"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]]},
        coordinate=coordinate,
        provider=provider,
        beta=1.0 / (KB_AU * 200.0),
        nbeads=128,
        xi_star=1.0,
    )
    core, m_info = mep_core(run, {"step": 10.0})
    path, p_info = pipeline._assemble(run, core, {"step": 10.0, "product_drop_kt": 15.0}, {"mep": m_info})
    assert p_info["termini_accepted"]

    opts = {**pipeline.DEFAULTS["quantum"], "environment": "adiabatic",
            **GATING, "path_info": p_info}
    model_1d, _, _ = pipeline.path_model(run, path, {**opts, "zero_point": "dvr"})
    assert np.isfinite(model_1d.values).all()
    zpe_min, zpe_max = model_1d.provenance["zero_point_range"]
    assert 0.008 < zpe_min < zpe_max < 0.016

    model_2d, _, info_2d, rpmd_2d = pipeline.rph_model(run, path, opts)
    assert np.isfinite(rpmd_2d.values).all()
    assert info_2d["half_width"] < 65.0
    assert abs(info_2d["development"]["turn_angle_degrees"]) == pytest.approx(120.0, abs=5.0)
    assert info_2d["retained_points"] < 8000

    # Direct batch vs scalar check on a smaller 2D slice of the same BKMP2 development.
    dev = develop_path(path, run.cartesian_masses, rigid_body=True, extend=40.0)
    _, t_bar, _ = dev.point(float(info_2d["barrier_arc"]))
    dev.orient(float(info_2d["barrier_arc"]), t_bar[0])
    bath = rph_bath(run.gradient, dev, np.linspace(info_2d["box_arc"][0], info_2d["box_arc"][1], 21),
                    run.cartesian_masses, rigid_body=True)
    gx = np.linspace(-30.0, 30.0, 25)
    gy = np.linspace(-15.0, 15.0, 19)
    v_scalar = project_rph_values(lambda x: run.energy(x), dev, (gx, gy), bath,
                                  arc_range=tuple(info_2d["box_arc"]), half_width=14.0,
                                  wall_energy=0.5, batch_potential=None)
    v_batch = project_rph_values(run.energy, dev, (gx, gy), bath, arc_range=tuple(info_2d["box_arc"]),
                                 half_width=14.0, wall_energy=0.5, batch_potential=run.energies)
    assert np.max(np.abs(v_batch - v_scalar)) < 1e-14
