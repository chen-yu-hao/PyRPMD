"""The instanton-anchor benchmark (``benchmarks/subspace_instanton``).

Each building block is tested against something it does not compute itself:
the Green's-function reference against analytic transmission probabilities,
the instanton against the scaling law and the Hessian of a parabolic barrier,
the minimum-energy path against the saddle-point expansion of its curvature.
The committed result files are tested for consistency with one another and
with a recomputation of their cheap entries.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.subspace_rpmd import free_ring_frequencies

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.subspace_instanton import greens, report, run_rpmd  # noqa: E402
from benchmarks.subspace_instanton.instanton import (  # noqa: E402
    find_instanton, ring_hessian, steepest_descent_factor)
from benchmarks.subspace_instanton.models import Valley, eckart, family  # noqa: E402
from benchmarks.subspace_instanton.reductions import (  # noqa: E402
    barrier_frequency, minimum_energy_path, reductions)
from benchmarks.subspace_instanton.run_qm import tabulated_instanton  # noqa: E402

RESULTS = ROOT / "benchmarks" / "subspace_instanton" / "results"


def eckart_transmission(energy, height, width):
    """Analytic transmission of ``height sech^2(x / width)`` (unit mass, hbar = 1); zero below threshold."""
    energy = np.asarray(energy, dtype=float)
    k = np.sqrt(2.0 * np.clip(energy, 0.0, None))
    numerator = np.sinh(np.pi * k * width) ** 2
    return numerator / (numerator + np.cosh(0.5 * np.pi * np.sqrt(8.0 * height * width**2 - 1.0)) ** 2)


def eckart_instanton(v, u, nbeads=128):
    _, slope, curvature, _ = eckart(v)
    return find_instanton(lambda b: slope(b[:, 0])[:, None], lambda b: curvature(b[:, 0])[:, None, None],
                          [0.0], [1.0], 1.0, u, nbeads=nbeads)


@pytest.fixture(scope="module")
def paths():
    return {name: minimum_energy_path(model) for name, model in family().items()}


@pytest.fixture(scope="module")
def qm():
    return json.loads((RESULTS / "qm.json").read_text())


@pytest.fixture(scope="module")
def skeleton():
    return json.loads((RESULTS / "skeleton.json").read_text())


# --- exact reference: N(E) with exact leads -----------------------------------------

def test_a_uniform_chain_transmits_every_wave_in_its_band():
    grid = np.arange(-5.0, 5.0 + 1e-9, 0.05)
    band = 4.0 / (2.0 * 0.05**2)
    probability = greens.cumulative_probability(np.linspace(0.01, 0.99 * band, 9), lambda s: np.zeros((1, 1)), grid)
    assert probability == pytest.approx(1.0, abs=1e-10)


def test_the_eckart_transmission_is_the_analytic_one():
    potential, _, _, width = eckart(2.1)
    grid = np.arange(-18.0, 18.005, 0.01)
    energies = np.array([0.3, 1.0, 2.0, 2.1, 2.5, 4.0])
    probability = greens.cumulative_probability(energies, lambda s: np.array([[potential(s)]]), grid)
    assert probability == pytest.approx(eckart_transmission(energies, 2.1, width), rel=5e-4)


def test_a_separable_valley_is_a_sum_of_shifted_channels():
    """Without coupling each oscillator state crosses the bare barrier at ``E - (n + 1/2) omega``."""
    model = Valley("separable", omega_inf=1.3)
    energies = np.array([1.5, 2.5, 3.5, 4.5])
    probability = greens.cumulative_probability(energies, greens.valley_channels(model, 6),
                                                np.arange(-14.0, 14.01, 0.02))
    expected = sum(np.where(energies > (n + 0.5) * 1.3,
                            eckart_transmission(energies - (n + 0.5) * 1.3, model.V0, model.a), 0.0)
                   for n in range(6))
    assert probability == pytest.approx(expected, rel=1e-3)


def test_the_oscillator_position_square_is_not_a_truncated_product():
    """``Q^2`` of the truncated basis must be the block of the exact operator, not ``Q @ Q``."""
    q, q2, h0 = greens.oscillator_matrices(8, 1.7)
    larger, _, _ = greens.oscillator_matrices(12, 1.7)
    assert q2 == pytest.approx((larger @ larger)[:8, :8], abs=1e-14)
    assert abs(q @ q - q2).max() > 1.0                       # the truncated product is wrong in the last state
    assert np.diag(h0) == pytest.approx(1.7 * (np.arange(8) + 0.5))
    # Virial theorem: <n| omega^2 Q^2 / 2 |n> is half the level energy.
    assert 0.5 * 1.7**2 * np.diag(q2) == pytest.approx(0.5 * np.diag(h0))


def test_the_rate_of_a_transparent_barrier_is_the_free_flux():
    """``N = 1``: ``k Q_r = 1 / (2 pi beta)``, i.e. ``k = sqrt(1 / (2 pi beta))`` per unit length."""
    energies = np.linspace(0.0, 60.0, 60001)                # trapezoid error (beta h)^2 / 12 = 3e-7
    rate = greens.thermal_rate(energies, np.ones_like(energies), 2.0, 0.0, np.sqrt(1.0 / (4.0 * np.pi)))
    assert rate == pytest.approx(np.sqrt(1.0 / (4.0 * np.pi)), rel=1e-6)


def test_a_separable_valley_has_the_rate_of_its_barrier():
    model = Valley("separable", omega_inf=1.3)
    betas = [5.0, 8.0]
    rates, _, _ = greens.valley_rates(model, betas, size=8, half_length=12.0, points=400)
    x = np.arange(-18.0, 18.005, 0.01)
    reference = greens.barrier_rates(x, model.barrier(x), betas, points=2000)
    for beta in betas:
        assert rates[beta] == pytest.approx(reference[beta], rel=3e-3)


# --- instanton -----------------------------------------------------------------------

def test_the_collapsed_ring_polymer_has_the_parabolic_barrier_spectrum():
    """At the top of a barrier of frequency 1 the eigenvalues are ``omega_k^2 - 1`` (free ring polymer)."""
    _, _, curvature, _ = eckart(2.1)
    nbeads, beta = 16, 4.0
    hessian = ring_hessian(lambda b: curvature(b[:, 0])[:, None, None], np.zeros((nbeads, 1)), nbeads / beta)
    expected = np.sort(free_ring_frequencies(nbeads, beta) ** 2 - 1.0)
    assert np.linalg.eigvalsh(hessian) == pytest.approx(expected, abs=1e-10)


def test_the_ring_hessian_is_the_derivative_of_the_ring_gradient():
    model = family()["M3"]
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.5, size=(6, 2))
    omega_n = 1.7

    def gradient(beads):
        return (model.gradient(beads) + omega_n**2 * (2.0 * beads - np.roll(beads, 1, 0) - np.roll(beads, -1, 0))).ravel()

    numeric = np.empty((12, 12))
    for j in range(12):
        step = np.zeros(12)
        step[j] = 1e-5
        numeric[:, j] = (gradient(x + step.reshape(6, 2)) - gradient(x - step.reshape(6, 2))) / 2e-5
    assert ring_hessian(model.hessian, x, omega_n) == pytest.approx(numeric, abs=1e-6)


def test_there_is_no_instanton_above_the_crossover_temperature():
    assert eckart_instanton(2.1, 6.0) is None                # u = beta omega_b < 2 pi
    assert steepest_descent_factor(6.0, 1.0, None) == 0.0


def test_the_instanton_is_a_first_order_saddle_with_a_permutation_zero_mode():
    found = eckart_instanton(2.1, 7.854)
    assert found["converged"] and found["negative_modes"] == 1
    assert abs(found["zero_mode"]) < 1e-9 * found["eta1"] ** 2
    assert found["gradient_max"] < 1e-9


def test_the_instanton_conserves_the_energy_of_the_inverted_potential():
    """``V(x) - xdot^2 / 2`` is constant along the periodic orbit, here to ``O(1 / N^2)``."""
    potential, _, _, _ = eckart(2.1)
    for nbeads, tolerance in ((128, 2e-3), (256, 5e-4)):
        x = eckart_instanton(2.1, 9.0, nbeads)["x"][:, 0]
        velocity = (np.roll(x, -1) - np.roll(x, 1)) * nbeads / (2.0 * 9.0)
        energy = potential(x) - 0.5 * velocity**2
        assert np.ptp(energy) < tolerance * energy.mean()


def test_the_eckart_unstable_frequency_depends_on_u_only():
    """``x = a y`` scales the height out of Newton's equation: eta_1 / omega_b is a function of ``u``."""
    for u in (6.4, 7.854, 11.42):
        low, high = eckart_instanton(0.8, u), eckart_instanton(3.6, u)
        assert low["eta1"] == pytest.approx(high["eta1"], rel=1e-9)
        assert low["spread"] / np.sqrt(1.6) == pytest.approx(high["spread"] / np.sqrt(7.2), rel=1e-9)


def test_the_steepest_descent_factor_is_continuous_at_the_crossover():
    factors = [steepest_descent_factor(u, 1.0, eckart_instanton(2.1, u, 256)) for u in (6.30, 6.35, 6.5, 7.0)]
    assert 0.0 < factors[0] < factors[1] < factors[2] < factors[3]
    assert factors[0] < 0.01                                 # u / 2 pi = 1.0027


def test_the_unstable_frequency_converges_as_the_inverse_square_of_the_bead_number():
    eta = [eckart_instanton(2.1, 9.0, n)["eta1"] for n in (64, 128, 256)]
    assert (eta[1] - eta[0]) / (eta[2] - eta[1]) == pytest.approx(4.0, rel=0.05)
    assert eta[2] == pytest.approx(eta[1], rel=2e-4)


def instanton_action(v, u, nbeads):
    """``S = beta_N U_N`` of the Eckart instanton."""
    potential, _, _, _ = eckart(v)
    x = eckart_instanton(v, u, nbeads)["x"][:, 0]
    return u / nbeads * (potential(x).sum() + 0.5 * (nbeads / u) ** 2 * ((x - np.roll(x, 1)) ** 2).sum())


def test_the_bead_numbers_of_the_rpmd_runs_resolve_the_instanton_action():
    """``exp(-S)`` is the leading factor of the RPMD rate below the crossover.

    Its bead error is far below the statistical errors (1 to 2%) at the bead
    numbers of ``run_rpmd.py`` and ``run_eckart.py``: 64 at 0.8 T_c, 96 or 128
    at 0.55 T_c.  It grows in proportion to ``v`` and as ``1 / P^2``.
    """
    for u, nbeads, bound in ((2.0 * np.pi / 0.8, 64, 2.5e-3), (2.0 * np.pi / 0.55, 96, 3e-3),
                             (2.0 * np.pi / 0.55, 128, 1.6e-3), (13.5, 128, 2.2e-3)):
        error = instanton_action(2.1, u, 1024) - instanton_action(2.1, u, nbeads)
        assert 0.0 < error < bound
    coarse = instanton_action(2.1, 9.0, 1024) - instanton_action(2.1, 9.0, 32)
    fine = instanton_action(2.1, 9.0, 1024) - instanton_action(2.1, 9.0, 64)
    assert coarse / fine == pytest.approx(4.0, rel=0.05)
    assert (instanton_action(3.6, 9.0, 1024) - instanton_action(3.6, 9.0, 64)) / fine == pytest.approx(3.6 / 2.1, rel=1e-6)


def test_a_transverse_mode_on_a_straight_path_does_not_move_the_instanton():
    """M1: the valley is straight, so the instanton and its unstable mode are those of the bare barrier."""
    model = family()["M1"]
    saddle, unstable, _, mode = model.saddle()
    beta = 2.0 * np.pi / 0.8
    full = find_instanton(model.gradient, model.hessian, saddle, mode, unstable, beta, nbeads=64)
    assert abs(full["x"][:, 1]).max() < 1e-12
    assert full["eta1"] == pytest.approx(eckart_instanton(2.1, beta, 64)["eta1"], rel=1e-6)
    assert full["negative_modes"] == 1


# --- minimum-energy path and reductions ---------------------------------------------------

def saddle_curvature(model, h=1e-3):
    """Curvature of the steepest-descent path at the saddle from the Taylor expansion of ``V``.

    With ``H t = -lambda^2 t`` and the third derivative ``T``, the path
    ``x = sigma t + sigma^2 c / 2`` satisfies ``(H_perp + 2 lambda^2) c = -T[t, t]_perp``.
    """
    saddle, unstable, stable, mode = model.saddle()
    normal = np.array([-mode[1], mode[0]])
    third = (model.hessian(saddle + h * normal) - model.hessian(saddle - h * normal)) / (2.0 * h)
    return abs(mode @ third @ mode) / (stable**2 + 2.0 * unstable**2)


def test_the_path_follows_the_gradient_and_ends_in_the_flat_valley(paths):
    for name, model in family().items():
        path = paths[name]
        points, sigma = path["points"], path["sigma"]
        assert np.diff(sigma) == pytest.approx(0.01)
        assert np.linalg.norm(np.diff(points, axis=0), axis=1) == pytest.approx(0.01, rel=2e-3)
        assert np.all(np.diff(path["energy"]) < 0.0)
        assert path["energy"][-1] < 3e-5 and path["end_s"] > 12.9      # V0 sech^2(13 / a) = 2.6e-5
        interior = slice(20, -200)                               # the gradient vanishes at both ends
        gradient = model.gradient(points[interior])
        tangent = np.gradient(points, sigma, axis=0)[interior]
        cross = gradient[:, 0] * tangent[:, 1] - gradient[:, 1] * tangent[:, 0]
        assert abs(cross).max() < 1e-3 * np.linalg.norm(gradient, axis=1).max()


def test_the_path_curvature_at_the_saddle_is_that_of_the_taylor_expansion(paths):
    """Continuing the path through the saddle with the wrong symmetry zeroes this curvature."""
    for name in ("M2", "M3"):
        expected = saddle_curvature(family()[name])
        assert expected > 0.1
        assert paths[name]["curvature"][0] == pytest.approx(expected, rel=2e-2)
    assert paths["M1"]["curvature"].max() < 1e-8
    # M4 is symmetric under inversion through the saddle: the path is odd in the arc length,
    # straight at the saddle and bent on either side of it.
    assert saddle_curvature(family()["M4"]) < 1e-6 and paths["M4"]["curvature"][0] < 1e-8
    assert paths["M4"]["curvature"].max() > 0.3
    assert saddle_curvature(family()["M2"]) == pytest.approx(1.5**2 * (2.0 / 4.2) / (1.5**2 + 2.0), rel=1e-5)


def test_the_transverse_frequency_on_the_path_runs_from_the_saddle_to_the_reactant_value(paths):
    for name, model in family().items():
        _, _, stable, _ = model.saddle()
        assert paths[name]["w_perp"][0] == pytest.approx(stable, rel=1e-6)
        assert paths[name]["w_perp"][-1] == pytest.approx(model.omega_inf, rel=1e-4)


def test_a_separable_valley_has_one_reduction():
    model = Valley("separable", omega_inf=1.3)
    reduced, mass = reductions(model, 7.0)
    x, mep = reduced["mep"]
    assert mass == pytest.approx(1.0)
    for key in ("fe", "fe+sct", "mep+sct"):
        assert reduced[key][1] == pytest.approx(mep, abs=1e-12)
    inside = abs(x) < 12.0
    # The path stops at s = 13, where the barrier is 2.6e-5; the reduction is shifted to zero there.
    assert mep[inside] == pytest.approx(model.barrier(x[inside]), abs=3e-5)
    assert barrier_frequency(x, mep) == pytest.approx(1.0, rel=1e-4)


def test_the_free_energy_reduction_carries_the_transverse_partition_function(paths):
    """M1: the barrier of ``fe`` is ``V0 + F(omega_saddle) - F(omega_inf)`` of the transverse oscillator."""
    model = family()["M1"]
    beta = 2.0 * np.pi / 0.8
    reduced, _ = reductions(model, beta, paths["M1"])
    free = lambda w: np.log(2.0 * np.sinh(0.5 * beta * w)) / beta
    assert reduced["fe"][1].max() == pytest.approx(model.V0 + free(1.1) - free(1.8), abs=1e-4)
    assert reduced["mep"][1].max() == pytest.approx(model.V0, abs=1e-4)


def test_the_sct_mass_shortens_the_coordinate_only_where_the_path_bends(paths):
    for name in ("M2", "M4"):
        reduced, mass = reductions(family()[name], 2.0 * np.pi / 0.8, paths[name])
        assert np.all(mass <= 1.0) and mass.min() < 0.9 and mass[-1] == pytest.approx(1.0, abs=1e-4)
        x, dressed = reduced["fe+sct"]
        _, skeleton = reduced["mep+sct"]
        # The skeleton has the coordinate of the reduction and the bare potential of "mep".
        assert skeleton.max() == pytest.approx(reduced["mep"][1].max(), abs=1e-6)
        ratio = barrier_frequency(x, skeleton) / barrier_frequency(*reduced["mep"])
        if name == "M2":
            # The path bends at the saddle: a lighter mass there is a higher barrier frequency.
            assert ratio == pytest.approx(mass[0] ** -0.5, rel=2e-3) and ratio > 1.2
        else:
            # M4 is straight at the saddle and bends beside it: the crossover temperature stays.
            assert mass[0] == pytest.approx(1.0, abs=1e-6) and ratio == pytest.approx(1.0, abs=2e-3)


def test_the_mep_skeleton_has_the_crossover_temperature_of_the_full_system(paths):
    """The path leaves the saddle along the unstable mode, so the bare potential on it has its frequency."""
    for name, model in family().items():
        reduced, _ = reductions(model, 2.0 * np.pi / 0.8, paths[name])
        assert barrier_frequency(*reduced["mep"]) == pytest.approx(model.saddle()[1], rel=2e-3)


# --- committed deterministic results ---------------------------------------------------------

def test_the_committed_exact_rates_are_converged_and_self_consistent(qm, skeleton):
    for name, record in qm["models"].items():
        model = family()[name]
        assert record["unstable_frequency"] == pytest.approx(model.saddle()[1], rel=1e-9)
        for ratio, entry in record["temperatures"].items():
            assert entry["beta"] == pytest.approx(2.0 * np.pi / (float(ratio) * record["unstable_frequency"]))
            assert abs(entry["basis_convergence"]) < 1e-3 and abs(entry["grid_convergence"]) < 3e-3
            below = "instanton" in entry
            assert below == (float(ratio) < 1.0)
            if below:
                eta = entry["instanton"]["eta1"]
                assert entry["instanton"]["converged"] and entry["instanton"]["negative_modes"] == 1
                assert entry["ln_eps_sd"] == pytest.approx(np.log(2.0 * np.pi / (entry["beta"] * eta)), abs=1e-12)
                assert eta == pytest.approx(entry["instanton"]["eta1_128"], rel=1e-3)
            else:
                assert entry["ln_eps_sd"] == 0.0
            assert skeleton["models"][name][ratio]["full"] == entry["ln_eps_sd"]
            for item in entry["reductions"].values():
                assert item["g_qm"] == pytest.approx(np.log(entry["k_qm"] / item["k_qm"]), abs=1e-12)
                assert item["delta_sd"] == pytest.approx(entry["ln_eps_sd"] - item["ln_eps_sd"], abs=1e-12)


def test_the_committed_reduction_entries_are_reproduced(qm, skeleton, paths):
    """One temperature of M2: 1D rates, instantons and skeleton factors from the current code."""
    model, entry = family()["M2"], qm["models"]["M2"]["temperatures"]["0.8"]
    beta = entry["beta"]
    reduced, mass = reductions(model, beta, paths["M2"])
    assert mass[0] == pytest.approx(entry["sct_mass_at_saddle"], rel=1e-6)
    for key in ("mep", "fe", "fe+sct"):
        x, potential = reduced[key]
        item = entry["reductions"][key]
        assert greens.barrier_rates(x, potential, [beta])[beta] == pytest.approx(item["k_qm"], rel=1e-6)
        found, frequency = tabulated_instanton(x, potential, beta, 256)
        assert steepest_descent_factor(beta, frequency, found) == pytest.approx(item["ln_eps_sd"], abs=1e-6)
    for key in ("mep", "mep+sct", "bare"):
        found, frequency = tabulated_instanton(*reduced[key], beta, 256)
        assert steepest_descent_factor(beta, frequency, found) == pytest.approx(
            skeleton["models"]["M2"]["0.8"][key], abs=1e-6)


def test_the_committed_full_instanton_is_reproduced(qm):
    for name in ("M2", "M4"):
        model, record = family()[name], qm["models"][name]
        entry = record["temperatures"]["0.8"]
        saddle, unstable, _, mode = model.saddle()
        found = find_instanton(model.gradient, model.hessian, saddle, mode, unstable, entry["beta"], nbeads=128)
        assert found["eta1"] == pytest.approx(entry["instanton"]["eta1_128"], rel=1e-7)


# --- committed RPMD results and the report ------------------------------------------------------

def test_independent_runs_are_averaged_in_the_logarithm_with_inverse_variance_weights():
    run = lambda rate, error, seed: {"rate": rate, "relative_error": error, "qtst": 2.0 * rate, "kappa": 0.5,
                                     "nbeads": 32, "seed": seed, "walkers": 32, "sweeps": 600, "seconds": 1.0}
    one = run_rpmd.combine([run(2.0, 0.02, 11)])
    assert one["rate"] == pytest.approx(2.0) and one["relative_error"] == pytest.approx(0.02)
    both = run_rpmd.combine([run(2.0, 0.02, 11), run(8.0, 0.02, 1011)])
    assert both["rate"] == pytest.approx(4.0) and both["qtst"] == pytest.approx(8.0)
    assert both["relative_error"] == pytest.approx(0.02 / np.sqrt(2.0))
    weighted = run_rpmd.combine([run(2.0, 0.01, 11), run(8.0, 0.02, 1011)])
    assert np.log(weighted["rate"]) == pytest.approx(0.8 * np.log(2.0) + 0.2 * np.log(8.0))
    assert weighted["seconds"] == 2.0 and len(weighted["runs"]) == 2
    # Only runs with the same beads and sampling are averaged with a new one.
    record = {"runs": [run(2.0, 0.02, 11), dict(run(2.0, 0.02, 5), nbeads=64)]}
    assert [r["seed"] for r in run_rpmd.earlier_runs(record, 32, run_rpmd.FULL)] == [11]
    assert run_rpmd.earlier_runs(None, 32, run_rpmd.FULL) == []


def test_every_rpmd_result_uses_the_committed_exact_rate(qm):
    files = sorted(RESULTS.glob("rpmd_*.json"))
    assert len(files) == 12
    for path in files:
        result = json.loads(path.read_text())
        name, ratio = result["model"], result["temperature_ratio"]
        entry = qm["models"][name]["temperatures"][str(ratio)]
        full = result["full"]
        assert result["beta"] == entry["beta"] and full["k_qm"] == entry["k_qm"]
        assert full["eps"] == pytest.approx(entry["k_qm"] * full["bead_factor"] / full["rate"], rel=1e-12)
        assert 0.0 < full["relative_error"] < 0.03 and 0.5 < full["kappa"] <= 1.0 + 3 * full["relative_error"]
        assert set(result["reductions"]) == set(run_rpmd.REDUCTIONS[name])
        for record in (full, *result["reductions"].values()):
            runs = record["runs"]
            assert len({run["seed"] for run in runs}) == len(runs) >= 1
            assert {run["nbeads"] for run in runs} == {run_rpmd.BEADS[name][ratio]}
            assert record["rate"] == pytest.approx(run_rpmd.combine(runs)["rate"], rel=1e-12)
            assert record["relative_error"] == pytest.approx(run_rpmd.combine(runs)["relative_error"], rel=1e-12)
            # Independent runs of one rate agree within their errors.
            logs = np.log([run["rate"] for run in runs])
            errors = np.array([run["relative_error"] for run in runs])
            assert np.all(abs(logs - np.log(record["rate"])) < 4.0 * errors)
        for key, item in result["reductions"].items():
            assert item["k_qm"] == entry["reductions"][key]["k_qm"]
            assert item["eps"] == pytest.approx(item["k_qm"] / item["rate"], rel=1e-12)
            assert 0.0 < item["relative_error"] < 0.03


def test_the_bare_reduction_of_m1_is_the_eckart_barrier():
    """M1 is straight: its ``mep`` reduction and ``run_eckart.py`` at ``v = 2.1`` solve one problem."""
    results = report.load_results()
    for row in report.transfer_rows(results):
        if row["model"] == "M1":
            eckart_run = report.eckart_entry(results, row["u"], 2.1)
            item = row["reductions"]["mep"]
            difference = item["ln_eps"] - np.log(eckart_run["eps"])
            assert abs(difference) < 3.5 * np.hypot(item["error_reduction"], eckart_run["relative_error"])


def test_the_eckart_runs_carry_their_steepest_descent_factor():
    results = report.load_results()
    assert len(results["eckart"]) == 15
    for run in results["eckart"]:
        found = eckart_instanton(run["v"], run["u"], 256) if run["u"] > 2.0 * np.pi else None
        assert run["ln_eps_sd"] == pytest.approx(steepest_descent_factor(run["u"], 1.0, found), abs=1e-9)
        assert run["remainder"] == pytest.approx(np.log(run["eps"]) - run["ln_eps_sd"], abs=1e-12)
        assert run["eps"] == pytest.approx(run["k_qm"] / run["rate"], rel=1e-12)


def test_the_anchors_vanish_above_the_crossover_and_for_an_exact_skeleton():
    rows = report.transfer_rows(report.load_results())
    assert {(r["model"], r["ratio"]) for r in rows} == {(m, t) for m in ("M1", "M2", "M3", "M4")
                                                         for t in (1.25, 0.8, 0.55)}
    for row in rows:
        assert "eckart" in row
        for key, item in row["reductions"].items():
            if row["ratio"] > 1.0 and key != "fe+sct":
                assert item["anchor_skeleton"] == 0.0 and item["anchor_naive"] == 0.0
            if row["model"] == "M1":
                assert abs(item["anchor_skeleton"]) < 1e-4
            assert item["transfer"] == pytest.approx(row["ln_eps"] - item["ln_eps"], abs=1e-12)
            assert item["error"] == pytest.approx(np.hypot(row["error"], item["error_reduction"]), rel=1e-12)


def test_the_naive_anchor_fails_where_the_skeleton_anchor_holds():
    """The finding of the report, with the statistics of the committed runs.

    A zero-point change (M1, M3) moves the instanton of the free-energy model
    but not the transfer error: the anchor that compares with that instanton
    is off by its whole size, the one that compares with the skeleton is not.
    """
    rows = report.transfer_rows(report.load_results())
    items = [r["reductions"]["fe"] for r in rows if r["model"] in ("M1", "M3") and r["ratio"] < 1.0]
    assert len(items) == 4
    for item in items:
        assert item["anchor_naive"] - item["anchor_skeleton"] > 0.06
        assert item["residual_naive"] < -2.5 * item["error"]
        assert abs(item["residual_skeleton"]) < 2.5 * item["error"]
    stats = report.summary(rows)["fe"]
    assert stats["skeleton"]["rms"] < stats["naive"]["rms"] / 2.0


def test_the_validation_report_is_the_generated_one():
    committed = ROOT / "docs" / "validation" / "2026-11-14-subspace-instanton-anchor" / "README.md"
    text = report.render()
    assert committed.read_text() == text
    assert "{{" not in text and "FINDINGS" not in text and "LIMITS" not in text
