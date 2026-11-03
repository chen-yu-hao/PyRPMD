"""Shared native CV mathematics must retain both descriptor entry points.

The Python integrator and dense analytic derivatives are independent references.
The two APIs deliberately retain their different validation/transaction rules.
"""

import numpy as np
import pytest

from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias
from pyrpmd.cv import BondDifference, Cartesian, COMDistance, CVStack, Distance, RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import FreePotential, HarmonicPotential


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native extension unavailable",
)

MASSES = np.array([2.4, 1.7, 1.7, 3.2])
REFERENCE = np.array([
    [-1.7, .3, -.1], [.1, -.2, .2], [1.8, .5, -.3], [2.9, 1.3, .7],
])


def _rate():
    return RPMDrateCoordinate(
        [[0], [1, 2, 3]], MASSES, REFERENCE,
        [[0, 1], [0, 2]], [[1, 3], [2, 3]], 12.,
        equivalent_paths=[[0, 2, 1, 3]],
    )


def _initial(beads):
    rng = np.random.default_rng(9103)
    q = REFERENCE[None] + rng.normal(0., .08, (beads, 4, 3))
    return State(q, rng.normal(0., .13, q.shape), step=7, time=.137)


def _advance(dynamics, state, api, cv, bias, steps=17):
    options = {"observe": cv} if api == "propagate" else {"cv": cv}
    return getattr(dynamics, api)(state, steps, bias=bias, **options)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("beads", [1, 5])
def test_both_descriptor_formats_share_all_cv_kinds_and_composite_bias(api, beads):
    rate = _rate()
    cv = CVStack([
        Distance([0, 1]), BondDifference([0, 1, 2]),
        COMDistance([[0, 3], [1, 2]], MASSES), Cartesian(2, "z"),
        rate, rate.dividing_surface(.73),
    ])
    actual, expected = _initial(beads), _initial(beads)
    center = cv.value(actual.q) + np.linspace(-.13, .17, cv.dimension)
    stiffness = np.diag(np.linspace(.01, .04, cv.dimension)) + .003
    bias = CompositeBias(
        UmbrellaBias(cv, center, stiffness), FluxMetricBias(rate, MASSES, 6.),
    )
    native = Dynamics(HarmonicPotential(k=.003), MASSES, 6., beads, .002,
                      loop_backend="native")
    reference = Dynamics(HarmonicPotential(k=.003), MASSES, 6., beads, .002,
                         native=False, loop_backend="python")
    result = _advance(native, actual, api, cv, bias)
    wanted = _advance(reference, expected, api, cv, bias)
    assert result.backend == "native" and wanted.backend == "python"
    for name in ("q", "p"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=2e-12, atol=2e-12)
    for name in ("values", "energies", "centroids"):
        np.testing.assert_allclose(getattr(result, name), getattr(wanted, name),
                                   rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(result.steps, wanted.steps)
    assert (actual.step, actual.time) == (expected.step, expected.time)


@pytest.mark.parametrize("beads", [1, 4, 7])
def test_block_umbrella_preserves_periodic_wrapping_and_mixed_estimators(beads):
    angle_like = Cartesian(0, "x", estimator="bead_average")
    # Built-in Cartesian descriptors are linear; periodicity is a bias property.
    angle_like.periods = (2.3,)
    cv = CVStack([
        angle_like, Distance([0, 1], estimator="bead_average"),
        BondDifference([0, 1, 2]),
        COMDistance([[0, 3], [1, 2]], MASSES, estimator="bead_average"),
    ])
    actual, expected = _initial(beads), _initial(beads)
    center = cv.value(actual.q) + np.array([16.1 + .31, -.13, .17, -.09])
    stiffness = np.diag([.04, .03, .02, .05]) + .004
    bias = UmbrellaBias(cv, center, stiffness)
    # This is an actual wrapped branch, well away from its half-period seam.
    raw = float(cv.value(actual.q)[0] - center[0])
    wrapped = (raw + 1.15) % 2.3 - 1.15
    assert abs(raw - wrapped) > 10. and abs(wrapped) < .4
    native = Dynamics(HarmonicPotential(k=.003), MASSES, 6., beads, .002,
                      loop_backend="native")
    reference = Dynamics(HarmonicPotential(k=.003), MASSES, 6., beads, .002,
                         native=False, loop_backend="python")
    result = native.propagate_block(actual, 19, bias=bias, cv=cv, record_states=True)
    wanted = reference.propagate_block(expected, 19, bias=bias, cv=cv, record_states=True)
    assert result.backend == "native"
    for name in ("q", "p", "values", "energies"):
        np.testing.assert_allclose(getattr(result, name), getattr(wanted, name),
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
def test_rate_ties_keep_first_path_and_bond_for_gradient_and_metric(api):
    masses = np.array([2., 1., 1.])
    q = np.array([[3., 6., 5.], [2., 4., 5.], [4., 4., 5.]])[None]
    rate = RPMDrateCoordinate(
        [[0], [1, 2]], masses, q[0], [[0, 1], [0, 2]], [[1, 2], [1, 2]],
        10., equivalent_paths=[[0, 2, 1]],
    )
    selected = RPMDrateCoordinate(
        [[0], [1, 2]], masses, q[0], [[0, 1]], [[1, 2]], 10.,
    )
    cv = CVStack([rate, rate.dividing_surface(.73)])
    selected_cv = CVStack([selected, selected.dividing_surface(.73)])
    center, stiffness = [1.1, 2.0], [[.4, .07], [.07, .3]]
    bias = CompositeBias(UmbrellaBias(cv, center, stiffness), FluxMetricBias(rate, masses, 6.))
    # A single explicitly selected branch avoids sharing the min/max selection
    # implementation with the native calculation. Its retained dense Hessian
    # also checks the HVP, at a seam where finite differences are inappropriate.
    _, gradient, hessian = selected.centroid_value_gradient_hessian(q[0])
    direction = gradient / masses[:, None]
    metric = np.sum(gradient * direction)
    expected_force = np.einsum("acbd,bd->ac", hessian, direction) / (6. * metric)
    expected_force = expected_force[None] + UmbrellaBias(selected_cv, center, stiffness).energy_force(q)[1]
    dynamics = Dynamics(FreePotential(), masses, 6., 1, 1e-10, loop_backend="native")
    state = State(q, np.zeros_like(q))
    result = _advance(dynamics, state, api, cv, bias, steps=1)
    # Tiny-step drift is below one ULP: p/dt directly measures the applied force.
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_allclose(state.p / dynamics.dt, expected_force, rtol=2e-12, atol=5e-14)
    np.testing.assert_allclose(result.values[0], selected_cv.value(q), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("surface", [False, True])
def test_unselected_coincident_rate_branch_keeps_each_api_validation_policy(surface):
    masses = np.array([2.4, 1.7, 1.7, 3.2])
    reference = np.array([[0., 2., 0.], [-1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    q = np.array([[2., 2., 2.], [6., 2., 2.], [2., 2., 2.], [7., 3., 2.]])[None]
    rate = RPMDrateCoordinate(
        [[0], [1, 2, 3]], masses, reference,
        [[0, 1], [0, 2]], [[1, 3], [2, 3]], 12.,
    )
    selected = RPMDrateCoordinate(
        [[0], [1, 2, 3]], masses, reference, [[0, 1]], [[1, 3]], 12.,
    )
    cv = rate.dividing_surface(.73) if surface else rate
    expected_cv = selected.dividing_surface(.73) if surface else selected
    # The coincident forming bond belongs to the larger, unselected candidate.
    values = [ref - np.linalg.norm(q[0, f[0]] - q[0, f[1]])
              + np.linalg.norm(q[0, b[0]] - q[0, b[1]])
              for ref, f, b in zip(rate.reference_difference, rate.forming, rate.breaking)]
    assert values[0] < values[1]
    dynamics = Dynamics(FreePotential(), masses, 6., 1, .01, loop_backend="native")
    block_state = State(q, np.zeros_like(q), 9, .2)
    result = dynamics.propagate_block(block_state, 3, cv=cv)
    np.testing.assert_allclose(result.values, np.tile(expected_cv.value(q), (3, 1)),
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(block_state.q, q)
    fused_state = State(q, np.zeros_like(q), 9, .2)
    with pytest.raises(RuntimeError, match="distance|coincident"):
        dynamics.propagate(fused_state, 3, observe=cv)
    np.testing.assert_array_equal(fused_state.q, q)
    np.testing.assert_array_equal(fused_state.p, np.zeros_like(q))
    assert (fused_state.step, fused_state.time) == (9, .2)


def test_numeric_descriptor_general_primitive_keeps_linear_term_and_many_norms():
    """The original numeric ABI permits more norms than a rate descriptor needs."""
    from pyrpmd.potentials._builtin_native import resolve_native_binding
    from pyrpmd.propagation import _atomic_binding

    weights = np.array([
        [1., -1., 0., 0.], [0., 1., -1., 0.], [0., 0., 1., -1.],
        [.2, .8, -.1, -.9], [.3, -.2, .4, -.5],
    ])
    factors = np.array([.3, -.2, 1.1, -.7, .4])
    linear = np.array([[.1, .2, -.1], [.3, -.2, .1], [-.1, .2, .3], [.2, -.3, .1]])
    q = (REFERENCE + 4.)[None]
    descriptor = {"kind": 0, "primitive": {
        "constant": 2.1, "linear": linear.ravel().tolist(),
        "norms": [{"factor": factor, "weights": weight.tolist()}
                  for factor, weight in zip(factors, weights)],
    }}
    value = 2.1 + np.sum(linear * q[0])
    gradient = linear.copy()
    hessian = np.zeros((4, 3, 4, 3))
    for factor, weight in zip(factors, weights):
        displacement = weight @ q[0]
        radius = np.linalg.norm(displacement)
        unit = displacement / radius
        value += factor * radius
        gradient += factor * weight[:, None] * unit
        hessian += factor * np.einsum("a,b,cd->acbd", weight, weight,
                                      (np.eye(3) - np.outer(unit, unit)) / radius)
    direction = gradient / MASSES[:, None]
    metric = np.sum(gradient * direction)
    expected_force = np.einsum("acbd,bd->ac", hessian, direction) / (6. * metric)
    dynamics = Dynamics(FreePotential(), MASSES, 6., 1, 1e-10)
    binding = _atomic_binding(resolve_native_binding(dynamics.potential, len(MASSES)))
    positions, momenta = q.copy(), np.zeros_like(q)
    energies, values = np.empty(1), np.empty((1, 1))
    spec = {"observers": [descriptor], "biases": [
        {"kind": 1, "cvs": [descriptor], "beta": 6.},
    ], "fragments": [], "time": 0.}
    with binding.acquire() as capsule, dynamics.integrator._lock:
        dynamics.integrator._native.run_block(
            dynamics.integrator._context, capsule, positions, momenta,
            1, 1, energies, None, values, spec,
        )
    np.testing.assert_array_equal(positions, q)
    np.testing.assert_allclose(momenta[0] / dynamics.dt, expected_force, rtol=2e-12, atol=5e-14)
    np.testing.assert_allclose(values[0, 0], value, rtol=2e-14, atol=2e-14)
    expected_energy = -np.log(np.sqrt(metric / (2. * np.pi * 6.))) / 6.
    np.testing.assert_allclose(energies[0], expected_energy, rtol=2e-14, atol=2e-14)
