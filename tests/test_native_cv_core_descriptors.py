"""The two native descriptor ABIs retain their parser and rollback contracts."""

from copy import deepcopy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.potentials import FreePotential
from pyrpmd.potentials._builtin_native import resolve_native_binding
from pyrpmd.potentials.native import descriptor_binding
from pyrpmd.propagation import _atomic_binding


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native extension unavailable",
)


def _primitive(constant=0., terms=()):
    return {"constant": constant, "linear": [0.] * 9,
            "norms": [{"factor": factor, "weights": weights}
                      for factor, weights in terms]}


def _descriptors(api):
    forming, breaking = [1., -1., 0.], [0., 1., -1.]
    if api == "numeric":
        norm = {"kind": 0, "primitive": _primitive(terms=[(1., forming)])}
        rate = {"kind": 1,
                "entry": _primitive(10., [(-1., [1., -.5, -.5])]),
                "paths": [[_primitive(0., [(-1., forming), (1., breaking)])]]}
        bias = {"kind": 0, "cvs": [deepcopy(norm)], "center": [1.], "stiffness": [.2]}
    else:
        norm = {"kind": "norm", "estimator": "centroid", "weights": forming}
        rate = {"kind": "rate", "estimator": "centroid", "weights": [1., -.5, -.5],
                "rinf": 10., "paths": [[{"forming": forming, "breaking": breaking,
                                        "reference": 0.}]]}
        bias = {"kind": "umbrella", "cvs": [deepcopy(norm)], "center": [1.],
                "stiffness": [.2], "periods": [0.]}
    return norm, rate, bias


@pytest.mark.parametrize("api", ["numeric", "tagged"])
@pytest.mark.parametrize("case,marker", [
    ("vector_length", "dimension|length"),
    ("unknown_kind", "kind"),
    ("empty_row", "pairs|paths"),
    ("unequal_rows", "rows|paths"),
    ("nonfinite_scalar", "scalar|finite"),
    ("missing_field", "descriptor|field"),
    ("bias_center_length", "dimension|length"),
])
def test_malformed_native_descriptors_fail_before_state_or_output_mutation(api, case, marker):
    norm, rate, bias = _descriptors(api)
    cv, biases = norm, []
    if case == "vector_length":
        weights = norm["primitive"]["norms"][0]["weights"] if api == "numeric" else norm["weights"]
        weights.pop()
    elif case == "unknown_kind":
        cv["kind"] = 99 if api == "numeric" else "unknown-cv"
    elif case == "empty_row":
        cv = rate
        cv["paths"] = [[]]
    elif case == "unequal_rows":
        cv = rate
        cv["paths"].append([deepcopy(cv["paths"][0][0]) for _ in range(2)])
    elif case == "nonfinite_scalar":
        cv = rate
        if api == "numeric":
            cv["entry"]["constant"] = float("nan")
        else:
            cv["rinf"] = float("nan")
    elif case == "missing_field":
        if api == "numeric":
            del cv["primitive"]["linear"]
        else:
            del cv["weights"]
    elif case == "bias_center_length":
        bias["center"].append(2.)
        biases = [bias]

    dynamics = Dynamics(FreePotential(), [2., 3., 4.], 6., 2, .01)
    binding = resolve_native_binding(dynamics.potential, 3)
    positions = np.array([
        [[-2., .1, .2], [.2, -.3, .1], [1.7, .2, -.1]],
        [[-1.9, .2, .3], [.1, -.2, .2], [1.8, .3, -.2]],
    ])
    momenta = np.arange(18, dtype=float).reshape(2, 3, 3) * .01
    before_q, before_p = positions.tobytes(), momenta.tobytes()
    # These initialized buffers also detect writes before descriptor validation.
    if api == "numeric":
        energies, centroids, values = np.full(2, -731.), np.full((2, 3, 3), -732.), np.full((2, 1), -733.)
        outputs = [energies, centroids, values]
        options = {"observers": [cv], "biases": biases, "fragments": [], "time": .25}
        lease = _atomic_binding(binding)
        error_type = RuntimeError
    else:
        records, snapshots = np.full((2, 21), -734.), np.full((2, 2, 2, 3, 3), -735.)
        outputs = [records, snapshots]
        options = {"beta": 6., "time": .25, "cvs": [cv], "biases": biases,
                   "constraint": None, "constraint_tolerance": 1e-10,
                   "fragments": [], "record_energy": True}
        lease = descriptor_binding(binding)
        error_type = ValueError
    before_outputs = [array.tobytes() for array in outputs]

    # Call the native functions directly so Python lowering cannot reject or
    # repair these deliberately malformed descriptors before the parser sees them.
    with lease.acquire() as native_binding, dynamics.integrator._lock:
        with pytest.raises(error_type, match=marker):
            if api == "numeric":
                dynamics.integrator._native.run_block(
                    dynamics.integrator._context, native_binding, positions, momenta,
                    2, 1, energies, centroids, values, options,
                )
            else:
                dynamics.integrator._native.advance(
                    dynamics.integrator._context, native_binding, positions, momenta,
                    options, np.ones(2, dtype=np.uint8), records, snapshots,
                )
    assert positions.tobytes() == before_q
    assert momenta.tobytes() == before_p
    assert [array.tobytes() for array in outputs] == before_outputs
