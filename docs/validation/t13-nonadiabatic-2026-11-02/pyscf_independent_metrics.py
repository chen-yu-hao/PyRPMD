"""Reproduce T13 PySCF acceptance numbers from the repository root.

Run with PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python
    docs/validation/t13-nonadiabatic-2026-11-02/pyscf_independent_metrics.py

The optional PySCF dependency must be installed. Results are JSON on stdout.
"""
import json
import time

import numpy as np
from pyscf import fci, gto

from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.potentials.pyscf_multistate import PySCFMultistatePotential, casscf_overlap
from examples.nonadiabatic.pyscf_lih_rpsh import run


def checked(result):
    if np.any(result.status):
        raise RuntimeError(str(result.messages))
    return result


started = time.perf_counter()
settings = dict(casscf_conv_tol=1e-12, casscf_conv_tol_grad=1e-8,
                response_atol=1e-12, response_rtol=1e-10, cache_size=32)
bridge = PySCFMultistatePotential(["Li", "H"], **settings)
q = np.array([[[0., 0., 0.], [1.5*ANGSTROM_TO_AU, 0., 0.]]])
center = checked(bridge.evaluate(q))
epsilon = 2e-4
rows = []
for atom in (0, 1):
    plus, minus = q.copy(), q.copy()
    plus[0, atom, 0] += epsilon
    minus[0, atom, 0] -= epsilon
    forward = checked(bridge.evaluate(plus, previous_q=q, previous_context=center.context))
    backward = checked(bridge.evaluate(minus, previous_q=q, previous_context=center.context))
    dnac = (forward.overlaps-backward.overlaps)[0]/(2*epsilon)
    denergy = (forward.energies-backward.energies)[0]/(2*epsilon)
    rows.append({
        "atom": atom, "component": "x",
        "analytic_nac": center.couplings[0, :, :, atom, 0].tolist(),
        "overlap_derivative": dnac.tolist(),
        "signed_nac_max_error_bohr_inverse": float(np.max(abs(dnac-center.couplings[0, :, :, atom, 0]))),
        "analytic_gradient": center.gradients[0, :, atom, 0].tolist(),
        "energy_derivative": denergy.tolist(),
        "gradient_max_error_hartree_per_bohr": float(np.max(abs(denergy-center.gradients[0, :, atom, 0]))),
    })

displaced = q.copy()
displaced[:, 0, 0] += .2
displaced[:, 1, 0] += .4
displaced[:, 1, 1] += .15
current = checked(bridge.evaluate(displaced))
old, new = center.context[0], current.context[0]
left, right = old.wavefunction, new.wavefunction
orbital = left.mo.T @ gto.intor_cross("int1e_ovlp", left.mol, right.mol) @ right.mo
determinants = []
for electrons in left.nelecas:
    strings = fci.cistring.gen_occslst(range(left.ncas), electrons)
    occupied = [np.r_[np.arange(left.ncore), left.ncore+string] for string in strings]
    determinants.append(np.array([[np.linalg.det(orbital[np.ix_(i, j)])
                                   for j in occupied] for i in occupied]))
reference = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        for a, b in np.ndindex(left.ci.shape[1:]):
            for c, d in np.ndindex(right.ci.shape[1:]):
                reference[i, j] += (left.ci[i, a, b]*right.ci[j, c, d]
                                    *determinants[0][a, c]*determinants[1][b, d])
reference = reference[np.ix_(old.order, new.order)]*old.phases[:, None]*new.phases[None]
exact_overlap_error = float(np.max(abs(casscf_overlap(old, new)-reference)))

h2 = PySCFMultistatePotential(["H", "H"], nstates=3, **settings)
hq = np.array([[[0., 0., 0.], [2.4, 0., 0.]]])
hc = checked(h2.evaluate(hq))
hp, hm = hq.copy(), hq.copy()
hp[0, 1, 0] += epsilon
hm[0, 1, 0] -= epsilon
hforward = checked(h2.evaluate(hp, previous_q=hq, previous_context=hc.context))
hbackward = checked(h2.evaluate(hm, previous_q=hq, previous_context=hc.context))
three_state_error = float(np.max(abs((hforward.overlaps-hbackward.overlaps)[0]/(2*epsilon)
                                    -hc.couplings[0, :, :, 1, 0])))
coarse = run(steps=6, dt=2., initial_state=1)
fine = run(steps=12, dt=1., initial_state=1)
report = {
    "provider": bridge.to_dict(), "finite_difference_step_bohr": epsilon,
    "lih_derivative_checks": rows,
    "lih_max_signed_nac_error_bohr_inverse": max(r["signed_nac_max_error_bohr_inverse"] for r in rows),
    "lih_max_gradient_error_hartree_per_bohr": max(r["gradient_max_error_hartree_per_bohr"] for r in rows),
    "full_occupied_determinant_overlap_max_error": exact_overlap_error,
    "h2_three_state_max_signed_nac_error_bohr_inverse": three_state_error,
    "rpsh_timestep_scan": {
        "duration_au": 12., "coarse_dt_au": 2., "fine_dt_au": 1.,
        "coarse_max_energy_drift_hartree": coarse["max_energy_drift_hartree"],
        "fine_max_energy_drift_hartree": fine["max_energy_drift_hartree"],
        "fine_over_coarse_error_ratio": fine["max_energy_drift_hartree"]/coarse["max_energy_drift_hartree"],
        "coarse_configuration": coarse["configuration"], "fine_configuration": fine["configuration"],
    },
    "wall_seconds": time.perf_counter()-started, "physical_converged": False,
}
bridge.close()
h2.close()
print(json.dumps(report, indent=2, allow_nan=False))
