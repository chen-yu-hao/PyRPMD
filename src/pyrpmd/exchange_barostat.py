"""All-bead isotropic BZP pressure control for periodic exchange sampling.

For beta_P=beta/P the target density is
exp[-beta_P*(H_P + P*p_ext*V + pi**2/(2*M))] dq dp dV dpi, with
every bead coordinate on the periodic cell.  Fractional coordinates therefore
have Jacobian V**(N*P).  No center-of-mass constraint is imposed.

Writing eta=pi/M, the all-bead dilation has phase-space compressibility
3*eta: the q and p divergences cancel, while dV/dt=3*eta*V.  The piston
force 2*K_all+Tr(W)-3*P*p_ext*V+3*P/beta gives exactly the compensating
log-density derivative.  The deterministic conserved quantity subtracts
(P/beta)*log(V).  A centroid dilation has a different coordinate domain
and must not be substituted for this generator.

The exact momentum/piston kick and drift/dilation subflows generalize
Eqs. (12)--(13) of Ceriotti, More and Manolopoulos, Comput. Phys. Commun.
185, 1019 (2014), doi:10.1016/j.cpc.2013.10.027, to all bead coordinates.
See also Bussi, Zykova-Timan and Parrinello, J. Chem. Phys. 130, 074101
(2009), doi:10.1063/1.3073889.  The latter removes the center of mass;
its extra measure correction must not be copied into this unconstrained map.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

from .barostat import BZPBarostat, _real, validate_bzp_snapshot
from .dynamics import Dynamics, _state_clock
from .geometry import validate_cell_pbc
from .pimd_methods import BosonicExchangePotential
from .potentials.base import PotentialError, validate_result


def validate_exchange_bzp_snapshot(snapshot, reference_cell):
    """Validate an all-bead piston checkpoint without constructing a PES.

    The coupling tag prevents a centroid checkpoint from silently selecting
    a different volume measure when an exchange run is resumed.
    """
    if (not isinstance(snapshot, Mapping)
            or set(snapshot) != {"method", "coupling", "cell", "momentum"}
            or snapshot["coupling"] != "all_beads"):
        raise ValueError("exchange BZP checkpoint requires coupling='all_beads'")
    validated = validate_bzp_snapshot(
        {key: value for key, value in snapshot.items() if key != "coupling"}, reference_cell)
    return {**validated, "coupling": "all_beads"}


class ExchangeBZPBarostat:
    """Isotropic BZP sampling on the complete periodic bead-coordinate domain.

    ``potential`` is a :class:`BosonicExchangePotential` with an explicit
    periodic action and physical configurational virials.  Fermionic
    observables may be reweighted from this same positive bosonic sampling.
    All beads have Maxwell momentum variance m*P/beta.  The piston mass is
    3*N*P*tau_au**2/beta and its OU variance is M*P/beta.

    The step has two full-action PES evaluations.  It reduces to ordinary
    velocity Verlet when the piston rate vanishes.  Only Andersen collisions
    or an explicit unthermostatted check are supported; no existing Dynamics
    thermostat, pressure operator, force splitting, constraint, or specialized
    dynamics subclass is bypassed.  Sampling accuracy still requires timestep
    and statistical convergence checks.
    """

    # Reuse pressure-state validation and the OU bath, but do not subclass the
    # centroid piston: the legacy composed engine dispatches that class through
    # a different generator and checkpoint schema.  It must reject this object
    # at binding rather than silently bypass the explicit all-bead step below.
    cell = BZPBarostat.cell
    volume = BZPBarostat.volume
    _check_state = BZPBarostat._check_state
    _thermostat = BZPBarostat._thermostat

    def __init__(self, potential, system, beta, nbeads, pressure_au, tau_au,
                 friction_au=None):
        if not isinstance(potential, BosonicExchangePotential):
            raise TypeError("exchange BZP requires BosonicExchangePotential")
        if not potential.pbc.all():
            raise ValueError("exchange BZP requires all three directions periodic")
        if not callable(getattr(potential, "evaluate_sampling", None)):
            raise PotentialError("exchange BZP requires a complete sampling-action evaluation")
        if potential.beta != beta or potential.nbeads != nbeads:
            raise ValueError("exchange action and BZP must share beta and bead count")
        get = system.get if isinstance(system, Mapping) else lambda key, default=None: getattr(system, key, default)
        if not np.array_equal(np.asarray(get("masses")), potential.masses):
            raise ValueError("exchange action and BZP must share atomic masses")
        if not np.array_equal(np.asarray(get("cell")), potential.cell):
            raise ValueError("exchange action and BZP initial cells differ")
        BZPBarostat.__init__(self, potential, system, beta, nbeads,
                            pressure_au, tau_au, friction_au)
        self.mass = _real(self.mass * self.nbeads, "exchange piston mass", positive=True)

    def _set_cell(self, cell):
        cell, _ = validate_cell_pbc(cell, [True, True, True])
        volume = abs(np.linalg.det(cell))
        if not np.isfinite(volume) or volume <= 0:
            raise FloatingPointError("exchange BZP cell has a nonpositive or nonfinite volume")
        # The explicit tag preserves the existing centroid-BZP rejection.
        self.potential.set_cell(cell.copy(), barostat_coupling="all_beads")
        self._cell = cell

    def snapshot(self):
        return {**BZPBarostat.snapshot(self), "coupling": "all_beads"}

    def restore(self, snapshot):
        saved = validate_exchange_bzp_snapshot(snapshot, self._reference_cell)
        self._set_cell(saved["cell"])
        self.momentum = saved["momentum"]

    def _sampling_result(self, q):
        result = validate_result(self.potential.evaluate_sampling(q), q.shape)
        if result.virial is None:
            raise PotentialError("exchange BZP requires physical and exchange configurational virials")
        return result

    def _pressure_numerator(self, state):
        self._check_state(state)
        result = self._sampling_result(state.q)
        numerator = (np.sum(state.p**2 / self.masses[None, :, None])
                     + np.trace(result.virial, axis1=1, axis2=2).sum())
        if not np.isfinite(numerator):
            raise FloatingPointError("nonfinite all-bead exchange BZP pressure")
        return float(numerator)

    def pressure(self, state):
        """All-bead mechanical sampling pressure in hartree/bohr**3.

        Its conditional momentum average equals the primitive quantum pressure
        estimator.  A finite free quantum gas can have a nonzero V=0 boundary
        weight, so its NPT mean pressure need not equal the external pressure.
        """
        return self._pressure_numerator(state) / (3 * self.nbeads * self.volume)

    def _scale(self, dt):
        exponent = dt * self.momentum / self.mass
        if not math.isfinite(exponent) or abs(exponent) > 100:
            raise FloatingPointError("exchange BZP cell change overflow; reduce timestep or increase time_constant")
        return exponent, math.exp(exponent)

    def _dilate(self, state, dt):
        """Exact all-bead dilation; unlike centroid BZP, offsets also scale."""
        _, scale = self._scale(dt)
        q, p = state.q * scale, state.p / scale
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite exchange BZP dilation")
        self._set_cell(self._cell * scale)
        state.q[:], state.p[:] = q, p

    def _drift_dilate(self, state, dt):
        """Exact combined free drift and all-bead dilation at fixed piston."""
        exponent, scale = self._scale(dt)
        if abs(exponent) < 1e-5:
            sinhc = 1 + exponent**2 / 6 + exponent**4 / 120
        else:
            sinhc = math.sinh(exponent) / exponent
        q = scale * state.q + dt * sinhc * state.p / self.masses[None, :, None]
        p = state.p / scale
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite exchange BZP drift/dilation")
        self._set_cell(self._cell * scale)
        state.q[:], state.p[:] = q, p

    def _force_piston_kick(self, state, result, dt):
        """Integrate the quadratic kinetic contribution during a force kick."""
        force = result.forces
        inverse_masses = 1 / self.masses[None, :, None]
        drive = (float(np.sum(state.p**2 * inverse_masses))
                 + float(np.trace(result.virial, axis1=1, axis2=2).sum())
                 - 3 * self.nbeads * self.pressure_au * self.volume
                 + 3 * self.nbeads / self.beta)
        self.momentum += (dt * drive
                          + dt**2 * float(np.sum(state.p * force * inverse_masses))
                          + dt**3 / 3 * float(np.sum(force**2 * inverse_masses)))
        state.p += dt * force
        if not math.isfinite(self.momentum) or not np.isfinite(state.p).all():
            raise FloatingPointError("nonfinite exchange BZP force/piston kick")

    def _check_dynamics(self, state, dynamics, thermostat, andersen_interval, method):
        if type(dynamics) is not Dynamics:
            raise TypeError("exchange BZP requires plain Dynamics, not a specialized subclass")
        dynamics._check_parameters()
        if (dynamics.potential is not self.potential or dynamics.beta != self.beta
                or dynamics.nbeads != self.nbeads
                or not np.array_equal(dynamics.masses, self.masses)):
            raise ValueError("exchange BZP and Dynamics must share action, masses, beta and bead count")
        if dynamics.integrator_name != "velocity-verlet":
            raise ValueError("exchange BZP requires integrator='velocity-verlet'")
        if dynamics.plan.active or dynamics._sampling_compositions:
            raise ValueError("exchange BZP cannot bypass existing Dynamics thermostat, pressure or splitting operators")
        if dynamics.fixed_atoms or dynamics.bond_constraints:
            raise ValueError("exchange BZP requires all atoms mobile and no constraints")
        if dynamics.loop_backend == "native" or dynamics._native_requested == "required":
            raise ValueError("exchange BZP requires the Python propagation loop")
        if thermostat not in ("andersen", None) or method is not None:
            raise ValueError("exchange BZP supports Andersen sampling or thermostat=None, without a middle method")
        if (isinstance(andersen_interval, (bool, np.bool_))
                or not isinstance(andersen_interval, (int, np.integer)) or andersen_interval < 1):
            raise ValueError("andersen_interval must be a positive integer")
        if not np.array_equal(self.cell, self.potential.cell):
            raise ValueError("exchange action cell changed outside its all-bead barostat")
        self._check_state(state)
        dynamics.integrator._check(state.q, state.p)
        step, time = _state_clock(state.step, state.time)
        dt = _real(dynamics.dt, "exchange BZP timestep", positive=True)
        if not math.isfinite(time + dt):
            raise FloatingPointError("nonfinite exchange BZP integration clock")
        return step, time, dt

    def step(self, state, dynamics, *, thermostat="andersen", gamma_centroid=None,
             andersen_interval=10, method=None):
        """Take one atomic all-bead step with exactly two action evaluations.

        RNG order is piston OU/2, a full Andersen collision when the *global*
        step clock selects one, then piston OU/2 after the reversible B/2 A
        B/2 map.  A failure restores q, p, clock, piston, cell, RNG and counters.
        No Dynamics operator is advanced, so operator restoration cannot
        accidentally install a centroid cell update.
        """
        if gamma_centroid is not None:
            raise ValueError("exchange BZP uses Andersen collisions; gamma_centroid is unsupported")
        step, time, dt = self._check_dynamics(
            state, dynamics, thermostat, andersen_interval, method)
        previous_state, previous_piston = state.copy(), self.snapshot()
        previous_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
        previous_stats = copy.deepcopy(dynamics.propagation_stats)
        previous_backend = dynamics.last_step_backend
        try:
            self._thermostat(dynamics.rng, dt / 2)
            if thermostat == "andersen" and step % andersen_interval == 0:
                state.p[:] = dynamics.sample_momenta()
            first = self._sampling_result(state.q)
            self._force_piston_kick(state, first, dt / 2)
            self._drift_dilate(state, dt)
            second = self._sampling_result(state.q)
            self._force_piston_kick(state, second, dt / 2)
            self._thermostat(dynamics.rng, dt / 2)
            self._check_state(state)
            state.step, state.time = step + 1, time + dt
            dynamics._record_python_steps(1)
            dynamics.last_step_backend = "python"
            if dynamics.loop_backend != "python":
                reason = "all-bead exchange BZP uses the Python propagation loop"
                reasons = dynamics.propagation_stats["fallback_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
        except BaseException:
            state.q[:], state.p[:] = previous_state.q, previous_state.p
            state.step, state.time = previous_state.step, previous_state.time
            dynamics.rng.bit_generator.state = previous_rng
            dynamics.propagation_stats = previous_stats
            dynamics.last_step_backend = previous_backend
            self.restore(previous_piston)
            raise
        return state

    def extended_energy(self, state, dynamics):
        """Deterministic invariant; fictitious thermostat heat is excluded."""
        self._check_state(state)
        if dynamics.potential is not self.potential:
            raise ValueError("extended energy requires the shared exchange action")
        result = self._sampling_result(state.q)
        kinetic = 0.5 * float(np.sum(state.p**2 / self.masses[None, :, None]))
        return float(kinetic + result.energy.sum()
                     + self.nbeads * self.pressure_au * self.volume
                     + self.momentum**2 / (2 * self.mass)
                     - self.nbeads / self.beta * math.log(self.volume))
