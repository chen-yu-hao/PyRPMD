"""Two-state MASH using direct adiabatic electronic data.

The active surface is the larger electronic population; there is no stochastic
hop decision. Initial spin sampling, NAC momentum rescaling and frustrated-hop
reflection follow Mannouch and Richardson, JCP 158, 104111 (2023).

This is a fixed-step endpoint-switching implementation. It requires timestep
convergence and does not claim the reversible/event-localized integrator of
Geuther et al., JCTC 21, 2179 (2025). P>1 is an experimental centroid extension,
not a claim to implement every published RP-MASH variant.
"""
import numpy as np

from .adiabatic import AdiabaticRPSHDynamics
from .potentials.base import PotentialError
from .surface_hopping import _surface


class MASHDynamics(AdiabaticRPSHDynamics):
    """Deterministic two-state MASH, with an explicit experimental P>1 extension.

    Pure initial surface populations sample the spin hemisphere with density
    proportional to |S_z|. ``populations`` reports active-surface indicators;
    electronic amplitudes are accessible separately on the state.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates=2, **kwargs):
        if isinstance(nstates, (bool, np.bool_)) or nstates != 2:
            raise ValueError("this MASH implementation requires exactly two states")
        if "frustrated_hops" in kwargs and kwargs["frustrated_hops"] != "reverse":
            raise ValueError("MASH requires NAC momentum reflection at frustrated hops")
        kwargs["frustrated_hops"] = "reverse"
        super().__init__(potential, masses, beta, nbeads, dt, 2, **kwargs)

    def initialize(self, nuclear, active_surface=0, *, adiabatic_coefficients=None, seed=None):
        active = _surface(active_surface, 2)
        if adiabatic_coefficients is None:
            rng = self.rng if seed is None else (seed if isinstance(seed, np.random.Generator)
                                                else np.random.default_rng(seed))
            z = np.sqrt(max(float(rng.random()), np.finfo(float).tiny))
            if active == 1:
                z = -z
            phi = rng.uniform(0., 2*np.pi)
            adiabatic_coefficients = np.array([np.sqrt((1+z)/2),
                                               np.sqrt((1-z)/2)*np.exp(1j*phi)])
        state = super().initialize(nuclear, active, adiabatic_coefficients=adiabatic_coefficients)
        self._check(state)
        return state

    def _check(self, state):
        super()._check(state)
        population = abs(state.coefficients)**2
        if population[state.active_surface] < .5-1e-12:
            raise ValueError("MASH active surface and spin hemisphere disagree")

    def populations(self, state):
        self._check(state)
        return np.eye(2)[state.active_surface]

    def _select_hop(self, trial, probabilities, surfaces):
        crossings = sum(a != b for a, b in zip(surfaces[:-1], surfaces[1:]))
        if crossings > 1:
            raise ValueError("multiple MASH hemisphere crossings in one step; reduce timestep")
        target = int(np.argmax(abs(trial.coefficients)**2))
        if target == trial.active_surface:
            return None
        nac = trial.centroid_frame.result.couplings[0, trial.active_surface, target]
        if np.linalg.norm(nac) <= np.finfo(float).tiny:
            raise PotentialError("MASH crossing has no endpoint NAC rescaling direction; reduce timestep")
        hop = self._hop(trial, target)
        if hop.frustrated:
            # Keep Sx and Sy, reverse Sz: preserve both coefficient phases
            # while swapping their moduli, rather than swapping coefficients.
            moduli = abs(trial.coefficients)
            phase = np.ones(2, dtype=complex)
            np.divide(trial.coefficients, moduli, out=phase, where=moduli > 0)
            trial.coefficients[:] = phase*moduli[::-1]
        return hop

    def attempt_hop(self, state, target):
        raise ValueError("MASH surface changes are determined by spin crossings in step()")
