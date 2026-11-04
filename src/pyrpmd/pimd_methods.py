"""Force-based equilibrium actions for the main :class:`Dynamics` engine.

These are sampling Hamiltonians, not Born--Oppenheimer potentials and not
real-time RPMD approximations.  They use the engine's sum-beads/beta-over-P
convention.  All arrays and imaginary times are in atomic units.

References: Kapil et al., JCP 145, 234103 (2016), doi:10.1063/1.4971438;
Feldman and Hirshberg, JCP 159, 154107 (2023), doi:10.1063/5.0173749;
Sarsa et al., JCP 113, 1366 (2000), doi:10.1063/1.481926.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp

from .path_integral import _integer, _masses, _positive
from .potentials.base import Potential, PotentialError, PESResult, coordinates, evaluate_checked, validate_result
from .potentials.contraction import _component


def _ring_terms(q, masses, beta):
    delta = q - np.roll(q, -1, axis=0)
    coefficient = masses[None, :, None] * (len(q) / beta) ** 2
    energy = 0.5 * float(np.sum(coefficient * delta * delta))
    force = -coefficient * (2 * q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0))
    return energy, force


class _SamplingPotential(Potential):
    """Borrow a physical provider; its lifecycle remains with the caller."""

    equilibrium_only = True

    def __init__(self, potential, masses, beta, nbeads):
        self.potential = _component(potential, "physical potential")
        self.masses = _masses(masses)
        self.beta = _positive(beta, "beta")
        self.nbeads = _integer(nbeads, "nbeads")
        self.metadata = {
            "provider": type(self).__name__, "batch_independent": False,
            "ring_polymer_coupled": True, "trajectory_batch_safe": False,
            "equilibrium_only": True, "virial": False,
            "energy_convention": "sum_beads_beta_over_P_sampling_action",
            "thread_safety": "context_serial",
        }

    def _q(self, positions):
        q = coordinates(positions)
        if q.shape != (self.nbeads, len(self.masses), 3):
            raise ValueError("sampling potential requires one complete ring with matching P,N")
        return q


class SuzukiChinPotential(_SamplingPotential):
    """Alternating Suzuki--Chin potential with symmetric finite-difference forces.

    Requires an even bead count.  ``fd_step`` is the RMS Cartesian displacement
    in bohr, normalized independently on each force-bearing bead.  Physical
    observables use even beads for alpha=0; averaging the modified potential
    is not an energy estimator.  :meth:`estimators` supplies thermodynamic
    energy, pressure and scaled-coordinate heat-capacity ingredients for all
    alpha.  When the physical provider supplies strain virials, the effective
    action virial is differentiated with the same displaced force calls.
    """

    def __init__(self, potential, masses, beta, nbeads, *, alpha=0.0, fd_step=1e-4):
        super().__init__(potential, masses, beta, nbeads)
        if self.nbeads % 2:
            raise ValueError("Suzuki--Chin requires an even bead count")
        if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
            raise ValueError("alpha must be finite and in [0,1]")
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1]")
        self.fd_step = _positive(fd_step, "fd_step")
        self.weights = np.where(np.arange(nbeads) % 2 == 0, 2 / 3, 4 / 3)
        d = np.where(np.arange(nbeads) % 2 == 0, self.alpha / 6, (1 - self.alpha) / 12)
        self.coefficients = self.weights * d * (self.beta / self.nbeads) ** 2
        self.weights.flags.writeable = self.coefficients.flags.writeable = False
        self.metadata.pop("virial", None)
        self.metadata.update({
            "action": "suzuki_chin",
            "virial_requires": "physical_provider_configurational_virials",
            "virial_convention": "negative_affine_strain_derivative_of_sampling_potential",
            "pressure_estimator": "thermodynamic_centroid_virial",
        })

    def set_cell(self, cell):
        """Forward a validated periodic cell; physical-provider ownership stays external."""
        from .geometry import validate_cell_pbc
        cell, _ = validate_cell_pbc(cell, [True, True, True])
        if cell is None:
            raise ValueError("Suzuki--Chin cell must be a finite nonsingular 3D cell")
        setter = getattr(self.potential, "set_cell", None)
        if not callable(setter):
            raise PotentialError("Suzuki--Chin NPT requires physical set_cell(cell_bohr) support")
        setter(cell.copy())
        self.cell = cell.copy()

    def evaluate(self, positions):
        q = self._q(positions)
        return self._evaluate_sc(q)[0]

    def _evaluate_sc(self, q):
        """Evaluate one or more whole rings as independent physical frames."""
        count = len(q) // self.nbeads
        if count < 1 or len(q) != count * self.nbeads:
            raise ValueError("Suzuki--Chin evaluation requires complete rings")
        weights = np.tile(self.weights, count)
        coefficients = np.tile(self.coefficients, count)
        result = evaluate_checked(self.potential, q)
        physical_energy, physical_force = result.energy.copy(), result.forces.copy()
        physical_virial = None if result.virial is None else result.virial.copy()
        acceleration = physical_force / self.masses[None, :, None]
        correction = coefficients * np.sum(physical_force * acceleration, axis=(1, 2))
        energy = weights * physical_energy + correction
        forces = weights[:, None, None] * physical_force
        virial = None if physical_virial is None else weights[:, None, None] * physical_virial
        rms = np.sqrt(np.mean(acceleration * acceleration, axis=(1, 2)))
        active = (coefficients != 0) & (rms > 0)
        if np.any(active):
            direction = acceleration[active]
            h = self.fd_step / rms[active]
            plus = q[active] + h[:, None, None] * direction
            minus = q[active] - h[:, None, None] * direction
            rp = evaluate_checked(self.potential, plus)
            fp = rp.forces.copy()
            wp = None if rp.virial is None else rp.virial.copy()
            rm = evaluate_checked(self.potential, minus)
            fm = rm.forces
            # dF/dq @ (F/m) = - Hess(V) @ (F/m).
            projected = (fp - fm) / (2 * h[:, None, None])
            forces[active] -= 2 * coefficients[active, None, None] * projected
            if virial is not None:
                if wp is None or rm.virial is None:
                    raise PotentialError("Suzuki--Chin displaced physical evaluations must preserve virial support")
                # W_G = -d(sum F^2/m)/d(strain) = 2*(F tensor F/m - D_(F/m) W).
                # The tensor term differentiates Cartesian vector components;
                # omitting it gives a wrong pressure even for a harmonic PES.
                projected_virial = (wp - rm.virial) / (2 * h[:, None, None])
                force_tensor = np.einsum("bna,bnc->bac", physical_force[active], direction)
                virial[active] += 2 * coefficients[active, None, None] * (force_tensor - projected_virial)
        effective = validate_result(PESResult(energy, forces, virial=virial), q.shape)
        physical = PESResult(physical_energy, physical_force, virial=physical_virial)
        return effective, physical, correction

    def estimators(self, positions, *, volume=None, finite_difference_step=1e-3,
                   require_pressure=True):
        """Physical SC observables and heat-capacity curvature for one ring.

        ``volume`` is in bohr^3; pressure requires a physical provider virial.
        Set ``require_pressure=False`` for NVT energy/heat-capacity analysis
        when a periodic physical provider does not supply virials.
        The beta derivative holds the cell fixed.  Heat capacities require
        an ensemble of these ingredients, not the fluctuation of the sampling
        Hamiltonian; use ``suzuki_chin_thermodynamics`` to combine samples.
        """
        from .suzuki_chin import suzuki_chin_estimators
        return suzuki_chin_estimators(self, positions, volume=volume,
                                      finite_difference_step=finite_difference_step,
                                      require_pressure=require_pressure)

    def estimators_batch(self, positions, *, volumes=None, finite_difference_step=1e-3,
                         require_pressure=True):
        """Vectorized fixed-cell estimator columns for paths shaped [sample,P,N,3]."""
        from .suzuki_chin import suzuki_chin_estimators_batch
        return suzuki_chin_estimators_batch(self, positions, volumes=volumes,
                                            finite_difference_step=finite_difference_step,
                                            require_pressure=require_pressure)


@dataclass(frozen=True)
class ExchangeResult:
    """Bosonic sampling action and optional fermionic reweighting ingredients.

    ``energy`` is the effective spring free energy in the engine's sum-beads
    convention. ``spring_energy`` is the probability-weighted quadratic spring
    energy; they are different quantities. ``virial`` is the negative affine
    strain derivative of ``energy`` (coordinates and cell strained together).
    The fermionic numerator is weighted by permutation parity and divided by
    the *bosonic* weight, so it remains finite at a fermionic node.
    """

    energy: float
    forces: np.ndarray
    connections: np.ndarray
    virial: np.ndarray | None = None
    spring_energy: float | None = None
    fermionic_sign: float | None = None
    fermionic_spring_energy_numerator: float | None = None
    winding_diagnostics: dict | None = None


def _exchange_geometry(cell, pbc, periodic_mode):
    from .geometry import FixedCellGeometry
    geometry = FixedCellGeometry(cell, pbc)
    if periodic_mode is not None and (not isinstance(periodic_mode, str)
                                      or periodic_mode not in ("minimum_image", "full_winding")):
        raise ValueError("periodic_mode must be 'minimum_image' or 'full_winding' for winding sums")
    if geometry.pbc.any() and periodic_mode is None:
        raise ValueError("periodic exchange requires explicit periodic_mode='minimum_image' or 'full_winding'")
    if not geometry.pbc.any() and periodic_mode is not None:
        raise ValueError("periodic_mode requires periodic boundary conditions")
    return geometry


def _exchange_statistics(statistics):
    if not isinstance(statistics, str) or statistics not in ("bosonic", "fermionic"):
        raise ValueError("statistics must be 'bosonic' or 'fermionic'")
    return statistics


def bosonic_exchange(positions, mass, beta, *, cell=None, pbc=None,
                     periodic_mode=None, statistics="bosonic",
                     winding_tolerance=1e-12, winding_max_terms=100000):
    """Feldman--Hirshberg springs, with O(N² + PN) exchange recursion.

    One group of equal-mass indistinguishable particles. The recurrence samples
    representative permutations; its pointwise potential is NOT the factorial
    sum over every permutation. ``connections[i,j]`` is the probability of
    joining last bead i to first j. The physical potential must be invariant
    under exchange within the group.

    Periodic cells require an explicit mode. ``minimum_image`` uses the
    shortest Euclidean lattice image for every link, an approximation with
    undefined derivatives on image ties. ``full_winding`` sums all periodic
    Gaussian images to controlled per-link truncation bounds, using a Poisson
    dual for broad Gaussians and separable sums for orthogonal cells. Its work
    includes the numerical terms per link; a budget failure raises ValueError.
    With ``statistics="fermionic"``, the forces still sample the positive
    bosonic density and the result additionally contains signed estimator
    ingredients. No division by a configuration's fermionic sign is performed.
    """
    from .exchange_winding import GaussianWindingKernel, validate_winding_options
    geometry = _exchange_geometry(cell, pbc, periodic_mode)
    tolerance, budget = validate_winding_options(winding_tolerance, winding_max_terms, periodic_mode)
    q, mass, beta = coordinates(positions), _positive(mass, "mass"), _positive(beta, "beta")
    statistics = _exchange_statistics(statistics)
    if periodic_mode == "full_winding":
        kernel = GaussianWindingKernel(geometry, mass * len(q) / (2 * beta),
                                       tolerance=tolerance, max_terms=budget)
        return _winding_exchange(q, mass, beta, kernel, statistics)
    return _bosonic_exchange(q, mass, beta, geometry, statistics)


def _bosonic_exchange(q, mass, beta, geometry, statistics):
    p, n, _ = q.shape
    coefficient = mass * (p / beta) ** 2
    beta_p = beta / p
    periodic = geometry.pbc.any()
    interior_delta = q[1:] - q[:-1]
    link_delta = q[-1, :, None] - q[0, None, :]
    if periodic:
        interior_delta = geometry.minimum_image(interior_delta)
        link_delta = geometry.minimum_image(link_delta)
    interior = 0.5 * coefficient * float(np.sum(interior_delta ** 2))
    # Internal springs occur in every representative permutation. Removing
    # their common offset before log recurrences prevents cancellation at low T.
    link = 0.5 * coefficient * np.sum(link_delta ** 2, axis=2)
    adjacent = np.concatenate(([0.0], np.cumsum(np.diag(link, k=1))))
    cycle = np.zeros((n, n))
    for end in range(n):
        starts = np.arange(end + 1)
        cycle[starts, end] = adjacent[end] - adjacent[starts] + link[end, starts]
    prefix = np.zeros(n + 1)
    signed = np.zeros(n + 1)
    signed[0] = 1.0
    signed_spring = np.zeros(n + 1)
    for end in range(1, n + 1):
        terms = prefix[:end] - beta_p * cycle[:end, end - 1]
        prefix[end] = logsumexp(terms) - math.log(end)
        if statistics == "fermionic":
            # Normalize by the positive bosonic prefix at each recursion level.
            # Parity of a cycle of length k is (-1)**(k-1). fsum limits loss
            # from cancellation without changing the requested float64 model.
            parity = np.where((end - np.arange(end) - 1) % 2, -1.0, 1.0)
            # Do not reconstruct the conditional normalization by subtracting
            # two large log partition values: that can move a valid sign
            # outside [-1,1]. Remove the common action before summation.
            raw_weight = np.exp(terms - np.max(terms))
            normalization = math.fsum(raw_weight)
            signed[end] = math.fsum(raw_weight * parity * signed[:end]) / normalization
            weight = raw_weight * parity / normalization
            signed_spring[end] = math.fsum(weight * (signed_spring[:end]
                + cycle[:end, end - 1] * signed[:end]))
    suffix = np.zeros(n + 1)
    log_indices = np.log(np.arange(1, n + 1))
    for start in range(n - 1, -1, -1):
        suffix[start] = logsumexp(-beta_p * cycle[start, start:] + suffix[start + 1:] - log_indices[start:])
    connections = np.zeros((n, n))
    for end in range(n):
        starts = np.arange(end + 1)
        connections[end, starts] = np.exp(prefix[starts] + suffix[end + 1]
            - beta_p * cycle[starts, end] - math.log(end + 1) - prefix[n])
        if end + 1 < n:
            exponent = prefix[end + 1] + suffix[end + 1] - prefix[n]
            connections[end, end + 1] = -np.expm1(min(0.0, exponent))
    force = np.zeros_like(q)
    force[:-1] += coefficient * interior_delta
    force[1:] -= coefficient * interior_delta
    if periodic:
        force[-1] -= coefficient * np.einsum("ij,ijc->ic", connections, link_delta)
        force[0] += coefficient * np.einsum("ij,ijc->jc", connections, link_delta)
    else:
        # Preserve the original arithmetic for old nonperiodic trajectories.
        force[-1] -= coefficient * (q[-1] - connections @ q[0])
        force[0] -= coefficient * (q[0] - connections.T @ q[-1])
    energy = interior - float(prefix[n]) / beta_p
    spring_energy = interior + float(np.sum(connections * link))
    virial = -coefficient * (np.einsum("pna,pnb->ab", interior_delta, interior_delta)
        + np.einsum("ij,ija,ijb->ab", connections, link_delta, link_delta))
    sign = float(signed[n]) if statistics == "fermionic" else None
    numerator = (float(signed_spring[n] + interior * signed[n])
                 if statistics == "fermionic" else None)
    if (not math.isfinite(energy) or not np.isfinite(force).all()
            or not math.isfinite(spring_energy) or not np.isfinite(virial).all()
            or (sign is not None and (not math.isfinite(sign) or not math.isfinite(numerator)))):
        raise ValueError("bosonic spring overflow; check mass, beta and coordinates")
    return ExchangeResult(energy, force, connections, virial, spring_energy, sign, numerator)


def _winding_exchange(q, mass, beta, kernel, statistics):
    """Representative exchange recurrence with separate log-weight and moments.

    A link's free energy is -log(K)/beta_P; its thermodynamic spring moment
    is coefficient*<delta squared>/2. Only the latter differentiates beta
    correctly. Keeping these channels separate is essential at small volume,
    where the link free energy is negative but its quadratic moment is positive.
    """
    p, n, _ = q.shape
    coefficient = mass * (p / beta) ** 2
    beta_p = beta / p
    interior_count = (p - 1) * n
    delta = np.concatenate(((q[1:] - q[:-1]).reshape(-1, 3),
                            (q[-1, :, None] - q[0, None, :]).reshape(-1, 3)))
    moments = kernel.evaluate(delta)
    inner_mean = moments.mean_displacement[:interior_count].reshape(p - 1, n, 3)
    link_mean = moments.mean_displacement[interior_count:].reshape(n, n, 3)
    inner_second = moments.second_moment[:interior_count]
    link_second = moments.second_moment[interior_count:].reshape(n, n, 3, 3)
    interior_free = -float(np.sum(moments.log_kernel[:interior_count])) / beta_p
    interior_spring = 0.5 * coefficient * float(np.trace(inner_second, axis1=1, axis2=2).sum())
    link = -moments.log_kernel[interior_count:].reshape(n, n) / beta_p
    spring = 0.5 * coefficient * np.trace(link_second, axis1=2, axis2=3)
    adjacent = np.concatenate(([0.0], np.cumsum(np.diag(link, k=1))))
    adjacent_spring = np.concatenate(([0.0], np.cumsum(np.diag(spring, k=1))))
    cycle, cycle_spring = np.zeros((n, n)), np.zeros((n, n))
    for end in range(n):
        starts = np.arange(end + 1)
        cycle[starts, end] = adjacent[end] - adjacent[starts] + link[end, starts]
        cycle_spring[starts, end] = adjacent_spring[end] - adjacent_spring[starts] + spring[end, starts]
    prefix = np.zeros(n + 1)
    signed = np.zeros(n + 1)
    signed[0] = 1.0
    signed_spring = np.zeros(n + 1)
    for end in range(1, n + 1):
        terms = prefix[:end] - beta_p * cycle[:end, end - 1]
        prefix[end] = logsumexp(terms) - math.log(end)
        if statistics == "fermionic":
            parity = np.where((end - np.arange(end) - 1) % 2, -1.0, 1.0)
            raw_weight = np.exp(terms - np.max(terms))
            normalization = math.fsum(raw_weight)
            signed[end] = math.fsum(raw_weight * parity * signed[:end]) / normalization
            weight = raw_weight * parity / normalization
            signed_spring[end] = math.fsum(weight * (signed_spring[:end]
                + cycle_spring[:end, end - 1] * signed[:end]))
    suffix = np.zeros(n + 1)
    log_indices = np.log(np.arange(1, n + 1))
    for start in range(n - 1, -1, -1):
        suffix[start] = logsumexp(-beta_p * cycle[start, start:] + suffix[start + 1:] - log_indices[start:])
    connections = np.zeros((n, n))
    for end in range(n):
        starts = np.arange(end + 1)
        connections[end, starts] = np.exp(prefix[starts] + suffix[end + 1]
            - beta_p * cycle[starts, end] - math.log(end + 1) - prefix[n])
        if end + 1 < n:
            exponent = prefix[end + 1] + suffix[end + 1] - prefix[n]
            connections[end, end + 1] = -np.expm1(min(0.0, exponent))
    force = np.zeros_like(q)
    force[:-1] += coefficient * inner_mean
    force[1:] -= coefficient * inner_mean
    force[-1] -= coefficient * np.einsum("ij,ijc->ic", connections, link_mean)
    force[0] += coefficient * np.einsum("ij,ijc->jc", connections, link_mean)
    energy = interior_free - float(prefix[n]) / beta_p
    spring_energy = interior_spring + float(np.sum(connections * spring))
    virial = -coefficient * (np.sum(inner_second, axis=0)
                            + np.einsum("ij,ijab->ab", connections, link_second))
    sign = float(signed[n]) if statistics == "fermionic" else None
    numerator = (float(signed_spring[n] + interior_spring * signed[n])
                 if statistics == "fermionic" else None)
    if (not math.isfinite(energy) or not np.isfinite(force).all()
            or not math.isfinite(spring_energy) or not np.isfinite(virial).all()
            or (sign is not None and (not math.isfinite(sign) or not math.isfinite(numerator)))):
        raise ValueError("full winding exchange overflow; check mass, beta, cell and coordinates")
    diagnostics = dict(kernel.metadata)
    diagnostics["representative_weight_relative_truncation_bound"] = math.expm1(
        p * n * math.log1p(kernel.error_bounds[0]))
    diagnostics["links_per_representative"] = p * n
    return ExchangeResult(energy, force, connections, virial, spring_energy, sign, numerator, diagnostics)


def _ring_virial(q, masses, beta):
    delta = q - np.roll(q, -1, axis=0)
    return -(len(q) / beta) ** 2 * np.einsum("n,pna,pnb->ab", masses, delta, delta)


class BosonicExchangePotential(_SamplingPotential):
    """Replace the engine's distinguishable springs by bosonic sampling springs.

    ``groups`` are disjoint equal-mass groups; unlisted particles remain
    distinguishable. ``statistics="fermionic"`` changes only estimator
    reweighting, never the positive sampling density. Distinct spin species
    can be represented by separate groups; spin degeneracies are not inferred.

    Periodic minimum-image sampling is an explicit approximation. Full winding
    sampling sums the periodic Gaussian images with controlled truncation.
    All links including those of unlisted particles are consistent. The raw
    springs added by Dynamics are subtracted exactly once. The affine virial
    describes this correction, but does not make the existing centroid BZP
    propagator compatible with the all-bead periodic coordinate domain.
    """

    def __init__(self, potential, masses, beta, nbeads, *, groups=None,
                 statistics="bosonic", cell=None, pbc=None, periodic_mode=None,
                 winding_tolerance=1e-12, winding_max_terms=100000):
        from .exchange_winding import validate_winding_options
        super().__init__(potential, masses, beta, nbeads)
        self.statistics = _exchange_statistics(statistics)
        self.geometry = _exchange_geometry(cell, pbc, periodic_mode)
        self.cell, self.pbc = self.geometry.cell, self.geometry.pbc
        self.periodic_mode = periodic_mode
        self.winding_tolerance, self.winding_max_terms = validate_winding_options(
            winding_tolerance, winding_max_terms, periodic_mode)
        if groups is None:
            groups = [list(range(len(self.masses)))]
        if not isinstance(groups, (list, tuple)) or not groups:
            raise ValueError("groups must be a nonempty sequence")
        self.groups, seen = [], set()
        for group in groups:
            values = list(group)
            if not values or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                    or not 0 <= i < len(self.masses) for i in values):
                raise ValueError("boson groups must contain in-range integer indices")
            if len(set(values)) != len(values) or seen.intersection(values):
                raise ValueError("boson groups must be disjoint without repeated particles")
            ids = np.asarray(values, dtype=int)
            if not np.all(self.masses[ids] == self.masses[ids[0]]):
                raise ValueError("each boson group requires exactly equal masses")
            self.groups.append(ids)
            seen.update(values)
        self._unlisted = np.asarray(sorted(set(range(len(self.masses))) - seen), dtype=int)
        self._sampling_groups = list(self.groups)
        if self.pbc.any():
            self._sampling_groups += [np.array([i]) for i in self._unlisted]
        self._winding_kernels = self._make_winding_kernels(self.geometry)
        self.metadata.pop("virial", None)
        self.metadata.update({
            "action": "bosonic_exchange", "statistics": self.statistics,
            "sampling_statistics": "bosonic", "periodic_mode": periodic_mode,
            "virial_requires": "physical_provider_configurational_virials",
            "virial_convention": "negative_affine_strain_derivative_of_sampling_potential",
            "pressure_estimator": "primitive_thermodynamic_given_cell",
            "npt_coordinate_coupling": "all_beads",
            "winding_sum": periodic_mode == "full_winding",
        })
        self._winding_metadata()

    def _make_winding_kernels(self, geometry):
        if self.periodic_mode != "full_winding":
            return {}
        from .exchange_winding import GaussianWindingKernel
        return {mass: GaussianWindingKernel(geometry, mass * self.nbeads / (2 * self.beta),
            tolerance=self.winding_tolerance, max_terms=self.winding_max_terms)
            for mass in dict.fromkeys(float(self.masses[ids[0]]) for ids in self._sampling_groups)}

    def _winding_metadata(self):
        if self.periodic_mode == "full_winding":
            self.metadata.update({"winding_tolerance": self.winding_tolerance,
                "winding_max_terms": self.winding_max_terms,
                "winding_kernels": [{"mass": mass, **kernel.metadata}
                                    for mass, kernel in self._winding_kernels.items()]})

    def _exchange_group(self, q, mass, statistics):
        if self.periodic_mode == "full_winding":
            return _winding_exchange(q, mass, self.beta, self._winding_kernels[float(mass)], statistics)
        return _bosonic_exchange(q, mass, self.beta, self.geometry, statistics)

    def set_cell(self, cell, *, barostat_coupling=None):
        """Transactionally update periodic geometry for explicit all-bead scaling.

        The keyword prevents the existing centroid-only BZP class from
        accidentally using an incompatible periodic coordinate measure.
        """
        if barostat_coupling != "all_beads":
            raise PotentialError("centroid-only BZP scaling is incompatible with periodic exchange; "
                                 "set_cell requires barostat_coupling='all_beads'")
        if not self.pbc.any():
            raise PotentialError("all-bead exchange cell updates require a periodic action")
        geometry = _exchange_geometry(cell, self.pbc, self.periodic_mode)
        kernels = self._make_winding_kernels(geometry)
        setter = getattr(self.potential, "set_cell", None)
        if not callable(setter):
            raise PotentialError("exchange cell updates require physical set_cell(cell_bohr) support")
        try:
            setter(geometry.cell.copy())
        except BaseException:
            try:
                setter(self.cell.copy())
            except BaseException as rollback_error:
                raise PotentialError("physical set_cell failed and restoring its previous cell also failed") from rollback_error
            raise
        self.geometry = geometry
        self.cell, self.pbc = geometry.cell, geometry.pbc
        self._winding_kernels = kernels
        self._winding_metadata()

    def evaluate_sampling(self, positions):
        """Return the complete physical-plus-exchange sampling action.

        Unlike :meth:`evaluate`, this entry point does not subtract any raw
        integrator springs. Its row sums provide the full potential energy
        and affine virial for a Cartesian all-bead periodic propagator.
        """
        q = self._q(positions)
        result = evaluate_checked(self.potential, q)
        energy, force = result.energy.copy(), result.forces.copy()
        virial = None if result.virial is None else result.virial.copy()
        for ids in self._sampling_groups:
            exchanged = self._exchange_group(q[:, ids], self.masses[ids[0]], "bosonic")
            energy += exchanged.energy / self.nbeads
            force[:, ids] += exchanged.forces
            if virial is not None:
                virial += exchanged.virial / self.nbeads
        if not self.pbc.any() and len(self._unlisted):
            ids = self._unlisted
            ring, ordinary = _ring_terms(q[:, ids], self.masses[ids], self.beta)
            energy += ring / self.nbeads
            force[:, ids] += ordinary
            if virial is not None:
                virial += _ring_virial(q[:, ids], self.masses[ids], self.beta) / self.nbeads
        return validate_result(PESResult(energy, force, virial=virial), q.shape)

    def evaluate(self, positions):
        q = self._q(positions)
        result = evaluate_checked(self.potential, q)
        energy, force = result.energy.copy(), result.forces.copy()
        virial = None if result.virial is None else result.virial.copy()
        for ids in self._sampling_groups:
            ring, ordinary = _ring_terms(q[:, ids], self.masses[ids], self.beta)
            exchanged = self._exchange_group(q[:, ids], self.masses[ids[0]], "bosonic")
            energy += (exchanged.energy - ring) / self.nbeads
            force[:, ids] += exchanged.forces - ordinary
            if virial is not None:
                virial += (exchanged.virial - _ring_virial(q[:, ids], self.masses[ids], self.beta)) / self.nbeads
        return validate_result(PESResult(energy, force, virial=virial), q.shape)

    def estimators(self, positions, *, volume=None, require_pressure=False,
                   physical_result=None):
        """Finite, sign-weighted primitive thermodynamic ingredients for one ring.

        Divide *ensemble* means of numerator columns by the ensemble mean sign;
        never divide individual configurations. ``physical_result`` can reuse
        a just-evaluated physical PES result for these same coordinates.
        Optional pressure requires a fully periodic cell, its volume (bohr³),
        and the physical configurational virial. The additional nonperiodic
        unshifted virial energy is valid for bound, integrable distributions.
        """
        q = self._q(positions)
        if not isinstance(require_pressure, (bool, np.bool_)):
            raise ValueError("require_pressure must be boolean")
        physical = validate_result(self.potential.evaluate(q) if physical_result is None
                                   else physical_result, q.shape)
        if volume is not None:
            volume = _positive(volume, "volume")
            if not self.pbc.all():
                raise ValueError("exchange pressure requires a fully periodic cell")
            if not np.isclose(volume, abs(np.linalg.det(self.cell)), rtol=1e-12, atol=0):
                raise ValueError("volume disagrees with the fixed exchange cell")
        if require_pressure and (volume is None or physical.virial is None):
            raise PotentialError("exchange pressure requires volume and physical configurational virials")
        signs, springs = [], []
        for ids in self._sampling_groups:
            exchanged = self._exchange_group(q[:, ids], self.masses[ids[0]], self.statistics)
            signs.append(exchanged.fermionic_sign if self.statistics == "fermionic" else 1.0)
            springs.append(exchanged.fermionic_spring_energy_numerator
                           if self.statistics == "fermionic" else exchanged.spring_energy)
        # Products excluding one group are built without division, including at
        # nodes where one or several of the group weights vanish exactly.
        prefix = np.concatenate(([1.0], np.cumprod(signs)))
        suffix = np.concatenate((np.cumprod(signs[::-1])[::-1], [1.0]))
        sign = float(prefix[-1])
        spring_numerator = math.fsum(spring * prefix[i] * suffix[i + 1]
                                     for i, spring in enumerate(springs))
        if not self.pbc.any() and len(self._unlisted):
            spring_numerator += sign * _ring_terms(q[:, self._unlisted],
                                                   self.masses[self._unlisted], self.beta)[0]
        potential = float(np.mean(physical.energy))
        kinetic_numerator = 3 * self.nbeads * len(self.masses) / (2 * self.beta) * sign - spring_numerator / self.nbeads
        observed = {
            "sign": sign,
            "potential_energy_numerator_hartree": sign * potential,
            "kinetic_energy_numerator_hartree": kinetic_numerator,
            "internal_energy_numerator_hartree": sign * potential + kinetic_numerator,
        }
        if not self.pbc.any():
            observed["virial_internal_energy_numerator_hartree"] = sign * (
                potential - float(np.sum(q * physical.forces)) / (2 * self.nbeads))
        if volume is not None and physical.virial is not None:
            observed["pressure_numerator_au"] = (self.nbeads * len(self.masses) / (self.beta * volume)
                + float(np.trace(physical.virial, axis1=1, axis2=2).sum()) / (3 * self.nbeads * volume)) * sign \
                - 2 * spring_numerator / (3 * self.nbeads * volume)
        if not all(math.isfinite(value) for value in observed.values()):
            raise ValueError("nonfinite exchange estimator; check beta, masses and coordinates")
        return observed


class GaussianTrialWavefunction:
    """Positive trial amplitude exp[-sum_i alpha_i |q_i-center_i|² / 2]."""

    def __init__(self, alpha, center=0.0):
        self.alpha = np.asarray(alpha, dtype=float)
        self.center = np.asarray(center, dtype=float)
        if (not np.isfinite(self.alpha).all() or np.any(self.alpha <= 0)
                or not np.isfinite(self.center).all()):
            raise ValueError("trial alpha must be positive; alpha and center must be finite")

    def evaluate(self, positions):
        q = coordinates(positions)
        alpha = self.alpha[:, None] if self.alpha.ndim == 1 else self.alpha
        displacement = q - self.center
        gradient = np.broadcast_to(-alpha * displacement, q.shape).copy()
        log_amplitude = -0.5 * np.sum(alpha * displacement ** 2, axis=(1, 2))
        return log_amplitude, gradient


class PIGSPotential(_SamplingPotential):
    """Primitive open-chain ground-state projection with positive endpoint trial.

    For L links and P=L+1 beads, tau=projection_time/L and the auxiliary
    engine beta=P*tau.  The density is Psi_T(q0) exp[-S_primitive] Psi_T(qL).
    Its midpoint has the pure projected distribution; endpoints do not.
    ``trial.evaluate([B,N,3])`` returns log amplitude[B], its gradient[B,N,3].
    """

    def __init__(self, potential, masses, nbeads, projection_time, trial):
        nbeads = _integer(nbeads, "nbeads", 3)
        if nbeads % 2 != 1:
            raise ValueError("PIGS requires odd nbeads (an even number of links)")
        self.projection_time = _positive(projection_time, "projection_time")
        self.tau = self.projection_time / (nbeads - 1)
        super().__init__(potential, masses, nbeads * self.tau, nbeads)
        if not callable(getattr(trial, "evaluate", None)):
            raise TypeError("trial must implement evaluate returning log amplitude and gradient")
        self.trial = trial
        self.weights = np.ones(nbeads)
        self.weights[[0, -1]] = 0.5
        self.metadata["topology"] = "fully_open_ground_state_projection"

    def evaluate(self, positions):
        q = self._q(positions)
        physical = evaluate_checked(self.potential, q)
        energy = self.weights * physical.energy
        force = self.weights[:, None, None] * physical.forces
        log_psi, gradient = self.trial.evaluate(q[[0, -1]])
        log_psi, gradient = np.asarray(log_psi), np.asarray(gradient)
        if (log_psi.dtype.kind not in "fiu" or gradient.dtype.kind not in "fiu"
                or log_psi.shape != (2,) or gradient.shape != (2, q.shape[1], 3)
                or not np.isfinite(log_psi).all() or not np.isfinite(gradient).all()):
            raise ValueError("trial must return finite log amplitude[2], gradient[2,N,3]")
        energy[[0, -1]] -= log_psi / self.tau
        force[[0, -1]] += gradient / self.tau
        # Remove only the closing link from the engine's periodic spring.
        delta = q[-1] - q[0]
        coefficient = self.masses[:, None] / self.tau ** 2
        closing = 0.5 * float(np.sum(coefficient * delta ** 2))
        energy -= closing / self.nbeads
        force[-1] += coefficient * delta
        force[0] -= coefficient * delta
        return validate_result(PESResult(energy, force), q.shape)

    def action(self, positions):
        q = self._q(positions)
        energy = float(self.evaluate(q).energy.sum()) + _ring_terms(q, self.masses, self.beta)[0]
        return self.tau * energy


__all__ = ["SuzukiChinPotential", "BosonicExchangePotential", "bosonic_exchange",
           "ExchangeResult", "GaussianTrialWavefunction", "PIGSPotential"]
