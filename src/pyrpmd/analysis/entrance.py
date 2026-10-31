"""Physical entrance normalisation for H + H2 centroid-bond RPMD rates.

All inputs use bohr, hartree, electron masses and atomic time. Never infer a
bimolecular rate from a finite r01/r12 PMF without an independently calibrated,
connected, *same-canonical-ensemble* reference-to-target flux bridge.
"""
from __future__ import annotations

import numpy as np

from ..constants import AU_RATE_TO_CM3_S
from .bond_flux import bond_difference_gradient
from .rate_postprocess import surface_rate_2d


def _number(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value):
        raise ValueError(f"{name} must be a scalar")
    x = float(value)
    if not np.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and positive")
    return x


def free_h3_entrance_flux(radius_bohr, beta, masses_au, *,
                          electronic_weight=1., symmetry_factor=1.):
    """Free H(0)+H(1)H(2) incoming flux through Rinf, bohr^3/atomic time."""
    R = _number(radius_bohr, "radius_bohr")
    beta = _number(beta, "beta")
    m = np.asarray(masses_au, float)
    if m.shape != (3,) or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError("masses_au must be three positive atomic-unit masses")
    g = _number(electronic_weight, "electronic_weight", allow_zero=True)
    sym = _number(symmetry_factor, "symmetry_factor")
    mu = float(m[0] * (m[1] + m[2]) / m.sum())
    v = float(np.sqrt(1 / (2 * np.pi * beta * mu)))
    k = float(4 * np.pi * R**2 * v * g * sym)
    return dict(reference_rate_au=k,
                reference_rate_cm3_s=k * AU_RATE_TO_CM3_S,
                reduced_mass_au=mu, incoming_half_maxwell_speed_au=v,
                radius_bohr=R, electronic_weight=g, symmetry_factor=sym)


def bond_entrance_observables(centroid, masses_au, beta):
    """R(0,COM12), both bond-difference surfaces and angle-dependent speeds.

    sigma_i = r(0,i)-r(1,2), i in {1,2}. Third-bond ordering prevents
    double-counting two atom-labelled transition arrangements.
    """
    q = np.asarray(centroid, float)
    if q.ndim == 3:
        if q.shape[0] == 0:
            raise ValueError("empty bead axis")
        q = q.mean(axis=0)
    m = np.asarray(masses_au, float)
    beta = _number(beta, "beta")
    if q.shape != (3, 3) or not np.isfinite(q).all():
        raise ValueError("centroid must be [3,3] or [P,3,3]")
    if m.shape != (3,) or np.any(m <= 0) or not np.isfinite(m).all():
        raise ValueError("invalid three atomic masses")
    centre = (m[1]*q[1] + m[2]*q[2]) / (m[1]+m[2])
    R = float(np.linalg.norm(q[0]-centre))
    if R <= 1e-12:
        raise ValueError("undefined reactant COM separation")
    r01 = float(np.linalg.norm(q[0]-q[1]))
    r02 = float(np.linalg.norm(q[0]-q[2]))
    r12 = float(np.linalg.norm(q[1]-q[2]))
    sigma, speed = [], []
    for partner in (1, 2):
        s, grad = bond_difference_gradient(q, bonds=((0, partner), (1, 2)))
        sigma.append(float(s))
        speed.append(float(np.sqrt(np.sum(grad**2/m[:,None])/(2*np.pi*beta))))
    return dict(R_bohr=R, sigma_bohr=np.asarray(sigma),
                target_speed=np.asarray(speed),
                reference_speed=float(np.sqrt(
                    (1/m[0] + 1/(m[1]+m[2]))/(2*np.pi*beta))),
                target_topology=np.asarray([
                    r02 > max(r01,r12), r01 > max(r02,r12)
                ], bool),
                r01_bohr=r01, r02_bohr=r02, r12_bohr=r12)


def _ess(weight):
    denom = np.dot(weight, weight)
    return float(weight.sum()**2/denom) if denom > 0 else 0.


def entrance_flux_ratio(reference_coordinate, target_coordinates, weights,
                        reference_speed, target_speed, *,
                        reference_radius, reference_width, target_width,
                        canonical_identity, measure, connected, target_topology,
                        masses_au, beta, min_effective=20,
                        electronic_weight=1., symmetry_factor=1.):
    """Calibrate physical bond fluxes against the free incoming sphere.

    One GLOBAL unbiased canonical weighting of ALL configurations is required,
    e.g. connected joint umbrella MBAR. A separate normalization of each
    PMF/window is invalid; checkpoint endpoints are not unbiased weights.
    For delta_h(s) = 1(|s|<=h)/(2h), k_i = k_in*
    sum(w delta_h(sigma_i) v_i topo_i)/sum(w delta_h(R-Rinf) v_R).

    Rates are strictly candidates until delta-shell/chain/entrance convergence.
    """
    if measure != "joint_unbiased_canonical" or connected is not True:
        raise ValueError("reference/target require connected joint_unbiased_canonical weights")
    if not isinstance(canonical_identity, str) or len(canonical_identity) < 16:
        raise ValueError("canonical_identity must bind the same Hamiltonian/T/P/biases")
    radius = _number(reference_radius, "reference_radius")
    dr = _number(reference_width, "reference_width")
    ds = _number(target_width, "target_width")
    if dr >= radius:
        raise ValueError("reference shell width must be smaller than its radius")
    ref = free_h3_entrance_flux(radius, beta, masses_au,
                               electronic_weight=electronic_weight,
                               symmetry_factor=symmetry_factor)
    R = np.asarray(reference_coordinate, float)
    S = np.asarray(target_coordinates, float)
    if R.ndim != 1 or len(R) < 2 or S.shape not in [(len(R),1),(len(R),2)]:
        raise ValueError("require R[N] and sigma[N,1 or 2]")
    n, channels = S.shape
    w = np.asarray(weights, float)
    vr = np.asarray(reference_speed, float)
    vs = np.asarray(target_speed, float)
    topo = np.asarray(target_topology)
    if (w.shape != (n,) or vr.shape != (n,) or vs.shape != (n,channels)
            or topo.shape != (n,channels) or topo.dtype.kind != 'b'):
        raise ValueError("flux weights, speeds and topology must match sample/channel shapes")
    if (not np.isfinite(R).all() or not np.isfinite(S).all()
            or not np.isfinite(w).all() or not np.isfinite(vr).all()
            or not np.isfinite(vs).all() or np.any(w < 0)
            or np.any(vr <= 0) or np.any(vs <= 0) or w.max() <= 0):
        raise ValueError("finite coordinates, nonnegative weights and positive velocities required")
    if np.any(topo.sum(axis=1)>1):
        raise ValueError("overlapping chemical channel assignment")
    w = w/w.max()  # invariant even for globally tiny MBAR-gauge weights
    ref_hit = np.abs(R-radius) <= dr
    tgt_hit = (np.abs(S) <= ds) & topo
    ref_flux = w*ref_hit*vr/(2*dr)
    tgt_flux = w[:,None]*tgt_hit*vs/(2*ds)
    iref = float(ref_flux.sum())
    itgt = tgt_flux.sum(axis=0)
    essr = _ess(ref_flux)
    esst = np.asarray([_ess(tgt_flux[:,c]) for c in range(channels)])
    threshold = _number(min_effective, "min_effective")
    if threshold < 2:
        raise ValueError("min_effective must be at least two")
    if iref <= 0 or np.any(itgt <= 0):
        raise ValueError("missing entrance or target support; bridge the physical entrance, not grey PMF bins")
    if essr < threshold or np.any(esst < threshold):
        raise ValueError("insufficient effective entrance or target shell samples")
    ratio = itgt/iref
    rates = ref['reference_rate_au']*ratio
    return {
        "schema":"pyrpmd.entrance_flux_bridge.v1",
        "status":"unconverged_candidate", "converged":False,
        "measure":measure,"overlap_connected":True,
        "canonical_identity":canonical_identity,"reference":ref,
        "reference_shell_half_width_bohr":dr,"target_shell_half_width_bohr":ds,
        "n_total_samples":n,"n_ref_support":int(ref_hit.sum()),
        "n_target_support":tgt_hit.sum(axis=0).astype(int).tolist(),
        "effective_reference_samples":essr,"effective_target_samples":esst.tolist(),
        "flux_ratio_each":ratio.tolist(),"rate_each_au":rates.tolist(),
        "rate_each_cm3_s":(rates*AU_RATE_TO_CM3_S).tolist(),
        "total_qtst_au":float(rates.sum()),
        "total_qtst_cm3_s":float(rates.sum()*AU_RATE_TO_CM3_S),
        "physical_scope":"candidate with common canonical flux weights; needs shell-width, chain, topology and physical convergence",
    }


def h3_entrance_from_centroid_samples(centroids, weights, masses_au, beta, *,
                                     radius_bohr, reference_width_bohr,
                                     target_width_bohr, canonical_identity,
                                     measure="joint_unbiased_canonical",
                                     connected=False, min_effective=20,
                                     electronic_weight=1., symmetry_factor=1.):
    """Compute both fluxes from every joint-distribution centroid configuration."""
    q=np.asarray(centroids,float)
    if q.ndim == 4:
        if q.shape[1] < 1:
            raise ValueError("empty bead axis")
        q = q.mean(axis=1)
    if q.ndim != 3 or q.shape[1:] != (3,3) or q.shape[0] < 2:
        raise ValueError("centroids must be [N,3,3] or [N,P,3,3]")
    obs=[bond_entrance_observables(c,masses_au,beta) for c in q]
    return entrance_flux_ratio(
        [o['R_bohr'] for o in obs],
        np.array([o['sigma_bohr'] for o in obs]), weights,
        [o['reference_speed'] for o in obs],
        np.array([o['target_speed'] for o in obs]),
        reference_radius=radius_bohr,reference_width=reference_width_bohr,
        target_width=target_width_bohr,canonical_identity=canonical_identity,
        measure=measure,connected=connected,
        target_topology=np.array([o['target_topology'] for o in obs]),
        masses_au=masses_au,beta=beta,min_effective=min_effective,
        electronic_weight=electronic_weight,symmetry_factor=symmetry_factor)


def bond_pmf_rate_from_entrance_bridge(free_energy, edges, *, beta, curve,
                                       positive_speed, anchor_curve,
                                       anchor_positive_speed, bridge,
                                       canonical_identity, mask=None):
    """Transfer a calibrated r01=r12 anchor flux to the SAME finite 2D PMF.

    The bridge must calibrate r01=r12 (first labelled exit). Both anchor
    and target curves must be completely supported; no arbitrary basin volume.
    A second exit r02=r12 needs its own appropriately remapped FES.
    """
    if (bridge.get("schema") != "pyrpmd.entrance_flux_bridge.v1"
            or bridge.get("canonical_identity") != canonical_identity
            or bridge.get("measure") != "joint_unbiased_canonical"):
        raise ValueError("PMF and entrance bridge need the same canonical physical identity")
    anchor=np.asarray(anchor_curve,float)
    if (anchor.ndim != 2 or anchor.shape[1] != 2 or len(anchor) < 2
            or not np.isfinite(anchor).all() or
            np.max(np.abs(anchor[:,0]-anchor[:,1])) > 1e-10):
        raise ValueError("anchor must lie on r01=r12")
    rate=float(bridge["rate_each_au"][0])
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("calibrated first exit rate must be finite positive")
    result=surface_rate_2d(
        free_energy, edges,beta=beta,curve=curve,normal_speed=positive_speed,
        reference_curve=anchor,reference_normal_speed=anchor_positive_speed,
        reference_rate_au=rate,mask=mask,measure="cv_density")
    result["surface_identity"]=canonical_identity
    result["entrance_reference"]=bridge["reference"]
    result["converged"]=False
    result["physical_scope"]="anchored 2D QTST candidate; bridge, third-bond and recrossing convergence required"
    return result
