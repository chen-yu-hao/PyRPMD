"""Explicit method and published-potential construction for saved workflows."""
import numpy as np

from .dynamics import State
from .parallel import execution_plan

METHODS = frozenset(("nrpmd", "sm-nrpmd", "rp-mash", "rpsh", "rpsh-ca",
                     "rpsh-ba", "rpsh-dm", "soc-rpsh", "iesh", "rp-iesh", "multistate-mash"))
EXTENDED_METHODS = METHODS - {"nrpmd", "rpsh"}


def potential_from_identity(identity):
    if identity.get("schema") == "pyrpmd.newns_anderson_model":
        from .iesh import NewnsAndersonModel
        return NewnsAndersonModel.from_dict(identity)
    if identity.get("schema") == "pyrpmd.molecular_potential":
        from .potentials.multistate_external import molecular_potential_from_dict
        return molecular_potential_from_dict(identity)
    from .potentials.nonadiabatic_models import model_from_dict
    return model_from_dict(identity)


def make_dynamics(config, system, rng, *, potential=None):
    """Return a driver; its ``potential`` is owned by the workflow caller.

    With no supplied provider this function constructs one and closes it if
    driver construction fails. Saved identities never select arbitrary imports.
    """
    own = potential is None
    potential = potential_from_identity(config["model"]) if own else potential
    try:
        if potential.natoms != system.natoms:
            raise ValueError("matrix provider atom count disagrees with System")
        if hasattr(potential, "symbols") and tuple(potential.symbols) != tuple(system.symbols):
            raise ValueError("matrix provider atom order disagrees with System.symbols")
        if hasattr(potential, "nuclear_masses") and not np.allclose(
                potential.nuclear_masses, system.masses, rtol=1e-12, atol=0.):
            raise ValueError("system masses disagree with model force-constant masses")
        method, options = config["method"], config["dynamics"]
        if method not in METHODS:
            raise ValueError("unsupported nonadiabatic method")
        from .iesh import NewnsAndersonModel
        if isinstance(potential, NewnsAndersonModel) and method not in {"iesh", "rp-iesh"}:
            raise ValueError("newns_anderson requires IESH to include its separate bare nuclear potential")
        common = dict(native=options.get("native", config.get("native", True)),
                      threads=execution_plan(config["parallel"]).threads)
        args = (potential,system.masses,config["beta"],config["beads"],
                config["dt_au"],potential.nstates)
        if method == "multistate-mash":
            from .mash import MultistateMASHDynamics
            if options["integrator"] != "adaptive-dop853":
                raise ValueError("multistate-mash requires integrator='adaptive-dop853'")
            return MultistateMASHDynamics(*args,seed=rng,**common,**{
                k:options[k] for k in ("rtol","atol","max_step","max_events","gap_tolerance") if k in options})
        if method in {"iesh", "rp-iesh"}:
            from .iesh import IESHDynamics
            cls = IESHDynamics
            if method == "rp-iesh":
                from .rp_iesh import RPIESHDynamics
                cls = RPIESHDynamics
            return cls(*args,seed=rng,**common,**{
                k:options[k] for k in ("integrator","nelectrons","electronic_substeps",
                                       "gap_tolerance","max_hop_probability") if k in options})
        if method == "rp-mash":
            from .rp_mash import RPMASHDynamics
            return RPMASHDynamics(*args,seed=rng,**common,**{
                k:options[k] for k in ("rtol","atol","max_step","max_events","gap_tolerance") if k in options})
        common["integrator"] = options["integrator"]
        if method in ("nrpmd", "sm-nrpmd"):
            from .nonadiabatic import NRPMDDynamics
            from .spin_mapping import SMNRPMDDynamics
            cls = NRPMDDynamics if method == "nrpmd" else SMNRPMDDynamics
            return cls(*args,**common,
                       electronic_backend=options.get("electronic_backend","numpy"),
                       electronic_device=options.get("electronic_device","cpu"))
        hopping = {k:options[k] for k in ("electronic_substeps","rescaling","frustrated_hops",
                                        "gap_tolerance","decoherence","decoherence_constant") if k in options}
        if method == "rpsh":
            from .surface_hopping import RPSHDynamics
            return RPSHDynamics(*args,seed=rng,**common,**hopping)
        from .advanced_surface_hopping import AdvancedRPSHDynamics
        variant = "ba" if method == "rpsh-ba" else "dm" if method == "rpsh-dm" else "ca"
        hopping.update({k:options[k] for k in ("dboc","dboc_step","geometric_force") if k in options})
        return AdvancedRPSHDynamics(*args,method=variant,seed=rng,**common,**hopping)
    except BaseException:
        if own:
            potential.close()
        raise


def initialize_state(config, system, dynamics, nuclear_rng, electronic_rng):
    """Prepare a nuclear parent and the explicitly requested method's state."""
    initial = config["initial"]
    if initial["kind"] == "point":
        q = np.broadcast_to(system.geometry,(config["beads"],system.natoms,3))
        nuclear = State(q,np.broadcast_to(initial["momenta"],q.shape))
    else:
        from .nonadiabatic_sampling import sample_harmonic_ring
        nuclear = sample_harmonic_ring(
            system.masses,config["beta"],config["beads"],initial["frequencies"],
            coordinates=initial["coordinates"],centers=initial["centers"],
            centroid_momenta=initial["centroid_momenta"],count=1,seed=nuclear_rng)[0]
    method = config["method"]
    if method in {"iesh", "rp-iesh"}:
        return dynamics.initialize(nuclear,occupied=initial["occupied"])
    if method == "multistate-mash":
        return dynamics.initialize(nuclear,initial["active_surface"],rng=electronic_rng)
    if method == "nrpmd":
        from .nonadiabatic import NRPMDState
        return NRPMDState.from_nuclear(nuclear,dynamics.nstates,initial["initial_state"],electronic_rng)
    if method == "sm-nrpmd":
        return dynamics.initialize(nuclear,initial_state=initial["initial_state"],rng=electronic_rng)
    if method == "rp-mash":
        return dynamics.initialize(nuclear,active_surface=initial["active_surface"],
                                   sampling="population",rng=electronic_rng)
    return dynamics.initialize(nuclear,active_surface=initial["active_surface"])
