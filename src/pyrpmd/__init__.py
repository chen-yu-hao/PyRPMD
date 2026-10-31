"""PyRPMD: ring-polymer dynamics with one Python and .inp task model."""

__version__ = "0.2.0"


def __getattr__(name):
    if name == "JobSpec":
        from .input import JobSpec

        return JobSpec
    if name == "System":
        from .system import System

        return System
    if name in ("State", "Dynamics"):
        from . import dynamics

        return getattr(dynamics, name)
    if name in ("NRPMDState", "NRPMDDynamics", "DiabaticResult"):
        from . import nonadiabatic
        return getattr(nonadiabatic, name)
    if name in ("ElectronicFrictionDynamics", "ConstantElectronicFriction", "FrictionResult"):
        from . import electronic_friction
        return getattr(electronic_friction, name)
    if name in ("RPSHDynamics", "RPSHState", "HopResult"):
        from . import surface_hopping
        return getattr(surface_hopping, name)
    if name in ("ThermalNRPMDSampler", "ThermalNRPMDDynamics", "ThermalEnsemble",
                "ThermalRateCorrelation", "thermal_weight", "thermal_flux_side", "thermal_population_rate"):
        from . import thermal_nrpmd
        return getattr(thermal_nrpmd, name)
    if name in ("MVRPMDState", "MVRPMDPotential", "MVRPMDDynamics", "MVRPMDSampler"):
        from . import mv_rpmd
        return getattr(mv_rpmd, name)
    if name in ("KCCalibration", "KCRPMDState", "KCRPMDPotential", "KCRPMDDynamics", "KCRPMDSampler"):
        from . import kc_rpmd
        return getattr(kc_rpmd, name)
    if name in ("IsomorphicPotential", "IsomorphicResult", "IsomorphicEhrenfestState", "IsomorphicEhrenfestDynamics"):
        from . import isomorphic
        return getattr(isomorphic, name)
    if name in ("run", "prepare", "resume_run", "analyze_run"):
        from . import runner

        return getattr(runner, name)
    if name == "analyze_diffusion":
        from .analysis.diffusion import analyze_diffusion

        return analyze_diffusion
    raise AttributeError(name)


__all__ = [
    "ElectronicFrictionDynamics", "ConstantElectronicFriction", "FrictionResult",
    "RPSHDynamics", "RPSHState", "HopResult",
    "ThermalNRPMDSampler", "ThermalNRPMDDynamics", "ThermalEnsemble", "ThermalRateCorrelation",
    "thermal_weight", "thermal_flux_side", "thermal_population_rate",
    "MVRPMDState", "MVRPMDPotential", "MVRPMDDynamics", "MVRPMDSampler",
    "KCCalibration", "KCRPMDState", "KCRPMDPotential", "KCRPMDDynamics", "KCRPMDSampler",
    "IsomorphicPotential", "IsomorphicResult", "IsomorphicEhrenfestState", "IsomorphicEhrenfestDynamics",
    "NRPMDState", "NRPMDDynamics", "DiabaticResult",
    "JobSpec", "System", "State", "Dynamics", "run", "prepare",
    "resume_run", "analyze_run", "analyze_diffusion", "__version__",
]
