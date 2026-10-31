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
    if name == "NoseHooverChain":
        from .thermostats import NoseHooverChain

        return NoseHooverChain
    if name in ("BatchedState", "BatchedDynamics", "BatchedPropagationError"):
        from . import batched
        return getattr(batched, name)
    if name == "MTSDynamics":
        from .multitime import MTSDynamics
        return MTSDynamics
    if name in ("RingPolymerContraction", "SplitPotential", "DifferencePotential"):
        from .potentials import contraction
        return getattr(contraction, name)
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
    if name in ("run", "prepare", "resume_run", "extend_run", "analyze_run"):
        from . import runner

        return getattr(runner, name)
    if name == "analyze_diffusion":
        from .analysis.diffusion import analyze_diffusion

        return analyze_diffusion
    if name == "OPESBias":
        from .opes import OPESBias

        return OPESBias
    if name in ("PlumedBias", "PlumedBiasResult"):
        from . import plumed

        return getattr(plumed, name)
    if name in ("TransitionInterfaceSampler", "TISPath", "TISResult", "TISMove",
                "PathGenerationError", "estimate_effective_flux", "combine_tis_rate"):
        from . import transition_interface

        return getattr(transition_interface, name)
    if name in ("plan_rate_study", "run_rate_study", "arrhenius_fit", "kinetic_isotope_effect"):
        from . import studies

        return getattr(studies, name)
    if name in ("optimize_instanton", "instanton_rate_1d", "instanton_seed", "crossover_temperature"):
        from . import instanton

        return getattr(instanton, name)
    if name in ("PathIntegralAction", "PathIntegralSamples", "OpenPathAction", "OpenPathSamples", "sample_path_integral", "sample_open_path"):
        from . import path_integral

        return getattr(path_integral, name)
    if name in ("energy_heat_capacity", "quantum_kinetic_energy", "isotope_substitution_free_energy", "isotope_fractionation", "fractionation_mass_integration", "momentum_distribution_from_open_paths"):
        from . import quantum_statistics

        return getattr(quantum_statistics, name)
    if name in ("normalize_initial_ensemble", "prepare_initial_ensemble"):
        from . import initial_states

        return getattr(initial_states, name)
    if name == "check_state_selected":
        from .workflows.state_selected import check_state_selected

        return check_state_selected
    raise AttributeError(name)


__all__ = [
    "ElectronicFrictionDynamics", "ConstantElectronicFriction", "FrictionResult",
    "RPSHDynamics", "RPSHState", "HopResult",
    "ThermalNRPMDSampler", "ThermalNRPMDDynamics", "ThermalEnsemble", "ThermalRateCorrelation",
    "thermal_weight", "thermal_flux_side", "thermal_population_rate",
    "MVRPMDState", "MVRPMDPotential", "MVRPMDDynamics", "MVRPMDSampler",
    "KCCalibration", "KCRPMDState", "KCRPMDPotential", "KCRPMDDynamics", "KCRPMDSampler",
    "IsomorphicPotential", "IsomorphicResult", "IsomorphicEhrenfestState", "IsomorphicEhrenfestDynamics",
    "BatchedState", "BatchedDynamics", "BatchedPropagationError", "MTSDynamics",
    "RingPolymerContraction", "SplitPotential", "DifferencePotential",
    "NRPMDState", "NRPMDDynamics", "DiabaticResult",
    "JobSpec", "System", "State", "Dynamics", "NoseHooverChain", "run", "prepare",
    "resume_run", "extend_run", "analyze_run", "analyze_diffusion", "__version__",
    "OPESBias", "PlumedBias", "PlumedBiasResult", "TransitionInterfaceSampler",
    "TISPath", "TISResult", "TISMove", "PathGenerationError",
    "estimate_effective_flux", "combine_tis_rate",
    "plan_rate_study", "run_rate_study", "arrhenius_fit", "kinetic_isotope_effect",
    "optimize_instanton", "instanton_rate_1d", "instanton_seed", "crossover_temperature",
    "PathIntegralAction", "PathIntegralSamples", "OpenPathAction", "OpenPathSamples", "sample_path_integral", "sample_open_path",
    "energy_heat_capacity", "quantum_kinetic_energy", "isotope_substitution_free_energy", "isotope_fractionation", "fractionation_mass_integration", "momentum_distribution_from_open_paths",
    "normalize_initial_ensemble", "prepare_initial_ensemble", "check_state_selected",
]
