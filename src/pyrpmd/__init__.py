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
    if name in ("run", "prepare", "resume_run", "analyze_run"):
        from . import runner

        return getattr(runner, name)
    if name == "analyze_diffusion":
        from .analysis.diffusion import analyze_diffusion

        return analyze_diffusion
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
    raise AttributeError(name)


__all__ = [
    "NRPMDState", "NRPMDDynamics", "DiabaticResult",
    "JobSpec", "System", "State", "Dynamics", "run", "prepare",
    "resume_run", "analyze_run", "analyze_diffusion", "__version__",
    "plan_rate_study", "run_rate_study", "arrhenius_fit", "kinetic_isotope_effect",
    "optimize_instanton", "instanton_rate_1d", "instanton_seed", "crossover_temperature",
    "PathIntegralAction", "PathIntegralSamples", "OpenPathAction", "OpenPathSamples", "sample_path_integral", "sample_open_path",
    "energy_heat_capacity", "quantum_kinetic_energy", "isotope_substitution_free_energy", "isotope_fractionation", "fractionation_mass_integration", "momentum_distribution_from_open_paths",
]
