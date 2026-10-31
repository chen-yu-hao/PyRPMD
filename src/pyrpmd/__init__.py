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
    if name in ("run", "prepare", "resume_run", "extend_run", "analyze_run"):
        from . import runner

        return getattr(runner, name)
    if name == "analyze_diffusion":
        from .analysis.diffusion import analyze_diffusion

        return analyze_diffusion
    if name in ("normalize_initial_ensemble", "prepare_initial_ensemble"):
        from . import initial_states

        return getattr(initial_states, name)
    if name == "check_state_selected":
        from .workflows.state_selected import check_state_selected

        return check_state_selected
    raise AttributeError(name)


__all__ = [
    "NRPMDState", "NRPMDDynamics", "DiabaticResult",
    "JobSpec", "System", "State", "Dynamics", "run", "prepare",
    "resume_run", "extend_run", "analyze_run", "analyze_diffusion", "__version__",
    "normalize_initial_ensemble", "prepare_initial_ensemble", "check_state_selected",
]
