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
    if name in ("run", "prepare", "resume_run", "analyze_run"):
        from . import runner

        return getattr(runner, name)
    if name == "analyze_diffusion":
        from .analysis.diffusion import analyze_diffusion

        return analyze_diffusion
    raise AttributeError(name)


__all__ = [
    "BatchedState", "BatchedDynamics", "BatchedPropagationError", "MTSDynamics",
    "RingPolymerContraction", "SplitPotential", "DifferencePotential",
    "NRPMDState", "NRPMDDynamics", "DiabaticResult",
    "JobSpec", "System", "State", "Dynamics", "run", "prepare",
    "resume_run", "analyze_run", "analyze_diffusion", "__version__",
]
