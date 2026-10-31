"""Optional offline PES generation; PySCF and Caracal are not core dependencies."""
from .reference import PySCFReference, PySCFSettings, ReferencePoint

__all__ = ["PySCFReference", "PySCFSettings", "ReferencePoint", "generate", "run_generated_job", "build_native"]


def __getattr__(name):
    if name in ("generate", "run_generated_job"):
        from . import workflow
        return getattr(workflow, name)
    if name == "build_native":
        from .build import build_native
        return build_native
    raise AttributeError(name)
