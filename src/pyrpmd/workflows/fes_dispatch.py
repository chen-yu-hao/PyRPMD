"""One public FES entry point with method-specific reconstruction hooks."""

from importlib import import_module

from ..input import fes_method


_WORKFLOWS = {
    "umbrella": ("fes", "run_fes", "reanalyze_fes", "resolve_umbrellas"),
    "metadynamics": (
        "metadynamics", "run_metadynamics", "reanalyze_metadynamics", "validate_metadynamics"
    ),
    "opes": ("opes", "run_opes", "reanalyze_opes", "validate_opes"),
}


def _operation(config, index):
    module, *operations = _WORKFLOWS[fes_method(config)]
    return getattr(import_module(f".{module}", __package__), operations[index])


def run_fes(config, system, output, **kwargs):
    """Run the selected sampler without changing its numerical protocol."""
    return _operation(config, 0)(config, system, output, **kwargs)


def reanalyze_fes(config, system, output, **kwargs):
    """Reconstruct the selected method from its saved sampling state."""
    return _operation(config, 1)(config, system, output, **kwargs)


def validate_fes(config, system):
    """Validate the selected method through the same hook as production."""
    return _operation(config, 2)(config, system)
