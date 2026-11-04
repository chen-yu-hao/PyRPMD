"""MEP-based TST, canonical VTST, ZCT and small-curvature tunnelling.

All numerical APIs use atomic units.  The implementation is split into focused
private modules; this module is the stable public import surface.
"""

from .vtst_core import (
    GeneralizedModes, VTSTError, cartesian_coordinates,
    evaluate_energy_gradient_hessian, generalized_normal_modes,
    mass_scaled_coordinates, rigid_body_basis,
)
from .vtst_path import (
    MEPOptions, MEPPath, trace_mep, validate_reaction_path_endpoints,
)
from .vtst_thermo import (
    CVTTemperatureResult, PartitionSpecies, PathModeAnalysis, analyze_path_modes,
    canonical_variational_profile, fragment_species_from_path_endpoint,
    harmonic_log_partition, moment_of_inertia, species_from_hessian,
)
from .vtst_tunnelling import (
    TunnellingCorrection, VTSTResult, compute_vtst,
    thermal_tunnelling_correction, wkb_transmission_probability,
)

__all__ = [
    "CVTTemperatureResult", "GeneralizedModes", "MEPOptions", "MEPPath",
    "PartitionSpecies", "PathModeAnalysis", "TunnellingCorrection",
    "VTSTError", "VTSTResult", "analyze_path_modes",
    "canonical_variational_profile", "cartesian_coordinates", "compute_vtst",
    "evaluate_energy_gradient_hessian", "fragment_species_from_path_endpoint",
    "generalized_normal_modes", "harmonic_log_partition",
    "mass_scaled_coordinates", "moment_of_inertia", "rigid_body_basis",
    "species_from_hessian", "thermal_tunnelling_correction", "trace_mep",
    "validate_reaction_path_endpoints", "wkb_transmission_probability",
]
