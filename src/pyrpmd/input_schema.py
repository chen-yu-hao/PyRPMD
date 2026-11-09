"""Input schema and error contracts shared by both frontends."""

from __future__ import annotations

from difflib import get_close_matches
from functools import partial

from .validation import checked_integer, finite_positive
from .potentials import _POTENTIAL_SPEC_FIELDS


class InputError(ValueError):
    """Actionable input error before any production simulation starts."""


def _unknown_fields(unknown, allowed, section):
    """Describe a misspelled field without silently accepting a different input."""
    names = sorted(str(key) for key in unknown)
    message = f"Unknown {section} fields: {', '.join(names)}"
    suggestions = []
    for name in names:
        matches = get_close_matches(name, sorted(allowed), n=1, cutoff=0.7)
        if matches:
            suggestions.append(f"{name!r} → {matches[0]!r}")
    if suggestions:
        message += ". Did you mean " + "; ".join(suggestions) + "?"
    return InputError(message)


_ROOT = set(
    "schema_version task preset temperature beads seed system potential dynamics sampling reaction preparation collision outcomes cvs umbrella analysis parallel output native description rate recrossing network metadynamics opes abf eabf steered ves adaptive_umbrella pcet diffusion site_rate surface nonadiabatic replica_exchange tempering barostat electronic_friction initial_ensemble fes path_integral active_learning".split()
)
_SECTIONS = {
    "path_integral": set("method alpha fd_step beta_fd_step groups projection_time trial periodic_mode block_size minimum_average_sign sign_sigma winding_tolerance winding_max_terms".split()),
    "active_learning": set("interval_steps force_threshold energy_threshold directory max_frames".split()),
    "electronic_friction": set("method provider gamma tensor factory params".split()),
    "initial_ensemble": set("method beta fragments j0_energy_policy".split()),
    "nonadiabatic": set("method nstates initial_state state_labels initialization asymptotic_coupling_tolerance".split()),
    "system": set(
        "geometry transition_state reactant product reactants separation geometry_unit isotopes masses masses_amu symbols fragments charge multiplicity cell cell_unit pbc fixed_atoms bond_constraints".split()
    ),
    "dynamics": set("timestep integrator native loop_backend block_steps".split()),
    "sampling": set(
        "engine device normal_mode_layout integrator thermostat timestep equilibration production save_interval interval interval_steps sample_interval equilibration_steps production_steps andersen_interval andersen_schedule thermostat_params gamma_centroid replicas parents".split()
    ),
    "potential": _POTENTIAL_SPEC_FIELDS,
    "parallel": set(
        "backend mode trajectory_groups trajectory_workers trajectory_batch_size workers bead_workers pes_ranks threads_per_rank devices trajectories_per_gpu coordinator_ranks scheduling reuse_potentials".split()
    ),
    "collision": set(
        "sampling impact_sampling bmax trajectories max_time max_steps energy orientation".split()
    ),
    "outcomes": set(
        "factory params channels exit_distance persistence persistence_steps check_interval bond_cutoffs".split()
    ),
    "output": set(
        "directory checkpoint_interval checkpoint_steps trajectory_stride trajectory_storage chunk_frames diagnostics store_configurations centroid_path".split()
    ),
    "reaction": set(
        "definition fragments product_fragments forming_bonds breaking_bonds electronic_weight electronic rinf Rinf equivalent_paths symmetry_factor preparation entry_side".split()
    ),
    "analysis": set(
        "method measure flux_axis bins bootstrap_samples bootstrap_replicates bootstrap seed overlap_threshold ranges min_count max_grid_points".split()
    ),
    "umbrella": set(
        "cvs coordinate grid centers centers_au regions windows force_constant force_matrix kappa stiffness stiffness_au start stop count sampling analysis".split()
    ),
    "rate": set(
        "recrossing pmf_points pmf_method pmf_search_range dividing_surface compare_ui pmf_source pmf".split()
    ),
    "recrossing": set(
        "parent_replicas parent_samples equilibration equilibration_steps parent_interval parent_interval_steps child_pairs child_pairs_per_parent child_steps child_time dividing_surface plateau_fraction plateau_tolerance bootstrap constraint_tolerance child_batch_pairs dynamics friction_scale integrator parent children analysis parent_thermostat".split()
    ),
    "surface": set("reactant_basin dividing_surface product_side max_relative_standard_error normalization free_energy".split()),
    "metadynamics": set(
        "cvs mode sigma sigma_au height height_au pace bias_factor walls walls_during_equilibration grid multiple_walkers block_steps shared_walkers freeze_after_steps freeze_equilibration_steps walkers_mode".split()
    ),
    "opes": set(
        "cvs sigma sigma_au barrier barrier_au bias_factor pace freeze_after_steps freeze_equilibration_steps mode adaptive_sigma sigma_min sigma_min_au walkers_mode".split()
    ),
    "abf": set("cvs ranges ranges_au bins full_samples pace wall_stiffness wall_stiffness_au derivative fd_step".split()),
    "eabf": set("cvs ranges ranges_au bins full_samples pace wall_stiffness wall_stiffness_au coupling coupling_au extended_mass extended_mass_au friction friction_au".split()),
    "steered": set("cvs centers centers_au start start_au end end_au stiffness stiffness_au direction thermostat gamma gamma_au initial_equilibrium".split()),
    "ves": set("cvs ranges ranges_au orders target bias_factor learning_rate learning_rate_au pace target_update_interval quadrature_order wall_stiffness wall_stiffness_au freeze_after_steps freeze_equilibration_steps".split()),
    "replica_exchange": set("enabled interval_steps pairs".split()),
    "tempering": set(
        "mode temperatures temperature_min temperature_max replicas spacing lambdas target exchange_interval_steps exchange_interval pairing seed".split()
    ),
    "barostat": set("method pressure time_constant friction mode tau gamma compressibility cell_mask reference_cell reference_cell_unit stress".split()),
    "adaptive_umbrella": set(
        "enabled pilot_steps pilot_equilibration_steps max_rounds max_windows max_pilot_steps target_overlap min_effective_samples min_spacing coverage_bins coverage_ranges min_bin_effective_samples target_coverage".split()
    ),
    "diffusion": set(
        "atom_indices axes reference_atoms fit_range max_lag bootstrap seed block_length".split()
    ),
    "site_rate": set("cv basin dividing_surface direction".split()),
}


# Bind the input-specific exception contract without duplicating validation.
_integer = partial(checked_integer, error_type=InputError)
_positive = partial(
    finite_positive, error_type=InputError, allow_bool=True, allow_strings=False,
    message="{name} must be positive and finite",
)
