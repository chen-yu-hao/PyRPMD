"""PyRPMD: ring-polymer dynamics with one Python and .inp task model."""

__version__ = "0.2.0"


# Keep optional property/CV backends lazy: importing pyrpmd never imports Torch
# or initializes an external electronic-structure or PLUMED process.
_TIER2_EXPORTS = {
    name: module
    for module, names in {
        "cmd": ("PACMDDynamics", "CentroidForceMatch", "QuadraticCentroidPotential",
                "centroid_force_training_data", "fit_quadratic_centroid_pmf", "ibi_update"),
        "properties": ("PropertyResult", "PropertyModel", "CallablePropertyModel",
                       "TorchScriptPropertyModel", "validate_property_result", "evaluate_property_trajectory"),
        "analysis.spectroscopy": ("relative_raman_spectrum", "relative_sfg_spectrum"),
        "pimd_methods": ("SuzukiChinPotential", "BosonicExchangePotential", "ExchangeResult",
                         "bosonic_exchange", "GaussianTrialWavefunction", "PIGSPotential"),
        "advanced_cv": ("Dihedral", "CoordinationNumber", "RMSD", "PathCV", "TorchScriptCV"),
        "opes": ("OPESExploreBias",),
        "shared_metadynamics": ("SharedMetadynamicsBias",),
        "replica_exchange": ("ReplicaExchange", "ReplicaExchangeMove", "reduced_configurational_energy"),
        "biased_kinetics": ("rescale_biased_time", "estimate_biased_rate"),
        "retis": ("ReplicaExchangeTIS", "RETISResult"),
        "adiabatic": ("AdiabaticResult", "AdiabaticPythonPotential", "ElectronicFrame",
                      "QDNRPMDState", "QDNRPMDDynamics", "AdiabaticRPSHState", "AdiabaticRPSHDynamics"),
        "mash": ("MASHDynamics",),
        "workflows.nonadiabatic_rate": ("run_nonadiabatic_rate",),
        "potentials.lifecycle": ("CommitteeResult", "CommitteePotential", "ActiveLearningMonitor", "DeltaLearningPotential"),
    }.items()
    for name in names
}


def __getattr__(name):
    if name in ("SMNRPMDState", "SMNRPMDDynamics"):
        from . import spin_mapping
        return getattr(spin_mapping, name)
    if name in ("RPMASHState", "RPMASHDynamics"):
        from . import rp_mash
        return getattr(rp_mash, name)
    if name in ("AdvancedRPSHDynamics", "RPSHBADynamics", "RPSHDMDynamics",
                "ComplexRPSHDynamics", "RPSHBAState", "RPSHDMState"):
        from . import advanced_surface_hopping
        return getattr(advanced_surface_hopping, name)
    if name in ("BatchedNRPMDState", "BatchedNRPMDDynamics"):
        from . import batched_nonadiabatic
        return getattr(batched_nonadiabatic, name)
    if name in ("electronic_mapping_flow", "torch_electronic_mapping_flow",
                "electronic_backend_capabilities"):
        from . import electronic_backends
        return getattr(electronic_backends, name)
    if name in ("NaH2LEPS5G", "ChemPotPyDPEM", "molecular_potential_from_dict"):
        from .potentials import multistate_external
        return getattr(multistate_external, name)
    if name in ("NonadiabaticEnsembleResult", "EnsemblePropagationError",
                "run_nonadiabatic_ensemble", "density_matrix"):
        from . import nonadiabatic_ensemble
        return getattr(nonadiabatic_ensemble, name)
    if name == "sample_harmonic_ring":
        from .nonadiabatic_sampling import sample_harmonic_ring
        return sample_harmonic_ring
    if name in ("TullyModelOne", "TullyModelTwo", "TullyModelThree",
                "LinearVibronicCoupling", "DiscreteSpinBoson", "model_from_dict"):
        from .potentials import nonadiabatic_models
        return getattr(nonadiabatic_models, name)
    if name in _TIER2_EXPORTS:
        from importlib import import_module

        return getattr(import_module("." + _TIER2_EXPORTS[name], __name__), name)
    if name == "JobSpec":
        from .input import JobSpec

        return JobSpec
    if name == "System":
        from .system import System

        return System
    if name in ("TrajectoryFrame", "write_trajectory", "frames_from_states", "export_materials_trajectory"):
        from . import trajectory_export
        return getattr(trajectory_export, name)
    if name in ("BondConstraints", "ConstraintError"):
        from . import bond_constraints
        return getattr(bond_constraints, name)
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
    if name in ("ForceSplitting", "PropagationPlan"):
        from . import engine
        return getattr(engine, name)
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
    if name in ("AdaptivePathOptions", "AdaptiveNRPMDPathResult", "ReweightedThermalEnsemble",
                "run_adaptive_nrpmd_path", "reweight_thermal_umbrellas"):
        from . import adaptive_nrpmd
        return getattr(adaptive_nrpmd, name)
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
    "SMNRPMDState", "SMNRPMDDynamics", "RPMASHState", "RPMASHDynamics",
    "AdvancedRPSHDynamics", "RPSHBADynamics", "RPSHDMDynamics", "ComplexRPSHDynamics",
    "RPSHBAState", "RPSHDMState", "BatchedNRPMDState", "BatchedNRPMDDynamics",
    "electronic_mapping_flow", "torch_electronic_mapping_flow", "electronic_backend_capabilities",
    "NaH2LEPS5G", "ChemPotPyDPEM", "molecular_potential_from_dict",
    "NonadiabaticEnsembleResult", "EnsemblePropagationError", "run_nonadiabatic_ensemble",
    "density_matrix", "sample_harmonic_ring", "TullyModelOne", "TullyModelTwo", "TullyModelThree",
    "LinearVibronicCoupling", "DiscreteSpinBoson", "model_from_dict",
    "TrajectoryFrame", "write_trajectory", "frames_from_states", "export_materials_trajectory",
    "BondConstraints", "ConstraintError",
    "ElectronicFrictionDynamics", "ConstantElectronicFriction", "FrictionResult",
    "RPSHDynamics", "RPSHState", "HopResult",
    "ThermalNRPMDSampler", "ThermalNRPMDDynamics", "ThermalEnsemble", "ThermalRateCorrelation",
    "thermal_weight", "thermal_flux_side", "thermal_population_rate",
    "AdaptivePathOptions", "AdaptiveNRPMDPathResult", "ReweightedThermalEnsemble",
    "run_adaptive_nrpmd_path", "reweight_thermal_umbrellas",
    "MVRPMDState", "MVRPMDPotential", "MVRPMDDynamics", "MVRPMDSampler",
    "KCCalibration", "KCRPMDState", "KCRPMDPotential", "KCRPMDDynamics", "KCRPMDSampler",
    "IsomorphicPotential", "IsomorphicResult", "IsomorphicEhrenfestState", "IsomorphicEhrenfestDynamics",
    "BatchedState", "BatchedDynamics", "BatchedPropagationError", "MTSDynamics", "ForceSplitting", "PropagationPlan",
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

__all__.extend(_TIER2_EXPORTS)
