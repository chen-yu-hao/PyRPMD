"""Opt-in Python workflow: native RPMDrate-compatible rate + shared-coordinate core.

Does not change the native `rate` input schema or default sampler/integrator.
The caller supplies a matched quantum-active solver; no default PMF tunneling
factor or fabricated kappa is inserted. Full bath quantum feedback is not
constructed merely by sharing a scalar coordinate.
"""
from __future__ import annotations
import json
import copy
from pathlib import Path
import numpy as np
from ..analysis.rpmdrate_shared import SharedRPMDrateCoordinate, _digest, _json_default
from ..analysis.rpmdrate_quantum import SharedRateContext, MatchedFluxPair, combine_shared_kappa


def bind_rate_context(shared, config, native_summary, *, pes_identity,
                      reactant_normalization, kinetic_mapping):
    if native_summary.get('task')!='rate':raise ValueError('Native gas-phase rate summary required')
    if not np.isclose(float(native_summary['pmf']['xi_star']),shared.xi_star,atol=1e-12,rtol=0):
        raise ValueError('Quantum core surface differs from PMF-selected native surface')
    reaction=config.get('reaction',{})
    return SharedRateContext(shared.coordinate_id,shared.surface_id,float(config['beta']),
        pes_identity,reactant_normalization,_digest(native_summary),kinetic_mapping,
        xi_star=shared.xi_star,electronic_weight=float(reaction.get('electronic_weight',1.)),
        symmetry_factor=float(reaction.get('symmetry_factor',1.)))


def run_shared_quantum_rate(config, system, output, quantum_backend, *,
                            pes_identity, reactant_normalization, kinetic_mapping,
                            plateau_start, bootstrap_samples=400):
    """Run native PMF/paired-child workflow unchanged, then an explicit backend.

    `config` must be a resolved native gas-phase rate configuration. The callback
    receives keyword arguments (context, coordinate, time_au, native_directory,
    native_summary, config, system), and returns MatchedFluxPair. It must never
    substitute a scattering probability for a normalized flux-side numerator.
    Outputs retain the original rate directory, exact scalar definition and
    native summary fingerprint. A bad backend never overwrites the native rate.
    """
    if not callable(quantum_backend):raise TypeError('An explicit matched active quantum backend is required')
    recrossing=config.get('recrossing',config.get('rate',{}).get('recrossing',{}))
    if config.get('task','rate')!='rate' or recrossing.get('dynamics','rpmd')!='rpmd':
        raise ValueError('Only standard gas-phase RPMD dynamics are supported')
    from .rate import run_rate
    from ..rate_artifacts import active_rate_directory
    out=Path(output);out.mkdir(parents=True,exist_ok=True);native=out/'rpmd'
    summary=run_rate(config,system,native)
    if summary.get('kappa') is None:raise ValueError('Native PMF/recrossing stage is not ready; quantum correction not attempted')
    # Native preparation may refine the TS or infer a reaction definition.
    # Rebind to its actual published output, never to the user's old TS guess.
    from .prepare import prepared_system
    prep_path=native/'preparation.json'
    if not prep_path.is_file():raise ValueError('Missing native preparation receipt')
    preparation=json.loads(prep_path.read_text())
    if preparation.get('ready') is not True or not isinstance(preparation.get('reaction'),dict):
        raise ValueError('Native preparation receipt is not ready')
    system=prepared_system(system,preparation)
    config=copy.deepcopy(config);config['reaction']=copy.deepcopy(preparation['reaction'])
    shared=SharedRPMDrateCoordinate.from_rate_job(config,system,summary['pmf']['xi_star'])
    context=bind_rate_context(shared,config,summary,pes_identity=pes_identity,
        reactant_normalization=reactant_normalization,kinetic_mapping=kinetic_mapping)
    root=active_rate_directory(native,summary)
    with np.load(root/'recrossing.npz',allow_pickle=False) as record:
        t=np.array(record['time_au']);base=np.array(record['kappa'])
    spec={'coordinate':shared.definition(),'surface_xi_star':shared.xi_star,'context':context.as_dict()}
    path=out/'shared_coordinate.json';text=json.dumps(spec,indent=2,allow_nan=False)
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Output belongs to a different shared rate definition')
    path.write_text(text)
    pair=quantum_backend(context=context,coordinate=shared,time_au=t.copy(),native_directory=root,
                         native_summary=json.loads(json.dumps(summary,default=_json_default,allow_nan=False)),config=config,system=system)
    if not isinstance(pair,MatchedFluxPair):raise TypeError('Backend must return MatchedFluxPair')
    shared._check_unchanged()
    result=combine_shared_kappa(context,summary,t,base,pair,plateau_start=plateau_start,bootstrap_samples=bootstrap_samples)
    np.savez_compressed(out/'active_flux_pair.npz',time_au=pair.time_au,quantum_numerator=pair.quantum_numerator,
        rpmd_numerator=pair.rpmd_numerator,reference_flux=pair.reference_flux,
        independent_block_ids=np.asarray(pair.independent_block_ids))
    (out/'active_flux_identity.json').write_text(json.dumps({'context':context.as_dict(),
        'active_hamiltonian_id':pair.active_hamiltonian_id,'reference_flux_id':pair.reference_flux_id,
        'equilibrium_normalization_id':pair.equilibrium_normalization_id,'evidence':pair.evidence},indent=2,allow_nan=False))
    target=out/'quantum_correction.json';temp=out/'quantum_correction.pending.json'
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(target)
    return result
