"""Opt-in quantum-flux extension around the unchanged native rate workflow.

The provider gets the actual RPMDrate coordinate, PMF-selected surface, grouped
recrossing numerators/denominators and constrained parent snapshots. It must
produce a matched conditional quantum-minus-RPMD flux difference. This module
never constructs a quantum PES from a PMF or silently supplies a bath closure.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from ..analysis.shared_rate_coordinate import SharedRateCoordinate
from ..analysis.shared_quantum_flux import MatchedFluxBlocks, apply_shared_flux_correction
from .rate_pmf import FLUX_MEASURE, configure_rate_umbrella
from .prepare import prepared_system
from .rate_options import _recrossing_options
from ..io import fingerprint as _fingerprint


@dataclass(frozen=True)
class SharedRateReference:
    coordinate: SharedRateCoordinate
    full_flux: MatchedFluxBlocks
    parent_snapshots: np.ndarray
    qtst_cm3_s: float
    native_summary: dict
    source_digest: str


def export_native_reference(config, system, directory):
    """Explicitly bind a native run to its matched quantum-provider request.

    Call on output produced with this config/system. The byte digest prevents
    accidental reuse after files change; it does not authenticate arbitrary
    user-supplied provenance. The production run wrapper establishes that link.
    """
    root=Path(directory);summary_path=root/'summary.json';flux_path=root/'recrossing.npz'
    summary=json.loads(summary_path.read_text());pmf=summary.get('pmf',{})
    if 'rate_extension' in summary:
        raise ValueError('Export the committed rate generation explicitly, not an extension pointer')
    prepared_path=root/'preparation.json'
    prepared=json.loads(prepared_path.read_text())
    if not prepared.get('ready'):
        raise ValueError('Native reaction preparation is not ready')
    system=prepared_system(system,prepared)
    config=configure_rate_umbrella(config,system,prepared['reaction'])
    if (summary.get('task')!='rate' or summary.get('dynamics','rpmd')!='rpmd'
            or pmf.get('measure')!=FLUX_MEASURE or pmf.get('status')!='ready'
            or not pmf.get('ready') or pmf.get('k_qtst_cm3_s') is None):
        raise ValueError('A supported gas-phase RPMD rate-flux PMF and completed children are required')
    for key,expected in [('rinf_bohr',config['reaction']['rinf']),
                         ('electronic_weight',config['reaction'].get('electronic_weight',1.)),
                         ('symmetry_factor',config['reaction'].get('symmetry_factor',1.))]:
        if key in pmf and pmf[key]!=expected:raise ValueError('Native PMF/config mismatch: '+key)
    digest_state=hashlib.sha256(summary_path.read_bytes()+flux_path.read_bytes()+prepared_path.read_bytes())
    digest=digest_state.hexdigest()
    shared=SharedRateCoordinate.from_job(config,system,xi_star=pmf['xi_star'],normalization_id='native-rate-flux:'+digest)
    with np.load(flux_path,allow_pickle=False) as z:
        full=MatchedFluxBlocks(shared.identity,z['time_au'],z['parent_ids'],z['parent_numerator'],z['parent_denominator'],'native recrossing.npz '+digest)
    if not np.allclose(full.time_au,np.arange(1,len(full.time_au)+1)*config['dt_au'],rtol=1e-13,atol=0):
        raise ValueError('Native time axis does not match job time step')
    parents=[];cache={};options=_recrossing_options(config)
    for row,(chain,parent) in enumerate(full.parent_ids):
        if int(chain) not in cache:
            with np.load(root/'checkpoints'/f'parent_{chain:04d}.npz',allow_pickle=False) as z:
                if float(z['xi_star'])!=shared.xi_star:raise ValueError('Parent dividing surface differs')
                cache[int(chain)]=np.array(z['snapshots'],copy=True)
            digest_state.update((root/'checkpoints'/f'parent_{chain:04d}.npz').read_bytes())
        if int(parent)>=len(cache[int(chain)]):raise ValueError('Unknown native parent index')
        q=cache[int(chain)][int(parent)]
        if q.shape!=(config['beads'],system.natoms,3):raise ValueError('Parent shape/atom/bead identity differs')
        if abs(shared.geometry(q.mean(0))['surface'])>1e-7:raise ValueError('Parent is off the shared surface')
        # Authenticate against the native child cache, not just caller labels.
        numerator=np.zeros_like(full.time_au);denominator=0.
        for pair in range(options['child_pairs']):
            checkpoint=root/'checkpoints'/f'child_{chain:04d}_{parent:05d}_{pair:04d}.npz'
            expected=_fingerprint(config,system,dict(stage='child',chain=int(chain),parent=int(parent),pair=pair,
                xi=pmf['xi_star'],child_steps=len(full.time_au),q=q))
            with np.load(checkpoint,allow_pickle=False) as z:
                if str(z['fingerprint'])!=expected:
                    raise ValueError('Native child physics/identity changed; config, PES or temperature mismatch')
                numerator+=z['numerator'];denominator+=float(z['denominator'])
            digest_state.update(checkpoint.read_bytes())
        if not np.array_equal(numerator,full.numerator[row]) or denominator!=full.denominator[row]:
            raise ValueError('Aggregated native flux does not match complete child checkpoints')
        parents.append(q)
    digest=digest_state.hexdigest()
    shared=SharedRateCoordinate.from_job(config,system,xi_star=pmf['xi_star'],normalization_id='native-rate-flux:'+digest)
    full=MatchedFluxBlocks(shared.identity,full.time_au,full.parent_ids,full.numerator,full.denominator,'authenticated native children '+digest)
    arrays=np.asarray(parents)
    parents=np.frombuffer(np.ascontiguousarray(arrays).tobytes(),dtype=arrays.dtype).reshape(arrays.shape)
    manifest=shared.manifest();manifest.update(native_source_digest=digest,qtst_cm3_s=pmf['k_qtst_cm3_s'],parent_count=len(parents))
    dest=root/'shared_coordinate.json'
    text=json.dumps(manifest,indent=2,allow_nan=False)
    if dest.exists() and dest.read_text()!=text:raise ValueError('Export identity changed; use a new correction directory')
    dest.write_text(text)
    return SharedRateReference(shared,full,parents,float(pmf['k_qtst_cm3_s']),summary,digest)


def apply_provider(reference, provider, output, *, bootstrap=256, seed=1938,
                   plateau_fraction=.5, plateau_tolerance=.1):
    """Provider returns active_rp, active_quantum and evidence on common parents."""
    if not callable(provider):raise TypeError('Explicit matched quantum provider required')
    out=Path(output)
    if (out/'summary.json').exists():raise ValueError('Refuse to overwrite a quantum correction result')
    reference.coordinate.identity
    result=provider(reference)
    if reference.coordinate.identity != reference.full_flux.identity:
        raise ValueError('Shared coordinate differs from native flux')
    if not isinstance(result,dict) or not {'active_rp','active_quantum','evidence'}<=result.keys():
        raise ValueError('Provider did not supply both matched dynamics and evidence')
    report=apply_shared_flux_correction(reference.full_flux,result['active_rp'],result['active_quantum'],
        k_qtst_cm3_s=reference.qtst_cm3_s,bootstrap=bootstrap,seed=seed,
        plateau_fraction=plateau_fraction,plateau_tolerance=plateau_tolerance,evidence=result['evidence'])
    report['native_status']=reference.native_summary['status']
    report['native_source_digest']=reference.source_digest
    report['native_qtst_uncertainty_included']=False
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Refuse to overwrite a quantum correction result')
    arrays={k:report.pop(k) for k in ('time_au','kappa_curve','baseline_kappa_curve')}
    np.savez_compressed(out/'shared_flux.npz',**arrays,active_rp_numerator=result['active_rp'].numerator,
        active_quantum_numerator=result['active_quantum'].numerator,parent_ids=reference.full_flux.parent_ids,
        reference_denominator=reference.full_flux.denominator)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def run_shared_rate(config,system,output,*,quantum_provider,executor=None,**correction_options):
    """Serial opt-in orchestrator: native PMF + parents/children, then quantum.

    Parallel native task execution can still be supplied via executor. No CLI
    schema/default integrator changes. A missing/unready PMF aborts correction.
    """
    from .rate import run_rate
    if not callable(quantum_provider):raise TypeError('Supply a matched quantum provider, not a transmission number')
    root=Path(output)
    run_rate(config,system,root/'rpmd',executor=executor)
    reference=export_native_reference(config,system,root/'rpmd')
    return apply_provider(reference,quantum_provider,root/'quantum',**correction_options)
