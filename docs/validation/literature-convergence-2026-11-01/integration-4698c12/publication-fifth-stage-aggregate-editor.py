"""Refresh live summaries after completed checks; preserve frozen evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path('/workspace/scratch/a5e4820daedd/pyrpmd-convergence-integration')
BASE = Path('docs/validation/literature-convergence-2026-11-01')
changed = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def edit(name, replacements):
    path = ROOT / name
    before = path.read_bytes()
    text = before.decode()
    for old, new in replacements:
        assert text.count(old) == 1, (name, old[:100], text.count(old))
        text = text.replace(old, new)
    path.write_text(text)
    changed[str(name)] = {'before_sha256': sha(before), 'after_sha256': sha(path.read_bytes())}


def main():
    edit(Path('docs/literature-validation.md'), [
        ('Pinned RPMDrate PES-2008; 15.0143 kcal/mol barrier versus paper 15.01; force/native/umbrella checks and complete 768-draw reference pilot | Converged thermal-rate sampling; the short P128 0.1 fs NVE gate fails and the reference pilot has severe weight concentration and unresolved tails',
         'Pinned RPMDrate PES-2008; 15.0143 kcal/mol barrier versus paper 15.01; force/native/umbrella checks and two complete excluded reference pilots | No converged rate; the new proposal improves effective sampling but its ξ=1.007 maximum-weight/channel screens still fail; domain tails and absolute normalization need separate validation'),
        ('Flexible force field, full PME, independent force checks, eight-seed OpenMM reference campaign running, completed velocity-recording check and prepared P32 reference jobs | Complete reference equilibrium/production gates, then quantum production and long-time comparison with PyRPMD\'s own transport workflow',
         'Complete eight-seed OpenMM classical reference: VACF D=0.19781 Å²/ps; exact raw-archive replay and a complete paired timestep control | Three preparation energy checks and the short-lag MSD screen fail; both timestep diffusion-equivalence intervals remain unresolved; quantum and PyRPMD production transport are unrun'),
        ('''The running classical water cohort retains its original 100 ps preparation
and 25 production segments for every seed. A separate all-eight continuation
coordinator has passed 35 orchestration fixtures, including real process-signal
and lease tests. It can extend every exact pre-production checkpoint by the
same 400 ps after a complete original assessment and a recorded adoption.
It has not yet been adopted or run. Neither this software gate nor a later
preparation extension can replace failed equilibrium checks with a diffusion
agreement claim.''',
         '''The complete original classical cohort retains all eight seeds, each with
100 ps preparation and 25 × 4 ps NVE segments. Its 2 ps VACF estimate is
**0.19780892 Å²/ps**, nominal 95% interval [0.18561141, 0.21000642];
the 1–2 ps MSD estimate is **0.19792159 Å²/ps**, interval
[0.18674048, 0.20910269]. The VACF point is 0.60% below the paper's
classical 0.199(2), but the comparison remains ineligible: three preparation
energy-stationarity checks fail, and the paired early/late MSD-window change
is +0.0171182 Å²/ps, interval [0.0119729, 0.0222635], outside ±0.01.
All eight chains and 200 production segments remain in the primary estimate.

The separate all-eight 4 ps paired timestep control is also complete.
Halving 0.25 fs to 0.125 fs passes both energy checks but does not establish
diffusion equivalence. Fine-minus-coarse VACF is −0.0141245 Å²/ps,
interval [−0.0412684, +0.0130193]; MSD is −0.0248398 Å²/ps,
interval [−0.0491921, −0.0004876]. Neither interval lies inside the
unchanged ±0.01 margin. The noisy VACF difference does not prove a timestep
bias, and the smaller timestep is not thereby declared converged.

Both full datasets pass actual archive-extraction replay. An independent
VACF–MSD integral identity accounts for the observed short-lag curvature;
it explains the failed screen without changing it. The separate all-eight
400 ps preparation and longer-lag proposals remain unadopted; their
prerequisite timestep-equivalence condition is not met. These are OpenMM
reference calculations, with long PyRPMD production transport, quantum
water and exact source-mass matching still outstanding.'''),
        ('''The independent eight-ladder umbrella estimate at 30,000 production sweeps
is 11.583% above the paper and fails its 3% standard-error target. It advances
to the registered 100,000-sweep prefix without dropping chains or changing
parameters. Agreement of the independent surface reference cannot replace
that sampler's own convergence checks. Equilibrium bead refinements also do
not establish full dynamical bead convergence.''',
         '''The independent eight-ladder umbrella calculation now has complete
30,000- and 100,000-production-sweep prefixes. At 100,000, QTST is
**2.58739680 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹**, nominal 95% interval
[2.29413296, 2.91814919] × 10⁻¹⁶, +3.084% from paper QTST.
The point criterion passes, but the primary log standard error is 0.050874,
above the unchanged 0.03 target. The entire cohort and its failed precision
gate reproduce exactly after actual archive extraction. The historical
30,000 estimate (+11.583%) and all chains remain intact. A future registered
300,000 prefix requires separate adoption from a new verified restore;
the original completed campaign remains unchanged after an unexplained
lease postcondition anomaly. Agreement of the independent surface reference
does not establish production MD umbrella precision or dynamic bead convergence.'''),
    ])
    edit(Path('benchmarks/literature/README.md'), [
        ('## Inspect the inventory\n', '''## Latest completed controls

The independent H + H₂ umbrella cohort is complete through 100,000
production sweeps per ladder. Its QTST point is +3.084% from the paper,
but the primary log standard error is 0.050874 against a fixed 0.03 limit.
Actual archive extraction and exact numerical replay preserve that failure.
The complete P256 dynamic comparison remains pending.

The original eight-chain classical water cohort gives VACF diffusion
0.19780892 Å²/ps versus the paper's 0.199(2), but three preparation checks
and the short-lag MSD screen fail. Its subsequent all-eight paired timestep
control passes energy checks while both diffusion-equivalence intervals
remain outside the fixed margin. These complete OpenMM reference datasets
do not establish quantum water or PyRPMD long-time transport agreement.

The second excluded H + CH₄ importance proposal uses 3,672 actual PES
evaluations within a 4,000-call cap. Effective sampling improves substantially,
but ξ=1.007 still fails the pooled maximum-weight and channel-2 screens.
Its saved calculations and actual extracted archive replay successfully;
no physical rate is promoted from either excluded pilot.

All case statuses below retain their full convergence scope. See the
[complete evidence record](../../docs/validation/literature-convergence-2026-11-01/README.md).

## Inspect the inventory
''')
    ])
    path = ROOT / 'benchmarks/literature/campaign.json'
    before = path.read_bytes()
    data = json.loads(before)
    cases = {row['id']: row for row in data['cases']}
    cases['h_ch4']['blockers'] = [
        'Converged methane thermal-rate sampling has not been run; the fixed0.1fs shortP128 NVE energy gate fails while smaller-step diagnostics remain separate.',
        'The improved excluded importance pilot still fails its xi1.007 pooled maximum-weight screen and channel2 ESS/maximum-weight screens. Absolute classical normalization, finite-domain tails and a fresh production cohort require separate validation.',
    ]
    cases['h_ch4']['available_evidence'].append('The second excluded pilot completed800 calibration,3072 main and128 matched control draws with3672 successfulPES evaluations. Pooled ESS is336.17,165.49,704.02; xi1.007 pooled maxweight0.045147 exceeds0.025, and channel2 ESS27.53<32/maxweight0.15743>0.10. Independent calculation and fresh archive-extraction replay agree; no physical rate is promoted.')
    cases['h_ch4']['evidence_path'] = str(BASE / 'review/hch4-reference-efficiency-replay/FINAL_STATUS.md')
    cases['h_h2']['blockers'][2] = 'Independent MC umbrella sampling fails the unchanged precision gate at complete30000 and100000 sweeps. At100000, logSE0.0508737487 exceeds0.03. A300000 continuation requires separate adoption from a new isolated verified restore; the anomalous original campaign remains unchanged. This MC/reference evidence does not validate production MD umbrella sampling.'
    cases['h_h2']['available_evidence'].append('Complete100000-sweep all-eight umbrella QTST2.5873968041771664e-16, nominal95%CI[2.294132956321048e-16,2.918149187395823e-16], +3.083538% from paperQTST. Seven criteria pass but statistical_precision fails. All checkpoint/RNG/prefix bytes and scientific analysis exactly replay after actual archive extraction.')
    cases['qtip4pf_h2o_diffusion']['blockers'] = [
        'Complete original classical cohort fails preparation potential-energy stationarity for seeds1102,1105,1106 and the short-lag MSD-window equivalence screen. These outcomes remain unchanged despite a paper-near diffusion point.',
        'Complete paired0.25/0.125fs timestep control passes energy checks but neither VACF norMSD95% difference interval lies inside+/-0.01A2/ps. The500ps preparation and longer-lag proposals remain unadopted because their timestep prerequisite is unmet.',
        'Long quantum production, exact source isotope masses, finite-size comparison and PyRPMD-own long-time transport validation remain incomplete.',
    ]
    cases['qtip4pf_h2o_diffusion']['available_evidence'] = [
        'All8 original OpenMM reference seeds and200 NVE segments complete:216 flexible molecules,298K,fullPME,100ps preparation+100ps production perchain. VACFD0.1978089161, nominal95%CI[0.1856114149,0.2100064173]A2/ps; MSDD0.1979215878,CI[0.1867404812,0.2091026944]. Literature comparison remainsnull andphysical_converged=false.',
        'All8 paired timestep seeds complete:16branches64ps. Fine-minus-coarse VACFmean-0.0141245323,95%CI[-0.0412683731,0.0130193084]; MSDmean-0.0248398419,95%CI[-0.0491920712,-0.0004876125]A2/ps. Both equivalence screens unresolved at+/-0.01; energy checks pass.',
        'Both original and paired datasets pass actual raw-archive extraction and complete scientific replay. A separately tested VACF-MSD integral identity explains early-time curvature while preserving every failed gate.',
    ]
    cases['qtip4pf_h2o_diffusion']['evidence_path'] = str(BASE / 'water/original-fixed-cohort/README.md')
    assert all(row['status'] == 'not_reproduced' for row in data['cases'])
    path.write_text(json.dumps(data, indent=2) + '\n')
    changed['benchmarks/literature/campaign.json'] = {'before_sha256': sha(before), 'after_sha256': sha(path.read_bytes())}
    edit(BASE / 'h3-quantum/README.md', [
        ('## Reproduction and retained initialization failures\n', '''The [complete 100,000-sweep prefix](umbrella-stage-100000/complete-cohort-analysis/README.md)
now retains all eight ladders and gives QTST=2.587396804e-16, nominal
95% interval [2.294132956e-16, 2.918149187e-16], +3.084% from the paper.
Seven registered criteria pass. Statistical precision alone still fails:
the primary log standard error is 0.05087375, above the fixed 0.03 target.
This is a QTST estimate; no recrossing factor or full rate is inferred.

The [complete archive and extracted-data replay](umbrella-stage-100000/raw-archive-v2/README.md)
preserve all checkpoint, RNG and sample-prefix states and reproduce every
scientific field and saved WHAM array exactly. The original campaign's
unexplained lease reappearance is explicitly retained in a separate reviewed
read-only admission. No further cleanup or original-directory writer was
admitted. Any future registered 300,000 prefix must use a new isolated
verified restore and a separate root adoption; it cannot be selected by
closeness to the paper.

## Reproduction and retained initialization failures
''')
    ])
    edit(BASE / 'README.md', [
        ('| Separate P=128 umbrella MC, 30,000 production sweeps per ladder | 2.80072503 × 10⁻¹⁶, 95% CI [2.33507586, 3.35923162] × 10⁻¹⁶ | All eight ladders complete; precision and paper-point criteria fail; registered 100,000-sweep continuation selected |',
         '''| Separate P=128 umbrella MC, 30,000 production sweeps per ladder | 2.80072503 × 10⁻¹⁶, 95% CI [2.33507586, 3.35923162] × 10⁻¹⁶ | Retained earlier complete cohort; precision and paper-point criteria fail |
| Separate P=128 umbrella MC, 100,000 production sweeps per ladder | 2.58739680 × 10⁻¹⁶, 95% CI [2.29413296, 2.91814919] × 10⁻¹⁶ | +3.084% from paper QTST; primary log SE 0.050874 still exceeds 0.03 |
| Original eight-chain classical water VACF diffusion | 0.19780892 Å²/ps, nominal 95% CI [0.18561141, 0.21000642] | Paper classical 0.199(2); preparation and short-lag eligibility checks fail, so no reproduction claim |
| Paired water timestep VACF difference, 0.125 fs minus 0.25 fs | −0.01412453 Å²/ps, nominal 95% CI [−0.04126837, +0.01301931] | All eight pairs complete; interval exceeds ±0.01 equivalence margin; energy checks pass |'''),
        ('''- [Water reference-engine, recording and checkpoint evidence](water/README.md).
  The fixed classical transport stage is still running; the complete quantum
  job specification is prepared but has not been launched.''',
         '''- [Complete original water cohort](water/original-fixed-cohort/README.md),
  [actual extracted-archive replay](water/original-cohort-archive-replay/README.md),
  and [paired timestep control](water/timestep-paired-control/completed-original-cohort/README.md).
  Both datasets are complete and exactly replayable; preparation, short-lag
  and paired diffusion checks retain their failed or unresolved outcomes.
  The [VACF–MSD identity diagnostic](water/vacf-msd-identity/README.md) explains
  short-time curvature without changing the fixed gate. Quantum water is unrun.
- [Complete 100,000-sweep quantum umbrella cohort](h3-quantum/umbrella-stage-100000/complete-cohort-analysis/README.md)
  and [exact archive replay](h3-quantum/umbrella-stage-100000/raw-archive-v2/README.md).
  The point criterion passes but statistical precision remains insufficient;
  every earlier prefix, independent ladder and failed result is retained.'''),
        ('''- [Methane reference pilot](review/hch4-reference-pilot/README.md).
  All 768 planned draws are retained, but their concentrated weights and
  unresolved domain tails do not establish a useful rate estimate. The
  [next efficiency design](review/hch4-reference-efficiency-design/README.md)
  has a separate measure/support review and a fixed excluded-sample budget;
  that design is not a new executed rate result.''',
         '''- [Initial methane reference pilot](review/hch4-reference-pilot/README.md)
  and [complete improved-proposal pilot](review/hch4-reference-efficiency-replay/FINAL_STATUS.md).
  The new fixed excluded pilot makes 3,672 PES calls within its 4,000 cap;
  effective sampling improves, but ξ=1.007 still fails the pooled maximum-weight
  and channel-2 screens. Saved calculations and actual archive extraction
  replay agree. Neither pilot supplies a promoted physical rate.'''),
        ('''still fails precision after all eight 30,000-sweep ladders; its continuation
retains every chain and the original thresholds.''',
         '''still fails precision after all eight 100,000-sweep ladders: its QTST
point now differs from the paper by +3.084%, but log SE 0.050874 exceeds
0.03. The exact archived replay retains every chain and earlier prefix.
The predeclared 300,000 endpoint is not an automatic extension; a new
isolated-restore adoption is required.'''),
        ('''independent seeds are scheduled
for 100 ps preparation plus 25 × 4 ps NVE segments each. No converged new
diffusion coefficient is claimed in this snapshot. CPU checkpoint restoration
and momentum conventions are audited separately from physical equilibration.''',
         '''independent seeds have completed
100 ps preparation plus 25 × 4 ps NVE segments each. The full-cohort VACF
estimate is 0.19780892 Å²/ps, interval [0.18561141, 0.21000642]; MSD
gives 0.19792159 Å²/ps, interval [0.18674048, 0.20910269]. These are
nominal t intervals over eight independent chains, not 200 independent
segments. No physically converged diffusion coefficient is claimed.
CPU checkpoint and momentum conventions remain separate from equilibration.'''),
        ('''potential-energy stationarity threshold; this failure remains part of the
fixed cohort. Completing the OpenMM reference would still require a separate
long-time comparison of PyRPMD's own production transport workflow.''',
         '''potential-energy stationarity threshold. The complete assessment also
fails this gate for seeds 1105 and 1106. The short-lag MSD contrast is
+0.0171182 Å²/ps, interval [0.0119729, 0.0222635], outside ±0.01.
All 200 production segments remain included. The paper-near point does not
override those failures, and a separate long-time PyRPMD comparison is unrun.

The complete paired timestep study halves 0.25 fs to 0.125 fs from each
seed's exact same saved state and momentum draw. All 16 branches (64 ps)
complete and pass their energy checks. Fine-minus-coarse VACF is
−0.0141245 Å²/ps, interval [−0.0412684, +0.0130193]; MSD is
−0.0248398 Å²/ps, interval [−0.0491921, −0.0004876]. Neither interval
fits inside ±0.01. This unresolved equivalence does not establish a converged
smaller timestep. The original and paired raw archives both pass actual
extraction and unchanged-analyzer replay, including their failures.'''),
        ('''launched for the cohort. The original potential-energy, temperature and
intramolecular-geometry thresholds remain fixed.''',
         '''launched for the cohort. The [completed timestep outcome](water/post-cohort-next-stage-timestep-outcome/README.md)
does not meet the prerequisite in the separately frozen preparation/long-lag
proposals. The original potential-energy, temperature and
intramolecular-geometry thresholds remain fixed.'''),
        ('[complete pilot and independent arithmetic replay](review/hch4-reference-pilot/README.md).',
         '''[complete pilot and independent arithmetic replay](review/hch4-reference-pilot/README.md).

The next excluded proposal pilot is now complete: 800 calibration draws,
3,072 fresh main draws and 128 matched separation-control draws, with 3,672
successful PES evaluations in total. Its pooled ESS values are 336.17,
165.49 and 704.02 out of 1,024 draws. The ξ=1.007 pooled ESS passes its
minimum, but maximum normalized weight 0.045147 exceeds 0.025; channel 2
also fails ESS 27.53 < 32 and maximum weight 0.15743 > 0.10. The aggregate
efficiency screen therefore remains false. Independent arithmetic and fresh
archive-extraction replay agree, with a separate reviewer checking the method
and archive. These are finite-domain dimensionless diagnostics. Absolute
rate normalization, unseen-tail bounds and fresh production are still
required; no calibration sample becomes production evidence.'''),
    ])
    edit(BASE / 'integration-4698c12/README.md', [
        ('## Documentation\n', '''## Further completed software gates

The [next additive gate](fifth-stage-compatibility/README.md) passes 135 new
cases per interpreter: 43 complete-P256-analysis admissions, 85 methane
efficiency cases and 7 analytic VACF–MSD identities. The subsequent
[archive-method gate](archive-method-compatibility/README.md) passes 65 new
cases, comprising the 28 original archive contracts and 37 explicit
read-only anomaly-admission cases. All three selected Python versions have
zero failures, errors or skips. The exact case-level union is **796 distinct
passing identities per interpreter**; the earlier cases were not rerun by
these additive gates. Imports, existing ABI products, recorded sources and
dependencies are checked in each actual test process and a fresh child.

Separately, the [corrected full-project Python 3.14 run](full-suite-36ec289-py314-v2/README.md)
passes **7,285 primary cases**, with **145 skips** and **486 passing subtests**,
zero failures or errors. It tests published head `36ec289` plus one validated
synthetic wall-fixture correction, before the new additive source files were
copied. Subtests are not counted as distinct primary test identities, and
the full-suite count is not added to the overlapping method-case union.
The [initial full-suite attempt](full-suite-36ec289-py314/README.md) remains
preserved with all 17 failures and their environmental/fixture diagnoses.

The test-only [wall-fixture correction](hch4-wall-fixture-supersession/README.md)
isolates its CPU clock from the outer pytest process lifetime while preserving
the real wall timeout, child draining and retry-refusal checks. All three
interpreters execute it. Production resource accounting and original active
physics sources are unchanged. Complete remote CI logs at head `36ec289`
show that same unpatched fixture as the sole failed case in five jobs;
those historical remote failures are retained and are not relabelled as passes.

## Documentation
''')
    ])
    record = {
        'schema': 'pyrpmd.fifth-stage-post-gate-aggregate-update.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'changed_live_aggregate_files': changed,
        'scope': 'Live documentation and descriptive catalog only, after completed full-project and additive method gates. Every full case status remains not_reproduced. Frozen sources, thresholds, execution receipts and historical inventories unchanged.',
        'new_method_union_per_interpreter': 796,
        'source_or_test_code_changed_by_this_update': False,
        'unrelated_full_suite_rerun_for_prose': False,
    }
    out = ROOT / BASE / 'integration-4698c12/publication-fifth-stage-aggregate-update.json'
    with out.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'updated_files': list(changed), 'receipt_sha256': sha(out.read_bytes())}))


if __name__ == '__main__':
    main()
