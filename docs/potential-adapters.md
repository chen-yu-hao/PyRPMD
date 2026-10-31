# Optional potential adapters

Named providers load their model or calculator once per provider context and
reuse it across evaluations. Optional packages are imported only when the
provider is requested. All PyRPMD-facing coordinates and cell vectors are in
bohr, energies and configurational virials in hartree, and forces in
hartree/bohr. Cell vectors are rows; trajectories remain unwrapped.

## Provider overview

| `potential.provider` | Optional installation | Boundary conditions | Evaluation route |
|---|---|---|---|
| `mace` | `pip install 'pyrpmd[mace]'` | Full, partial, or no PBC, subject to model | Reused MACE ASE calculator; frames evaluated in order |
| `nequip` | `pip install 'pyrpmd[nequip]'` | Full, partial, or no PBC, subject to model | Reused compiled-model NequIP ASE calculator |
| `deepmd` | `pip install 'pyrpmd[deepmd]'` | Full PBC or isolated; partial PBC rejected | One native `DeepPot.eval` call per batch |
| `fairchem` | `pip install 'pyrpmd[fairchem]'` | Subject to the selected model and task | Reused FAIRChem ASE calculator |
| `tblite` / `xtb` | `pip install 'pyrpmd[tblite]'` | Full, partial, or no PBC | Reused tblite xTB calculator; sequential electronic calculations |
| `pyscf` | `pip install 'pyrpmd[pyscf]'` | Isolated molecules | Reused HF/DFT object with fresh geometry and analytic gradients |
| `torch` | `pip install 'pyrpmd[gpu]'` | Explicit periodic model contract below | Native tensor batch |

ASE-backed models accept a batch at the PyRPMD interface but call the calculator
once per frame. They do not advertise native GPU batching. This choice reuses
the established neighbor-list, species, units and stress interfaces without
silently assuming a particular graph-model format. Models retain their own
training domain, reference electronic state and numerical precision. Converting
outputs to NumPy float64 does not increase a model's inference precision.

The fixed System supplies atom identities, cell, PBC, charge and multiplicity.
`D` and `T` map to the chemical element `H` in potential engines; isotope masses
remain in PyRPMD dynamics. Do not repeat System fields inside `potential.params`.

### ASE energy and force consistency

ASE-backed adapters request `atoms.get_potential_energy(force_consistent=True)` so that a
calculator with electronic smearing returns the electronic free energy differentiated by
its forces and stress. This is the calculator's electronic thermodynamic potential, not
a nuclear free energy or a PyRPMD PMF. Electronic smearing temperature and nuclear thermostat
temperature remain separate model inputs. See the
[ASE energy contract](https://docs.ase-lib.org/ase/atoms.html#ase.Atoms.get_potential_energy).

If a calculator does not advertise `free_energy` and raises ASE's
`PropertyNotImplementedError`, the adapter falls back to its ordinary energy. A calculator
that advertises `free_energy` but fails to provide it raises an error; other calculator
failures are not swallowed. Classical calculators whose two energies coincide retain their
energy convention. The ASE energy convention is part of the run identity: older ASE-backed
checkpoints cannot be resumed or recomputed with the new adapter, including classical runs
where the two energies happen to coincide. Their existing summary can still be read. Start
a new output directory; new-convention runs resume normally. In particular, do not splice
old zero-temperature extrapolated energies or free-energy weights into a trajectory using
the corrected convention.

The identity guard recognizes the explicit `ase`, `mace`, `nequip` and `fairchem` providers;
it cannot discover an ASE calculator hidden inside arbitrary Python factory code. Keep the
source and energy convention of such plugins in the model provenance as well.

## MACE and NequIP

Python input dictionaries:

```python
mace = {
    "provider": "mace",
    "model": "/models/material.model",
    "device": "cuda:0",
    "params": {"default_dtype": "float64"},
}

nequip = {
    "provider": "nequip",
    "model": "/models/material.nequip.pt2",
    "device": "cuda:0",
    "params": {
        "chemical_species_to_atom_type_map": {"H": "H", "O": "O"},
    },
}
```

MACE uses `MACECalculator(model_paths=..., device=...)`. Its adapter defaults to
`default_dtype="float64"`; other calculator options can be passed in `params`.
NequIP uses `NequIPCalculator.from_compiled_model(compile_path=..., device=...)`.
Compile the NequIP checkpoint for the target integration and hardware; AOT
artifacts require the ASE compile target. The default species mapping is an
explicit identity mapping. Supply a mapping when trained type names differ
from chemical symbols. NequIP's `energy_units_to_eV` and `length_units_to_A`
options handle models trained in other units before the ASE-to-AU conversion.

These routes use the maintained upstream interfaces described in the
[MACE calculator source](https://github.com/ACEsuit/mace/blob/develop/mace/calculators/mace.py)
and [NequIP ASE guide](https://nequip.readthedocs.io/en/latest/integrations/ase.html).

## DeePMD

```python
deepmd = {
    "provider": "deepmd",
    "model": "/models/graph.pb",
    "params": {
        "model_kwargs": {"auto_batch_size": True},
        "eval_kwargs": {},
    },
}
```

The adapter obtains species indices from `DeepPot.get_type_map()`. An explicit
`params.type_map` can override that mapping for checkpoints whose type labels
are not chemical symbols. The map must contain each system element exactly
once. Full cells are passed as `[B,9]`, with row vectors flattened in C order;
isolated calculations pass `cells=None`. Partial PBC is rejected because this
API has no independent boundary flags.

Inputs default to the usual DeePMD dataset units of angstrom and eV. A model
trained in different consistent units requires an explicit `potential.units`
mapping. `eval_kwargs` can contain trained frame or atom parameters such as
`fparam` or `aparam`; they cannot replace geometry, type mapping or the expected
energy/force/virial output layout. The System's charge and multiplicity do not
automatically become a model input: electronic-state-dependent models must
receive the appropriate trained parameters.

`DeepPot` chooses its execution device through its backend. The adapter reports
`execution_device="backend"`; it does not claim that a generic `device` argument
selects a particular GPU. Configure the DeePMD runtime before starting the
process and use PyRPMD serial or bead mode. Explicit PyRPMD GPU scheduling for
this adapter is rejected. Native model batching remains active in serial mode.

See the [official inference API](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/autoapi/deepmd/infer/deep_pot/index.html).

## fairchem

```python
fairchem = {
    "provider": "fairchem",
    "model": "/models/uma-checkpoint.pt",
    "device": "cuda:0",
    "params": {
        "task_name": "omat",
        "inference_settings": "high-precision",
    },
}
```

The adapter uses `FAIRChemCalculator.from_model_checkpoint`. A local checkpoint
or an upstream registered model name is accepted. Registered models may require
an authorized upstream download; PyRPMD supplies no weights or model access.
`task_name` is required so the electronic-structure reference is an explicit
choice. The calculator receives integer total `charge` and `spin` (multiplicity)
through `Atoms.info` for `task_name="omol"`. Other UMA tasks use the upstream
`charge=0, spin=0` model-input sentinels. Keep System's neutral/default values
(`charge=0, multiplicity=1`) for these tasks: they cannot select an electronic
charge or magnetic state, and other System values are rejected before model
loading. The zero model spin is distinct from a molecular singlet constraint.

Select inference settings supported by the installed fairchem version.
Current fairchem's `high-precision` mode is full FP32, not FP64; defaults and
available acceleration modes can change between releases. The adapter preserves
upstream choices and does not enable untrained stress predictions. A task without
stress produces no virial through the ASE route. Check a model's energy drift
and gradient accuracy before using its trajectories as production evidence.

References: [FAIRChem calculator API](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/calculate/ase_calculator.py),
[inference guide](https://fair-chem.github.io/core/common_tasks/ase_calculator.html),
the [UMA task and electronic-state conventions](https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma.md),
and [UMA precision changelog](https://facebookresearch.github.io/fairchem/uma-changelog/).

## On-the-fly xTB through tblite

```python
xtb = {
    "provider": "tblite",
    "params": {
        "method": "GFN2-xTB",
        "accuracy": 0.1,
        "max_iterations": 250,
        "electronic_temperature": 300.0,  # kelvin, distinct from nuclear T
        "warm_start": False,
    },
}
```

Available methods are `GFN2-xTB`, `GFN1-xTB` and `IPEA1-xTB`. System multiplicity
maps to `uhf=multiplicity-1`. Settings use tblite's native `Calculator.set`
interface; `params.settings` allows additional supported native options. The
electronic temperature is converted from kelvin to hartree. Convergence or
engine failures raise `PotentialError` and discard cached electronic state.

The calculator object survives geometry and cell updates. `warm_start=False`
starts each electronic solve without a previous result; this is the default.
With `warm_start=True`, result objects are cached by batch index. Preserve bead
ordering when using this option and assess electronic-state convergence.

tblite's result named `virial` is the tensile energy-strain derivative. PyRPMD
negates it, matching the upstream ASE relation `stress=tblite_virial/volume` and
PyRPMD's `W=-volume*stress` convention. No coordinate wrapping, kinetic pressure,
spring pressure or bead normalization is added.

References: [tblite Python API](https://tblite.readthedocs.io/en/latest/api/python.html)
and [upstream ASE conversion](https://tblite.readthedocs.io/en/latest/_modules/tblite/ase.html).

## On-the-fly molecular PySCF

```python
pyscf = {
    "provider": "pyscf",
    "params": {
        "method": "DFT",
        "xc": "pbe0",
        "basis": "def2-svp",
        "conv_tol": 1e-10,
        "conv_tol_grad": 1e-6,
        "grid_level": 4,
        "grid_response": True,
        "density_fit": True,
        "warm_start": False,
    },
}
```

`HF` and `DFT` choose RHF/RKS for singlets and UHF/UKS otherwise. Explicit
`RHF`, `UHF`, `RKS` and `UKS` are also accepted. DFT requires an explicit `xc`;
restricted methods require a singlet. Basis, optional ECP, density fitting and
auxiliary basis are delegated to PySCF. The default basis is `def2-svp`; the
appropriate scientific basis/grid/convergence study remains system dependent.

One SCF object is reused, while geometry-dependent integrals and DFT grids are
reset on each geometry change. Analytic nuclear gradients are negated to obtain
forces. DFT grid response is included by default. The adapter checks
`engine.converged` before requesting gradients; unconverged energies cannot be
silently accepted. CPU thread limits belong to the execution configuration,
not to global settings modified inside this adapter.

With `warm_start=True`, converged density matrices are cached by batch index;
the default disables this history dependence. SCF failure clears the cache.
The reported isolated-molecule virial is `sum_i F_i outer r_i`. Periodic PySCF,
post-HF gradients, excited-state dynamics and GPU4PySCF are outside this adapter.

References: [PySCF gradient example](https://github.com/pyscf/pyscf/blob/master/examples/grad/16-scan_force.py)
and [gradient API](https://pyscf.org/pyscf_api_docs/pyscf.grad.html).

## Periodic Torch model contract

The existing nonperiodic `forward(q)` interface remains unchanged. A periodic
model requires `potential.periodic=true` and a nonsingular System cell:

```python
potential = {
    "provider": "torch",
    "factory": "my_models:make_model",
    "device": "cpu",
    "periodic": True,
    "virial": "auto",
    "units": {"positions": "angstrom", "energy": "eV"},
}
```

The model is called positionally as `model(q, cell, pbc)`. Inputs are
float64 `q[B,N,3]`, float64 `cell[B,3,3]` and boolean `pbc[B,3]`, on the selected
device. Cell and coordinates use the declared provider length unit. Models
must honor partial boundary flags, construct any required neighbor lists and
keep frames independent. No cutoff or minimum-image approximation is imposed
by the adapter.

The energy output has shape `[B]` or `[B,1]`. A dictionary may also contain
`forces[B,N,3]` and either `virial[B,3,3]` or tensile-positive
`stress[B,3,3]`. All numerical tensor outputs must be float64. Forces can be
derived from a differentiable energy graph; detached energy requires explicit
forces. Stress is converted to virial in the provider's consistent units.

Virial modes:

| Mode | Behavior |
|---|---|
| `auto` (default) | Use returned virial/stress; otherwise differentiate periodic energy with respect to strain when its graph is available |
| `autograd` | Require a differentiable energy and obtain virial from strain, even if the model also provides one |
| `provided` | Require returned virial or stress |
| `none` | Return no virial |

For autograd, every frame has an independent strain tensor. The adapter uses
`q'=q@(I+strain).T`, `cell'=cell@(I+strain).T` and
`W[a,b]=-dE/dstrain[a,b]`. It includes the cell contribution, which cannot in
general be reconstructed from `sum F outer r` for a periodic system. Energy
must remain differentiable through both coordinates and cell. Automatic mode
returns no virial when a direct-force model's energy is detached and no stress
or virial is supplied; this does not establish pressure support.

A factory returning a Python provider continues through `PythonPotential` and
must declare its own periodic binding or keyword contract. It is not treated as
a Torch tensor model.

## Verification and scope

`tests/test_torch_periodic.py` uses actual Torch CPU differentiation, partial
PBC, a triclinic cell, periodic translations, force finite differences,
all-nine-component strain derivatives, a deliberately nonsymmetric tensor,
stress units and detached-energy failures.

`tests/test_potential_adapters.py` checks named ML loader arguments and model
reuse with dependency doubles, including a real ASE conversion path. These
tests do not validate pretrained MACE, NequIP, DeePMD or fairchem weights.
Optional actual-runtime checks exercise tblite H2 forces, periodic strained-cell
virial and calculator reuse, and PySCF H2/H HF/DFT forces, convergence failure
and object reuse. No GPU speedup or production-model accuracy is established
by these software tests.
