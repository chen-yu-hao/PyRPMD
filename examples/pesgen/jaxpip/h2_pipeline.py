"""Actual RHF/STO-3G -> fitted JaxPIP -> optional native dynamics pilot.

The small H2 bond scan demonstrates interfaces, not a global reactive potential,
state-of-the-art electronic structure or a converged physical RPMD observable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from pyrpmd.pesgen.formats import write_xyz
from pyrpmd.pesgen.jaxpip_data import collect_dataset
from pyrpmd.pesgen.jaxpip_fit import fit_jaxpip
from pyrpmd.pesgen.jaxpip_export import export_jaxpip, native_potential_spec


def run(output, kind="linear", exporter=None, library=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    radii = {"train": np.linspace(1.0, 2.2, 16),
             "validation": np.array([1.03, 1.39, 1.75, 2.11]),
             "test": np.array([1.07, 1.43, 1.79, 2.15])}
    for split, values in radii.items():
        q = np.zeros((len(values), 2, 3)); q[:, 1, 0] = values
        write_xyz(output/f"{split}.xyz", ("H", "H"), q)
    data_spec = output/"data.toml"
    data_spec.write_text('''schema="pyrpmd.jaxpip-data/1"
[geometries]
train="train.xyz"
validation="validation.xyz"
test="test.xyz"
[qm]
method="RHF"
basis="sto-3g"
conv_tol=1e-12
''')
    collect_dataset(data_spec, output/"data")
    # A diatomic PIP basis, exactly invariant under exchange of the two atoms.
    (output/"basis.json").write_text(json.dumps([[[k]] for k in range(8)]))
    fit_spec = output/"fit.toml"
    fit_spec.write_text(f'''schema="pyrpmd.jaxpip-fit/1"
dataset="data"
basis="basis.json"
basis_symbols=["H", "H"]
[fit]
kind="{kind}"
ridge=1e-14
hidden_layers=[8]
max_steps=250
[acceptance]
energy_rmse_max=3e-5
force_rmse_max=3e-4
''')
    fitted = fit_jaxpip(fit_spec, output/"fit")
    report = json.loads((output/"fit/metrics.json").read_text())
    report["kind"] = kind
    report["fit_manifest"] = fitted
    if exporter:
        export_jaxpip(output/"fit", exporter, output/"native")
    if library:
        if not exporter:
            raise ValueError("native propagation needs both --exporter and --library")
        from pyrpmd.dynamics import Dynamics, State
        from pyrpmd.potentials import CABIPotential
        from pyrpmd.system import System
        from pyrpmd.constants import AMU_TO_AU
        q0 = np.array([[0., 0., 0.], [1.4, 0., 0.]])
        system = System(("H", "H"), np.full(2, 1.00782503223*AMU_TO_AU), q0)
        spec = native_potential_spec(output/"native", library, allow_unvalidated=True)
        calls = []
        def profile(frame, event, arg):
            if event == "call" and frame.f_code is CABIPotential.evaluate.__code__:
                calls.append(1)
        with CABIPotential(spec["library"], spec["params"], system=system) as pes:
            dyn = Dynamics(pes, system.masses, 300., 8, 0.1)
            q = np.repeat(q0[None], 8, axis=0)
            p = np.random.default_rng(123).normal(scale=.05, size=q.shape)
            a, b = State(q.copy(), p.copy()), State(q.copy(), p.copy())
            previous = sys.getprofile()
            try:
                sys.setprofile(profile)
                native = dyn.propagate_block(a, 100, backend="native", record_states=True)
            finally:
                sys.setprofile(previous)
            assert native.backend == "native" and not calls
            reference = dyn.propagate_block(b, 100, backend="python", record_states=True)
            np.testing.assert_allclose(native.q, reference.q, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(native.p, reference.p, atol=1e-9, rtol=1e-10)
            report["native"] = {"steps": 100, "beads": 8, "python_pes_calls": len(calls),
                                "q_max_error_bohr": float(np.max(np.abs(native.q-reference.q))),
                                "p_max_error_au": float(np.max(np.abs(native.p-reference.p))),
                                "energy_range_hartree": float(np.ptp(native.energies))}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("linear", "nn"), default="linear")
    parser.add_argument("--exporter")
    parser.add_argument("--library")
    run(**vars(parser.parse_args()))
