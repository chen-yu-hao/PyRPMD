from pathlib import Path

import numpy as np
import pytest

from pyrpmd.potentials.excited_state import (
    BackendFailure,
    ExcitedStateProvider,
    ExcitedStateStatus,
    OpenMolcasExcitedStateBackend,
    ORCAExcitedStateBackend,
    QChemExcitedStateBackend,
)

FIXTURES = Path(__file__).parent / "fixtures" / "excited_state"


def _backend(name, tmp_path):
    common = dict(symbols=["H", "H"], nstates=3, program_version="recorded-test")
    if name == "orca":
        return ORCAExcitedStateBackend(**common)
    if name == "qchem":
        return QChemExcitedStateBackend(**common)
    if name == "openmolcas":
        return OpenMolcasExcitedStateBackend(**common)
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name,ground,excited",
    [
        ("orca", -75.0, [-74.9, -74.82]),
        ("qchem", -40.0, [-39.9, -39.8]),
        ("openmolcas", -108.0, [-107.9, -107.82]),
    ],
)
def test_recorded_success_outputs(name, ground, excited, tmp_path):
    backend = _backend(name, tmp_path)
    parsed = backend.parse_output((FIXTURES / name / "success.out").read_text())
    np.testing.assert_allclose(parsed.energies, [ground, *excited], atol=2e-12, rtol=0.0)
    assert parsed.gradients.shape == (3, 2, 3)
    np.testing.assert_allclose(parsed.gradients.sum(axis=1), 0.0, atol=0.003, rtol=0.0)
    assert parsed.oscillator_strengths.shape == (3,)
    assert parsed.transition_dipoles.shape == (3, 3)


@pytest.mark.parametrize("name", ["orca", "qchem", "openmolcas"])
def test_recorded_scf_failures_have_explicit_status(name, tmp_path):
    backend = _backend(name, tmp_path)
    with pytest.raises(BackendFailure) as caught:
        backend.parse_output((FIXTURES / name / "scf_failure.out").read_text())
    assert caught.value.status is ExcitedStateStatus.SCF_NOT_CONVERGED


@pytest.mark.parametrize("name", ["orca", "qchem", "openmolcas"])
def test_recorded_state_shortage_has_explicit_status(name, tmp_path):
    backend = _backend(name, tmp_path)
    with pytest.raises(BackendFailure) as caught:
        backend.parse_output((FIXTURES / name / "state_shortage.out").read_text())
    assert caught.value.status is ExcitedStateStatus.STATE_COUNT_MISMATCH


def test_input_generators_are_deterministic_and_use_bohr_coordinates(tmp_path):
    q = np.array([[0.0, 0.0, 0.0], [1.4, 0.1, -0.2]])
    orca = ORCAExcitedStateBackend(["H", "H"], 3, program_version="6.0")
    qchem = QChemExcitedStateBackend(["H", "H"], 3, program_version="6.2")
    molcas = OpenMolcasExcitedStateBackend(["H", "H"], 3, program_version="24.10")
    orca_input = orca.render_input(q)
    qchem_input = qchem.render_input(q)
    assert "nroots 2" in orca_input
    assert "Bohrs" in orca_input
    coordinate_lines = [
        line for line in orca_input.splitlines() if line.strip().startswith("H")
    ]
    assert len(coordinate_lines) == 2
    assert float(coordinate_lines[1].split()[1]) == pytest.approx(1.4)
    assert "CIS_N_ROOTS 2" in qchem_input
    assert "INPUT_BOHR TRUE" in qchem_input
    molcas.prepare_files(q, tmp_path)
    assert (tmp_path / "geometry.xyz").read_text().splitlines()[1].startswith("bohr")
    assert "CIROOT=3 3 1" in molcas.render_input(q)


def test_template_and_ecp_hashes_enter_external_identity(tmp_path):
    ecp = tmp_path / "model.ecp"
    ecp.write_text("first")
    first = ORCAExcitedStateBackend(
        ["H", "H"], 2, program_version="6.0", ecp_files=[ecp]
    )
    first_identity = first.scientific_identity()
    ecp.write_text("second")
    second = ORCAExcitedStateBackend(
        ["H", "H"], 2, program_version="6.0", ecp_files=[ecp]
    )
    assert first_identity["ecp_sha256"] != second.scientific_identity()["ecp_sha256"]
    custom = ORCAExcitedStateBackend(
        ["H", "H"], 2, program_version="6.0",
        input_template=ORCAExcitedStateBackend.DEFAULT_TEMPLATE + "\n%pal nprocs 2 end\n",
    )
    assert first_identity["input_template_sha256"] != custom.scientific_identity()["input_template_sha256"]


def test_file_backend_executes_argument_vectors_restores_wavefunction_and_runs_overlap(tmp_path):
    import sys

    program = tmp_path / "program.py"
    program.write_text(
        """from pathlib import Path
import sys
fixture, input_name, previous = sys.argv[1:4]
assert Path(input_name).is_file()
previous_path = Path(previous)
if previous_path.name == 'previous':
    assert (previous_path / 'job.gbw').read_bytes().startswith(b'wavefunction')
Path('job.gbw').write_bytes(b'wavefunction-current')
print(Path(fixture).read_text(), end='')
"""
    )
    overlap = tmp_path / "overlap.py"
    overlap.write_text(
        """from pathlib import Path
import sys
old_dir, new_dir, matrix, nstates = sys.argv[1:5]
assert (Path(old_dir) / 'job.gbw').is_file()
assert (Path(new_dir) / 'job.gbw').is_file()
n = int(nstates)
Path(matrix).write_text('\\n'.join(' '.join('1' if i == j else '0' for j in range(n)) for i in range(n)) + '\\n')
"""
    )
    backend = ORCAExcitedStateBackend(
        ["H", "H"],
        3,
        program_version="recorded-wrapper",
        command=(
            sys.executable,
            str(program),
            str(FIXTURES / "orca" / "success.out"),
            "{input}",
            "{previous}",
        ),
        overlap_command=(
            sys.executable,
            str(overlap),
            "{old_dir}",
            "{new_dir}",
            "{matrix}",
            "{nstates}",
        ),
    )
    provider = ExcitedStateProvider(
        backend,
        ["H", "H"],
        3,
        require_couplings=False,
        scratch_root=tmp_path / "scratch",
    )
    q0 = np.array([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]])
    q1 = q0.copy()
    q1[0, 1, 0] += 1e-3
    first = provider.evaluate(q0)
    assert first.status.tolist() == [ExcitedStateStatus.SUCCESS]
    assert first.context[0].payload.artifacts["job.gbw"] == b"wavefunction-current"
    second = provider.evaluate(q1, previous_q=q0, previous_context=first.context)
    assert second.status.tolist() == [ExcitedStateStatus.SUCCESS]
    np.testing.assert_allclose(second.overlaps[0], np.eye(3), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(second.couplings[0], 0.0, atol=0.0, rtol=0.0)
    assert provider.statistics().call_count == 2
    provider.close()


def test_basis_files_and_environment_are_hashed_without_exposing_values(tmp_path):
    basis_file = tmp_path / "custom.basis"
    basis_file.write_text("basis-v1")
    first = ORCAExcitedStateBackend(
        ["H", "H"],
        2,
        program_version="6.0",
        basis_files=[basis_file],
        environment={"PRIVATE_SETTING": "value-one"},
    ).scientific_identity()
    assert first["basis_sha256"]["files"][0]["name"] == "custom.basis"
    assert "value-one" not in str(first)
    basis_file.write_text("basis-v2")
    second = ORCAExcitedStateBackend(
        ["H", "H"],
        2,
        program_version="6.0",
        basis_files=[basis_file],
        environment={"PRIVATE_SETTING": "value-two"},
    ).scientific_identity()
    assert first["basis_sha256"] != second["basis_sha256"]
    assert first["environment_sha256"] != second["environment_sha256"]


def test_external_initial_nac_uses_centered_overlap_finite_difference(tmp_path):
    import json

    from pyrpmd.potentials.excited_state import (
        BackendResult,
        ExternalWavefunction,
        FileProgramBackend,
    )

    class RotatingFileBackend(FileProgramBackend):
        name = "rotating-file-test"
        wavefunction_filenames = ()

        def __init__(self):
            super().__init__(
                ["H"],
                2,
                program_version="1",
                method="analytic-test",
                basis="none",
                command=("true",),
                input_template="test input",
                initial_nac="finite_difference",
                nac_finite_difference_step=1e-5,
            )

        def render_input(self, geometry):
            return "test input"

        def parse_output(self, text):
            raise AssertionError("not used")

        @staticmethod
        def _vectors(geometry):
            angle = float(geometry[0, 0])
            return np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

        def calculate(self, geometry, scratch, previous_payload):
            vectors = self._vectors(geometry)
            payload = ExternalWavefunction(
                self.name,
                {"vectors.json": json.dumps(vectors.tolist()).encode()},
            )
            overlap = None
            if previous_payload is not None:
                previous = np.asarray(
                    json.loads(previous_payload.artifacts["vectors.json"]), dtype=float
                )
                overlap = previous.T @ vectors
            return BackendResult(
                np.array([0.0, 1.0]),
                np.zeros((2, 1, 3)),
                payload,
                overlap=overlap,
            )

    backend = RotatingFileBackend()
    q = np.zeros((1, 3))
    center = backend.calculate(q, tmp_path, None)
    coupling = backend.initial_derivative_couplings(q, center.payload, tmp_path)
    expected = np.zeros((2, 2, 1, 3))
    expected[:, :, 0, 0] = [[0.0, -1.0], [1.0, 0.0]]
    np.testing.assert_allclose(coupling, expected, atol=3e-10, rtol=0.0)


def test_recorded_wrapper_markers_reject_invalid_state_and_atom_indices():
    from pyrpmd.potentials.excited_state.external import (
        parse_gradient_blocks,
        parse_transition_properties,
    )

    with pytest.raises(BackendFailure) as caught:
        parse_transition_properties(
            "PYRPMD PROPERTY STATE -1 F 0.1 MU 0.0 0.0 0.0", nstates=2
        )
    assert caught.value.status is ExcitedStateStatus.OUTPUT_PARSE_ERROR

    mixed = """PYRPMD GRADIENT STATE 0
0 0.0 0.0 0.0
2 0.0 0.0 0.0
"""
    with pytest.raises(BackendFailure) as caught:
        parse_gradient_blocks(mixed, nstates=1, natoms=2)
    assert caught.value.status is ExcitedStateStatus.OUTPUT_PARSE_ERROR


def test_openmolcas_post_rasscf_requires_explicit_template():
    with pytest.raises(ValueError, match="explicit input_template"):
        OpenMolcasExcitedStateBackend(
            ["H", "H"], 2, program_version="24.10", method="MS-CASPT2"
        )


def test_identity_files_are_staged_from_the_hashed_snapshot(tmp_path):
    import sys

    basis = tmp_path / "custom.basis"
    ecp = tmp_path / "model.ecp"
    basis.write_bytes(b"basis-v1\n")
    ecp.write_bytes(b"ecp-v1\n")
    program = tmp_path / "check_staged_files.py"
    program.write_text(
        """from pathlib import Path
import sys
fixture, input_name = sys.argv[1:3]
assert Path(input_name).is_file()
assert Path('custom.basis').read_bytes() == b'basis-v1\\n'
assert Path('model.ecp').read_bytes() == b'ecp-v1\\n'
print(Path(fixture).read_text(), end='')
"""
    )
    backend = ORCAExcitedStateBackend(
        ["H", "H"],
        3,
        program_version="recorded-wrapper",
        basis_files=[basis],
        ecp_files=[ecp],
        command=(
            sys.executable,
            str(program),
            str(FIXTURES / "orca" / "success.out"),
            "{input}",
        ),
    )
    # Mutating the source after construction must not change the bytes paired
    # with the already-issued scientific identity.
    basis.write_bytes(b"basis-v2\n")
    ecp.write_bytes(b"ecp-v2\n")
    scratch = tmp_path / "scratch-staging"
    scratch.mkdir()
    result = backend.calculate(
        np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]), scratch, None
    )
    assert result.energies.shape == (3,)


def test_external_parsers_reject_gapped_root_labels():
    cases = [
        (
            ORCAExcitedStateBackend(["H", "H"], 3, program_version="6.0"),
            (FIXTURES / "orca" / "success.out").read_text().replace(
                "STATE   2: E=", "STATE   3: E="
            ),
        ),
        (
            QChemExcitedStateBackend(["H", "H"], 3, program_version="6.2"),
            (FIXTURES / "qchem" / "success.out").read_text().replace(
                "Excited state   2:", "Excited state   3:"
            ),
        ),
        (
            OpenMolcasExcitedStateBackend(
                ["H", "H"], 3, program_version="24.10"
            ),
            (FIXTURES / "openmolcas" / "success.out").read_text().replace(
                "root number  2", "root number  4"
            ),
        ),
    ]
    for backend, text in cases:
        with pytest.raises(BackendFailure) as caught:
            backend.parse_output(text)
        assert caught.value.status is ExcitedStateStatus.STATE_COUNT_MISMATCH


def test_basis_and_ecp_staging_rejects_managed_filename_collisions(tmp_path):
    managed = tmp_path / "job.gbw"
    managed.write_bytes(b"collision")
    with pytest.raises(ValueError, match="collide"):
        ORCAExcitedStateBackend(
            ["H", "H"], 2, program_version="6.0", basis_files=[managed]
        )
