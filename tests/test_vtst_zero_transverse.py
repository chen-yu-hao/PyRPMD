import json
import math

import numpy as np

from pyrpmd.constants import AMU_TO_AU
from pyrpmd.vtst import MEPPath, PartitionSpecies, compute_vtst


def _diatomic_stretch_path() -> tuple[MEPPath, np.ndarray, PartitionSpecies]:
    masses = np.full(2, AMU_TO_AU)
    scaling_mass = AMU_TO_AU
    s = np.linspace(-1.0, 1.0, 9)
    tangent = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    base = np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    geometries = np.array(
        [
            (base.ravel() + 0.2 * coordinate * tangent).reshape(2, 3)
            for coordinate in s
        ]
    )
    barrier = 0.005
    energies = barrier * (1.0 - s**2) ** 2
    path = MEPPath(
        s_bohr=s,
        geometries_bohr=geometries,
        energies_hartree=energies,
        gradients_hartree_per_bohr=np.zeros_like(geometries),
        hessians_hartree_per_bohr2=np.zeros((len(s), 6, 6)),
        tangent_scaled=np.tile(tangent, (len(s), 1)),
        side_converged=(True, True),
        saddle_eigenvalues_omega2=np.array([-1.0e-6]),
        scaling_mass=scaling_mass,
    )
    reactant = PartitionSpecies(
        geometry_bohr=geometries[0],
        masses=masses,
        frequencies_au=np.array([0.01]),
    )
    return path, masses, reactant


def test_diatomic_zero_transverse_subspace_is_json_safe_and_sct_equals_zct():
    path, masses, reactant = _diatomic_stretch_path()
    result = compute_vtst(
        path,
        masses,
        [reactant],
        [300.0],
        tunnelling_quadrature_points=16,
    )

    assert all(frequencies.size == 0 for frequencies in result.modes.frequencies_au)
    np.testing.assert_array_equal(
        result.modes.curvature_bohr_inverse, np.zeros(len(path.s_bohr))
    )
    np.testing.assert_array_equal(
        result.modes.effective_frequency_au, np.zeros(len(path.s_bohr))
    )
    np.testing.assert_array_equal(
        result.modes.transverse_turning_point_bohr, np.zeros(len(path.s_bohr))
    )
    np.testing.assert_array_equal(
        result.modes.effective_mass_ratio_sct, np.ones(len(path.s_bohr))
    )

    row = result.temperatures[0]
    assert row["zct"]["log_kappa"] == row["sct"]["log_kappa"]
    assert row["zct"]["kappa"] == row["sct"]["kappa"]
    assert row["cvt_zct"]["log_rate_au"] == row["cvt_sct"]["log_rate_au"]
    json.dumps(result.to_dict(), allow_nan=False)
