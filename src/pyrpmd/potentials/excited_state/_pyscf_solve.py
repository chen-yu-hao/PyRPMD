"""Electronic-structure solve for the PySCF TDA/TDDFT backend."""

from __future__ import annotations

import numpy as np

from ._pyscf_overlap import PySCFTDWavefunction, auxiliary_wavefunction_overlap, _normalized_auxiliary_amplitudes
from ._types import BackendFailure, BackendResult, ExcitedStateStatus


class _PySCFSolveMixin:
    def _solve(
        self,
        geometry: np.ndarray,
        previous_payload: PySCFTDWavefunction | None,
        *,
        gradients: bool,
        properties: bool,
    ) -> BackendResult:
        from pyscf import dft, scf

        q = np.asarray(geometry)
        if q.shape != (len(self.symbols), 3) or q.dtype.kind not in "fiu" or not np.isfinite(q).all():
            raise ValueError(f"geometry must be finite [{len(self.symbols)},3]")
        try:
            mol = self._molecule(q)
            if self.xc is None:
                mean_field = scf.RHF(mol)
            else:
                mean_field = dft.RKS(mol)
                mean_field.xc = self.xc
                mean_field.grids.level = self.grid_level
            mean_field.conv_tol = self.scf_conv_tol
            mean_field.max_cycle = self.max_scf_cycles
            dm0 = None
            if previous_payload is not None:
                if not isinstance(previous_payload, PySCFTDWavefunction):
                    raise BackendFailure(
                        ExcitedStateStatus.INVALID_WAVEFUNCTION,
                        "previous PySCF TDDFT payload has the wrong type",
                    )
                try:
                    old_density = (
                        previous_payload.mo_coeff * previous_payload.mo_occ[np.newaxis, :]
                    ) @ previous_payload.mo_coeff.T
                    dm0 = scf.addons.project_dm_nr2nr(
                        previous_payload.mol, old_density, mol
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                    raise BackendFailure(
                        ExcitedStateStatus.INVALID_WAVEFUNCTION,
                        f"cannot project the previous PySCF density to the current AO basis: {exc}",
                    ) from exc
                if dm0.shape != (mol.nao_nr(), mol.nao_nr()) or not np.isfinite(dm0).all():
                    raise BackendFailure(
                        ExcitedStateStatus.INVALID_WAVEFUNCTION,
                        "projected previous PySCF density has invalid shape or nonfinite values",
                    )
            ground_energy = mean_field.kernel(dm0=dm0)
            if not bool(mean_field.converged) or not np.isfinite(ground_energy):
                raise BackendFailure(
                    ExcitedStateStatus.SCF_NOT_CONVERGED,
                    f"PySCF SCF did not converge in {self.max_scf_cycles} cycles",
                )

            response = mean_field.TDA() if self.tda else mean_field.TDDFT()
            response.nstates = self.nstates - 1
            if hasattr(response, "conv_tol"):
                response.conv_tol = self.td_conv_tol
            if hasattr(response, "max_cycle"):
                response.max_cycle = self.max_td_cycles
            excitation, xy = response.kernel()
            excitation = np.atleast_1d(np.asarray(excitation, dtype=float))
            if len(excitation) < self.nstates - 1 or not np.isfinite(excitation[: self.nstates - 1]).all():
                raise BackendFailure(
                    ExcitedStateStatus.STATE_COUNT_MISMATCH,
                    f"PySCF returned {len(excitation)} excitation energies; expected {self.nstates - 1}",
                )
            converged = np.asarray(getattr(response, "converged", True))
            if not bool(np.all(converged)):
                raise BackendFailure(
                    ExcitedStateStatus.EXCITED_STATE_NOT_CONVERGED,
                    "one or more PySCF TDA/TDDFT roots did not converge",
                )
            xy = tuple(xy[: self.nstates - 1])
            amplitudes = tuple(
                _normalized_auxiliary_amplitudes(pair[0], pair[1], tda=self.tda)
                for pair in xy
            )
            energies = np.r_[float(ground_energy), float(ground_energy) + excitation[: self.nstates - 1]]

            gradient_values = np.zeros((self.nstates, len(self.symbols), 3))
            if gradients:
                try:
                    gradient_values[0] = mean_field.nuc_grad_method().kernel()
                    gradient_method = response.nuc_grad_method()
                    for root in range(1, self.nstates):
                        gradient_values[root] = gradient_method.kernel(state=root)
                except (NotImplementedError, AttributeError, RuntimeError) as exc:
                    raise BackendFailure(
                        ExcitedStateStatus.GRADIENT_UNAVAILABLE,
                        f"PySCF does not provide the requested TD gradient: {exc}",
                    ) from exc
                if not np.isfinite(gradient_values).all():
                    raise BackendFailure(
                        ExcitedStateStatus.GRADIENT_UNAVAILABLE,
                        "PySCF returned nonfinite nuclear gradients",
                    )

            strengths = None
            dipoles = None
            if properties:
                try:
                    excited_strengths = np.asarray(
                        response.oscillator_strength(gauge="length"), dtype=float
                    )[: self.nstates - 1]
                    excited_dipoles = np.asarray(
                        response.transition_dipole(), dtype=float
                    )[: self.nstates - 1]
                except (NotImplementedError, AttributeError) as exc:
                    raise BackendFailure(
                        ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
                        f"PySCF transition properties are unavailable: {exc}",
                    ) from exc
                if (
                    excited_strengths.shape != (self.nstates - 1,)
                    or excited_dipoles.shape != (self.nstates - 1, 3)
                    or not np.isfinite(excited_strengths).all()
                    or not np.isfinite(excited_dipoles).all()
                ):
                    raise BackendFailure(
                        ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
                        "PySCF transition properties have invalid shape or nonfinite values",
                    )
                strengths = np.r_[0.0, excited_strengths]
                dipoles = np.vstack([np.zeros(3), excited_dipoles])

            payload = PySCFTDWavefunction(
                mol.dumps(),
                q,
                mean_field.mo_coeff,
                mean_field.mo_occ,
                amplitudes,
                self.tda,
            )
            overlap = None
            if previous_payload is not None:
                overlap = auxiliary_wavefunction_overlap(
                    previous_payload,
                    payload,
                    max_determinant_pairs=self.max_determinant_pairs,
                )
            return BackendResult(
                energies,
                gradient_values,
                payload,
                overlap=overlap,
                oscillator_strengths=strengths,
                transition_dipoles=dipoles,
                metadata={
                    "method": "TDA" if self.tda else "TDDFT",
                    "xc": self.xc,
                    "overlap_model": (
                        "cis-determinant-x-v1"
                        if self.tda
                        else "casida-auxiliary-determinant-x-plus-y-v1"
                    ),
                },
            )
        except BackendFailure:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise BackendFailure(
                ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
                f"PySCF TDA/TDDFT calculation failed: {exc}",
            ) from exc
