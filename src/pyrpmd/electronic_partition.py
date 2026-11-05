"""Temperature-dependent electronic partition functions for rate prefactors.

Single-surface RPMD, dRPMD and QTST rates are computed on one adiabatic
potential. The electronic factor that multiplies such a rate is

    w_el(T) = Q_el^TS(T) / prod_r Q_el^r(T),
    Q_el(T) = sum_i g_i exp(-E_i / k_B T),

where every species lists its electronic levels relative to a reference the
user chooses. Presets measure energies from the species' own lowest level;
an explicit level list may shift that reference, for example to a
spin-orbit-free PES asymptote. Nothing is inferred from geometry or atom
labels: each species is selected explicitly.

Preset level energies are the NIST Atomic Spectra Database fine-structure
levels (Kramida, Ralchenko, Reader and NIST ASD Team) in cm^-1, with
degeneracy 2J + 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers

from .constants import KB_AU, WAVENUMBER_TO_AU
from .units import parse_quantity

__all__ = [
    "ELECTRONIC_PRESETS", "ElectronicLevels", "ElectronicModel",
    "electronic_levels", "electronic_partition_function", "electronic_weight",
]

# name -> ((energy / cm^-1, degeneracy), ...), ground level first.
_PRESET_CM = {
    "singlet": ((0.0, 1),),
    "doublet": ((0.0, 2),),
    "triplet": ((0.0, 3),),
    "H(2S)": ((0.0, 2),),
    "F(2P)": ((0.0, 4), (404.141, 2)),
    "Cl(2P)": ((0.0, 4), (882.3515, 2)),
    "Br(2P)": ((0.0, 4), (3685.24, 2)),
    "I(2P)": ((0.0, 4), (7603.15, 2)),
    "O(3P)": ((0.0, 5), (158.265, 3), (226.977, 1)),
    "S(3P)": ((0.0, 5), (396.055, 3), (573.640, 1)),
}
ELECTRONIC_PRESETS = {
    name: tuple((e * WAVENUMBER_TO_AU, g) for e, g in levels)
    for name, levels in _PRESET_CM.items()
}
_PRESET_KEYS = {name.lower(): name for name in ELECTRONIC_PRESETS}


def _degeneracy(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _temperature(value):
    temperature = parse_quantity(value, "temperature")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return temperature


@dataclass(frozen=True)
class ElectronicLevels:
    """Electronic levels of one species; energies are hartree."""

    label: str
    levels: tuple[tuple[float, int], ...]
    preset: str | None = None

    def log_partition_function(self, temperature) -> float:
        beta = 1.0 / (KB_AU * _temperature(temperature))
        # Shift by the lowest level so a large common offset cannot overflow.
        low = min(e for e, _ in self.levels)
        return math.log(sum(g * math.exp(-beta * (e - low)) for e, g in self.levels)) - beta * low

    def partition_function(self, temperature) -> float:
        log_q = self.log_partition_function(temperature)
        if log_q > 700:
            raise OverflowError(f"{self.label}: electronic partition function overflows")
        return math.exp(log_q)

    def to_dict(self):
        out = {"label": self.label,
               "levels": [{"energy_hartree": e, "energy_cm-1": e / WAVENUMBER_TO_AU, "degeneracy": g}
                          for e, g in self.levels]}
        if self.preset is not None:
            out["preset"] = self.preset
        return out


def electronic_levels(spec, label="species") -> ElectronicLevels:
    """Build levels from a preset name, a level list, or a ``{preset|levels}`` table.

    A level is ``{energy=..., degeneracy=...}`` or an ``[energy, degeneracy]``
    pair. Energies follow PyRPMD quantity rules: a bare number is hartree and
    a string such as ``"882.35 cm-1"`` carries its unit.
    """
    if isinstance(spec, ElectronicLevels):
        return spec
    if isinstance(spec, str):
        key = _PRESET_KEYS.get(spec.strip().lower())
        if key is None:
            raise ValueError(f"{label}: unknown electronic preset {spec!r}; "
                             f"choose one of {', '.join(ELECTRONIC_PRESETS)} or give explicit levels")
        return ElectronicLevels(label, ELECTRONIC_PRESETS[key], key)
    if isinstance(spec, Mapping):
        if set(spec) - {"preset", "levels", "label"} or ("preset" in spec) == ("levels" in spec):
            raise ValueError(f"{label}: give exactly one of preset or levels (optional label)")
        name = str(spec.get("label", label))
        if "preset" in spec:
            return electronic_levels(spec["preset"], name)
        spec = spec["levels"]
        label = name
    if isinstance(spec, (str, bytes)) or not isinstance(spec, Sequence) or not spec:
        raise ValueError(f"{label}: electronic levels must be a nonempty list")
    levels = []
    for i, level in enumerate(spec):
        where = f"{label} level {i}"
        if isinstance(level, Mapping):
            if set(level) != {"energy", "degeneracy"}:
                raise ValueError(f"{where} needs exactly energy and degeneracy")
            energy, g = level["energy"], level["degeneracy"]
        elif isinstance(level, Sequence) and not isinstance(level, (str, bytes)) and len(level) == 2:
            energy, g = level
        else:
            raise ValueError(f"{where} must be {{energy, degeneracy}} or [energy, degeneracy]")
        energy = parse_quantity(energy, "energy")
        levels.append((float(energy), _degeneracy(g, where + " degeneracy")))
    return ElectronicLevels(label, tuple(levels))


def electronic_partition_function(spec, temperature) -> float:
    """Q_el(T) = sum_i g_i exp(-E_i / k_B T) for one species."""
    return electronic_levels(spec).partition_function(temperature)


@dataclass(frozen=True)
class ElectronicModel:
    """Transition-state and reactant electronic levels for one reaction channel."""

    transition_state: ElectronicLevels
    reactants: tuple[ElectronicLevels, ...]

    @classmethod
    def from_mapping(cls, mapping, *, reactant_count=None):
        if not isinstance(mapping, Mapping) or set(mapping) != {"transition_state", "reactants"}:
            raise ValueError("electronic model needs exactly transition_state and reactants")
        reactants = mapping["reactants"]
        if isinstance(reactants, (str, bytes, Mapping)) or not isinstance(reactants, Sequence) or not reactants:
            raise ValueError("electronic reactants must be a nonempty list with one entry per fragment")
        if reactant_count is not None and len(reactants) != reactant_count:
            raise ValueError(f"electronic reactants needs {reactant_count} entries, one per reactant fragment")
        return cls(electronic_levels(mapping["transition_state"], "transition_state"),
                   tuple(electronic_levels(r, f"reactant {i}") for i, r in enumerate(reactants)))

    def weight(self, temperature) -> float:
        return self.evaluate(temperature)["electronic_weight"]

    def evaluate(self, temperature) -> dict:
        temperature = _temperature(temperature)
        log_ts = self.transition_state.log_partition_function(temperature)
        log_r = [r.log_partition_function(temperature) for r in self.reactants]
        log_weight = log_ts - sum(log_r)
        if not math.isfinite(log_weight) or abs(log_weight) > 700:
            raise ValueError("electronic weight is not finite and positive")
        weight = math.exp(log_weight)
        return {
            "model": "Q_el_TS/prod(Q_el_reactants)",
            "temperature_K": temperature,
            "electronic_weight": weight,
            "log_electronic_weight": log_weight,
            "transition_state": {**self.transition_state.to_dict(), "log_partition_function": log_ts},
            "reactants": [{**r.to_dict(), "log_partition_function": q}
                          for r, q in zip(self.reactants, log_r)],
        }


def electronic_weight(transition_state, reactants, temperature) -> float:
    """Electronic rate factor Q_el^TS(T) / prod_r Q_el^r(T)."""
    return ElectronicModel.from_mapping(
        {"transition_state": transition_state, "reactants": list(reactants)}
    ).weight(temperature)
