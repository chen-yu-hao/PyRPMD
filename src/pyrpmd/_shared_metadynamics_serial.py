"""Synchronous multiple walkers sharing one metadynamics history."""
from __future__ import annotations

import numpy as np

from .bias import MetadynamicsBias
from .path_integral import _integer


class SharedMetadynamicsBias(MetadynamicsBias):
    """Deposit a complete walker batch against the same pre-deposition bias.

    All walkers see one frozen history for an entire MD step. At a deposition
    barrier, each contributes the configured hill height (WT-scaled using the
    bias BEFORE any hill in this batch). Total deposition grows with walker
    count; choose height/pace accordingly. There is one correlated bias history.
    """

    _shared_clocks = True

    def __init__(self, *args, walkers, **kwargs):
        self.walkers = _integer(walkers, "walkers", 2)
        super().__init__(*args, **kwargs)
        if self.hill_count % self.walkers:
            raise ValueError("shared history must end at a complete walker deposition")

    def deposit(self, *args, **kwargs):
        raise ValueError("shared bias requires deposit_walkers with all walkers")

    def truncate_history(self, count):
        """Rollback only to a complete synchronized deposition barrier."""
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or count % self.walkers):
            raise ValueError("shared history rollback must retain complete walker batches")
        super().truncate_history(count)

    def deposit_walkers(self, values, *, step, time):
        centers = np.asarray(values)
        if centers.dtype.kind not in "fiu" or centers.shape != (self.walkers, self.cv.dimension) or not np.isfinite(centers).all():
            raise ValueError("shared deposition needs finite [walkers,dimension] centers")
        before = np.asarray(self.physical_energy(centers), dtype=float)
        heights = np.full(self.walkers, self.height)
        if self.mode == "well_tempered":
            heights *= np.exp(-self.beta * before / (self.bias_factor - 1))
        old_count = self.hill_count
        try:
            for walker in range(self.walkers):
                self._append({"center": centers[walker], "height": heights[walker],
                    "step": step, "time": time, "bias_before": before[walker], "walker": walker})
        except BaseException:
            self.truncate_history(old_count)
            raise
        return self.hills[-self.walkers:]


__all__ = ["SharedMetadynamicsBias"]
