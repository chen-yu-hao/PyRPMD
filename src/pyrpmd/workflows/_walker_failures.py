"""Failure evidence for coupled walker histories."""
from __future__ import annotations

from pathlib import Path

from ..io import write_report_json
from .biased_sampling import SamplingRuntime, sampling_failure_record


def walker_failure_path(output, method, mode, walker):
    return Path(output) / "failures" / f"{method}_{mode}_{walker:04d}.json"


def record_walker_failure(output, method, mode, walker, dynamics, cv, error, *,
                          state, elapsed, samples, options, checkpoint, bias_state):
    """Record a rolled-back walker without replacing the sampling error."""
    if not isinstance(error, Exception):
        return
    try:
        runtime = SamplingRuntime(state, elapsed, values=list(samples))
        record = sampling_failure_record(
            runtime, dynamics, cv, error,
            identity={"method": method, "walkers_mode": mode, "walker": walker},
            neq=options["neq"], nprod=options["nprod"],
            checkpoint=checkpoint, bias_state=bias_state)
        # MPI root owns the stored arrays, but every rank has this sample clock.
        record["state"]["recorded_samples"] = max(0, elapsed - options["neq"]) // options["interval"]
        path = walker_failure_path(output, method, mode, walker)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_report_json(path, record)
    except Exception as report_error:
        error.add_note(f"failure artifact not written: {report_error}")


def record_async_checkpoint_failure(output, method, walker, dynamics, cv, error, *,
                                    options, checkpoint):
    """Publication failures refer to the durable pending-record checkpoint."""
    import json

    from ..checkpoint import load_checkpoint
    from ..dynamics import State

    try:
        if not Path(checkpoint).exists():
            return
        with load_checkpoint(checkpoint, require_integrity=True) as data:
            elapsed = int(data["elapsed"])
            pending = json.loads(str(data["pending_record"]))
            if elapsed == options["neq"] + options["nprod"] and pending is None:
                return  # completed-history analysis errors are not sampling failures
            state = State(data["q"].copy(), data["p"].copy(), elapsed, float(data["time"]))
            samples = data["samples"].copy()
            bias_state = {"pending_record": pending}
            if "bias_state" in data:
                bias_state["bias_state"] = json.loads(str(data["bias_state"]))
            else:
                bias_state["record_count"] = len(json.loads(str(data["records"])))
        record_walker_failure(
            output, method, "async-files", walker, dynamics, cv, error,
            state=state, elapsed=elapsed, samples=samples, options=options,
            checkpoint=checkpoint, bias_state=lambda: bias_state)
    except Exception as report_error:
        error.add_note(f"failure artifact not written: {report_error}")
