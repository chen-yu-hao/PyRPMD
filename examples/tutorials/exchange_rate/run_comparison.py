"""Run fresh UI, MBAR and reactive dRPMD inputs and compare their rates."""
from pathlib import Path
import argparse
import json
import math

from pyrpmd.input import JobSpec
from pyrpmd.runner import run
from build import build
from compare import compare


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    build()
    summaries = {}
    for name in ("drpmd", "ui", "mbar"):
        spec = JobSpec.from_file(here / ("drpmd.inp" if name == "drpmd" else "rate.inp"))
        if name != "drpmd":
            spec.data["rate"]["pmf"]["estimator"] = name
        print(json.dumps({"method": name, "status": "running"}), flush=True)
        summaries[name] = run(spec, output=output / name)
        print(json.dumps({"method": name, "status": "complete"}), flush=True)
    checks = {name: compare(output / "drpmd", bc_run=output / name) for name in ("ui", "mbar")}
    ui, mbar = summaries['ui'], summaries['mbar']
    rates = [s.get('rate_cm3_s') for s in (ui, mbar)]
    errors = [s.get('rate_standard_error_cm3_s') for s in (ui, mbar)]
    ready = all(v is not None and math.isfinite(v) for v in rates + errors)
    ready &= all(s.get('rate_uncertainty', {}).get('available', False) for s in (ui, mbar))
    combined = math.hypot(*errors) if ready else 0.
    delta = abs(rates[0] - rates[1]) / combined if ready and combined > 0 else None
    checks['ui_mbar'] = {'passed': delta is not None and delta <= 3,
                         'deviation_in_combined_standard_errors': delta,
                         'uncertainty': 'quadrature of full marginal errors; not a paired-bootstrap estimate'}
    report = {'passed': all(c['passed'] for c in checks.values()), 'checks': checks,
              'physical_convergence': False, 'seeds_retried': False}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output))
