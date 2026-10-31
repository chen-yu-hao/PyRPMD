"""Reproduce finite-budget OGRe double-well metadynamics parameter trials.

height04 was repeated with seeds 4101, 4102 and 4103 for well_tempered;
other parameter combinations and standard height04 are exploratory seed 4101.
No setting here changes the library defaults or certifies convergence.
"""
import argparse
from pathlib import Path
from optimization import run_case
from pyrpmd.constants import AU_TIME_FS

OPTIONS = {
    "dt04": {"dt_au": .4/AU_TIME_FS, "sampling": {"andersen_interval": 25}},
    "height04": {"metadynamics": {"height": .4/2625.4996394799}},
    "andersen500": {"sampling": {"andersen_interval": 500}},
    "andersen500_height04": {"sampling": {"andersen_interval": 500},
                             "metadynamics": {"height": .4/2625.4996394799}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=OPTIONS, required=True)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--mode", choices=["standard", "well_tempered"], default="well_tempered")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    run_case("double_well", args.mode, args.seed, args.output, args.library,
             f"{args.mode}-{args.case}", OPTIONS[args.case])


if __name__ == "__main__":
    main()
