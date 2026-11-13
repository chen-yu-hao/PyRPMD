"""Check the saved tutorial result against its analytic reference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_checks import main

if __name__ == "__main__":
    main('double_well_metad')
