"""Reproduce the finite-box HBr rovibrational preparation reference."""
import json
from pathlib import Path
from dynamics import reactant_reference
if __name__=='__main__':
    out=Path(__file__).resolve().parents[2]/'results';out.mkdir(exist_ok=True)
    data={str(T):reactant_reference(T) for T in (150,300,600)}
    (out/'reactant_reference.json').write_text(json.dumps(data,indent=2))
    print({T:r['internal_energy_Eh'] for T,r in data.items()})
