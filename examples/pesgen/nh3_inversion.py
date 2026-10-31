"""Generate a reproducible NH3 inversion scan and fit TREQ or dE-EVB.

RHF/STO-3G and a linear constrained scan are deliberately inexpensive controls,
not a converged ab-initio IRC, production PES, or computed inversion rate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pyrpmd.pesgen import PySCFReference, PySCFSettings, generate
from pyrpmd.pesgen.formats import write_xyz


def prepare(directory, native_build, *, method="treq", points=9):
    directory=Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    if method not in ("treq","de_evb") or type(points) is not int or points<5 or points%2!=1:
        raise ValueError("method must be treq/de_evb; points must be odd and >=5")
    directory.mkdir(parents=True)
    engine=PySCFReference(PySCFSettings(basis="sto-3g"))
    symbols=("N","H","H","H")
    q=engine.optimize(symbols,[[0,0,.6],[1.75,0,-.2],[-.875,1.5155444566,-.2],[-.875,-1.5155444566,-.2]])
    q-=q.mean(axis=0);right=q.copy();right[:,2]*=-1
    write_xyz(directory/"left.xyz",symbols,q)
    write_xyz(directory/"right.xyz",symbols,right)
    write_xyz(directory/"path.xyz",symbols,[(1-t)*q+t*right for t in np.linspace(0,1,points)])
    (directory/"coordinates.inp").write_text("1 2\n1 3\n1 4\n2 1 3\n2 1 4\n1 1 2 3 4\n")
    text=f'''schema="pyrpmd.pesgen/1"
method="{method}"
endpoints=["left.xyz", "right.xyz"]
path="path.xyz"
native_build={json.dumps(str(Path(native_build).resolve()))}
'''
    if method=="treq":text+='coordinates="coordinates.inp"\nforce_step=0.0001\n'
    text+='[qm]\nmethod="RHF"\nbasis="sto-3g"\n'
    (directory/"pes.toml").write_text(text)
    return directory/"pes.toml"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-build",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--method",choices=("treq","de_evb"),default="treq")
    parser.add_argument("--points",type=int,default=9)
    args=parser.parse_args()
    spec=prepare(args.output,args.native_build,method=args.method,points=args.points)
    print(json.dumps(generate(spec,Path(args.output)/"fit"),indent=2))
