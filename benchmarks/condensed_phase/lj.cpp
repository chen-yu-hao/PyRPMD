// Independent smooth Lennard-Jones solid reference, reduced units (hbar=1).
// ABI is PyRPMD's public thermal callback. No third-party implementation used.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>

extern "C" int32_t condensed_lj_v1(void *context, int64_t beads, int64_t atoms,
    const double *q, const double *cell, const int32_t *pbc,
    double *energy, double *force, double *virial, char *error, int64_t capacity) {
  // context=0 full LJ, 1 repulsion, 2 attraction. Exact additive split.
  const intptr_t kind = reinterpret_cast<intptr_t>(context);
  const double rc=2.5, rs=2.0;
  if (kind<0 || kind>2) return 1;
  for (int a=0;a<3;++a) {
    if (!pbc[a] || cell[3*a+a]<=rc) {
      std::snprintf(error,capacity,"requires cubic PBC cell side > cutoff=2.5"); return 2;
    }
    for (int b=0;b<3;++b) if(a!=b && cell[3*a+b]!=0) {
      std::snprintf(error,capacity,"requires orthorhombic cell"); return 2;
    }
  }
  std::fill(force,force+beads*atoms*3,0.);
  std::fill(virial,virial+beads*9,0.);
  std::fill(energy,energy+beads,0.);
  for(int64_t bead=0;bead<beads;++bead) {
    const double *x=q+bead*atoms*3;
    double *f=force+bead*atoms*3, *w=virial+bead*9;
    for(int64_t i=0;i<atoms;++i) for(int64_t j=i+1;j<atoms;++j) {
      double base[3];
      for(int a=0;a<3;++a) {
        double delta=x[3*i+a]-x[3*j+a], length=cell[3*a+a];
        base[a]=delta-length*std::nearbyint(delta/length);
      }
      // A small supercell can have more than one interacting image of a pair.
      // Enumerate all images; never apply an invalid half-box cutoff shortcut.
      for(int ix=-1;ix<=1;++ix) for(int iy=-1;iy<=1;++iy) for(int iz=-1;iz<=1;++iz) {
        double r[3]={base[0]+ix*cell[0],base[1]+iy*cell[4],base[2]+iz*cell[8]};
        const double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        if(r2>=rc*rc) continue;
        if(r2<1.e-10) { std::snprintf(error,capacity,"coincident LJ atoms"); return 3; }
        const double distance=std::sqrt(r2), inv2=1./r2, inv6=inv2*inv2*inv2;
        const double rep=4*inv6*inv6, attr=-4*inv6;
        const double u=(kind==2?0:rep)+(kind==1?0:attr);
        const double du=(-12*(kind==2?0:rep)-6*(kind==1?0:attr))/distance;
        double sw=1, dsw=0;
        if(distance>rs) {
          double t=(distance-rs)/(rc-rs);
          sw=1-10*t*t*t+15*t*t*t*t-6*t*t*t*t*t;
          dsw=(-30*t*t+60*t*t*t-30*t*t*t*t)/(rc-rs);
        }
        const double radial=-(du*sw+u*dsw)/distance;
        energy[bead]+=u*sw;
        for(int a=0;a<3;++a) {
          const double fij=radial*r[a];
          f[3*i+a]+=fij; f[3*j+a]-=fij;
          for(int b=0;b<3;++b) w[3*a+b]+=fij*r[b];
        }
      }
    }
  }
  return 0;
}
