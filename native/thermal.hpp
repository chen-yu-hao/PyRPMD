/* Native, callback-free thermostat/barostat trajectory engine.
 * The BZP and upper-triangular flexible-cell equations follow the i-PI
 * implementation (MIT option), ipi/engine/barostats.py, blob
 * 98cf6e6c38366d9580cb48b69e36e29a48e1bb76. See thermal-NOTICE.md.
 * Native PES modes keep stochastic draws, forces, transforms and steps in C++.
 * An explicitly selected interpreter bridge is installed by thermal.cpp;
 * this header itself uses no Python API. Native scratch is allocated once.
 */
#ifndef PYRPMD_THERMAL_HPP
#define PYRPMD_THERMAL_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <fstream>
#include <functional>
#ifdef _WIN32
#include <filesystem>
#endif
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "ring_fft.h"
#include "philox.h"
#include "include/pyrpmd/pes_api_v1.h"
#include "include/pyrpmd/thermal_pes.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace thermal {
using Vec = std::vector<double>;
using Mat = std::array<double,9>;
inline void require(bool b,const char* s){if(!b)throw std::invalid_argument(s);}
inline bool finite(const Vec& v){return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);});}
inline Mat identity(){return {1,0,0,0,1,0,0,0,1};}
inline Mat transpose(const Mat& a){Mat b{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[3*i+j]=a[3*j+i];return b;}
inline Mat product(const Mat&a,const Mat&b){Mat c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c[3*i+j]+=a[3*i+k]*b[3*k+j];return c;}
inline double det(const Mat& a){return a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);}
inline Mat inverse(const Mat&a){double d=det(a);require(std::isfinite(d)&&d>0,"cell must have finite positive determinant");Mat b{
    a[4]*a[8]-a[5]*a[7],a[2]*a[7]-a[1]*a[8],a[1]*a[5]-a[2]*a[4],
    a[5]*a[6]-a[3]*a[8],a[0]*a[8]-a[2]*a[6],a[2]*a[3]-a[0]*a[5],
    a[3]*a[7]-a[4]*a[6],a[1]*a[6]-a[0]*a[7],a[0]*a[4]-a[1]*a[3]};for(double&x:b)x/=d;return b;}
inline double norm(const Mat&a){double n=0;for(int i=0;i<3;++i)n=std::max(n,std::abs(a[3*i])+std::abs(a[3*i+1])+std::abs(a[3*i+2]));return n;}
// Scaling/Taylor/squaring remains well-defined for repeated eigenvalues and
// nilpotent shear matrices. No diagonalization or inverse of the strain rate.
inline void matrix_functions(Mat a,Mat& ep,Mat& em,Mat& sh){
    double n=norm(a);require(std::isfinite(n)&&n<1.e6,"unstable cell strain; reduce timestep");
    int scale=n>.25?static_cast<int>(std::ceil(std::log2(n/.25))):0;
    double divisor=std::ldexp(1.,scale);for(double&x:a)x/=divisor;
    ep=em=sh=identity();Mat term=identity(),a2=product(a,a),st=identity();
    for(int k=1;k<=22;++k){term=product(term,a);for(int j=0;j<9;++j){term[j]/=k;ep[j]+=term[j];em[j]+=(k%2?-1.:1.)*term[j];}}
    for(int k=1;k<=12;++k){st=product(st,a2);double z=(2.*k)*(2.*k+1.);for(int j=0;j<9;++j){st[j]/=z;sh[j]+=st[j];}}
    for(int k=0;k<scale;++k){Mat ch{};for(int j=0;j<9;++j)ch[j]=.5*(ep[j]+em[j]);sh=product(sh,ch);ep=product(ep,ep);em=product(em,em);}
    for(double x:ep)require(std::isfinite(x),"cell exponential overflow");
}

enum class Thermostat { None, Andersen, Langevin, PileL, PileG, SVR, GLE, NMGLE, NMGLEG, Piglet, TRPMD, FFL, NHC };
enum class Barostat { None, BZP, MTK, RGB, SCR };
struct Config {
    size_t atoms=0,beads=0; double beta=1,dt=.01;
    int threads=1;
    Vec masses; std::vector<uint8_t> fixed;
    Thermostat thermostat=Thermostat::PileL; Barostat barostat=Barostat::None;
    bool cayley=true,initialize_momenta=true,has_cell=false,fuse_modes=true,middle_npt=false;
    double gamma=1,gamma0=1,pile_scale=1,tau=1;
    uint64_t interval=10,seed=0;
    bool philox=false;
    size_t gle_size=0; Vec gle_T,gle_S,gle_Croot;
    size_t nhc_length=3,nhc_substeps=1;
    double pressure=0,baro_tau=1,baro_gamma=1,compressibility=0;
    std::array<int32_t,3> pbc{0,0,0};
    std::array<uint8_t,9> cell_mask{1,1,1,0,1,1,0,0,1};
    Mat reference_cell=identity(),external_stress{};
    // 0 free, 1 harmonic, 2 fractional cosine, 3 native thermal callback,
    // 4 existing C ABI v1, 5 existing ISO_C_BINDING Fortran batch v1.
    int potential=0; uintptr_t address=0,context=0; Vec parameters{1,1,1};
    // Explicit interpreter bridge only; it is not the trusted thermal C ABI.
    std::function<void(const Vec&,const Mat&,Vec&,Vec&,Vec&)> python_force;
};
struct State {
    Vec q,p,aux,energy,force,virial; Mat cell=identity(),cell_p{};
    uint64_t step=0,evaluations=0,rng_counter=0,rng_seed=0; bool force_valid=false;
    std::mt19937_64 rng; std::normal_distribution<double> gaussian;
};
struct CommonObservations {
    std::vector<uint8_t> mask;
    bool record_energy=true,check_energy=false,energy_parts=false,ring_radii=false;
};
struct Output { Vec samples,centroids,energy_parts; size_t frames=0; uint64_t completed=0; std::string error; };
static constexpr size_t ncolumns=18; // step,t,U,Kprim,Kcv,H_P,V,P,cell[9],K_fict
class Engine {
    friend struct KernelProfile; // Standalone benchmarks/efficiency/thermal_kernels.cpp.
public:
    const Config c;
    State s;
private:
    size_t d,l,mobile;
    double theta,baro_mass;
    RingFFT fft;
    Vec qn,pn,centroid_q,centroid_p,centroid_f;
    Vec parallel_xr,parallel_xi,parallel_tmp,pile_noise,partial_sums;
    Vec omega,a,b,cc,ah,bh,ch,y,z,noise,nhc_mass;
    std::vector<int32_t> status;
    Mat reference_inverse;
    double reference_volume;
    bool normal_thermostat() const {
        auto t=c.thermostat;
        return t==Thermostat::PileL||t==Thermostat::PileG||t==Thermostat::NMGLE||t==Thermostat::NMGLEG||t==Thermostat::Piglet||t==Thermostat::TRPMD;
    }
    bool gle()const{return c.thermostat==Thermostat::GLE||c.thermostat==Thermostat::NMGLE||c.thermostat==Thermostat::NMGLEG||c.thermostat==Thermostat::Piglet;}
    bool active(size_t j)const{return !c.fixed[j/3];}
    size_t max_threads()const{
#ifdef _OPENMP
        return static_cast<size_t>(c.threads);
#else
        return 1;
#endif
    }
    void ensure_parallel_scratch(){
        const size_t need=max_threads()*c.beads*8;
        if(parallel_xr.size()!=need){
            parallel_xr.resize(need);parallel_xi.resize(need);parallel_tmp.resize(2*need);
        }
    }
    template<class F> void parallel_for(size_t n,F&& fn){
#ifdef _OPENMP
        if(l<32768||max_threads()==1){for(size_t i=0;i<n;++i)fn(i);return;}
#pragma omp parallel for schedule(static) num_threads(c.threads)
        for(long long i=0;i<static_cast<long long>(n);++i)fn(static_cast<size_t>(i));
#else
        for(size_t i=0;i<n;++i)fn(i);
#endif
    }
    template<class F> double fixed_sum(size_t count,F&& term){
        // philox-v1 only: fixed 2048-element blocks in every thread layout.
        const size_t blocks=(count+2047)/2048;partial_sums.resize(blocks);
        parallel_for(blocks,[&](size_t block){
            double value=0.;const size_t end=std::min((block+1)*2048,count);
            for(size_t i=block*2048;i<end;++i)value+=term(i);
            partial_sums[block]=value;
        });
        return std::accumulate(partial_sums.begin(),partial_sums.end(),0.);
    }
    double g(){return s.gaussian(s.rng);}
    pyrpmd::philox::Stream begin_random(){
        if(!c.philox)return {};
        require(s.rng_counter<UINT64_MAX,"Philox event counter overflow");
        return pyrpmd::philox::Stream(s.rng_seed,s.step,s.rng_counter++);
    }
    double random_normal(const pyrpmd::philox::Stream& stream,size_t bead,size_t dof,uint32_t purpose,uint32_t component=0){
        if(!c.philox)return g();
        return stream.normal(static_cast<uint32_t>(bead),static_cast<uint32_t>(dof),purpose,component);
    }
    void means(){
        std::fill(centroid_q.begin(),centroid_q.end(),0.);
        std::fill(centroid_p.begin(),centroid_p.end(),0.);
        std::fill(centroid_f.begin(),centroid_f.end(),0.);
        if(fft.radix2&&l>=32768){
            // Each freedom still sums beads in order, including subnormals
            // in builds which enable fused multiply-add instructions.
            parallel_for((d+255)/256,[&](size_t block){
                const size_t begin=block*256,end=std::min(begin+256,d);
                for(size_t k=0;k<c.beads;++k)for(size_t j=begin;j<end;++j){
                    size_t i=k*d+j;
                    centroid_q[j]+=s.q[i]/c.beads;centroid_p[j]+=s.p[i]/c.beads;centroid_f[j]+=s.force[i]/c.beads;
                }
            });
        }else for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j){size_t i=k*d+j;centroid_q[j]+=s.q[i]/c.beads;centroid_p[j]+=s.p[i]/c.beads;centroid_f[j]+=s.force[i]/c.beads;}
    }
    void tile_fft(double* real,double* imag,size_t width,bool inverse){
        const size_t n=c.beads;
        for(size_t k=0;k<n;++k)if(k<size_t(fft.reverse[k]))for(size_t j=0;j<width;++j){
            std::swap(real[k*8+j],real[fft.reverse[k]*8+j]);
            std::swap(imag[k*8+j],imag[fft.reverse[k]*8+j]);
        }
        for(size_t length=2;length<=n;length*=2){
            const size_t half=length/2,stride=n/length;
            for(size_t start=0;start<n;start+=length)for(size_t k=0;k<half;++k){
                const size_t lo=(start+k)*8,hi=(start+k+half)*8;
                const double wr=fft.cosine[k*stride],wi=(inverse?1.:-1.)*fft.sine[k*stride];
                for(size_t j=0;j<width;++j){
                    const double tr=wr*real[hi+j]-wi*imag[hi+j];
                    const double ti=wr*imag[hi+j]+wi*real[hi+j];
                    real[hi+j]=real[lo+j]-tr;imag[hi+j]=imag[lo+j]-ti;
                    real[lo+j]+=tr;imag[lo+j]+=ti;
                }
            }
            if(length==n)break;
        }
    }
    // Eight adjacent freedoms share FFT butterflies. Each scalar follows
    // RingFFT's original operation order; SIMD operates across freedoms only.
    void modes_tile(const Vec& v,Vec& out,size_t first,bool inverse){
#ifdef _OPENMP
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
#else
        const size_t tid=0;
#endif
        const size_t n=c.beads,width=std::min(size_t(8),d-first);
        double* real=parallel_xr.data()+tid*n*8;
        double* imag=parallel_xi.data()+tid*n*8;
        const double sc=inverse?std::sqrt(double(n)):std::sqrt(1./n);
        const double pair=inverse?std::sqrt(n/2.):std::sqrt(2./n);
        if(!inverse){
            for(size_t k=0;k<n;++k)for(size_t j=0;j<width;++j){real[k*8+j]=v[k*d+first+j];imag[k*8+j]=0.;}
        }else{
            std::fill(real,real+n*8,0.);std::fill(imag,imag+n*8,0.);
            for(size_t j=0;j<width;++j)real[j]=sc*v[first+j];
            for(size_t k=1;k<=(n-1)/2;++k)for(size_t j=0;j<width;++j){
                real[k*8+j]=real[(n-k)*8+j]=pair*v[(2*k-1)*d+first+j];
                imag[k*8+j]=-pair*v[2*k*d+first+j];imag[(n-k)*8+j]=-imag[k*8+j];
            }
            if(n%2==0)for(size_t j=0;j<width;++j)real[(n/2)*8+j]=sc*v[(n-1)*d+first+j];
        }
        tile_fft(real,imag,width,inverse);
        if(inverse){
            for(size_t k=0;k<n;++k)for(size_t j=0;j<width;++j)out[k*d+first+j]=real[k*8+j]/n;
        }else{
            for(size_t j=0;j<width;++j)out[first+j]=sc*real[j];
            for(size_t k=1;k<=(n-1)/2;++k)for(size_t j=0;j<width;++j){
                out[(2*k-1)*d+first+j]=pair*real[k*8+j];out[2*k*d+first+j]=-pair*imag[k*8+j];
            }
            if(n%2==0)for(size_t j=0;j<width;++j)out[(n-1)*d+first+j]=sc*real[(n/2)*8+j];
        }
    }
    void to_modes(const Vec& v,Vec& out){
        if(c.beads==1){std::copy(v.begin(),v.end(),out.begin());return;}
        double sc=std::sqrt(1./c.beads),pair=std::sqrt(2./c.beads);
        ensure_parallel_scratch();
        if(fft.radix2&&c.beads>=8&&d>=24){parallel_for((d+7)/8,[&](size_t block){modes_tile(v,out,block*8,false);});return;}
        parallel_for(d,[&](size_t j){
#ifdef _OPENMP
            const size_t tid=static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid=0;
#endif
            double* real=parallel_xr.data()+tid*c.beads;
            double* imag=parallel_xi.data()+tid*c.beads;
            double* work=parallel_tmp.data()+2*tid*c.beads;
            for(size_t k=0;k<c.beads;++k){real[k]=v[k*d+j];imag[k]=0;}
            fft.transform(real,imag,work);out[j]=sc*real[0];
            for(size_t k=1;k<=(c.beads-1)/2;++k){out[(2*k-1)*d+j]=pair*real[k];out[2*k*d+j]=-pair*imag[k];}
            if(c.beads%2==0)out[(c.beads-1)*d+j]=sc*real[c.beads/2];
        });
    }
    void from_modes(const Vec& v,Vec& out){
        if(c.beads==1){std::copy(v.begin(),v.end(),out.begin());return;}
        double sc=std::sqrt(double(c.beads)),pair=std::sqrt(c.beads/2.);
        ensure_parallel_scratch();
        if(fft.radix2&&c.beads>=8&&d>=24){parallel_for((d+7)/8,[&](size_t block){modes_tile(v,out,block*8,true);});return;}
        parallel_for(d,[&](size_t j){
#ifdef _OPENMP
            const size_t tid=static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid=0;
#endif
            double* real=parallel_xr.data()+tid*c.beads;
            double* imag=parallel_xi.data()+tid*c.beads;
            double* work=parallel_tmp.data()+2*tid*c.beads;
            std::fill(real,real+c.beads,0.);std::fill(imag,imag+c.beads,0.);real[0]=sc*v[j];
            for(size_t k=1;k<=(c.beads-1)/2;++k){real[k]=real[c.beads-k]=pair*v[(2*k-1)*d+j];imag[k]=-pair*v[2*k*d+j];imag[c.beads-k]=-imag[k];}
            if(c.beads%2==0)real[c.beads/2]=sc*v[(c.beads-1)*d+j];
            fft.transform(real,imag,work,true);
            for(size_t k=0;k<c.beads;++k)out[k*d+j]=real[k];
        });
    }
    void free_modes(bool half,bool internal_only=false){
        const Vec& aa=half?ah:a;const Vec& bb=half?bh:b;const Vec& cl=half?ch:cc;
        const size_t begin=internal_only?1:0;
        parallel_for(c.beads-begin,[&](size_t offset){size_t k=begin+offset;
            for(size_t j=0;j<d;++j)if(active(j)){
                size_t i=k*d+j;double x=qn[i],p=pn[i],m=c.masses[j/3];
                qn[i]=aa[k]*x+bb[k]*p/m;pn[i]=aa[k]*p-cl[k]*m*x;
            }
        });
    }
    void pin(){
        // FFT roundoff must never move a pinned atom.
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(!active(j)){s.q[k*d+j]=fixed_q[j];s.p[k*d+j]=0.;}
    }
    void free_ring(bool half,bool internal_only=false){
        to_modes(s.q,qn);to_modes(s.p,pn);
        free_modes(half,internal_only);
        from_modes(qn,s.q);from_modes(pn,s.p);pin();
    }
    // Keep q and p in the SAME real, orthonormal basis across C-O-C/A-O-A.
    // Normal-mode baths need four transforms instead of ten, with the same map.
    // RNG draw ordering is shared with the unfused reference implementation.
    void philox_middle_tile(size_t first,const Vec& decay,const Vec& amplitude,
                            const std::vector<pyrpmd::philox::Stream>& streams){
#ifdef _OPENMP
        const size_t tid=static_cast<size_t>(omp_get_thread_num());
#else
        const size_t tid=0;
#endif
        const size_t n=c.beads,width=std::min(size_t(8),d-first);
        double* real=parallel_xr.data()+tid*n*8;
        double* imag=parallel_xi.data()+tid*n*8;
        const double sc=std::sqrt(1./n),pair=std::sqrt(2./n);
        // The optional v1 stream also fixes this packed two-real-FFT rounding
        // path. Legacy mt19937 keeps its original four separate transforms.
        for(size_t k=0;k<n;++k)for(size_t j=0;j<width;++j){real[k*8+j]=s.q[k*d+first+j];imag[k*8+j]=s.p[k*d+first+j];}
        tile_fft(real,imag,width,false);
        for(size_t j=0;j<width;++j){qn[first+j]=sc*real[j];pn[first+j]=sc*imag[j];}
        for(size_t k=1;k<=(n-1)/2;++k)for(size_t j=0;j<width;++j){
            qn[(2*k-1)*d+first+j]=pair*(.5*(real[k*8+j]+real[(n-k)*8+j]));
            qn[2*k*d+first+j]=-pair*(.5*(imag[k*8+j]-imag[(n-k)*8+j]));
            pn[(2*k-1)*d+first+j]=pair*(.5*(imag[k*8+j]+imag[(n-k)*8+j]));
            pn[2*k*d+first+j]=-pair*(.5*(real[(n-k)*8+j]-real[k*8+j]));
        }
        if(n%2==0)for(size_t j=0;j<width;++j){qn[(n-1)*d+first+j]=sc*real[(n/2)*8+j];pn[(n-1)*d+first+j]=sc*imag[(n/2)*8+j];}
        for(size_t k=0;k<n;++k){
            double draws[8];
            const double rate=k==0?(c.thermostat==Thermostat::TRPMD?0.:c.gamma0):2*c.pile_scale*omega[k];
            if(rate!=0.)for(size_t j=0;j<width;j+=2){
                const auto pair=streams[k].normal_pair(static_cast<uint32_t>(k),static_cast<uint32_t>((first+j)/2),10);
                draws[j]=pair[0];if(j+1<width)draws[j+1]=pair[1];
            }
            for(size_t j=0;j<width;++j)if(active(first+j)){
                const size_t i=k*d+first+j;const double m=c.masses[(first+j)/3];
                double x=qn[i],p=pn[i];
                qn[i]=ah[k]*x+bh[k]*p/m;pn[i]=ah[k]*p-ch[k]*m*x;
                if(rate!=0.)pn[i]=decay[k]*pn[i]+amplitude[k]*std::sqrt(theta*m)*draws[j];
                x=qn[i];p=pn[i];
                qn[i]=ah[k]*x+bh[k]*p/m;pn[i]=ah[k]*p-ch[k]*m*x;
            }
        }
        const double isc=std::sqrt(double(n)),ipair=std::sqrt(n/2.);
        for(size_t j=0;j<width;++j){real[j]=isc*qn[first+j];imag[j]=isc*pn[first+j];}
        for(size_t k=1;k<=(n-1)/2;++k)for(size_t j=0;j<width;++j){
            const double qr=ipair*qn[(2*k-1)*d+first+j],qi=-ipair*qn[2*k*d+first+j];
            const double pr=ipair*pn[(2*k-1)*d+first+j],pi=-ipair*pn[2*k*d+first+j];
            real[k*8+j]=qr-pi;imag[k*8+j]=qi+pr;
            real[(n-k)*8+j]=qr+pi;imag[(n-k)*8+j]=-qi+pr;
        }
        if(n%2==0)for(size_t j=0;j<width;++j){real[(n/2)*8+j]=isc*qn[(n-1)*d+first+j];imag[(n/2)*8+j]=isc*pn[(n-1)*d+first+j];}
        tile_fft(real,imag,width,true);
        for(size_t k=0;k<n;++k)for(size_t j=0;j<width;++j){s.q[k*d+first+j]=real[k*8+j]/n;s.p[k*d+first+j]=imag[k*8+j]/n;}
    }
    void pile_middle_tiles(){
        // Preserve mode-major mt19937 draw order, then perform each tile's
        // complete C-O-C map while its four modal arrays are still in cache.
        Vec decay(c.beads),amplitude(c.beads);
        std::vector<pyrpmd::philox::Stream> streams(c.beads);
        for(size_t k=0;k<c.beads;++k){
            const double rate=k==0?(c.thermostat==Thermostat::TRPMD?0.:c.gamma0):2*c.pile_scale*omega[k];
            decay[k]=std::exp(-rate*c.dt);amplitude[k]=std::sqrt(-std::expm1(-2*rate*c.dt));
            if(rate==0)continue;
            if(c.philox)streams[k]=begin_random();
            else for(size_t j=0;j<d;++j)if(active(j))pile_noise[k*d+j]=g();
        }
        ensure_parallel_scratch();
        parallel_for((d+7)/8,[&](size_t block){
            const size_t first=block*8,end=std::min(first+8,d);
            if(c.philox){philox_middle_tile(first,decay,amplitude,streams);return;}
            modes_tile(s.q,qn,first,false);modes_tile(s.p,pn,first,false);
            for(size_t k=0;k<c.beads;++k)for(size_t j=first;j<end;++j)if(active(j)){
                const size_t i=k*d+j;const double m=c.masses[j/3];
                double x=qn[i],p=pn[i];
                qn[i]=ah[k]*x+bh[k]*p/m;pn[i]=ah[k]*p-ch[k]*m*x;
                const double rate=k==0?(c.thermostat==Thermostat::TRPMD?0.:c.gamma0):2*c.pile_scale*omega[k];
                if(rate!=0.){
                    const double draw=c.philox?random_normal(streams[k],k,j,10):pile_noise[i];
                    pn[i]=decay[k]*pn[i]+amplitude[k]*std::sqrt(theta*m)*draw;
                }
                x=qn[i];p=pn[i];
                qn[i]=ah[k]*x+bh[k]*p/m;pn[i]=ah[k]*p-ch[k]*m*x;
            }
            modes_tile(qn,s.q,first,true);modes_tile(pn,s.p,first,true);
        });
        pin();
    }
    void free_bath_free(){
        if(fft.radix2&&c.beads>=8&&d>=24&&(c.thermostat==Thermostat::PileL||c.thermostat==Thermostat::TRPMD)){
            pile_middle_tiles();return;
        }
        to_modes(s.q,qn);to_modes(s.p,pn);
        free_modes(true);bath(c.dt,true);free_modes(true);
        from_modes(qn,s.q);from_modes(pn,s.p);pin();
    }
    Vec fixed_q;
    void refresh(){
        double nan=std::numeric_limits<double>::quiet_NaN();
        std::fill(s.energy.begin(),s.energy.end(),nan);std::fill(s.force.begin(),s.force.end(),nan);std::fill(s.virial.begin(),s.virial.end(),nan);
        if(c.potential<=2){
            std::fill(s.energy.begin(),s.energy.end(),0.);std::fill(s.force.begin(),s.force.end(),0.);std::fill(s.virial.begin(),s.virial.end(),0.);
            if(c.potential==1){double k=c.parameters[0];parallel_for(c.beads,[&](size_t b0){for(size_t atom=0;atom<c.atoms;++atom){
                size_t i=b0*d+3*atom;for(int j=0;j<3;++j){s.energy[b0]+=.5*k*s.q[i+j]*s.q[i+j];s.force[i+j]=-k*s.q[i+j];}
                for(int j=0;j<3;++j)for(int k0=0;k0<3;++k0)s.virial[9*b0+3*j+k0]+=s.q[i+j]*s.force[i+k0];
            }});}
            if(c.potential==2){Mat inv=inverse(s.cell);double pi=std::acos(-1.);
                parallel_for(c.beads,[&](size_t b0){for(size_t atom=0;atom<c.atoms;++atom){size_t i=b0*d+3*atom;double grad[3]{};
                    for(int j=0;j<3;++j){double frac=0;for(int k0=0;k0<3;++k0)frac+=s.q[i+k0]*inv[3*k0+j];double phase=2*pi*frac;s.energy[b0]+=.5*c.parameters[j]*(1-std::cos(phase));grad[j]=pi*c.parameters[j]*std::sin(phase);}
                    for(int j=0;j<3;++j)for(int k0=0;k0<3;++k0)s.force[i+j]-=inv[3*j+k0]*grad[k0];
                }});
            }
        }else if(c.potential==6){
            c.python_force(s.q,s.cell,s.energy,s.force,s.virial);
        }else if(c.potential==3){
            char message[512]{};
            auto fn=reinterpret_cast<pyrpmd_thermal_evaluate_v1>(c.address);
            int32_t rc=fn(reinterpret_cast<void*>(c.context),static_cast<int64_t>(c.beads),static_cast<int64_t>(c.atoms),s.q.data(),s.cell.data(),c.pbc.data(),s.energy.data(),s.force.data(),s.virial.data(),message,512);
            message[511]=0;if(rc)throw std::runtime_error(std::string("native thermal PES failed: ")+message);
        }else{
            std::fill(status.begin(),status.end(),-1);
            if(c.potential==4){
                using F=pyrpmd_rc_v1(*)(pyrpmd_pes_context_v1*,const pyrpmd_pes_input_v1*,pyrpmd_pes_output_v1*,pyrpmd_error_v1*);
                pyrpmd_pes_input_v1 in{1,sizeof(pyrpmd_pes_input_v1),1,1,static_cast<int64_t>(c.beads),static_cast<int64_t>(c.atoms),s.q.data(),static_cast<int64_t>(l)};
                pyrpmd_pes_output_v1 out{1,sizeof(pyrpmd_pes_output_v1),s.energy.data(),static_cast<int64_t>(c.beads),s.force.data(),static_cast<int64_t>(l),status.data(),static_cast<int64_t>(c.beads),nullptr,0,0};
                char message[512]{};pyrpmd_error_v1 err{message,512,0};
                int32_t rc=reinterpret_cast<F>(c.address)(reinterpret_cast<pyrpmd_pes_context_v1*>(c.context),&in,&out,&err);message[511]=0;
                if(rc)throw std::runtime_error(std::string("native C ABI PES failed: ")+message);
            }else{
                using F=void(*)(int32_t,const double*,double*,double*,int32_t*);
                reinterpret_cast<F>(c.address)(static_cast<int32_t>(c.beads),s.q.data(),s.energy.data(),s.force.data(),status.data());
            }
            for(int32_t rc:status)require(rc==0,"native PES returned a failed bead");
            // Nonperiodic v1 supplies no virial. This path cannot be used for NPT.
            std::fill(s.virial.begin(),s.virial.end(),0.);
        }
        require(finite(s.energy)&&finite(s.force)&&finite(s.virial),"native PES left missing/nonfinite energy, force or virial");
        ++s.evaluations;s.force_valid=true;
    }
    void kick(double h){
        if(mobile==c.atoms)parallel_for(l,[&](size_t i){s.p[i]+=h*s.force[i];});
        else parallel_for(l,[&](size_t i){if(active(i%d))s.p[i]+=h*s.force[i];});
    }
    void svr(Vec& p,size_t begin,size_t end,double h){
        size_t dof=0;double k=0;
        if(c.philox&&l>=32768){
            dof=3*mobile*((end-begin)/d);
            k=fixed_sum(end-begin,[&](size_t offset){const size_t i=begin+offset;return active(i%d)?p[i]*p[i]/(2*c.masses[(i%d)/3]):0.;});
        }else for(size_t i=begin;i<end;++i)if(active(i%d)){++dof;k+=p[i]*p[i]/(2*c.masses[(i%d)/3]);}
        if(!dof)return;
        const auto epoch=begin_random();
        if(k<=std::numeric_limits<double>::min()){
            // Exact zero-energy limit of the CIR kinetic-energy transition:
            // K' = theta*(1-exp(-h/tau))*chi_square(dof)/2, not a full redraw.
            double variance=theta*(-std::expm1(-h/c.tau));
            for(size_t i=begin;i<end;++i)if(active(i%d))p[i]=std::sqrt(c.masses[(i%d)/3]*variance)*random_normal(epoch,i/d,i%d,3);
            return;
        }
        double decay=std::exp(-h/c.tau),one=-std::expm1(-h/c.tau),r=random_normal(epoch,0,0,3);
        double chi=0;if(dof>1){
            if(c.philox){for(size_t i=0,draw=0;i<end-begin;++i)if(active((begin+i)%d)){
                if(draw++>0) {const double z=random_normal(epoch,(begin+i)/d,(begin+i)%d,3,1);chi+=z*z;}
            }}else{std::gamma_distribution<double> dist(.5*(dof-1),2.);chi=dist(s.rng);}
        }
        double newk=decay*k+.5*one*theta*(r*r+chi)+r*std::sqrt(2*decay*one*theta*k);
        require(newk>=-1.e-13*theta&&std::isfinite(newk),"invalid SVR kinetic energy");
        double scale=std::sqrt(std::max(0.,newk)/k);
        if(one>0&&r+std::sqrt(2*decay*k/(one*theta))<0)scale=-scale;
        for(size_t i=begin;i<end;++i)if(active(i%d))p[i]*=scale;
    }
    void ou(Vec& p,size_t begin,size_t end,double rate,double h){
        double decay=std::exp(-rate*h),amplitude=std::sqrt(-std::expm1(-2*rate*h));
        if(rate==0)return;
        const auto epoch=begin_random();
        if(c.philox)parallel_for((end-begin)/3,[&](size_t offset){
            const size_t i=begin+3*offset,atom=(i%d)/3;
            if(!c.fixed[atom]){
                const double scale=amplitude*std::sqrt(theta*c.masses[atom]);
                for(size_t j=0;j<3;++j)p[i+j]=decay*p[i+j]+scale*random_normal(epoch,i/d,i%d+j,10);
            }
        });
        // Keep this legacy expression: extracting the scale changes FMA rounding.
        else for(size_t i=begin;i<end;++i)if(active(i%d))p[i]=decay*p[i]+amplitude*std::sqrt(theta*c.masses[(i%d)/3])*g();
    }
    // Global Nose--Hoover chain, in the same bead Hamiltonian convention as
    // the public Python thermostat: kT_P=P/beta and 3*P*N_mobile freedoms.
    // The chain occupies aux=[eta[chain_length], momentum[chain_length]],
    // so it shares the complete transactional/checkpoint state with GLE.
    void nhc_kick(Vec& p,size_t j,double h){
        size_t n=c.nhc_length;
        double damping=j+1<n?std::exp(-.5*h*s.aux[n+j+1]/nhc_mass[j+1]):1.;
        double force=-theta;
        if(j==0){
            force=-3.*c.beads*mobile*theta;
            for(size_t i=0;i<l;++i)if(active(i%d))force+=p[i]*p[i]/c.masses[(i%d)/3];
        }else force+=s.aux[n+j-1]*s.aux[n+j-1]/nhc_mass[j-1];
        s.aux[n+j]=damping*(damping*s.aux[n+j]+h*force);
    }
    void nhc_bath(Vec& p,double dt){
        size_t n=c.nhc_length;
        const double w=1./(2.-std::cbrt(2.));
        const std::array<double,3> weights{w,1.-2.*w,w};
        for(size_t sub=0;sub<c.nhc_substeps;++sub)for(double weight:weights){
            double h=weight*dt/c.nhc_substeps;
            for(size_t j=n;j-->0;)nhc_kick(p,j,.5*h);
            for(size_t j=0;j<n;++j)s.aux[j]+=h*s.aux[n+j]/nhc_mass[j];
            double scale=std::exp(-h*s.aux[n]/nhc_mass[0]);
            for(size_t i=0;i<l;++i)if(active(i%d))p[i]*=scale;
            for(size_t j=0;j<n;++j)nhc_kick(p,j,.5*h);
        }
    }
    void bath(double h,bool modes_ready=false){
        auto kind=c.thermostat;if(kind==Thermostat::None)return;
        bool nm=normal_thermostat();Vec& p=nm?pn:s.p;if(nm&&!modes_ready)to_modes(s.p,pn);
        if(kind==Thermostat::Andersen){
            if(s.step%c.interval==0){
                const auto epoch=begin_random();
                if(c.philox)parallel_for(l,[&](size_t i){if(active(i%d))p[i]=std::sqrt(theta*c.masses[(i%d)/3])*random_normal(epoch,i/d,i%d,2);});
                else for(size_t i=0;i<l;++i)if(active(i%d))p[i]=std::sqrt(theta*c.masses[(i%d)/3])*g();
            }
        }else if(kind==Thermostat::NHC)nhc_bath(p,h);
        else if(kind==Thermostat::SVR)svr(p,0,l,h);
        else if(kind==Thermostat::Langevin)ou(p,0,l,c.gamma,h);
        else if(kind==Thermostat::FFL){
            double decay=std::exp(-c.gamma*h),amp=std::sqrt(-std::expm1(-2*c.gamma*h));
            // FFL rescale variant: retain direction, update Maxwellian speed.
            const auto epoch=begin_random();
            if(c.philox)parallel_for(c.atoms*c.beads,[&](size_t atom_index){size_t i=atom_index*3;if(active(i%d)){
                double old[3]{p[i],p[i+1],p[i+2]},old2=0,new2=0;
                for(int j=0;j<3;++j){old2+=old[j]*old[j];p[i+j]=decay*old[j]+amp*std::sqrt(theta*c.masses[(i%d)/3])*random_normal(epoch,i/d,i%d,5,j);new2+=p[i+j]*p[i+j];}
                if(old2>std::numeric_limits<double>::min()){double r=std::sqrt(new2/old2);for(int j=0;j<3;++j)p[i+j]=old[j]*r;}
            }});
            else for(size_t i=0;i<l;i+=3)if(active(i%d)){
                double old[3]{p[i],p[i+1],p[i+2]},old2=0,new2=0;
                for(int j=0;j<3;++j){old2+=old[j]*old[j];p[i+j]=decay*old[j]+amp*std::sqrt(theta*c.masses[(i%d)/3])*g();new2+=p[i+j]*p[i+j];}
                if(old2>std::numeric_limits<double>::min()){double r=std::sqrt(new2/old2);for(int j=0;j<3;++j)p[i+j]=old[j]*r;}
            }
        }else if(gle()){
            size_t n=c.gle_size;
            const auto epoch=begin_random();
            if(c.philox)parallel_for(l,[&](size_t i){if(active(i%d)){
                size_t mode=nm?i/d:0;
                if(kind==Thermostat::NMGLEG&&mode==0)return;
                double local_y[64],local_z[64],local_noise[64];
                size_t offset=mode*n*n;double scale=std::sqrt(theta*c.masses[(i%d)/3]);
                local_y[0]=p[i]/scale;for(size_t j=1;j<n;++j)local_y[j]=s.aux[i*(n-1)+j-1];
                for(size_t j=0;j<n;++j)local_noise[j]=random_normal(epoch,i/d,i%d,4,static_cast<uint32_t>(j));
                for(size_t j=0;j<n;++j){local_z[j]=0;for(size_t k=0;k<n;++k)local_z[j]+=c.gle_T[offset+j*n+k]*local_y[k]+c.gle_S[offset+j*n+k]*local_noise[k];}
                p[i]=scale*local_z[0];for(size_t j=1;j<n;++j)s.aux[i*(n-1)+j-1]=local_z[j];
            }});
            else for(size_t i=0;i<l;++i)if(active(i%d)){
                size_t mode=nm?i/d:0;
                if(kind==Thermostat::NMGLEG&&mode==0)continue;
                size_t offset=mode*n*n;double scale=std::sqrt(theta*c.masses[(i%d)/3]);
                y[0]=p[i]/scale;for(size_t j=1;j<n;++j)y[j]=s.aux[i*(n-1)+j-1];
                for(size_t j=0;j<n;++j)noise[j]=random_normal(epoch,i/d,i%d,4,static_cast<uint32_t>(j));
                for(size_t j=0;j<n;++j){z[j]=0;for(size_t k=0;k<n;++k)z[j]+=c.gle_T[offset+j*n+k]*y[k]+c.gle_S[offset+j*n+k]*noise[k];}
                p[i]=scale*z[0];for(size_t j=1;j<n;++j)s.aux[i*(n-1)+j-1]=z[j];
            }
            if(kind==Thermostat::NMGLEG)svr(p,0,d,h);
        }else{
            if(kind==Thermostat::PileG)svr(p,0,d,h);
            else if(kind!=Thermostat::TRPMD)ou(p,0,d,c.gamma0,h);
            for(size_t k=1;k<c.beads;++k)ou(p,k*d,(k+1)*d,2*c.pile_scale*omega[k],h);
        }
        if(nm&&!modes_ready)from_modes(pn,s.p);
        if(!modes_ready)for(size_t i=0;i<l;++i)if(!active(i%d))s.p[i]=0.;
    }
    Mat stress_volume(){
        means();Mat v{};
        for(size_t k=0;k<c.beads;++k)for(int i=0;i<9;++i)v[i]+=s.virial[9*k+i];
        for(size_t k=0;k<c.beads;++k)for(size_t atom=0;atom<c.atoms;++atom)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            v[3*i+j]-=(s.q[k*d+3*atom+i]-centroid_q[3*atom+i])*s.force[k*d+3*atom+j];
        for(size_t atom=0;atom<c.atoms;++atom)for(int i=0;i<3;++i)
            v[3*i+i]+=c.beads*centroid_p[3*atom+i]*centroid_p[3*atom+i]/c.masses[atom];
        return v;
    }
    void baro_bath(double h){
        if(c.barostat==Barostat::None||c.barostat==Barostat::SCR||c.baro_gamma==0)return;
        double decay=std::exp(-c.baro_gamma*h),amp=std::sqrt(baro_mass*theta*(-std::expm1(-2*c.baro_gamma*h)));
        if(c.philox){
            const auto epoch=begin_random();
            if(c.barostat==Barostat::BZP)s.cell_p[0]=decay*s.cell_p[0]+amp*random_normal(epoch,0,0,6);
            else for(int i=0;i<9;++i)if(c.cell_mask[i])s.cell_p[i]=decay*s.cell_p[i]+amp*random_normal(epoch,0,static_cast<size_t>(i),6);
        }else{
            if(c.barostat==Barostat::BZP)s.cell_p[0]=decay*s.cell_p[0]+amp*g();
            else for(int i=0;i<9;++i)if(c.cell_mask[i])s.cell_p[i]=decay*s.cell_p[i]+amp*g();
        }
    }
    void coupled_kick(double h){
        if(c.barostat==Barostat::None||c.barostat==Barostat::SCR){kick(h);return;}
        Mat v=stress_volume();double vol=det(s.cell);
        if(c.barostat==Barostat::BZP){
            double a0=0,a1=0;for(size_t j=0;j<d;++j){double f=centroid_f[j];a0+=centroid_p[j]*f/c.masses[j/3];a1+=f*f/c.masses[j/3];}
            s.cell_p[0]+=h*(v[0]+v[4]+v[8]-3*c.beads*c.pressure*vol+3*theta)+c.beads*h*h*(a0+h*a1/3.);
        }else{
            Mat external{};
            if(c.barostat==Barostat::MTK){external[0]=external[4]=external[8]=c.pressure*vol;}
            else {Mat strain=product(transpose(s.cell),reference_inverse);external=product(product(strain,c.external_stress),transpose(strain));for(double& x:external)x*=reference_volume;}
            for(int i=0;i<3;++i)for(int j=i;j<3;++j)if(c.cell_mask[3*i+j]){
                double count=0;if(i==j)for(int k=i;k<3;++k)count+=c.cell_mask[3*i+k];
                double a0=0,a1=0;for(size_t atom=0;atom<c.atoms;++atom){a0+=centroid_f[3*atom+i]*centroid_p[3*atom+j]/c.masses[atom];a1+=centroid_f[3*atom+i]*centroid_f[3*atom+j]/c.masses[atom];}
                s.cell_p[3*i+j]+=h*(v[3*i+j]-c.beads*external[3*i+j]+theta*count)+c.beads*h*h*(a0+h*a1/3.);
            }
        }
        kick(h);
    }
    void cell_drift(double h){
        means();Mat ep{},em{},sh{};
        if(c.barostat==Barostat::BZP){
            double x=h*s.cell_p[0]/baro_mass;require(std::abs(x)<50,"unstable BZP cell; reduce timestep");
            double sinhc=std::abs(x)<1.e-5?1+x*x/6+x*x*x*x/120:std::sinh(x)/x;
            for(int j=0;j<3;++j){ep[3*j+j]=std::exp(x);em[3*j+j]=std::exp(-x);sh[3*j+j]=h*sinhc;}
        }else{
            Mat v=s.cell_p;for(double& x:v)x*=h/baro_mass;matrix_functions(v,ep,em,sh);for(double& x:sh)x*=h;
        }
        for(size_t atom=0;atom<c.atoms;++atom){double dq[3]{},dp[3]{};
            for(int i=0;i<3;++i){for(int j=0;j<3;++j){dq[i]+=ep[3*i+j]*centroid_q[3*atom+j]+sh[3*i+j]*centroid_p[3*atom+j]/c.masses[atom];dp[i]+=em[3*i+j]*centroid_p[3*atom+j];}dq[i]-=centroid_q[3*atom+i];dp[i]-=centroid_p[3*atom+i];}
            for(size_t k=0;k<c.beads;++k)for(int i=0;i<3;++i){s.q[k*d+3*atom+i]+=dq[i];s.p[k*d+3*atom+i]+=dp[i];}
        }
        s.cell=product(s.cell,transpose(ep));require(det(s.cell)>0&&std::isfinite(det(s.cell)),"invalid propagated cell");
    }
    void scr_drift(){
        // Bernetti--Bussi sqrt(V) Euler proposal, with symmetric drift/rescale.
        // First-order weak finite-step bias: no claim of Metropolis correction.
        Mat v=stress_volume();double vol=det(s.cell),lam=std::sqrt(vol);
        double press=(v[0]+v[4]+v[8])/(3*vol),D=theta*c.compressibility/(4*c.baro_tau);
        double force=-2*lam*(c.pressure-press-theta/(2*vol));
        const auto epoch=begin_random();
        double next=lam+D*force*c.dt/theta+std::sqrt(2*D*c.dt)*random_normal(epoch,0,0,7);
        require(next>0&&std::isfinite(next),"SCR proposed non-positive sqrt(volume); reduce timestep");
        double halfscale=std::exp(std::log(next/lam)/3.);
        require(halfscale>.01&&halfscale<100,"unstable SCR cell; reduce timestep");
        for(size_t j=0;j<d;++j){s.q[j]+=.5*c.dt*s.p[j]/c.masses[j/3];s.q[j]*=halfscale;s.p[j]/=halfscale;}
        for(double& x:s.cell)x*=halfscale;
        for(size_t j=0;j<d;++j){s.q[j]*=halfscale;s.p[j]/=halfscale;s.q[j]+=.5*c.dt*s.p[j]/c.masses[j/3];}
        for(double& x:s.cell)x*=halfscale;
    }
    // Symmetric centroid BZP wrapper around a fixed-volume BCOCB/BAOAB step.
    // Keep the old coupled-kick NPT map unchanged unless explicitly selected.
    // Piston kicks here do NOT contain the coupled physical-kick h^2/h^3 terms.
    void piston_kick(double h){
        Mat v=stress_volume();
        s.cell_p[0]+=h*(v[0]+v[4]+v[8]-3*c.beads*c.pressure*det(s.cell)+3*theta);
    }
    void centroid_dilate(double h){
        means();double x=h*s.cell_p[0]/baro_mass;
        require(std::isfinite(x)&&std::abs(x)<50,"unstable BZP dilation; reduce timestep");
        double a=std::exp(x),b=std::exp(-x);
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j){
            s.q[k*d+j]+=(a-1)*centroid_q[j];
            s.p[k*d+j]+=(b-1)*centroid_p[j];
        }
        for(double& v:s.cell)v*=a;
        require(det(s.cell)>0&&std::isfinite(det(s.cell)),"invalid propagated cell");
        s.force_valid=false;
    }
    void one_step(){
        if(!s.force_valid)refresh();
        if(c.middle_npt){
            baro_bath(.5*c.dt);piston_kick(.5*c.dt);centroid_dilate(.5*c.dt);
            refresh();kick(.5*c.dt);
            if(c.fuse_modes)free_bath_free();
            else {free_ring(true);bath(c.dt);free_ring(true);}
            refresh();kick(.5*c.dt);centroid_dilate(.5*c.dt);
            refresh();piston_kick(.5*c.dt);baro_bath(.5*c.dt);
        }else if(c.barostat==Barostat::None&&c.thermostat==Thermostat::NHC){
            // Preserve the established symmetric O-B-C-B-O chain splitting.
            bath(.5*c.dt);kick(.5*c.dt);free_ring(false);
            refresh();kick(.5*c.dt);bath(.5*c.dt);
        }else if(c.barostat==Barostat::None){
            kick(.5*c.dt);
            if(c.fuse_modes&&(normal_thermostat()||c.thermostat==Thermostat::None))free_bath_free();
            else {free_ring(true);bath(c.dt);free_ring(true);}
            refresh();kick(.5*c.dt);
        }else{
            // Andersen redraws once at the full-step boundary, not twice.
            if(c.thermostat==Thermostat::Andersen)bath(c.dt);else bath(.5*c.dt);
            baro_bath(.5*c.dt);coupled_kick(.5*c.dt);
            if(c.barostat==Barostat::SCR)scr_drift();
            else {cell_drift(.5*c.dt);free_ring(false,true);cell_drift(.5*c.dt);}
            refresh();coupled_kick(.5*c.dt);baro_bath(.5*c.dt);
            if(c.thermostat!=Thermostat::Andersen)bath(.5*c.dt);
        }
        require(finite(s.q)&&finite(s.p)&&finite(s.aux),"nonfinite trajectory state; reduce timestep");
        for(double x:s.cell_p)require(std::isfinite(x),"nonfinite cell momentum");
        ++s.step;
    }
    std::array<double,3> mode_energy_sums(bool virial){
        partial_sums.resize(3*c.beads);const double wn=c.beads/c.beta;
        parallel_for(c.beads,[&](size_t k){
            double spring=0.,kinetic=0.,cv=0.;const size_t next=(k+1)%c.beads;
            for(size_t j=0;j<d;++j)if(active(j)){
                const size_t i=k*d+j;const double delta=s.q[i]-s.q[next*d+j];
                spring+=.5*c.masses[j/3]*wn*wn*delta*delta;
                kinetic+=.5*s.p[i]*s.p[i]/c.masses[j/3];
                if(virial)cv+=.5*(s.q[i]-centroid_q[j])*s.force[i]/c.beads;
            }
            partial_sums[3*k]=spring;partial_sums[3*k+1]=kinetic;partial_sums[3*k+2]=cv;
        });
        std::array<double,3> total{};
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<3;++j)total[j]+=partial_sums[3*k+j];
        return total;
    }
    void observe(double* row,double* cq,bool moments=false){
        means();double spring=0,kinetic=0,cv=1.5*mobile/c.beta,U=std::accumulate(s.energy.begin(),s.energy.end(),0.);
        double wn=c.beads/c.beta;
        if(c.philox&&l>=32768){
            const auto sums=mode_energy_sums(true);spring=sums[0];kinetic=sums[1];cv-=sums[2];
        }else for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(active(j)){
            size_t i=k*d+j;double delta=s.q[i]-s.q[((k+1)%c.beads)*d+j];spring+=.5*c.masses[j/3]*wn*wn*delta*delta;
            kinetic+=.5*s.p[i]*s.p[i]/c.masses[j/3];cv-=.5*(s.q[i]-centroid_q[j])*s.force[i]/c.beads;
        }
        row[0]=static_cast<double>(s.step);row[1]=s.step*c.dt;row[2]=U/c.beads;
        row[3]=1.5*mobile*c.beads/c.beta-spring/c.beads;row[4]=cv;row[5]=kinetic+spring+U;
        row[6]=c.has_cell?det(s.cell):std::numeric_limits<double>::quiet_NaN();
        row[7]=std::numeric_limits<double>::quiet_NaN();
        if(c.has_cell){Mat stress=stress_volume();row[7]=(stress[0]+stress[4]+stress[8])/(3*c.beads*row[6]);}
        std::copy(s.cell.begin(),s.cell.end(),row+8);row[17]=kinetic;
        // Finite coordinates/momenta do not guarantee finite reductions (for
        // example p*p can overflow). Reject the observation inside run's
        // transaction so neither the state nor a partial stream is accepted.
        for(size_t i=0;i<ncolumns;++i){
            if(!c.has_cell&&(i==6||i==7))continue; // undefined molecular volume/pressure
            require(std::isfinite(row[i]),"nonfinite ensemble observation; reduce timestep or check initial state");
        }
        if(cq){
            std::copy(centroid_q.begin(),centroid_q.end(),cq);
            if(moments){
                for(size_t j=0;j<d;++j)cq[d+j]=centroid_p[j]/c.masses[j/3];
                std::fill(cq+2*d,cq+2*d+c.atoms,0.);
                for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j){
                    const double delta=s.q[k*d+j]-centroid_q[j];
                    cq[2*d+j/3]+=delta*delta/c.beads;
                }
            }
            const size_t width=moments?2*d+c.atoms:d;
            for(size_t i=0;i<width;++i)require(std::isfinite(cq[i]),"nonfinite centroid observation; check trajectory moments");
        }
    }
    // The common propagation API observes only explicitly selected quantities.
    // Thermal estimator columns are deliberately not evaluated here: finite
    // physical state need not have finite squared moments or virial estimators.
    double nuclear_energy(double* parts=nullptr){
        double spring=0,kinetic=0,U=std::accumulate(s.energy.begin(),s.energy.end(),0.);
        const double wn=c.beads/c.beta;
        if(c.philox&&l>=32768){
            const auto sums=mode_energy_sums(false);spring=sums[0];kinetic=sums[1];
        }else for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(active(j)){
            const size_t i=k*d+j;
            const double delta=s.q[i]-s.q[((k+1)%c.beads)*d+j];
            spring+=.5*c.masses[j/3]*wn*wn*delta*delta;
            kinetic+=.5*s.p[i]*s.p[i]/c.masses[j/3];
        }
        const double total=U+kinetic+spring;
        require(std::isfinite(total),"nonfinite total ring energy");
        if(parts){
            double centroid=0,internal=0;
            for(size_t j=0;j<d;++j)if(active(j))centroid+=.5*centroid_p[j]*centroid_p[j]/c.masses[j/3];
            for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(active(j)){
                const double delta=s.p[k*d+j]-centroid_p[j];
                internal+=.5*delta*delta/c.masses[j/3];
            }
            parts[0]=U/c.beads;parts[1]=centroid;parts[2]=internal/c.beads;parts[3]=spring/c.beads;
            for(size_t i=0;i<4;++i)require(std::isfinite(parts[i]),"nonfinite nuclear energy diagnostics");
        }
        return total;
    }
    void observe_common(double* row,double* cq,double* parts,const CommonObservations& options){
        const double nan=std::numeric_limits<double>::quiet_NaN();
        std::fill(row,row+ncolumns,nan);
        std::fill(cq,cq+2*d+c.atoms,nan);
        row[0]=static_cast<double>(s.step);row[1]=s.step*c.dt;
        means();
        std::copy(centroid_q.begin(),centroid_q.end(),cq);
        // Record momenta directly. Dividing by masses to form velocities would
        // add an unrequested reduction and can overflow a finite momentum.
        std::copy(centroid_p.begin(),centroid_p.end(),cq+d);
        for(size_t j=0;j<2*d;++j)require(std::isfinite(cq[j]),"nonfinite centroid observation");
        if(options.record_energy||options.energy_parts){
            row[5]=nuclear_energy(parts);
            if(parts)row[2]=parts[0];
        }
        if(options.ring_radii){
            std::fill(cq+2*d,cq+2*d+c.atoms,0.);
            for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j){
                const double delta=s.q[k*d+j]-centroid_q[j];
                cq[2*d+j/3]+=delta*delta/c.beads;
            }
            for(size_t atom=0;atom<c.atoms;++atom)require(std::isfinite(cq[2*d+atom]),"nonfinite ring radius diagnostics");
        }
    }
public:
    Engine(Config config,Vec q,Vec p,Mat cell):c(std::move(config)),d(3*c.atoms),l(d*c.beads),mobile(c.atoms),theta(c.beads/c.beta),baro_mass((c.barostat==Barostat::BZP?3.:1.)*c.atoms*theta*c.baro_tau*c.baro_tau),fft(c.beads),qn(l),pn(l),centroid_q(d),centroid_p(d),centroid_f(d),omega(c.beads),a(c.beads),b(c.beads),cc(c.beads),ah(c.beads),bh(c.beads),ch(c.beads),y(c.gle_size),z(c.gle_size),noise(c.gle_size),status(c.beads),reference_inverse(inverse(transpose(c.reference_cell))),reference_volume(det(c.reference_cell)),fixed_q(d){
        require(c.threads>0,"threads must be positive");
#ifndef _OPENMP
        require(c.threads==1,"threads>1 requires the compiled OpenMP kernel");
#endif
        require(c.atoms>0&&c.beads>0&&c.masses.size()==c.atoms&&c.fixed.size()==c.atoms,"invalid dimensions");
        if(c.philox)require(c.beads<=UINT32_MAX&&d<=UINT32_MAX,"Philox index exceeds uint32");
        require(c.beta>0&&c.dt>0&&std::isfinite(c.beta)&&std::isfinite(c.dt),"beta and dt must be positive and finite");
        require(c.interval>0&&c.tau>0&&c.gamma>=0&&c.gamma0>=0&&c.pile_scale>=0,"invalid thermostat parameters");
        require(std::isfinite(theta)&&std::isfinite(c.tau)&&std::isfinite(c.gamma)&&std::isfinite(c.gamma0)&&std::isfinite(c.pile_scale),"nonfinite thermostat parameters");
        for(double m:c.masses)require(m>0&&std::isfinite(m),"invalid mass");
        require(q.size()==l&&p.size()==l&&finite(q)&&finite(p),"invalid q/p arrays");
        require(c.potential>=0&&c.potential<=6,"unsupported potential");
        if(c.potential==6)require(static_cast<bool>(c.python_force),"missing Python force bridge");
        if(c.potential>=3&&c.potential<=5)require(c.address!=0,"missing native PES address");
        if(c.potential==5)require(c.beads<=INT32_MAX,"Fortran bead count exceeds int32");
        if(c.potential>=4&&c.potential<=5)require(!c.has_cell,"molecular C ABI/Fortran v1 cannot accept a cell");
        if(c.potential==1)require(c.parameters.size()>=1&&c.parameters[0]>=0&&finite(c.parameters)&&!c.has_cell,"harmonic model is nonperiodic");
        if(c.potential==2)require(c.parameters.size()==3&&finite(c.parameters)&&c.has_cell,"cosine model requires a cell and three heights");
        for(int32_t pbc:c.pbc)require(pbc==0||pbc==1,"invalid PBC flag");
        if(c.has_cell){inverse(cell);for(double x:cell)require(std::isfinite(x),"nonfinite cell");}
        for(size_t atom=0;atom<c.atoms;++atom)if(c.fixed[atom])--mobile;
        require(mobile>0,"at least one mobile atom is required");
        if(c.middle_npt)require(c.barostat==Barostat::BZP&&c.thermostat==Thermostat::PileL,
                               "middle NPT requires isotropic BZP and PILE-L");
        if(c.barostat!=Barostat::None){
            require(c.has_cell&&c.pbc==std::array<int32_t,3>{1,1,1}&&mobile==c.atoms,"NPT requires 3D PBC and all atoms mobile");
            require(c.baro_tau>0&&c.baro_gamma>=0&&std::isfinite(baro_mass)&&baro_mass>0&&std::isfinite(c.baro_gamma)&&std::isfinite(c.pressure),"invalid barostat parameters");
            require(c.thermostat!=Thermostat::TRPMD,"TRPMD NPT is unsupported; use equilibrium PIMD");
            if(c.thermostat==Thermostat::Piglet)require(c.barostat==Barostat::BZP,"PIGLET NPT requires isotropic BZP");
            if(c.barostat==Barostat::SCR)require(c.beads==1&&c.compressibility>0&&std::isfinite(c.compressibility),"SCR requires P=1 and positive compressibility");
            if(c.barostat==Barostat::MTK||c.barostat==Barostat::RGB){
                require(cell[0]>0&&cell[4]>0&&cell[8]>0&&cell[1]==0&&cell[2]==0&&cell[5]==0,"flexible cell requires lower-triangular ROW lattice vectors with positive diagonal");
                require(c.cell_mask[3]==0&&c.cell_mask[6]==0&&c.cell_mask[7]==0,"cell mask must be upper triangular");
                require(std::any_of(c.cell_mask.begin(),c.cell_mask.end(),[](uint8_t x){return x!=0;}),"empty cell mask");
            }
        }
        if(gle()){
            size_t n=c.gle_size,blocks=normal_thermostat()?c.beads:1;
            require(n>0&&n<=64,"GLE dimension must be 1..64");
            require(c.gle_T.size()==blocks*n*n&&c.gle_S.size()==blocks*n*n&&c.gle_Croot.size()==blocks*n*n,"GLE matrix dimensions differ");
            require(finite(c.gle_T)&&finite(c.gle_S)&&finite(c.gle_Croot),"nonfinite GLE matrices");
            require(l<=std::numeric_limits<size_t>::max()/n,"GLE state size overflow");
            if(c.thermostat==Thermostat::Piglet&&c.barostat!=Barostat::None){
                // The centroid momentum is canonical and uncorrelated with
                // its auxiliaries. Internal PIGLET modes are noncanonical.
                for(size_t i=0;i<n;++i){
                    double covariance=0;
                    for(size_t j=0;j<n;++j)covariance+=c.gle_Croot[j]*c.gle_Croot[i*n+j];
                    require(std::abs(covariance-(i==0?1.:0.))<1.e-9,"PIGLET NPT requires canonical centroid covariance");
                }
            }
        }
        s.q=std::move(q);s.p=std::move(p);s.cell=cell;s.rng.seed(c.seed);s.rng_seed=c.seed;
        s.energy.resize(c.beads);s.force.resize(l);s.virial.resize(9*c.beads);
        if(!c.philox&&fft.radix2&&c.beads>=8&&d>=24&&(c.thermostat==Thermostat::PileL||c.thermostat==Thermostat::TRPMD))pile_noise.resize(l);
        if(gle())s.aux.assign(l*(c.gle_size-1),0.);
        if(c.thermostat==Thermostat::NHC){
            require(c.nhc_length>0&&c.nhc_length<=64&&c.nhc_substeps>0&&c.nhc_substeps<=100000,"invalid NHC chain length/substeps");
            nhc_mass.assign(c.nhc_length,theta*c.tau*c.tau);
            nhc_mass[0]*=3.*c.beads*mobile;
            require(finite(nhc_mass)&&std::all_of(nhc_mass.begin(),nhc_mass.end(),[](double x){return x>0;}),"invalid NHC bath masses");
            s.aux.assign(2*c.nhc_length,0.);
        }
        std::copy(s.q.begin(),s.q.begin()+d,fixed_q.begin());
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(!active(j))require(s.q[k*d+j]==fixed_q[j]&&s.p[k*d+j]==0,"fixed rings must be coincident with zero momenta");
        double pi=std::acos(-1.);omega[0]=0;
        for(size_t k=1;k<=(c.beads-1)/2;++k)omega[2*k-1]=omega[2*k]=2*c.beads/c.beta*std::sin(pi*k/c.beads);
        if(c.beads%2==0)omega.back()=2*c.beads/c.beta;
        for(size_t k=0;k<c.beads;++k){double w=omega[k],t=.5*c.dt*w;
            if(c.cayley){double den=1+t*t,root=std::sqrt(den);a[k]=(1-t*t)/den;b[k]=c.dt/den;cc[k]=c.dt*w*w/den;ah[k]=1/root;bh[k]=.5*c.dt/root;ch[k]=.5*c.dt*w*w/root;}
            else{a[k]=std::cos(2*t);b[k]=w==0?c.dt:std::sin(2*t)/w;cc[k]=w*std::sin(2*t);ah[k]=std::cos(t);bh[k]=w==0?.5*c.dt:std::sin(t)/w;ch[k]=w*std::sin(t);}
        }
        if(c.initialize_momenta){
            const auto epoch=begin_random();
            if(gle()){
                bool nm=normal_thermostat();Vec& p0=nm?pn:s.p;size_t n=c.gle_size;
                for(size_t i=0;i<l;++i)if(active(i%d)){
                    size_t offset=(nm?i/d:0)*n*n;for(size_t j=0;j<n;++j)noise[j]=random_normal(epoch,i/d,i%d,1,static_cast<uint32_t>(j));
                    for(size_t j=0;j<n;++j){z[j]=0;for(size_t k=0;k<n;++k)z[j]+=c.gle_Croot[offset+j*n+k]*noise[k];}
                    p0[i]=std::sqrt(theta*c.masses[(i%d)/3])*z[0];for(size_t j=1;j<n;++j)s.aux[i*(n-1)+j-1]=z[j];
                }
                if(nm)from_modes(pn,s.p);
            }else for(size_t i=0;i<l;++i)if(active(i%d))s.p[i]=std::sqrt(theta*c.masses[(i%d)/3])*random_normal(epoch,i/d,i%d,1);
        }
        refresh();
    }
    void set_state(Vec q,Vec p,uint64_t step){
        require(q.size()==l&&p.size()==l&&finite(q)&&finite(p),"invalid external q/p arrays");
        require(std::isfinite(static_cast<double>(step)*c.dt),"external state time overflow");
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(!active(j))
            require(q[k*d+j]==fixed_q[j]&&p[k*d+j]==0,"external state changes fixed atoms");
        State before=s;
        try{s.q=std::move(q);s.p=std::move(p);s.step=step;refresh();}
        catch(...){s=std::move(before);throw;}
    }
    void copy_thermostat_from(const Engine& other,bool has_step=false,uint64_t step=0){
        const Config& o=other.c;
        require(c.atoms==o.atoms&&c.beads==o.beads&&c.beta==o.beta&&c.masses==o.masses&&c.fixed==o.fixed
                &&c.thermostat==o.thermostat&&c.gamma==o.gamma&&c.gamma0==o.gamma0&&c.pile_scale==o.pile_scale
                &&c.tau==o.tau&&c.interval==o.interval&&c.gle_size==o.gle_size&&c.philox==o.philox
                &&c.gle_T==o.gle_T&&c.gle_S==o.gle_S&&c.gle_Croot==o.gle_Croot
                &&c.nhc_length==o.nhc_length&&c.nhc_substeps==o.nhc_substeps,
                "thermostat state transfer parameters differ");
        for(size_t j=0;j<d;++j)if(!active(j))require(fixed_q[j]==other.fixed_q[j],"thermostat state transfer changes fixed positions");
        require(s.aux.size()==other.s.aux.size(),"thermostat state transfer shape mismatch");
        // Allocate before mutating; RNG and distribution assignment are values.
        Vec aux=other.s.aux;
        s.aux.swap(aux);s.rng=other.s.rng;s.gaussian=other.s.gaussian;s.rng_counter=other.s.rng_counter;s.rng_seed=other.s.rng_seed;
        if(has_step)s.step=step;
    }
    Vec apply_thermostat(Vec q,Vec p,double dt,bool has_step=false,uint64_t step=0){
        require(c.barostat==Barostat::None,"a standalone thermostat cannot apply a barostat");
        require(std::isfinite(dt)&&dt>0,"thermostat duration must be positive and finite");
        require(!gle()||dt==c.dt,"GLE operator duration differs from its prepared transition");
        require(q.size()==l&&p.size()==l&&finite(q)&&finite(p),"invalid thermostat q/p arrays");
        require((has_step?step:s.step)<UINT64_MAX,"thermostat application counter overflow");
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(!active(j))
            require(q[k*d+j]==fixed_q[j]&&p[k*d+j]==0,"thermostat input changes fixed atoms");
        State before=s;
        try{
            // q only validates the frozen-atom contract. Keeping the engine's
            // own coordinates unchanged also keeps its cached forces valid.
            s.p=std::move(p);if(has_step)s.step=step;bath(dt);
            require(finite(s.p)&&finite(s.aux),"nonfinite thermostat state; reduce duration or increase damping time");
            ++s.step;
            return s.p;
        }catch(...){s=std::move(before);throw;}
    }
    Output run(uint64_t steps,uint64_t stride,bool centroids,const std::string& path="",bool moments=false,uint64_t sample_start=0,bool keep_prefix=false,const CommonObservations* common=nullptr){
        require(!moments||centroids,"record_moments requires centroid records");
        require(!keep_prefix||path.empty(),"valid-prefix propagation requires in-memory observations");
        if(common){
            require(path.empty()&&centroids&&moments&&stride==1&&sample_start==0,"common observations require in-memory centroid/momentum records and a relative mask");
            require(common->mask.size()==steps,"observation mask length must equal steps");
            for(uint8_t observed:common->mask)require(observed<=1,"observation mask must be binary");
        }
        const size_t width=moments?2*d+c.atoms:d;
        require(width<=static_cast<size_t>(std::numeric_limits<std::streamsize>::max())/sizeof(double),"record width overflow");
        require(stride>0,"stride must be positive");require(steps<=UINT64_MAX-s.step,"step counter overflow");
        require(std::isfinite(static_cast<double>(s.step+steps)*c.dt),"trajectory time overflow");
        // Absolute observation clock. Splitting a run never inserts extra frames.
        auto observations=[&](uint64_t step){return step>sample_start?(step-sample_start)/stride:uint64_t(0);};
        uint64_t count=common?std::accumulate(common->mask.begin(),common->mask.end(),uint64_t(0)):observations(s.step+steps)-observations(s.step);
        require(count<=std::numeric_limits<size_t>::max()/ncolumns,"output size overflow");
        if(centroids)require(count<=std::numeric_limits<size_t>::max()/width,"centroid output size overflow");
        Output out;out.frames=static_cast<size_t>(count);
        std::ofstream stream;
        std::array<double,ncolumns> row{};Vec coordinate(centroids?width:0);
        if(path.empty()){
            out.samples.resize(out.frames*ncolumns);if(centroids)out.centroids.resize(out.frames*width);
            if(common&&common->energy_parts)out.energy_parts.resize(out.frames*4);
        }else{
            stream.exceptions(std::ios::badbit|std::ios::failbit);
            #ifdef _WIN32
            stream.open(std::filesystem::u8path(path),std::ios::binary|std::ios::trunc);
#else
            // UTF-8 paths are native POSIX byte strings. Avoid std::filesystem
            // here so macOS wheels retain their 10.13 deployment target.
            stream.open(path,std::ios::binary|std::ios::trunc);
#endif
            const uint16_t endian=1;
            // NPY v1 header; records are native-endian IEEE binary64.
            std::string header="{'descr': '"+std::string(*reinterpret_cast<const uint8_t*>(&endian)?"<f8":">f8")+"', 'fortran_order': False, 'shape': ("+std::to_string(count)+", "+std::to_string(ncolumns+(centroids?width:0))+"), }";
            while((10+header.size()+1)%64)header+=' ';
            header+='\n';require(header.size()<65536,"NPY header too large");
            const char magic[8]={char(0x93),'N','U','M','P','Y',1,0};
            char length[2]={static_cast<char>(header.size()&255),static_cast<char>((header.size()>>8)&255)};
            stream.write(magic,8);stream.write(length,2);stream.write(header.data(),static_cast<std::streamsize>(header.size()));
        }
        State before=s,accepted;
        try{
            size_t frame=0;
            for(uint64_t k=0;k<steps;++k){
                if(keep_prefix)accepted=s;
                try{
                    if(common&&common->check_energy&&k==0)nuclear_energy();
                    one_step();
                    const bool observed=common?common->mask[k]!=0:s.step>sample_start&&(s.step-sample_start)%stride==0;
                    if(common&&common->check_energy&&!(observed&&(common->record_energy||common->energy_parts)))nuclear_energy();
                    if(observed){
                        if(common)observe_common(out.samples.data()+frame*ncolumns,out.centroids.data()+frame*width,common->energy_parts?out.energy_parts.data()+frame*4:nullptr,*common);
                        else if(path.empty())observe(out.samples.data()+frame*ncolumns,centroids?out.centroids.data()+frame*width:nullptr,moments);
                        else {observe(row.data(),centroids?coordinate.data():nullptr,moments);stream.write(reinterpret_cast<const char*>(row.data()),sizeof(double)*ncolumns);if(centroids)stream.write(reinterpret_cast<const char*>(coordinate.data()),static_cast<std::streamsize>(sizeof(double)*width));}
                        ++frame;
                    }
                    ++out.completed;
                }catch(const std::exception& error){
                    if(!keep_prefix)throw;
                    s=std::move(accepted);out.error=error.what();break;
                }
            }
            if(keep_prefix){out.frames=frame;out.samples.resize(frame*ncolumns);if(centroids)out.centroids.resize(frame*width);if(common&&common->energy_parts)out.energy_parts.resize(frame*4);}
            if(!path.empty()){stream.flush();stream.close();}
        }catch(...){s=std::move(before);throw;}
        return out;
    }
    std::string rng_state() const {
        if(c.philox)return std::string("philox-v1\n")+std::to_string(s.rng_seed)+"\n"+std::to_string(s.rng_counter);
        std::ostringstream text;text.imbue(std::locale::classic());text<<s.rng<<'\n'<<s.gaussian;return text.str();
    }
    void set_rng(const std::string& text,State& state)const{
        require(text.size()<100000,"RNG state too large");
        if(c.philox){
            std::istringstream stream(text);std::string version;stream>>version>>state.rng_seed>>state.rng_counter;
            require(version=="philox-v1"&&!stream.fail(),"invalid Philox RNG state");stream>>std::ws;require(stream.eof(),"trailing RNG state data");
            state.rng.seed(c.seed);state.gaussian.reset();return;
        }
        std::istringstream stream(text);stream.imbue(std::locale::classic());
        stream>>state.rng>>state.gaussian;require(!stream.fail(),"invalid RNG state");stream>>std::ws;require(stream.eof(),"trailing RNG state data");
    }
    void restore(State state){
        require(state.q.size()==l&&state.p.size()==l&&state.aux.size()==s.aux.size()&&state.energy.size()==c.beads&&state.force.size()==l&&state.virial.size()==9*c.beads,"checkpoint shape mismatch");
        require(finite(state.q)&&finite(state.p)&&finite(state.aux)&&finite(state.energy)&&finite(state.force)&&finite(state.virial),"nonfinite checkpoint");
        for(double x:state.cell)require(std::isfinite(x),"nonfinite checkpoint cell");
        for(double x:state.cell_p)require(std::isfinite(x),"nonfinite checkpoint cell momentum");
        if(c.has_cell)inverse(state.cell);
        if(c.barostat==Barostat::None)require(state.cell==s.cell,"fixed-cell checkpoint changes cell");
        if(c.barostat==Barostat::MTK||c.barostat==Barostat::RGB){require(state.cell[1]==0&&state.cell[2]==0&&state.cell[5]==0&&state.cell[0]>0&&state.cell[4]>0&&state.cell[8]>0,"invalid triangular checkpoint cell");for(int i=0;i<9;++i)if(!c.cell_mask[i])require(state.cell_p[i]==0,"checkpoint changes a fixed cell component");}
        for(size_t k=0;k<c.beads;++k)for(size_t j=0;j<d;++j)if(!active(j))require(state.q[k*d+j]==fixed_q[j]&&state.p[k*d+j]==0,"checkpoint changes fixed atoms");
        state.force_valid=true;s=std::move(state);
    }
};
} // namespace thermal
#endif
