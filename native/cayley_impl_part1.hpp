/* PyRPMD reentrant real-mass ring-polymer propagation.
 * Cyclic midpoint/Thomas + rank-two correction follows chen-yu-hao/RPMDrate
 * commit 19ba2602bad5b2145b0d0ba0716d7ba3d90a1003, _main.f90, and the
 * thermal-dRPMD implementation at 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78.
 * Original RPMDrate copyright (c) 2012 Joshua W. Allen, William H. Green,
 * Yury V. Suleimanov. Licensed under MIT; see THIRD_PARTY_NOTICES.md.
 * This implementation uses the PyRPMD [bead, atom, xyz] C-contiguous layout.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ring_fft.h"
#include "builtin_pes.hpp"

struct Cayley {
    Py_ssize_t beads, nd;
    int threads, scheme; // 0: Cayley BCB; 1: exact normal-mode BAB; 2: full VV.
    double dt, alpha, omega2;
    std::vector<double> mass, coordinate_mass, inverse_coordinate_mass, invdiag, lower, z1, z2;
    std::vector<double> exact_a, exact_b, exact_c;
    std::vector<double> half_a, half_b, half_c;
    std::unique_ptr<RingFFT> fourier;
    bool spectral_exact = false;
    double si[4];
    Cayley(std::vector<double> m, double beta, Py_ssize_t p, double h, int nt, int mode=0)
      : beads(p), nd(3 * static_cast<Py_ssize_t>(m.size())), threads(nt), scheme(mode),
        dt(h), alpha(.25*h*h*(p/beta)*(p/beta)), omega2((p/beta)*(p/beta)),
        mass(std::move(m)), invdiag(p,1.), lower(p,0.), z1(p,0.), z2(p,0.) {
        si[0]=si[3]=1.; si[1]=si[2]=0.;
        coordinate_mass.resize(nd);
        inverse_coordinate_mass.resize(nd);
        for (Py_ssize_t j = 0; j < nd; ++j) {
            coordinate_mass[j] = mass[j/3];
            inverse_coordinate_mass[j] = 1. / coordinate_mass[j];
        }
        if (scheme != 2) {
            fourier = std::make_unique<RingFFT>(p);
            half_a.resize(p); half_b.resize(p); half_c.resize(p);
            const double pi=std::acos(-1.);
            for (Py_ssize_t k=0;k<p;++k) {
                const double w=2.*p/beta*std::sin(pi*std::min(k,p-k)/p);
                const double theta=.5*h*w;
                if (scheme == 1) {
                    half_a[k]=std::cos(theta);
                    half_b[k]=theta==0. ? .5*h : .5*h*std::sin(theta)/theta;
                    half_c[k]=w*std::sin(theta);
                } else {
                    const double den=std::sqrt(1.+theta*theta);
                    half_a[k]=1./den; half_b[k]=.5*h/den; half_c[k]=.5*h*w*w/den;
                }
            }
        }
        if (scheme == 1) {
            spectral_exact = p >= 32 && fourier->radix2;
            if (spectral_exact) {
                exact_a.resize(p); exact_b.resize(p); exact_c.resize(p);
                const double pi=std::acos(-1.);
                for (Py_ssize_t k=0;k<p;++k) {
                    const double w=2.*p/beta*std::sin(pi*std::min(k,p-k)/p), theta=h*w;
                    exact_a[k]=std::cos(theta);
                    exact_b[k]=theta==0. ? h : h*std::sin(theta)/theta;
                    exact_c[k]=w*std::sin(theta);
                }
                return;
            }
            exact_a.assign(p,0.); exact_b.assign(p,0.); exact_c.assign(p,0.);
            const double pi=std::acos(-1.);
            for (Py_ssize_t delta=0;delta<p;++delta) {
                for (Py_ssize_t k=0;k<p;++k) {
                    const double w=2.*p/beta*std::sin(pi*k/p), theta=h*w;
                    const double weight=std::cos(2.*pi*k*delta/p)/p;
                    const double b=theta==0. ? h : h*(std::sin(theta)/theta);
                    exact_a[delta]+=weight*std::cos(theta);
                    exact_b[delta]+=weight*b;
                    exact_c[delta]+=weight*w*std::sin(theta);
                }
            }
            double sa=0.,sb=0.,sc=0.;
            for(Py_ssize_t k=0;k<p;++k){sa+=exact_a[k];sb+=exact_b[k];sc+=exact_c[k];}
            exact_a[0]+=1.-sa;exact_b[0]+=h-sb;exact_c[0]-=sc;
            return;
        }
        if (scheme == 2) return;
        if (p < 3) return;
        const double d=1.+2.*alpha;
        invdiag[0]=1./d;
        for (Py_ssize_t k=1;k<p;k++) {
            lower[k]=-alpha*invdiag[k-1];
            invdiag[k]=1./(d+alpha*lower[k]);
        }
        z1[0]=-alpha; z2[p-1]=-alpha;
        solve(z1.data()); solve(z2.data());
        const double a=1.+z1[p-1], b=z2[p-1], c=z1[0], d2=1.+z2[0];
        const double det=a*d2-b*c;
        if (!(det>0.) || !std::isfinite(det)) throw std::runtime_error("Cayley factorization lost precision");
        si[0]=d2/det; si[1]=-b/det; si[2]=-c/det; si[3]=a/det;
    }
    void solve(double* x) const {
        for (Py_ssize_t k=1;k<beads;k++) x[k]-=lower[k]*x[k-1];
        x[beads-1]*=invdiag[beads-1];
        for (Py_ssize_t k=beads-1;k-->0;) x[k]=(x[k]+alpha*x[k+1])*invdiag[k];
    }
    Py_ssize_t workspace_size() const { return (scheme==1 ? 2 : 1)*beads*nd; }
    void spectral_pair(double* real, double* imag, Py_ssize_t k) const {
        const Py_ssize_t n = beads, dims = nd;
        const Py_ssize_t mirror = (n - k) % n;
        const Py_ssize_t base = k * dims;
        const Py_ssize_t other_base = mirror * dims;
        const double a = exact_a[k], b = exact_b[k], c = exact_c[k];
        if (k == mirror) {
            for (Py_ssize_t j = 0; j < dims; ++j) {
                const Py_ssize_t index = base + j;
                const double qr = real[index], pr = imag[index];
                real[index] = a * qr + b * pr * inverse_coordinate_mass[j];
                imag[index] = a * pr - c * coordinate_mass[j] * qr;
            }
            return;
        }
        for (Py_ssize_t j = 0; j < dims; ++j) {
            const Py_ssize_t index = base + j;
            const Py_ssize_t other = other_base + j;
            const double ar = real[index], ai = imag[index];
            const double br = real[other], bi = imag[other];
            const double qr = .5 * (ar + br);
            const double qi = .5 * (ai - bi);
            const double pr = .5 * (ai + bi);
            const double pi = .5 * (br - ar);
            const double next_qr = a * qr + b * pr * inverse_coordinate_mass[j];
            const double next_qi = a * qi + b * pi * inverse_coordinate_mass[j];
            const double next_pr = a * pr - c * coordinate_mass[j] * qr;
            const double next_pi = a * pi - c * coordinate_mass[j] * qi;
            real[index] = next_qr - next_pi;
            imag[index] = next_qi + next_pr;
            real[other] = next_qr + next_pi;
            imag[other] = -next_qi + next_pr;
        }
    }
    void propagate(double* q,double* p,double* workspace) const {
        const Py_ssize_t n=beads, dims=nd;
        if (scheme == 1 && spectral_exact) {
            const Py_ssize_t size = n * dims;
            double* real = workspace;
            double* imag = real + size;
            const Py_ssize_t pairs = n / 2 + 1;
            if (threads == 1) {
                std::copy(q, q + size, real);
                std::copy(p, p + size, imag);
                fourier->transform_batched_serial(real, imag, dims);
                for (Py_ssize_t k=0; k<pairs; ++k)
