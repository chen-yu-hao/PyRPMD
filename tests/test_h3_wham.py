"""Non-Gaussian density and exact unbinned-MBAR checks for histogram WHAM."""
import numpy as np
import pytest

from benchmarks.h3.mixing_analysis import histogram_samples, log_density_at, solve_wham
from pyrpmd.analysis.fes import solve_mbar


def test_exact_non_gaussian_density_is_recovered_from_biased_probabilities():
    edges=np.linspace(-2.005,2.005,402)
    x=(edges[:-1]+edges[1:])/2
    centers=np.array([-1.4,-.7,0.,.7,1.4]);k=np.array([3.,5.,2.,7.,4.]);beta=1.3
    logtarget=-5*(x*x-1)**2-.3*x
    target=np.exp(logtarget);target/=target.sum()
    bias=np.exp(-.5*beta*k[:,None]*(x[None]-centers[:,None])**2)
    probabilities=bias*target;probabilities/=probabilities.sum(axis=1)[:,None]
    fit=solve_wham(probabilities*100000,edges,centers,k,beta)
    assert fit['converged'] and fit['overlap_connected']
    wanted=np.log(target)-np.log(np.diff(edges))
    np.testing.assert_allclose(fit['log_density'],wanted,rtol=0,atol=2e-9)


def test_binned_wham_equals_exact_mbar_when_samples_are_at_bin_centers():
    rng=np.random.default_rng(78162)
    centers=np.array([-.75,0.,.8]);k=np.array([3.,6.,4.]);beta=1.2
    x=np.linspace(-1.5,1.5,31);edges=np.r_[x-.05,x[-1]+.05]
    target=np.exp(-x**4)
    biased=target[None]*np.exp(-.5*beta*k[:,None]*(x[None]-centers[:,None])**2)
    biased/=biased.sum(axis=1)[:,None]
    H=np.array([rng.multinomial(n,p) for n,p in zip((240,320,280),biased)])
    samples=np.concatenate([np.repeat(x,h) for h in H])
    fit=solve_wham(H,edges,centers,k,beta,tolerance=1e-11)
    u=.5*beta*k[:,None]*(samples[None]-centers[:,None])**2
    exact=solve_mbar(u,H.sum(axis=1),tolerance=1e-11,return_state_weights=False)
    assert exact.converged and fit['converged']
    np.testing.assert_allclose(fit['free_energies'],exact.free_energies,atol=2e-10,rtol=0)
    exact_bins=np.histogram(samples,edges,weights=exact.weights)[0]
    occupied=fit['occupied']
    observed=np.exp(fit['log_density'][occupied])*np.diff(edges)[occupied]
    np.testing.assert_allclose(observed,exact_bins[occupied],atol=2e-11,rtol=0)


def test_local_quadratic_log_density_fit_has_no_gaussian_window_assumption():
    grid=np.arange(-.02,.021,.001)
    density=3-7*grid+200*grid**2
    fit={'grid':grid,'log_density':density,'histogram':np.ones((2,len(grid)))*500}
    result=log_density_at(fit,.003,halfwidth=.004)
    assert result['log_density']==pytest.approx(3-7*.003+200*.003**2,abs=1e-13)


def test_histogram_never_silently_drops_tail_samples():
    data=np.array([[-.1,1.02],[.03,1.1],[.04,1.3]])
    H,edges=histogram_samples(data,.001)
    assert H.sum()==data.size
    with pytest.raises(ValueError,match='discard'):
        histogram_samples(data,edges=np.linspace(0,1,100))
