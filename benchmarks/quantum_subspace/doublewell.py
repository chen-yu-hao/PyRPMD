"""Double well with a linearly coupled harmonic bath; hbar=m=1.

This validation engine requires NumPy, SciPy and optional numba. Equilibrium
sampling is Metropolis-corrected preconditioned HMC with independent chains.
Snapshots within a chain are correlated. Real-time RPMD is unthermostatted.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, null_space
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh
from numba import njit, prange


@dataclass(frozen=True)
class Model:
    quartic: float = .4
    well: float = 1.4
    frequencies: tuple = (1.5, 2.3)
    displacements: tuple = (.8, .02)

    def __post_init__(self):
        if self.quartic <= 0 or self.well <= 0 or len(self.frequencies) != len(self.displacements):
            raise ValueError('Positive double-well parameters and matching bath arrays required')
        if any(w <= 0 for w in self.frequencies):
            raise ValueError('Positive bath frequencies required')

    @property
    def dimension(self):
        return 1 + len(self.frequencies)

    @property
    def barrier(self):
        return self.quartic * self.well**4

    def stiffness(self):
        K = np.zeros((self.dimension, self.dimension))
        K[0, 0] = -4*self.quartic*self.well**2
        for i, (w, c) in enumerate(zip(self.frequencies, self.displacements), 1):
            K[0, 0] += w*w*c*c
            K[i, i] = w*w
            K[0, i] = K[i, 0] = -w*w*c
        return K

    def force(self, z):
        f = -z @ self.stiffness().T
        f[..., 0] -= 4*self.quartic*z[..., 0]**3
        return f


def oscillator(n, w):
    x = np.diag(np.sqrt(np.arange(1, n+5)/(2*w)), 1)
    x = x+x.T
    x2, x4 = x@x, x@x@x@x
    h = np.diag(w*(np.arange(n)+.5))
    return h, x[:n, :n], x2[:n, :n], x4[:n, :n]


def quantum(model, beta, basis, states=80):
    """Sparse spectral reference; converge both basis and retained transitions."""
    D = model.dimension
    if len(basis) != D:
        raise ValueError('One basis cutoff per physical dimension required')
    ops = [oscillator(basis[0], 2.)] + [oscillator(n, w) for n, w in zip(basis[1:], model.frequencies)]
    def embed(op, i):
        result = csr_matrix([[1.]])
        for j, n in enumerate(basis):
            result = kron(result, csr_matrix(op) if j == i else eye(n), format='csr')
        return result
    h, x, x2, x4 = ops[0]
    hq = h-2*x2+model.quartic*(x4-2*model.well**2*x2+model.well**4*np.eye(len(x)))
    H, X, X2 = embed(hq, 0), embed(x, 0), embed(x2, 0)
    for i, (w, c) in enumerate(zip(model.frequencies, model.displacements), 1):
        H = H+embed(ops[i][0], i)+.5*w*w*c*c*X2-w*w*c*(X@embed(ops[i][1], i))
    if H.shape[0] < 400:
        E, U = eigh(H.toarray(), check_finite=False)
        E, U = E[:states], U[:, :states]
    else:
        E, U = eigsh(H, k=min(states, H.shape[0]-2), which='SA', tol=1e-10,
                      v0=np.random.default_rng(901).normal(size=H.shape[0]))
        order = np.argsort(E)
        E, U = E[order], U[:, order]
    p = np.exp(-beta*(E-E[0])); p /= p.sum()
    operators = [U.T@(X@U)] + [U.T@(embed(ops[j][1], j)@U) for j in range(1, D)]
    ii, jj = np.triu_indices(len(E), 1)
    gap = E[jj]-E[ii]
    amplitude = np.array([op[ii, jj] for op in operators])
    coefficient = 2*(p[ii]-p[jj])[None, None, :]*amplitude[:, None, :]*amplitude[None, :, :]
    keep = (gap > 1e-10) & (np.max(np.abs(coefficient), axis=(0, 1)) > 1e-14)
    gap, coefficient = gap[keep], coefficient[..., keep]
    parity = np.array([(-1.)**sum(index) for index in np.ndindex(*basis)])
    metadata = {
        'basis': list(basis), 'states': len(E), 'E0': float(E[0]), 'E1': float(E[1]),
        'gap': float(E[1]-E[0]), 'parity_lowest': np.sum(U**2*parity[:, None], axis=0)[:2].tolist(),
        'thermal_tail_last10_retained': float(p[-min(10, len(p)):].sum()),
        'response_sum_rule': float(coefficient[0, 0]@gap),
        'C0_Kubo': float(np.sum(coefficient[0, 0]/(beta*gap))),
        'q2': float(np.sum(p*np.diag(U.T@(X2@U)))),
        'barrier_along_minimum_path': model.barrier,
        'above_barrier_first_two_after_subtracting_bath_ZPE':
            (E[:2]-.5*sum(model.frequencies)-model.barrier).tolist(),
    }
    return {'gap': gap, 'coefficient': coefficient[0, 0],
            'matrix_coefficients': coefficient, 'energies': E, 'metadata': metadata}


def exact_response(spec, frequencies, damping):
    s = damping-1j*np.asarray(frequencies)
    return np.sum(spec['coefficient'][:, None]*spec['gap'][:, None]
                  /(spec['gap'][:, None]**2+s[None, :]**2), axis=0)


def exact_correlation(spec, beta, times):
    return np.cos(np.outer(times, spec['gap']))@(spec['coefficient']/(beta*spec['gap']))


def sample_hmc(model, beta, beads, count, seed, chains=64, burn=600,
               stride=4, step=.12, leapfrog=12):
    """Sample q marginal exactly in target measure, then conditional Gaussian bath.

    Finite chains still require convergence checks. The harmonic bath is
    integrated out during sampling only; all coordinates are restored before
    RPMD propagation. Global parity flips exploit an exact model symmetry.
    """
    if count % chains:
        raise ValueError('Trajectory count must be a multiple of chain count')
    rng = np.random.default_rng(seed)
    frequencies = 2*beads/beta*np.sin(np.pi*np.arange(beads//2+1)/beads)
    kappa = .5
    effective = frequencies**2+kappa
    for w, c in zip(model.frequencies, model.displacements):
        effective += w*w*c*c*frequencies**2/(frequencies**2+w*w)
    scale = np.sqrt(beads/beta/effective)
    beta_bead = beta/beads
    def transform(a):
        return np.fft.irfft(np.fft.rfft(a, axis=1, norm='ortho')*scale[None],
                            n=beads, axis=1, norm='ortho')
    def evaluate(x):
        q = transform(x)
        potential = .5*np.sum(x*x, axis=1)+beta_bead*np.sum(
            model.quartic*(q*q-model.well**2)**2-.5*kappa*q*q, axis=1)
        gradient = x+beta_bead*transform(4*model.quartic*q*(q*q-model.well**2)-kappa*q)
        return potential, gradient
    x = rng.normal(size=(chains, beads))
    potential, gradient = evaluate(x)
    saved = []; accepted = attempted = 0
    for iteration in range(burn+stride*(count//chains)):
        initial_p = rng.normal(size=x.shape)
        p, proposed, force = initial_p.copy(), x.copy(), gradient.copy()
        for _ in range(leapfrog):
            p -= .5*step*force; proposed += step*p
            new_potential, force = evaluate(proposed)
            p -= .5*step*force
        log_acceptance = potential+.5*np.sum(initial_p**2, axis=1)-new_potential-.5*np.sum(p**2, axis=1)
        keep = np.log(rng.random(chains)) < log_acceptance
        x[keep], potential[keep], gradient[keep] = proposed[keep], new_potential[keep], force[keep]
        signs = np.where(rng.random(chains) < .5, -1., 1.)
        x *= signs[:, None]; gradient *= signs[:, None]
        if iteration >= burn:
            accepted += int(keep.sum()); attempted += chains
            if (iteration-burn+1) % stride == 0:
                saved.append(transform(x).copy())
    paths = np.stack(saved, axis=1).reshape(count, beads)
    modes = np.fft.rfft(paths, axis=1, norm='ortho')
    q = np.empty((count, beads, model.dimension)); q[..., 0] = paths
    for j, (w, c) in enumerate(zip(model.frequencies, model.displacements), 1):
        noise = np.fft.rfft(rng.normal(size=(count, beads)), axis=1, norm='ortho')
        noise *= np.sqrt(beads/beta/(frequencies**2+w*w))[None]
        q[..., j] = np.fft.irfft(noise+modes*(w*w*c/(frequencies**2+w*w))[None],
                                 n=beads, axis=1, norm='ortho')
    p = rng.normal(size=q.shape)*np.sqrt(beads/beta)
    return np.ascontiguousarray(q), np.ascontiguousarray(p), {
        'sampler': 'Metropolized q-marginal HMC + conditional Gaussian bath',
        'acceptance': accepted/attempted, 'chains': chains, 'draws_per_chain': count//chains,
        'burn': burn, 'stride': stride, 'step': step, 'leapfrog': leapfrog,
        'seed': seed, 'beads': beads, 'not_independent_snapshots': True,
    }


@njit(cache=True)
def forces(q, K, quartic, spring_frequency_squared, out):
    beads, dimension = q.shape
    for j in range(beads):
        for a in range(dimension):
            value = -spring_frequency_squared*(2*q[j, a]-q[(j-1)%beads, a]-q[(j+1)%beads, a])
            for b in range(dimension): value -= K[a, b]*q[j, b]
            if a == 0: value -= 4*quartic*q[j, 0]**3
            out[j, a] = value


@njit(cache=True, parallel=True)
def propagate(q0, p0, K, quartic, beta, dt, steps, stride, nblocks=32):
    count, beads, dimension = q0.shape
    samples_per_block = count//nblocks
    blocks = np.zeros((nblocks, steps//stride+1))
    spring_frequency_squared = (beads/beta)**2
    for block in prange(nblocks):
        for i in range(block*samples_per_block, (block+1)*samples_per_block):
            initial = np.mean(q0[i, :, 0])
            for sign in range(2):
                q, p = q0[i].copy(), p0[i].copy()
                if sign: p = -p
                force = np.empty_like(q)
                forces(q, K, quartic, spring_frequency_squared, force)
                blocks[block, 0] += initial*initial/(2*samples_per_block)
                index = 1
                for step_index in range(1, steps+1):
                    for j in range(beads):
                        for a in range(dimension):
                            p[j, a] += .5*dt*force[j, a]
                            q[j, a] += dt*p[j, a]
                    forces(q, K, quartic, spring_frequency_squared, force)
                    for j in range(beads):
                        for a in range(dimension): p[j, a] += .5*dt*force[j, a]
                    if step_index % stride == 0:
                        blocks[block, index] += initial*np.mean(q[:, 0])/(2*samples_per_block)
                        index += 1
    return blocks


def reduced_model(model, U, stiffness=None):
    """Relax residual harmonic coordinates; refuse leaked quartic coordinates."""
    U = np.asarray(U, float)
    if abs(abs(U[0, 0])-1) > 1e-8 or np.linalg.norm(U[0, 1:]) > 1e-8:
        raise ValueError('First active direction must contain the whole nonlinearity')
    W = null_space(U.T)
    K = model.stiffness() if stiffness is None else np.asarray(stiffness, float)
    A, C, B = U.T@K@U, U.T@K@W, W.T@K@W
    if len(B) and np.linalg.eigvalsh(B).min() <= 0:
        raise ValueError('Nonpositive residual bath stiffness')
    effective = A-C@np.linalg.solve(B, C.T) if len(B) else A
    rotation = np.eye(len(effective))
    if len(effective) > 1:
        eigenvalues, R = eigh(effective[1:, 1:])
        rotation[1:, 1:] = R
        effective = rotation.T@effective@rotation
        frequencies = np.sqrt(eigenvalues)
        shifts = -effective[0, 1:]/eigenvalues
    else:
        frequencies, shifts = np.empty(0), np.empty(0)
    bare = effective[0, 0]-np.sum((frequencies*shifts)**2)
    if not np.isclose(bare, -4*model.quartic*model.well**2, atol=1e-8):
        raise AssertionError('Static counterterm mismatch; this model-family reduction is invalid')
    return (Model(model.quartic, model.well, tuple(frequencies), tuple(shifts)),
            U@rotation, W, rotation.T@C, B)
