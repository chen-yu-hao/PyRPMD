/* Immutable Fourier plans for independent real ring coordinates.
 * Power-of-two sizes use radix-2 butterflies. Other sizes retain an exact
 * direct DFT (small/odd rings are supported without a new FFT dependency).
 * Scratch buffers belong to the caller, so plans are reentrant and immutable.
 */
#ifndef PYRPMD_RING_FFT_H
#define PYRPMD_RING_FFT_H

struct RingFFT {
    Py_ssize_t size;
    bool radix2;
    std::vector<Py_ssize_t> reverse, swap_first, swap_second;
    std::vector<double> cosine, sine;

    explicit RingFFT(Py_ssize_t n)
        : size(n), radix2((n & (n - 1)) == 0), cosine(n), sine(n) {
        const double pi = std::acos(-1.);
        for (Py_ssize_t k = 0; k < n; ++k) {
            const double angle = 2. * pi * k / n;
            cosine[k] = std::cos(angle);
            sine[k] = std::sin(angle);
        }
        if (radix2) {
            reverse.assign(n, 0);
            for (Py_ssize_t k = 1; k < n; ++k)
                reverse[k] = (reverse[k >> 1] >> 1) | ((k & 1) ? n >> 1 : 0);
            for (Py_ssize_t k = 0; k < n; ++k) if (k < reverse[k]) {
                swap_first.push_back(k);
                swap_second.push_back(reverse[k]);
            }
        }
    }

    // Forward: sum_j x_j exp(-2 pi i j k / n). Inverse divides by n.
    // temporary needs 2*n doubles only for the direct-DFT path.
    void transform(double* real, double* imag, double* temporary, bool inverse=false) const {
        const Py_ssize_t n = size;
        const double sign = inverse ? 1. : -1.;
        if (radix2) {
            for (Py_ssize_t j = 0; j < n; ++j) {
                if (j < reverse[j]) {
                    std::swap(real[j], real[reverse[j]]);
                    std::swap(imag[j], imag[reverse[j]]);
                }
            }
            for (Py_ssize_t length = 2; length <= n; length *= 2) {
                const Py_ssize_t half = length / 2, stride = n / length;
                for (Py_ssize_t start = 0; start < n; start += length) {
                    for (Py_ssize_t j = 0; j < half; ++j) {
                        const Py_ssize_t lo = start + j, hi = lo + half, k = j * stride;
                        const double wi = sign * sine[k];
                        const double tr = cosine[k] * real[hi] - wi * imag[hi];
                        const double ti = cosine[k] * imag[hi] + wi * real[hi];
                        real[hi] = real[lo] - tr; imag[hi] = imag[lo] - ti;
                        real[lo] += tr; imag[lo] += ti;
                    }
                }
                if (length == n) break; // Avoid an integer overflow after the final stage.
            }
        } else {
            double* tr = temporary;
            double* ti = temporary + n;
            for (Py_ssize_t k = 0; k < n; ++k) {
                double r = 0., im = 0.;
                Py_ssize_t phase = 0;
                for (Py_ssize_t j = 0; j < n; ++j) {
                    const double wi = sign * sine[phase];
                    r += cosine[phase] * real[j] - wi * imag[j];
                    im += cosine[phase] * imag[j] + wi * real[j];
                    phase = (phase + k) % n;
                }
                tr[k] = r; ti[k] = im;
            }
            std::copy(tr, tr + n, real); std::copy(ti, ti + n, imag);
        }
        if (inverse) {
            for (Py_ssize_t j = 0; j < n; ++j) { real[j] /= n; imag[j] /= n; }
        }
    }

    // Radix-2 transform for a [bead,width] batch. Every thread in the
    // enclosing OpenMP team must call this method in the same order; its work
    // sharing loops retain one team across bit reversal and all FFT stages.
    void transform_batched_team(double* real, double* imag, Py_ssize_t width,
                                bool inverse=false) const {
        const Py_ssize_t n = size;
        const double sign = inverse ? 1. : -1.;
        const Py_ssize_t swaps = static_cast<Py_ssize_t>(swap_first.size());
        #pragma omp for schedule(static)
        for (Py_ssize_t pair = 0; pair < swaps; ++pair)
            for (Py_ssize_t j = 0; j < width; ++j) {
                const Py_ssize_t lo = swap_first[pair] * width + j;
                const Py_ssize_t hi = swap_second[pair] * width + j;
                std::swap(real[lo], real[hi]);
                std::swap(imag[lo], imag[hi]);
            }
        for (Py_ssize_t length = 2; length <= n; length *= 2) {
            const Py_ssize_t half = length / 2, stride = n / length;
            const Py_ssize_t butterflies = n / 2;
            #pragma omp for schedule(static)
            for (Py_ssize_t pair = 0; pair < butterflies; ++pair) {
                const Py_ssize_t group = pair / half;
                const Py_ssize_t offset = pair - group * half;
                const Py_ssize_t lo_base = (group * length + offset) * width;
                const Py_ssize_t hi_base = lo_base + half * width;
                const Py_ssize_t k = offset * stride;
                for (Py_ssize_t j = 0; j < width; ++j) {
                    const Py_ssize_t lo = lo_base + j;
                    const Py_ssize_t hi = hi_base + j;
                    const double wi = sign * sine[k];
                    const double tr = cosine[k] * real[hi] - wi * imag[hi];
                    const double ti = cosine[k] * imag[hi] + wi * real[hi];
                    real[hi] = real[lo] - tr; imag[hi] = imag[lo] - ti;
                    real[lo] += tr; imag[lo] += ti;
                }
            }
            if (length == n) break;
        }
        if (inverse) {
            #pragma omp for schedule(static)
            for (Py_ssize_t index = 0; index < n * width; ++index) {
                real[index] /= n;
                imag[index] /= n;
            }
        }
    }

    void transform_batched_serial(double* real, double* imag, Py_ssize_t width,
                                  bool inverse=false) const {
        const Py_ssize_t n = size;
        const double sign = inverse ? 1. : -1.;
        for (size_t pair = 0; pair < swap_first.size(); ++pair) {
            const Py_ssize_t lo_base = swap_first[pair] * width;
            const Py_ssize_t hi_base = swap_second[pair] * width;
            for (Py_ssize_t j = 0; j < width; ++j) {
                std::swap(real[lo_base + j], real[hi_base + j]);
                std::swap(imag[lo_base + j], imag[hi_base + j]);
            }
        }
        for (Py_ssize_t length = 2; length <= n; length *= 2) {
            const Py_ssize_t half = length / 2, stride = n / length;
            for (Py_ssize_t start = 0; start < n; start += length)
                for (Py_ssize_t offset = 0; offset < half; ++offset) {
                    const Py_ssize_t lo_base = (start + offset) * width;
                    const Py_ssize_t hi_base = lo_base + half * width;
                    const Py_ssize_t k = offset * stride;
                    const double wr = cosine[k], wi = sign * sine[k];
                    for (Py_ssize_t j = 0; j < width; ++j) {
                        const Py_ssize_t lo = lo_base + j, hi = hi_base + j;
                        const double tr = wr * real[hi] - wi * imag[hi];
                        const double ti = wr * imag[hi] + wi * real[hi];
                        real[hi] = real[lo] - tr; imag[hi] = imag[lo] - ti;
                        real[lo] += tr; imag[lo] += ti;
                    }
                }
            if (length == n) break;
        }
        if (inverse) {
            const double scale = 1. / static_cast<double>(n);
            for (Py_ssize_t index = 0; index < n * width; ++index) {
                real[index] *= scale;
                imag[index] *= scale;
            }
        }
    }

    // Transform two real sequences with one complex FFT. The four arrays hold
    // X.real, X.imag, Y.real and Y.imag on return. This is exact algebra around
    // the same transform and does not change the normal-mode definition.
    void forward_two_real(double* first_real, double* first_imag,
                          double* second_real, double* second_imag,
                          double* temporary) const {
        const Py_ssize_t n = size;
        for (Py_ssize_t k = 0; k < n; ++k) first_imag[j] = second_real[j];
        transform(first_real, first_imag, temporary);
        for (Py_ssize_t k = 0; k < n; ++k) {
            const Py_ssize_t mirror = (n - k) % n;
            if (k > mirror) continue;
            const double ar = first_real[k], ai = first_imag[k];
            if (k == mirror) {
                first_real[k] = ar;
                first_imag[k] = 0.;
                second_real[k] = ai;
                second_imag[k] = 0.;
                continue;
            }
            const double br = first_real[mirror], bi = first_imag[mirror];
            const double xr = .5 * (ar + br);
            const double xi = .5 * (ai - bi);
            const double yr = .5 * (ai + bi);
            const double yi = .5 * (br - ar);
            first_real[k] = xr; first_imag[k] = xi;
            second_real[k] = yr; second_imag[k] = yi;
            first_real[mirror] = xr; first_imag[mirror] = -xi;
            second_real[mirror] = yr; second_imag[mirror] = -yi;
        }
    }

    void inverse_two_real(double* first_real, double* first_imag,
                          double* second_real, double* second_imag,
                          double* temporary) const {
        const Py_ssize_t n = size;
        for (Py_ssize_t k = 0; k < n; ++k) {
            const double zr = first_real[k] - second_imag[k];
            const double zi = first_imag[k] + second_real[k];
            first_real[k] = zr;
            first_imag[k] = zi;
        }
        transform(first_real, first_imag, temporary, true);
        std::copy(first_imag, first_imag + n, second_real);
    }
};
#endif
