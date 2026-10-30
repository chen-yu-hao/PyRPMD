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
    std::vector<Py_ssize_t> reverse;
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
};
#endif
