// Native fixed-grid subspace propagation. Python buffer API; no NumPy ABI.
// Numerical definitions follow analysis/subspace_rpmd.py and GridPotential.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "ring_fft.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr const char* capsule_name = "pyrpmd.subspace.v1";

struct Buffer {
    Py_buffer b{};
    Buffer(PyObject* obj, int ndim, bool writable = false, bool boolean = false) {
        int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES;
        if (writable) flags |= PyBUF_WRITABLE;
        if (PyObject_GetBuffer(obj, &b, flags) != 0) throw std::runtime_error("buffer acquisition failed");
        const char* format = boolean ? "?" : "d";
        const auto bytes = boolean ? sizeof(bool) : sizeof(double);
        if (b.ndim != ndim || !b.format || std::strcmp(b.format, format) != 0 ||
            b.itemsize != static_cast<Py_ssize_t>(bytes) || !PyBuffer_IsContiguous(&b, 'C') ||
            (!boolean && reinterpret_cast<uintptr_t>(b.buf) % alignof(double) != 0)) {
            PyBuffer_Release(&b); b = {};
            throw std::invalid_argument("expected native-endian C-contiguous float64/bool buffer of the documented rank");
        }
    }
    ~Buffer() { if (b.obj) PyBuffer_Release(&b); }
    Buffer(const Buffer&) = delete;
    double* data() { return static_cast<double*>(b.buf); }
    const double* data() const { return static_cast<const double*>(b.buf); }
    Py_ssize_t size() const { return b.len / b.itemsize; }
    void shape(std::initializer_list<Py_ssize_t> expected) const {
        if (static_cast<int>(expected.size()) != b.ndim) throw std::invalid_argument("incorrect buffer rank");
        int k = 0;
        for (auto n : expected) if (b.shape[k++] != n) throw std::invalid_argument("incompatible buffer shape");
    }
};

void finite(const Buffer& a) {
    for (Py_ssize_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a.data()[i])) throw std::invalid_argument("non-finite input buffer");
}
void disjoint(const std::vector<const Buffer*>& arrays) {
    for (size_t i = 0; i < arrays.size(); ++i) for (size_t j = 0; j < i; ++j) {
        const auto a = reinterpret_cast<uintptr_t>(arrays[i]->b.buf);
        const auto b = reinterpret_cast<uintptr_t>(arrays[j]->b.buf);
        if (arrays[i]->b.len && arrays[j]->b.len && a < b + arrays[j]->b.len && b < a + arrays[i]->b.len)
            throw std::invalid_argument("native input/output buffers must not overlap");
    }
}

struct Basis {
    Py_ssize_t first;
    double n[4]{}, derivative[4]{};
};
struct Spline {
    int d;
    std::array<Py_ssize_t, 3> sizes{}, stride{};
    std::array<std::vector<double>, 3> knots;
    std::array<double, 3> lower{}, upper{};
    std::vector<double> coefficients;

    Spline(PyObject* knot_object, PyObject* coefficient_object) {
        if (!PyTuple_Check(knot_object)) throw std::invalid_argument("knots must be a tuple");
        d = static_cast<int>(PyTuple_Size(knot_object));
        if (d < 1 || d > 3) throw std::invalid_argument("only 1D-3D subspaces are supported");
        Buffer coeff(coefficient_object, d); finite(coeff);
        coefficients.assign(coeff.data(), coeff.data() + coeff.size());
        Py_ssize_t product = 1;
        for (int axis = d - 1; axis >= 0; --axis) {
            sizes[axis] = coeff.b.shape[axis]; stride[axis] = product;
            if (sizes[axis] < 4) throw std::invalid_argument("cubic spline needs at least four coefficients per axis");
            if (product > std::numeric_limits<Py_ssize_t>::max() / sizes[axis]) throw std::overflow_error("spline size overflow");
            product *= sizes[axis];
            Buffer t(PyTuple_GetItem(knot_object, axis), 1); finite(t);
            if (t.size() != sizes[axis] + 4) throw std::invalid_argument("inconsistent spline knots and coefficients");
            knots[axis].assign(t.data(), t.data() + t.size());
            if (!std::is_sorted(knots[axis].begin(), knots[axis].end())) throw std::invalid_argument("knots must be nondecreasing");
            lower[axis] = knots[axis][3]; upper[axis] = knots[axis][sizes[axis]];
            if (!(upper[axis] > lower[axis])) throw std::invalid_argument("empty spline domain");
            // Interior multiplicity above degree would make derivative support ambiguous.
            for (Py_ssize_t i = 3; i < sizes[axis]; ++i)
                if (!(knots[axis][i + 1] > knots[axis][i])) throw std::invalid_argument("interior spline knots must be strictly increasing");
        }
    }

    Basis basis(int axis, double x) const {
        const auto& t = knots[axis];
        const Py_ssize_t count = sizes[axis];
        auto span = static_cast<Py_ssize_t>(std::upper_bound(t.begin() + 3, t.begin() + count + 1, x) - t.begin()) - 1;
        span = std::max<Py_ssize_t>(3, std::min(count - 1, span));
        Basis b{}; b.first = span - 3; b.n[0] = 1.;
        double left[4]{}, right[4]{}, lower_basis[3]{};
        for (int j = 1; j <= 3; ++j) {
            left[j] = x - t[span + 1 - j]; right[j] = t[span + j] - x;
            double saved = 0.;
            for (int r = 0; r < j; ++r) {
                const double den = right[r + 1] + left[j - r];
                const double temp = den == 0. ? 0. : b.n[r] / den;
                b.n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            b.n[j] = saved;
            if (j == 2) std::copy(b.n, b.n + 3, lower_basis);
        }
        for (int r = 0; r < 4; ++r) {
            if (r > 0) {
                const double den = t[span + r] - t[span + r - 3];
                if (den != 0.) b.derivative[r] += 3. * lower_basis[r - 1] / den;
            }
            if (r < 3) {
                const double den = t[span + r + 1] - t[span + r - 2];
                if (den != 0.) b.derivative[r] -= 3. * lower_basis[r] / den;
            }
        }
        return b;
    }

    bool evaluate(const double* point, double& energy, double* gradient, bool& outside) const {
        Basis bases[3]; bool clamped[3]{}; outside = false; energy = 0.;
        for (int axis = 0; axis < d; ++axis) {
            if (!std::isfinite(point[axis])) return false;
            clamped[axis] = point[axis] < lower[axis] || point[axis] > upper[axis];
            outside |= clamped[axis]; gradient[axis] = 0.;
            bases[axis] = basis(axis, std::clamp(point[axis], lower[axis], upper[axis]));
        }
        const int combinations = 1 << (2 * d);
        for (int code = 0; code < combinations; ++code) {
            int local[3]; Py_ssize_t index = 0; double weight = 1.;
            for (int axis = 0; axis < d; ++axis) {
                local[axis] = (code >> (2 * axis)) & 3;
                index += (bases[axis].first + local[axis]) * stride[axis];
                weight *= bases[axis].n[local[axis]];
            }
            const double c = coefficients[index]; energy += c * weight;
            for (int derivative_axis = 0; derivative_axis < d; ++derivative_axis) {
                if (clamped[derivative_axis]) continue;
                double w = c;
                for (int axis = 0; axis < d; ++axis)
                    w *= axis == derivative_axis ? bases[axis].derivative[local[axis]] : bases[axis].n[local[axis]];
                gradient[derivative_axis] += w;
            }
        }
        if (!std::isfinite(energy)) return false;
        for (int axis = 0; axis < d; ++axis) if (!std::isfinite(gradient[axis])) return false;
        return true;
    }
};

struct Context {
    Spline spline;
    Py_ssize_t p;
    int threads;
    std::vector<double> mass, c, omega;
    std::unique_ptr<RingFFT> fft;
    Context(PyObject* knots, PyObject* coeff, PyObject* mass_obj, PyObject* matrix_obj, PyObject* omega_obj, int nthreads)
        : spline(knots, coeff), threads(nthreads) {
        if (threads < 1) throw std::invalid_argument("threads must be positive");
#ifndef _OPENMP
        if (threads != 1) throw std::invalid_argument("multiple threads require an OpenMP build");
#endif
        Buffer m(mass_obj, 1), matrix(matrix_obj, 2), w(omega_obj, 1);
        p = w.size(); if (p < 1) throw std::invalid_argument("empty ring");
        m.shape({spline.d}); matrix.shape({p, p}); finite(m); finite(matrix); finite(w);
        for (int a = 0; a < spline.d; ++a) if (m.data()[a] <= 0.) throw std::invalid_argument("masses must be positive");
        for (Py_ssize_t k = 0; k < p; ++k) if (w.data()[k] < 0.) throw std::invalid_argument("frequencies must be nonnegative");
        if (w.data()[0] != 0.) throw std::invalid_argument("centroid frequency must be zero");
        mass.assign(m.data(), m.data() + m.size()); c.assign(matrix.data(), matrix.data() + matrix.size());
        omega.assign(w.data(), w.data() + w.size());
        // FFT is valid only for the canonical real-mode ordering. Low-level
        // callers may supply a different orthonormal basis: retain its matrix.
        bool canonical = p >= 32 && (p & (p - 1)) == 0;
        const double pi = std::acos(-1.);
        for (Py_ssize_t j = 0; canonical && j < p; ++j) for (Py_ssize_t k = 0; k < p; ++k) {
            const double expected = k == 0 ? 1. / std::sqrt(static_cast<double>(p)) :
                (2*k < p ? std::sqrt(2. / p) * std::cos(2*pi*j*k/p) :
                 (2*k == p ? (j % 2 ? -1. : 1.) / std::sqrt(static_cast<double>(p)) :
                  std::sqrt(2. / p) * std::sin(2*pi*j*k/p)));
            if (std::abs(c[j*p+k] - expected) > 1e-12) { canonical = false; break; }
        }
        if (canonical) fft = std::make_unique<RingFFT>(p);
    }
    void transform(const double* input, double* output, bool to_modes,
                   double* real, double* imag, double* temporary) const {
        const int d = spline.d;
        if (fft) {
            const double scale = std::sqrt(2. / p), sqrtp = std::sqrt(static_cast<double>(p));
            for (int a = 0; a < d; ++a) {
                if (to_modes) {
                    for (Py_ssize_t j = 0; j < p; ++j) { real[j] = input[j * d + a]; imag[j] = 0.; }
                    fft->transform(real, imag, temporary);
                    output[a] = real[0] / sqrtp;
                    for (Py_ssize_t k = 1; k < p; ++k)
                        output[k * d + a] = 2 * k < p ? scale * real[k] :
                            (2 * k == p ? real[k] / sqrtp : -scale * imag[k]);
                } else {
                    real[0] = sqrtp * input[a]; imag[0] = 0.;
                    for (Py_ssize_t k = 1; k < p / 2; ++k) {
                        real[k] = real[p-k] = input[k * d + a] / scale;
                        imag[k] = input[(p-k) * d + a] / scale; imag[p-k] = -imag[k];
                    }
                    real[p/2] = sqrtp * input[(p/2) * d + a]; imag[p/2] = 0.;
                    fft->transform(real, imag, temporary, true);
                    for (Py_ssize_t j = 0; j < p; ++j) output[j * d + a] = real[j];
                }
            }
            return;
        }
        for (Py_ssize_t j = 0; j < p; ++j) for (int a = 0; a < d; ++a) {
            double value = 0.;
            for (Py_ssize_t k = 0; k < p; ++k)
                value += (to_modes ? c[k * p + j] : c[j * p + k]) * input[k * d + a];
            output[j * d + a] = value;
        }
    }
};
struct Workspace {
    std::vector<double> q, p, force, beads, gradient, real, imag, temporary;
    Workspace(size_t n, size_t beads_count) : q(n), p(n), force(n), beads(n), gradient(n),
        real(beads_count), imag(beads_count), temporary(2 * beads_count) {}
};
Context* context(PyObject* capsule) {
    auto* ptr = static_cast<Context*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!ptr) throw std::invalid_argument("invalid subspace context");
    return ptr;
}
void destroy(PyObject* capsule) { delete static_cast<Context*>(PyCapsule_GetPointer(capsule, capsule_name)); }
PyObject* failure() {
    try { throw; }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::invalid_argument& e) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, e.what()); }
    catch (const std::exception& e) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what()); }
    return nullptr;
}
PyObject* make_context(PyObject*, PyObject* args) {
    PyObject *knots, *coeff, *m, *matrix, *w; int threads;
    if (!PyArg_ParseTuple(args, "OOOOOi", &knots, &coeff, &m, &matrix, &w, &threads)) return nullptr;
    try {
        auto c = std::make_unique<Context>(knots, coeff, m, matrix, w, threads);
        PyObject* capsule = PyCapsule_New(c.get(), capsule_name, destroy);
        if (capsule) c.release();
        return capsule;
    } catch (...) { return failure(); }
}

PyObject* evaluate(PyObject*, PyObject* args) {
    PyObject *capsule, *points, *energies, *gradients;
    if (!PyArg_ParseTuple(args, "OOOO", &capsule, &points, &energies, &gradients)) return nullptr;
    try {
        const auto& c = *context(capsule);
        Buffer q(points, 2), e(energies, 1, true), g(gradients, 2, true);
        const auto n = q.b.shape[0]; q.shape({n, c.spline.d}); e.shape({n}); g.shape({n, c.spline.d});
        disjoint({&q, &e, &g});
        unsigned long long outside = 0; int failed = 0;
        const int threads = static_cast<int>(std::min<Py_ssize_t>(c.threads, std::max<Py_ssize_t>(1, n)));
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:outside) reduction(|:failed)
        for (Py_ssize_t i = 0; i < n; ++i) {
            bool out = false;
            if (!c.spline.evaluate(q.data() + i * c.spline.d, e.data()[i], g.data() + i * c.spline.d, out)) failed = 1;
            outside += out;
        }
        Py_END_ALLOW_THREADS
        if (failed) { PyErr_SetString(PyExc_FloatingPointError, "non-finite subspace spline evaluation"); return nullptr; }
        return Py_BuildValue("KK", static_cast<unsigned long long>(n), outside);
    } catch (...) { return failure(); }
}

PyObject* advance(PyObject*, PyObject* args) {
    PyObject *capsule, *q_obj, *p_obj, *f_obj, *e_obj, *g_obj, *side_obj;
    double dt, surface; int steps, constrained;
    if (!PyArg_ParseTuple(args, "OOOOOOdiiOd", &capsule, &q_obj, &p_obj, &f_obj, &e_obj, &g_obj,
                          &dt, &steps, &constrained, &side_obj, &surface)) return nullptr;
    try {
        const auto& c = *context(capsule);
        if (!std::isfinite(dt) || dt <= 0. || steps < 1 || !std::isfinite(surface) || (constrained != 0 && constrained != 1))
            throw std::invalid_argument("invalid propagation controls");
        Buffer q(q_obj, 3, true), p(p_obj, 3, true), f(f_obj, 3, true), e(e_obj, 1, true), g(g_obj, 3, true);
        const auto batch = q.b.shape[0]; const int d = c.spline.d; const auto size = c.p * d;
        q.shape({batch, c.p, d}); p.shape({batch, c.p, d}); f.shape({batch, c.p, d});
        e.shape({batch}); g.shape({batch, c.p, d}); finite(q); finite(p); finite(f);
        std::unique_ptr<Buffer> side;
        std::vector<const Buffer*> arrays{&q, &p, &f, &e, &g};
        if (side_obj != Py_None) {
            side = std::make_unique<Buffer>(side_obj, 2, true, true);
            side->shape({static_cast<Py_ssize_t>(steps) + 1, batch}); arrays.push_back(side.get());
        }
        disjoint(arrays);
        bool* indicators = side ? static_cast<bool*>(side->b.buf) : nullptr;
        std::vector<double> cosine(size), q_from_p(size), p_from_q(size);
        for (Py_ssize_t k = 0; k < c.p; ++k) for (int a = 0; a < d; ++a) {
            const auto i = k * d + a; const double w = c.omega[k];
            cosine[i] = std::cos(w * dt);
            q_from_p[i] = w == 0. ? dt / c.mass[a] : std::sin(w * dt) / (c.mass[a] * w);
            p_from_q[i] = -c.mass[a] * w * std::sin(w * dt);
        }
        const int threads = static_cast<int>(std::min<Py_ssize_t>(c.threads, std::max<Py_ssize_t>(1, batch)));
        std::vector<Workspace> work;
        for (int i = 0; i < threads; ++i) work.emplace_back(size, c.p);
        unsigned long long outside = 0, evaluations = 0; int failed = 0;
        const double sqrtp = std::sqrt(static_cast<double>(c.p));
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel num_threads(threads) reduction(+:outside,evaluations) reduction(|:failed)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& s = work[tid];
            #pragma omp for schedule(static)
            for (Py_ssize_t ring = 0; ring < batch; ++ring) {
                std::copy(q.data() + ring * size, q.data() + (ring + 1) * size, s.q.begin());
                std::copy(p.data() + ring * size, p.data() + (ring + 1) * size, s.p.begin());
                std::copy(f.data() + ring * size, f.data() + (ring + 1) * size, s.force.begin());
                if (indicators) indicators[ring] = s.q[0] / sqrtp > surface;
                bool ok = true; double energy = 0.;
                for (int step = 0; step < steps; ++step) {
                    for (Py_ssize_t i = 0; i < size; ++i) {
                        const double momentum = s.p[i] + 0.5 * dt * s.force[i];
                        s.p[i] = cosine[i] * momentum + p_from_q[i] * s.q[i];
                        s.q[i] = cosine[i] * s.q[i] + q_from_p[i] * momentum;
                    }
                    c.transform(s.q.data(), s.beads.data(), false, s.real.data(), s.imag.data(), s.temporary.data()); energy = 0.;
                    for (Py_ssize_t bead = 0; bead < c.p; ++bead) {
                        bool out = false; double v = 0.; ++evaluations;
                        ok &= c.spline.evaluate(s.beads.data() + bead * d, v, s.gradient.data() + bead * d, out);
                        outside += out; energy += v;
                    }
                    if (!ok || !std::isfinite(energy)) { ok = false; failed = 1; break; }
                    c.transform(s.gradient.data(), s.force.data(), true, s.real.data(), s.imag.data(), s.temporary.data());
                    for (auto& value : s.force) value = -value;
                    if (constrained) s.force[0] = 0.;
                    for (Py_ssize_t i = 0; i < size; ++i) s.p[i] += 0.5 * dt * s.force[i];
                    if (indicators) indicators[(static_cast<Py_ssize_t>(step) + 1) * batch + ring] = s.q[0] / sqrtp > surface;
                }
                if (ok) {
                    std::copy(s.q.begin(), s.q.end(), q.data() + ring * size);
                    std::copy(s.p.begin(), s.p.end(), p.data() + ring * size);
                    std::copy(s.force.begin(), s.force.end(), f.data() + ring * size);
                    std::copy(s.gradient.begin(), s.gradient.end(), g.data() + ring * size);
                    e.data()[ring] = energy;
                }
            }
        }
        Py_END_ALLOW_THREADS
        if (failed) { PyErr_SetString(PyExc_FloatingPointError, "non-finite subspace propagation"); return nullptr; }
        return Py_BuildValue("KK", evaluations, outside);
    } catch (...) { return failure(); }
}
PyMethodDef methods[] = {
    {"make_context", make_context, METH_VARARGS, "Build an immutable 1D-3D cubic spline/ring context."},
    {"evaluate", evaluate, METH_VARARGS, "Evaluate energy and analytic gradient together."},
    {"advance", advance, METH_VARARGS, "Fuse ring propagation steps; independent rings use OpenMP."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_subspace_native", "Subspace native kernels (API 1).", -1, methods};
}  // namespace
PyMODINIT_FUNC PyInit__subspace_native() {
    PyObject* m = PyModule_Create(&module);
    if (!m) return nullptr;
    if (PyModule_AddIntConstant(m, "API_VERSION", 1) < 0) { Py_DECREF(m); return nullptr; }
#ifdef _OPENMP
    const int openmp = 1;
#else
    const int openmp = 0;
#endif
    if (PyModule_AddIntConstant(m, "OPENMP_ENABLED", openmp) < 0) { Py_DECREF(m); return nullptr; }
    return m;
}
