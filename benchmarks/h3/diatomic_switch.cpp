/* Experimental isolated, collinear diatomic ring-polymer preparation.
 * No production integrator is modified.  All time steps, including the
 * explicitly time-dependent force endpoints and switching-work quadrature,
 * execute here.  Atomic units are used throughout this C ABI.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

using Py_ssize_t = std::ptrdiff_t;
#include "../../native/ring_fft.h"

namespace {
constexpr int WIDTH = 18;
constexpr int HOLD_FIELDS = 6;
using Radial = void (*)(const double*, double*, double*, std::int32_t*);

struct Options {
    std::int64_t beads, switch_steps, hold_steps, trace_stride;
    double dt, beta, mass, omega, equilibrium, energy_zero, asymptote;
    double target_omega, target_equilibrium, target_offset;
    std::int32_t target_kind;  // 0: existing BKMP2 radial C binding; 1: harmonic oracle.
};

struct Output {
    std::int64_t completed_steps, trace_count;
    std::int32_t status;
    double initial_harmonic_energy, final_energy, work;
    double max_abs_residual, max_abs_hold_drift, hold_initial_energy;
    double min_bond, max_bond, hold_duration;
    double hold_integrals[HOLD_FIELDS];
};

void message(char* error, std::size_t capacity, const char* text) {
    if (error && capacity) std::snprintf(error, capacity, "%s", text);
}

struct Schedule { double value, derivative; };
Schedule schedule(std::int64_t step, const Options& o) {
    if (o.switch_steps == 0 || step >= o.switch_steps) return {1., 0.};
    const double s = static_cast<double>(step) / o.switch_steps;
    return {s*s*s*(10. + s*(-15. + 6.*s)),
            30.*s*s*(1.-s)*(1.-s)/(o.switch_steps*o.dt)};
}

struct Endpoint { double harmonic, full; };
bool force(const std::vector<double>& r, double lambda, const Options& o,
           Radial radial, std::vector<double>& f, Endpoint& e,
           char* error, std::size_t error_capacity) {
    double vh = 0., vf = 0.;
    for (std::size_t j = 0; j < r.size(); ++j) {
        if (!std::isfinite(r[j]) || r[j] <= 0.) {
            message(error, error_capacity, "Nonpositive or nonfinite radial bond coordinate");
            return false;
        }
        const double x = r[j] - o.equilibrium;
        const double gh = o.mass*o.omega*o.omega*x;
        const double harmonic = o.energy_zero + .5*gh*x;
        double full, gf;
        if (o.target_kind == 0) {
            const double distances[3] = {o.asymptote, r[j], o.asymptote};
            double gradients[3];
            std::int32_t status = 0;
            radial(distances, &full, gradients, &status);
            if (status != 0) {
                if (error && error_capacity)
                    std::snprintf(error, error_capacity,
                        "BKMP2 radial evaluation failed at bead %zu, r=%.17g bohr, status=%d (including at lambda=0)",
                        j, r[j], status);
                return false;
            }
            gf = gradients[1];
        } else {
            const double y = r[j] - o.target_equilibrium;
            gf = o.mass*o.target_omega*o.target_omega*y;
            full = o.energy_zero + o.target_offset + .5*gf*y;
        }
        f[j] = -((1.-lambda)*gh + lambda*gf);
        if (!(std::isfinite(harmonic) && std::isfinite(full) && std::isfinite(f[j]))) {
            message(error, error_capacity, "Nonfinite potential or force in diatomic switch");
            return false;
        }
        vh += harmonic; vf += full;
    }
    e = {vh/r.size(), vf/r.size()};
    if (!std::isfinite(e.harmonic) || !std::isfinite(e.full)) {
        message(error, error_capacity, "Nonfinite bead-averaged diatomic potential");
        return false;
    }
    return true;
}

bool observe(const std::vector<double>& r, const std::vector<double>& p,
             const Endpoint& e, const Options& o, std::int64_t step,
             double work, double reference, double hold_reference,
             double* row) {
    const std::size_t n = r.size();
    double kinetic = 0., spring = 0., mean_r = 0., mean_p = 0.;
    double lo = r[0], hi = r[0];
    const double spring_frequency = n/o.beta;
    for (std::size_t j = 0; j < n; ++j) {
        kinetic += p[j]*p[j]/(2.*o.mass);
        const double difference = r[j]-r[(j+1)%n];
        spring += .5*o.mass*spring_frequency*spring_frequency*difference*difference;
        mean_r += r[j]; mean_p += p[j];
        lo = std::min(lo, r[j]); hi = std::max(hi, r[j]);
    }
    kinetic /= n; spring /= n; mean_r /= n; mean_p /= n;
    double spread = 0.;
    for (double value : r) spread += (value-mean_r)*(value-mean_r);
    spread = std::sqrt(spread/n);
    const auto s = schedule(step, o);
    const double energy = kinetic+spring+(1.-s.value)*e.harmonic+s.value*e.full;
    row[0] = step*o.dt;
    row[1] = s.value;
    row[2] = kinetic; row[3] = e.harmonic; row[4] = e.full; row[5] = spring;
    row[6] = energy; row[7] = work; row[8] = energy-reference-work;
    // This uses the full potential even DURING the ramp. It is a diagnostic,
    // not the instantaneous driven Hamiltonian nor a pure-state energy.
    row[9] = kinetic+e.full-o.energy_zero-spring;
    row[10] = mean_r; row[11] = mean_p; row[12] = spread;
    row[13] = lo; row[14] = hi;
    row[15] = mean_p*mean_p/(2.*o.mass)
        + .5*o.mass*o.omega*o.omega*(mean_r-o.equilibrium)*(mean_r-o.equilibrium);
    row[16] = s.derivative;
    row[17] = step >= o.switch_steps ? energy-hold_reference
                                          : std::numeric_limits<double>::quiet_NaN();
    for (int j = 0; j < WIDTH-1; ++j) if (!std::isfinite(row[j])) return false;
    // The switch endpoint defines its own hold reference after observation.
    // Every subsequent accepted hold endpoint must have a finite drift.
    if (step > o.switch_steps && !std::isfinite(row[17])) return false;
    return true;
}

void update_extrema(const double* row, Output& out, bool holding) {
    out.final_energy = row[6]; out.work = row[7];
    out.max_abs_residual = std::max(out.max_abs_residual, std::abs(row[8]));
    out.min_bond = std::min(out.min_bond, row[13]);
    out.max_bond = std::max(out.max_bond, row[14]);
    if (holding) out.max_abs_hold_drift = std::max(out.max_abs_hold_drift, std::abs(row[17]));
}

void free_ring(std::vector<double>& r, std::vector<double>& p,
               const RingFFT& fft, const std::vector<double>& cosines,
               const std::vector<double>& q_from_p,
               const std::vector<double>& p_from_q,
               std::vector<double>& ri, std::vector<double>& pi,
               std::vector<double>& temporary) {
    std::fill(ri.begin(), ri.end(), 0.); std::fill(pi.begin(), pi.end(), 0.);
    fft.transform(r.data(), ri.data(), temporary.data());
    fft.transform(p.data(), pi.data(), temporary.data());
    for (std::size_t k = 0; k < r.size(); ++k) {
        const double qr = r[k], qi = ri[k], pr = p[k], pii = pi[k];
        r[k] = cosines[k]*qr + q_from_p[k]*pr;
        ri[k] = cosines[k]*qi + q_from_p[k]*pii;
        p[k] = cosines[k]*pr - p_from_q[k]*qr;
        pi[k] = cosines[k]*pii - p_from_q[k]*qi;
    }
    fft.transform(r.data(), ri.data(), temporary.data(), true);
    fft.transform(p.data(), pi.data(), temporary.data(), true);
}
} // namespace

extern "C" int pyrpmd_diatomic_switch_v1(
    const Options* options, void* radial_address,
    const double* r_input, const double* p_input, double* r_output, double* p_output,
    double* trace, std::int64_t trace_capacity, Output* output,
    char* error, std::size_t error_capacity) noexcept {
    if (!output) return 3;
    *output = Output{};
    output->initial_harmonic_energy = output->final_energy =
        output->hold_initial_energy = std::numeric_limits<double>::quiet_NaN();
    output->min_bond = std::numeric_limits<double>::infinity();
    output->max_bond = -std::numeric_limits<double>::infinity();
    message(error, error_capacity, "");
    try {
        if (!options || !r_input || !p_input || !r_output || !p_output || !trace)
            throw std::invalid_argument("Null switch argument");
        const Options& o = *options;
        if (o.beads < 1 || o.beads > 1000000 || o.switch_steps < 0 || o.hold_steps < 0
            || o.switch_steps > std::numeric_limits<std::int64_t>::max()-o.hold_steps
            || o.trace_stride < 1 || trace_capacity < 1
            || !(o.dt > 0.) || !(o.beta > 0.) || !(o.mass > 0.) || !(o.omega > 0.)
            || !(o.equilibrium > 0.) || !(o.asymptote > 0.)
            || (o.target_kind != 0 && o.target_kind != 1)
            || (o.target_kind == 0 && !radial_address)
            || (o.target_kind == 1 && (!(o.target_omega > 0.) || !(o.target_equilibrium > 0.))))
            throw std::invalid_argument("Invalid diatomic switch options");
        const double scalars[] = {o.dt, o.beta, o.mass, o.omega, o.equilibrium,
            o.energy_zero, o.asymptote, o.target_omega, o.target_equilibrium, o.target_offset};
        for (double value : scalars) if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite diatomic switch option");
        const std::int64_t total = o.switch_steps+o.hold_steps;
        // Include the initial/end/switch observations without signed overflow
        // for a direct ABI caller supplying an extreme int64 step budget.
        const std::uint64_t required = 1+static_cast<std::uint64_t>(total/o.trace_stride)
            +(total%o.trace_stride != 0)
            +(o.switch_steps > 0 && o.switch_steps < total && o.switch_steps%o.trace_stride != 0);
        if (static_cast<std::uint64_t>(trace_capacity) < required)
            throw std::invalid_argument("Insufficient trace capacity");
        const std::size_t n = static_cast<std::size_t>(o.beads);
        std::copy(r_input, r_input+n, r_output);
        std::copy(p_input, p_input+n, p_output);
        std::vector<double> r(r_input, r_input+n), p(p_input, p_input+n);
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(r[j]) || !std::isfinite(p[j]))
                throw std::invalid_argument("Nonfinite input radial phase point");
        std::vector<double> f(n), f_next(n), trial_r(n), trial_p(n);
        std::vector<double> ri(n), pi(n), temporary(2*n), cosines(n), q_from_p(n), p_from_q(n);
        RingFFT fft(static_cast<Py_ssize_t>(n));
        const double circle_pi = std::acos(-1.);
        for (std::size_t k = 0; k < n; ++k) {
            const double w = 2.*n/o.beta*std::sin(circle_pi*std::min(k, n-k)/n);
            cosines[k] = std::cos(w*o.dt);
            q_from_p[k] = w == 0. ? o.dt/o.mass : std::sin(w*o.dt)/(o.mass*w);
            p_from_q[k] = o.mass*w*std::sin(w*o.dt);
        }
        Radial radial = reinterpret_cast<Radial>(radial_address);
        Endpoint endpoint;
        if (!force(r, schedule(0,o).value, o, radial, f, endpoint, error, error_capacity)) {
            output->status = 1; return 1;
        }
        double row[WIDTH], next_row[WIDTH];
        // First evaluate the same phase point in the initial harmonic system.
        if (!observe(r, p, endpoint, o, 0, 0., 0., 0., row)) {
            message(error, error_capacity, "Nonfinite initial diatomic energy");
            output->status = 2; return 2;
        }
        const double reference = row[2]+row[5]+endpoint.harmonic;
        double work = o.switch_steps == 0 ? endpoint.full-endpoint.harmonic : 0.;
        double hold_reference = o.switch_steps == 0 ? row[6]
            : std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(reference) || !std::isfinite(work)
                || !observe(r, p, endpoint, o, 0, work, reference, hold_reference, row)) {
            message(error, error_capacity, "Nonfinite initial diatomic energy reference or switching work");
            output->status = 2; return 2;
        }
        output->initial_harmonic_energy = reference;
        output->hold_initial_energy = hold_reference;
        update_extrema(row, *output, o.switch_steps == 0);
        std::copy(row, row+WIDTH, trace); output->trace_count = 1;
        const int hold_indices[HOLD_FIELDS] = {9, 2, 4, 5, 15, 12};
        for (std::int64_t step = 0; step < total;) {
            ++step;
            const auto previous = schedule(step-1,o), current = schedule(step,o);
            trial_r = r; trial_p = p;
            for (std::size_t j = 0; j < n; ++j) trial_p[j] += .5*o.dt*f[j];
            free_ring(trial_r, trial_p, fft, cosines, q_from_p, p_from_q, ri, pi, temporary);
            Endpoint next_endpoint;
            if (!force(trial_r, current.value, o, radial, f_next, next_endpoint, error, error_capacity)) {
                output->status = 1; break;
            }
            for (std::size_t j = 0; j < n; ++j) trial_p[j] += .5*o.dt*f_next[j];
            const double next_work = work + .5*o.dt*(
                previous.derivative*(endpoint.full-endpoint.harmonic)
                + current.derivative*(next_endpoint.full-next_endpoint.harmonic));
            if (!observe(trial_r, trial_p, next_endpoint, o, step,
                         next_work, reference, hold_reference, next_row)) {
                message(error, error_capacity, "Nonfinite diatomic phase point or energy after integration step");
                output->status = 2; break;
            }
            if (step == o.switch_steps) {
                hold_reference = next_row[6]; next_row[17] = 0.;
                output->hold_initial_energy = hold_reference;
            }
            double next_hold_integrals[HOLD_FIELDS];
            const double next_hold_duration = (step-o.switch_steps)*o.dt;
            if (step > o.switch_steps) {
                bool finite_hold = std::isfinite(next_hold_duration) && next_hold_duration > 0.;
                for (int j = 0; j < HOLD_FIELDS; ++j) {
                    // Average before multiplying by dt so a representable
                    // trapezoid does not overflow while adding its endpoints.
                    next_hold_integrals[j] = output->hold_integrals[j]
                        + o.dt*(.5*row[hold_indices[j]]+.5*next_row[hold_indices[j]]);
                    finite_hold = finite_hold && std::isfinite(next_hold_integrals[j])
                        && std::isfinite(next_hold_integrals[j]/next_hold_duration);
                }
                if (!finite_hold) {
                    message(error, error_capacity, "Nonfinite diatomic hold time integral or average");
                    output->status = 2; break;
                }
            }
            r.swap(trial_r); p.swap(trial_p); f.swap(f_next);
            endpoint = next_endpoint; work = next_work;
            std::copy(next_row, next_row+WIDTH, row);
            if (step > o.switch_steps) {
                std::copy(next_hold_integrals, next_hold_integrals+HOLD_FIELDS, output->hold_integrals);
                output->hold_duration = next_hold_duration;
            }
            output->completed_steps = step;
            update_extrema(row, *output, step >= o.switch_steps);
            if (step%o.trace_stride == 0 || step == o.switch_steps || step == total) {
                std::copy(row, row+WIDTH, trace+output->trace_count*WIDTH);
                ++output->trace_count;
            }
        }
        // On a rejected endpoint the last accepted state is returned, never
        // clipped, reflected, rescaled, resampled, or silently thrown away.
        std::copy(r.begin(), r.end(), r_output); std::copy(p.begin(), p.end(), p_output);
        if (output->status && output->completed_steps > 0
                && trace[(output->trace_count-1)*WIDTH] != output->completed_steps*o.dt) {
            if (output->trace_count < trace_capacity) {
                std::copy(row, row+WIDTH, trace+output->trace_count*WIDTH);
                ++output->trace_count;
            }
        }
        return output->status;
    } catch (const std::invalid_argument& e) {
        message(error, error_capacity, e.what()); output->status = 3; return 3;
    } catch (const std::exception& e) {
        message(error, error_capacity, e.what()); output->status = 4; return 4;
    } catch (...) {
        message(error, error_capacity, "Unknown exception in isolated diatomic switch");
        output->status = 4; return 4;
    }
}
