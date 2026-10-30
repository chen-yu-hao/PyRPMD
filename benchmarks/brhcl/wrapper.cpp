// PyRPMD MIT C ABI; the independently fetched LEPS source retains its terms.
#include "pyrpmd/pes_api_v1.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

extern "C" void pyrpmd_brhcl_bare_frame(const double*, double*, double*, double*, int32_t*);
static std::mutex legacy_mutex;
struct pyrpmd_pes_context_v1 { double radius, kappa; std::string metadata; };

static void message(pyrpmd_error_v1* error, const char* text) {
  if (!error) return;
  error->required_capacity = static_cast<int64_t>(std::strlen(text)) + 1;
  if (error->message && error->capacity > 0)
    std::snprintf(error->message, static_cast<size_t>(error->capacity), "%s", text);
}

// Only two numeric options are consumed; the loader supplies the JSON object.
static double number(const std::string& json, const char* key, double fallback) {
  std::string token = std::string("\"") + key + "\"";
  auto position = json.find(token);
  if (position == std::string::npos) return fallback;
  position = json.find(':', position + token.size());
  if (position == std::string::npos) throw std::invalid_argument("missing numeric option value");
  const char* start = json.c_str() + position + 1;
  char* end = nullptr;
  double value = std::strtod(start, &end);
  if (end == start || !std::isfinite(value)) throw std::invalid_argument("nonfinite/nonnumeric wall option");
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
  if (*end != ',' && *end != '}') throw std::invalid_argument("invalid numeric wall option");
  return value;
}

extern "C" int32_t pyrpmd_pes_abi_version_v1(void) { return 1; }
extern "C" pyrpmd_rc_v1 pyrpmd_pes_create_v1(
    const pyrpmd_create_options_v1* options, pyrpmd_pes_context_v1** output, pyrpmd_error_v1* error) {
  if (!output) return PYRPMD_INVALID_ARGUMENT_V1;
  *output = nullptr;
  if (!options || options->abi_version != 1 || options->struct_size != sizeof(*options))
    return PYRPMD_ABI_MISMATCH_V1;
  if (options->config_json_bytes < 0 || options->config_json_bytes > 1024*1024 ||
      (!options->config_json && options->config_json_bytes)) return PYRPMD_INVALID_CONFIG_V1;
  try {
    std::string json(options->config_json ? options->config_json : "", static_cast<size_t>(options->config_json_bytes));
    double radius = number(json, "wall_radius_bohr", 0.0);
    double kappa = number(json, "wall_k_hartree_bohr2", 0.0);
    if (radius < 0 || kappa < 0) throw std::invalid_argument("wall options must be nonnegative");
    std::ostringstream metadata;
    metadata << std::setprecision(17)
      << R"({"schema":"pyrpmd.pes/1","abi_version":1,"provider_name":"BrHCl_LEPS1","provider_version":"1","natoms":3,"atomic_numbers":[35,1,17],"state":{"charge":0,"multiplicity":2},"layout":"B,N,3:C","dtype":"float64","position_unit":"bohr","energy_unit":"hartree","force_unit":"hartree/bohr","boundary_conditions":"nonperiodic","capabilities":{"native_batch":true,"force_kind":"analytic","thread_safety":"process_serial","stateful":false,"execution_device":"cpu"},"common_pair_wall":{"radius_bohr":)"
      << radius << ",\"kappa_hartree_per_bohr2\":" << kappa
      << ",\"enabled\":" << ((radius > 0 && kappa > 0) ? "true" : "false")
      << R"(,"retained_in_target_pmf":true,"pairs":[[0,1],[1,2],[0,2]]}})";
    *output = new pyrpmd_pes_context_v1{radius, kappa, metadata.str()};
    message(error, "");
    return 0;
  } catch (const std::exception& exception) { message(error, exception.what()); return PYRPMD_INVALID_CONFIG_V1; }
  catch (...) { message(error, "context creation failed"); return PYRPMD_INTERNAL_ERROR_V1; }
}

extern "C" pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(
    const pyrpmd_pes_context_v1* context, char* output, int64_t capacity,
    int64_t* required, pyrpmd_error_v1* error) {
  if (!context || !required || capacity < 0) return PYRPMD_INVALID_ARGUMENT_V1;
  *required = static_cast<int64_t>(context->metadata.size()) + 1;
  if (capacity < *required) {
    if (output && capacity) output[0] = 0;
    return PYRPMD_BUFFER_TOO_SMALL_V1;
  }
  if (!output) return PYRPMD_INVALID_ARGUMENT_V1;
  std::memcpy(output, context->metadata.c_str(), static_cast<size_t>(*required));
  message(error, ""); return 0;
}

extern "C" pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(
    pyrpmd_pes_context_v1* context, const pyrpmd_pes_input_v1* input,
    pyrpmd_pes_output_v1* output, pyrpmd_error_v1* error) {
  if (!context || !input || !output) return PYRPMD_INVALID_ARGUMENT_V1;
  if (input->abi_version != 1 || output->abi_version != 1 ||
      input->struct_size != sizeof(*input) || output->struct_size != sizeof(*output))
    return PYRPMD_ABI_MISMATCH_V1;
  const int64_t n = input->nstructures;
  if (n < 1 || n > INT64_MAX/9 || input->natoms != 3 || input->layout != 1 || input->scalar_type != 1 ||
      input->positions_count != 9*n || output->energies_count != n || output->forces_count != 9*n ||
      output->structure_status_count != n || !input->positions || !output->energies ||
      !output->forces || !output->structure_status) return PYRPMD_INVALID_ARGUMENT_V1;
  if (output->structure_messages && (output->structure_message_stride < 1 ||
      n > INT64_MAX/output->structure_message_stride ||
      output->structure_messages_capacity < n*output->structure_message_stride)) return PYRPMD_INVALID_ARGUMENT_V1;
  if (!output->structure_messages && (output->structure_message_stride || output->structure_messages_capacity))
    return PYRPMD_INVALID_ARGUMENT_V1;
  try {
    std::lock_guard<std::mutex> guard(legacy_mutex);
    for (int64_t frame = 0; frame < n; ++frame) {
      const double* q = input->positions + 9*frame;
      double* f = output->forces + 9*frame;
      double& e = output->energies[frame];
      int32_t& status = output->structure_status[frame];
      double gradient[3];
      pyrpmd_brhcl_bare_frame(q, &e, f, gradient, &status);
      if (!status && context->radius > 0 && context->kappa > 0) {
        for (int a = 0; a < 2; ++a) for (int b = a+1; b < 3; ++b) {
          double delta[3], distance2 = 0;
          for (int c = 0; c < 3; ++c) { delta[c] = q[3*a+c]-q[3*b+c]; distance2 += delta[c]*delta[c]; }
          const double distance = std::sqrt(distance2);
          const double excess = std::max(0.0, distance-context->radius);
          e += 0.5*context->kappa*excess*excess;
          for (int c = 0; c < 3; ++c) {
            double contribution = -context->kappa*excess*delta[c]/distance;
            f[3*a+c] += contribution; f[3*b+c] -= contribution;
          }
        }
      }
      if (!std::isfinite(e) || !std::all_of(f, f+9, [](double v) { return std::isfinite(v); }))
        if (!status) status = PYRPMD_STRUCTURE_NONFINITE_V1;
      if (status) { e = std::numeric_limits<double>::quiet_NaN(); std::fill(f, f+9, e); }
      if (output->structure_messages) {
        const char* reason = !status ? "" : (status == 2 ? "pair distance below 0.2 bohr or nonfinite geometry" : "nonfinite legacy PES (including a singular LEPS seam)");
        std::snprintf(output->structure_messages + frame*output->structure_message_stride,
                      static_cast<size_t>(output->structure_message_stride), "%s", reason);
      }
    }
    message(error, ""); return 0;
  } catch (...) { message(error, "native BrHCl evaluation failed"); return PYRPMD_INTERNAL_ERROR_V1; }
}

// Auxiliary bare radial oracle, r=(Br-H,H-Cl,Br-Cl), independent of a wall context.
extern "C" void pyrpmd_brhcl_radial(const double* r, double* energy, double* gradient, int32_t* status) {
  *energy = std::numeric_limits<double>::quiet_NaN(); std::fill(gradient, gradient+3, *energy); *status = 2;
  if (!std::all_of(r, r+3, [](double v) { return std::isfinite(v) && v >= 0.2; })) return;
  const double largest = *std::max_element(r, r+3);
  if (2*largest > r[0]+r[1]+r[2]+1e-10) return;
  const double x = (r[0]*r[0]+r[2]*r[2]-r[1]*r[1])/(2*r[0]);
  const double y2 = r[2]*r[2]-x*x;
  if (y2 < -1e-8) return;
  const double q[9] = {0,0,0,r[0],0,0,x,std::sqrt(std::max(0.0,y2)),0};
  double force[9];
  try { std::lock_guard<std::mutex> guard(legacy_mutex); pyrpmd_brhcl_bare_frame(q,energy,force,gradient,status); }
  catch (...) { *status = 3; }
}

extern "C" void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1* context) { delete context; }
