/* Optional batched ONNX Runtime PES. No Python/JAX calls in this library.
 * SPDX-License-Identifier: MIT
 * Contract: ../ABI.md. Each context owns its session and reusable host buffers.
 */
#include "pyrpmd/pes_api_v1.h"
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace {
struct Error : std::runtime_error {
    int32_t code;
    Error(int32_t c, const std::string& s) : std::runtime_error(s), code(c) {}
};
void require(bool ok, const char* text, int32_t code = PYRPMD_INVALID_ARGUMENT_V1) {
    if (!ok) throw Error(code, text);
}
// Diagnostics never allocate or throw; ASCII sanitization also makes truncation
// safe when an external runtime's what() contains multi-byte UTF-8.
void diagnostic(pyrpmd_error_v1* e, const char* text) noexcept {
    if (!e) return;
    const size_t len = std::strlen(text);
    e->required_capacity = static_cast<int64_t>(len + 1);
    if (!e->message || e->capacity <= 0) return;
    const size_t n = std::min(len, static_cast<size_t>(e->capacity - 1));
    for (size_t i=0; i<n; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        e->message[i] = c < 128 ? static_cast<char>(c) : '?';
    }
    e->message[n] = '\0';
}
template<class F> int32_t guarded(pyrpmd_error_v1* e, int32_t fallback, F&& f) noexcept {
    if (e && (e->capacity < 0 || (e->capacity > 0 && !e->message)))
        return PYRPMD_INVALID_ARGUMENT_V1;
    diagnostic(e, "");
    try { f(); return PYRPMD_OK_V1; }
    catch (const Error& x) { diagnostic(e, x.what()); return x.code; }
    catch (const std::bad_alloc&) { diagnostic(e, "allocation failed"); return PYRPMD_INTERNAL_ERROR_V1; }
    catch (const std::exception& x) { diagnostic(e, x.what()); return fallback; }
    catch (...) { diagnostic(e, "unknown native exception"); return PYRPMD_INTERNAL_ERROR_V1; }
}
int64_t integer(const json& j, int64_t low, int64_t high, const char* text) {
    require(j.is_number_integer(), text, PYRPMD_INVALID_CONFIG_V1);
    if (j.is_number_unsigned())
        require(j.get<uint64_t>() <= static_cast<uint64_t>(high), text, PYRPMD_INVALID_CONFIG_V1);
    const int64_t n = j.get<int64_t>();
    require(n >= low && n <= high, text, PYRPMD_INVALID_CONFIG_V1);
    return n;
}
double constant(const json& c, const char* name, double reference) {
    const auto& item = c.at(name);
    require(item.is_number(), "invalid conversion constant", PYRPMD_INVALID_CONFIG_V1);
    const double v = item.get<double>();
    require(std::isfinite(v) && std::abs(v/reference - 1.) < 1.e-6,
            "conversion constant outside supported CODATA range", PYRPMD_INVALID_CONFIG_V1);
    return v;
}
void tensor_contract(const Ort::TypeInfo& type, const std::vector<int64_t>& expected) {
    require(type.GetONNXType() == ONNX_TYPE_TENSOR, "ONNX I/O must be tensors", PYRPMD_INVALID_CONFIG_V1);
    const auto t = type.GetTensorTypeAndShapeInfo();
    require(t.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            "ONNX positions, energy and forces must all be float64", PYRPMD_INVALID_CONFIG_V1);
    const auto dims = t.GetShape();
    require(dims.size() == expected.size() && dims[0] == -1,
            "ONNX I/O must have a dynamic leading batch dimension", PYRPMD_INVALID_CONFIG_V1);
    for (size_t i=1; i<dims.size(); ++i)
        require(dims[i] == expected[i], "ONNX atom/Cartesian dimensions mismatch", PYRPMD_INVALID_CONFIG_V1);
}
}

struct pyrpmd_pes_context_v1 {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "pyrpmd-onnx"};
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session> session;
    std::mutex mutex;
    int64_t natoms = 0, cached_batch = 0;
    double coordinate_factor = 1., energy_factor = 1.;
    std::string metadata, input_name, energy_name, force_name;
    std::vector<double> positions, energy, force;
    Ort::Value input{nullptr};
    std::vector<Ort::Value> outputs;

    explicit pyrpmd_pes_context_v1(const json& config) {
        require(config.is_object(), "config must be a JSON object", PYRPMD_INVALID_CONFIG_V1);
        const std::vector<std::string> allowed = {"model", "device", "intra_op_threads", "allow_cpu_fallback",
            "natoms", "symbols", "atomic_numbers", "charge", "multiplicity"};
        for (const auto& item : config.items())
            require(std::find(allowed.begin(), allowed.end(), item.key()) != allowed.end(),
                    "unknown ONNX configuration key", PYRPMD_INVALID_CONFIG_V1);
        const std::string model = config.at("model").get<std::string>();
        require(!model.empty(), "model path is empty", PYRPMD_INVALID_CONFIG_V1);
        const std::string device = config.value("device", std::string("cpu"));
        const bool cpu_fallback = config.value("allow_cpu_fallback", false);
        const int threads = static_cast<int>(integer(config.value("intra_op_threads", json(1)), 1, 1024,
                                                      "intra_op_threads must be an integer in [1,1024]"));
        env.DisableTelemetryEvents();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (device != "cpu") {
            require(device.rfind("cuda:", 0) == 0 && device.size() > 5,
                    "device must be cpu or cuda:<nonnegative index>", PYRPMD_INVALID_CONFIG_V1);
            const auto index = device.substr(5);
            require(std::all_of(index.begin(), index.end(), [](char c){ return c >= '0' && c <= '9'; }),
                    "invalid CUDA device index", PYRPMD_INVALID_CONFIG_V1);
            require(index.size() < 10, "CUDA device index too large", PYRPMD_INVALID_CONFIG_V1);
            const auto providers = Ort::GetAvailableProviders();
            require(std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end(),
                    "CUDA requested but this ONNX Runtime has no CUDA provider", PYRPMD_BACKEND_FAILURE_V1);
            OrtCUDAProviderOptionsV2* raw = nullptr;
            const auto& api = Ort::GetApi();
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
            const auto release = [&api](OrtCUDAProviderOptionsV2* p){ api.ReleaseCUDAProviderOptions(p); };
            std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release)> cuda(raw, release);
            const char* keys[] = {"device_id"};
            const char* values[] = {index.c_str()};
            Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda.get(), keys, values, 1));
            options.AppendExecutionProvider_CUDA_V2(*cuda);
            if (!cpu_fallback) options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
        const std::filesystem::path model_path(model);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto model_metadata = session->GetModelMetadata();
        const auto text = model_metadata.LookupCustomMetadataMapAllocated("pyrpmd.onnx", allocator);
        require(text && std::strlen(text.get()) <= 1048576,
                "missing/oversized pyrpmd.onnx model metadata; use export_jaxpip.py", PYRPMD_INVALID_CONFIG_V1);
        const auto spec = json::parse(text.get());
        require(spec.at("schema") == "pyrpmd.onnx/1" && spec.at("batch_independent") == true,
                "model must declare schema pyrpmd.onnx/1 and independent structures", PYRPMD_INVALID_CONFIG_V1);
        const auto atoms = spec.at("atomic_numbers");
        require(atoms.is_array() && !atoms.empty() && atoms.size() <= 1000000,
                "invalid model atomic_numbers", PYRPMD_INVALID_CONFIG_V1);
        for (const auto& z : atoms) integer(z, 1, 118, "invalid atomic number");
        natoms = static_cast<int64_t>(atoms.size());
        const auto state = spec.at("state");
        integer(state.at("charge"), -1000000, 1000000, "invalid charge");
        integer(state.at("multiplicity"), 1, 1000000, "invalid multiplicity");
        if (config.contains("natoms")) require(config.at("natoms") == natoms, "atom count mismatch", PYRPMD_INVALID_CONFIG_V1);
        if (config.contains("atomic_numbers")) require(config.at("atomic_numbers") == atoms, "element ordering mismatch", PYRPMD_INVALID_CONFIG_V1);
        for (const char* key : {"charge", "multiplicity"})
            if (config.contains(key)) require(config.at(key) == state.at(key), "electronic state mismatch", PYRPMD_INVALID_CONFIG_V1);
        const std::string length = spec.at("position_unit"), eunit = spec.at("energy_unit");
        require((length == "angstrom" || length == "bohr") && (eunit == "eV" || eunit == "hartree"),
                "unsupported model units", PYRPMD_INVALID_CONFIG_V1);
        require(spec.at("force_unit") == eunit + "/" + length,
                "forces must be negative Cartesian gradients in energy/length", PYRPMD_INVALID_CONFIG_V1);
        // Constants are copied from the exporting PyRPMD/SciPy environment so
        // that a change of CODATA revision cannot silently change a model.
        if (length == "angstrom") coordinate_factor = constant(spec.at("constants"), "bohr_angstrom", 0.529177210544);
        if (eunit == "eV") energy_factor = constant(spec.at("constants"), "ev_to_hartree", 0.03674932217566422);
        require(session->GetInputCount() == 1 && session->GetOutputCount() == 2,
                "model must have one input and ordered (energy, forces) outputs", PYRPMD_INVALID_CONFIG_V1);
        tensor_contract(session->GetInputTypeInfo(0), {-1, natoms, 3});
        tensor_contract(session->GetOutputTypeInfo(0), {-1});
        tensor_contract(session->GetOutputTypeInfo(1), {-1, natoms, 3});
        input_name = session->GetInputNameAllocated(0, allocator).get();
        energy_name = session->GetOutputNameAllocated(0, allocator).get();
        force_name = session->GetOutputNameAllocated(1, allocator).get();
        metadata = json{{"schema","pyrpmd.pes/1"}, {"abi_version",1},
            {"provider_name","onnxruntime"}, {"provider_version",Ort::GetVersionString()},
            {"natoms",natoms}, {"atomic_numbers",atoms}, {"state",state},
            {"layout","B,N,3:C"}, {"dtype","float64"}, {"position_unit","bohr"},
            {"energy_unit","hartree"}, {"force_unit","hartree/bohr"},
            {"boundary_conditions","nonperiodic"}, {"execution_device",device},
            {"thread_safety","context_serial"}, {"model_spec",spec},
            {"capabilities",{{"stateful",false}, {"native_batch",true},
                {"independent_contexts",true}, {"host_memory",true},
                {"allow_cpu_fallback",cpu_fallback}}}}.dump();
    }
    void buffers(int64_t b, size_t n) {
        if (cached_batch == b) return;
        cached_batch = 0;
        outputs.clear();
        input = Ort::Value(nullptr);
        positions.resize(n); energy.resize(static_cast<size_t>(b)); force.resize(n);
        const std::array<int64_t,3> shape{b,natoms,3};
        const std::array<int64_t,1> eshape{b};
        input = Ort::Value::CreateTensor<double>(memory, positions.data(), n, shape.data(), shape.size());
        outputs.push_back(Ort::Value::CreateTensor<double>(memory, energy.data(), energy.size(), eshape.data(), 1));
        outputs.push_back(Ort::Value::CreateTensor<double>(memory, force.data(), n, shape.data(), shape.size()));
        cached_batch = b;
    }
};

extern "C" {
int32_t pyrpmd_pes_abi_version_v1(void) { return PYRPMD_PES_ABI_VERSION_V1; }
int32_t pyrpmd_pes_create_v1(const pyrpmd_create_options_v1* o, pyrpmd_pes_context_v1** out, pyrpmd_error_v1* e) {
    if (out) *out = nullptr;
    return guarded(e, PYRPMD_INVALID_CONFIG_V1, [&]{
        require(o && out, "null create argument");
        require(o->abi_version == 1 && o->struct_size == sizeof(*o), "create ABI mismatch", PYRPMD_ABI_MISMATCH_V1);
        require(o->config_json && o->config_json_bytes > 0 && o->config_json_bytes <= 1048576, "invalid config buffer");
        const auto cfg = json::parse(o->config_json, o->config_json + o->config_json_bytes);
        auto ctx = std::make_unique<pyrpmd_pes_context_v1>(cfg);
        *out = ctx.release();
    });
}
int32_t pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1* c, char* buf, int64_t cap,
                              int64_t* required, pyrpmd_error_v1* e) {
    return guarded(e, PYRPMD_INTERNAL_ERROR_V1, [&]{
        require(c && required && cap >= 0 && (cap == 0 || buf), "invalid metadata buffer");
        *required = static_cast<int64_t>(c->metadata.size() + 1);
        if (cap < *required) {
            if (cap > 0) buf[0] = '\0';
            throw Error(PYRPMD_BUFFER_TOO_SMALL_V1, "metadata buffer too small");
        }
        std::memcpy(buf, c->metadata.c_str(), static_cast<size_t>(*required));
    });
}
int32_t pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1* c, const pyrpmd_pes_input_v1* in,
                              pyrpmd_pes_output_v1* out, pyrpmd_error_v1* e) {
    return guarded(e, PYRPMD_BACKEND_FAILURE_V1, [&]{
        require(c && in && out, "null evaluate argument");
        require(in->abi_version == 1 && out->abi_version == 1 && in->struct_size == sizeof(*in)
                && out->struct_size == sizeof(*out), "evaluate ABI mismatch", PYRPMD_ABI_MISMATCH_V1);
        require(in->layout == 1 && in->scalar_type == 1, "expected contiguous float64 B,N,3");
        const int64_t b = in->nstructures, n = in->natoms;
        require(b > 0 && n == c->natoms && n <= INT64_MAX/3 && b <= INT64_MAX/(3*n), "invalid/overflowing batch shape");
        const int64_t count = b*n*3;
        require(static_cast<uint64_t>(count) <= std::numeric_limits<size_t>::max()/sizeof(double), "batch byte size overflow");
        require(in->positions && out->energies && out->forces && out->structure_status, "null tensor buffer");
        require(in->positions_count == count && out->forces_count == count && out->energies_count == b
                && out->structure_status_count == b, "tensor buffer count mismatch");
        if (out->structure_messages) {
            require(out->structure_message_stride > 0 && b <= INT64_MAX/out->structure_message_stride
                    && out->structure_messages_capacity >= b*out->structure_message_stride, "invalid frame message buffer");
        } else require(out->structure_message_stride == 0 && out->structure_messages_capacity == 0, "invalid omitted message buffer");
        std::lock_guard<std::mutex> lock(c->mutex);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::fill_n(out->energies, b, nan);
        std::fill_n(out->forces, count, nan);
        std::fill_n(out->structure_status, b, PYRPMD_STRUCTURE_FAILED_V1);
        for (int64_t k=0; k<b; ++k)
            if (out->structure_messages) out->structure_messages[k*out->structure_message_stride] = '\0';
        c->buffers(b, static_cast<size_t>(count));
        for (int64_t i=0; i<count; ++i) {
            const double x = in->positions[i] * c->coordinate_factor;
            require(std::isfinite(x), "nonfinite input coordinate");
            c->positions[static_cast<size_t>(i)] = x;
        }
        const char* input_names[] = {c->input_name.c_str()};
        const char* output_names[] = {c->energy_name.c_str(), c->force_name.c_str()};
        // Synchronous, with preallocated HOST outputs. CUDA transfers are handled
        // by ORT; this ABI does not accept device pointers or promise zero-copy.
        c->session->Run(Ort::RunOptions{nullptr}, input_names, &c->input, 1, output_names, c->outputs.data(), 2);
        const double ff = c->energy_factor * c->coordinate_factor;
        for (int64_t k=0; k<b; ++k) {
            const auto offset = static_cast<size_t>(k*n*3);
            const double energy = c->energy[static_cast<size_t>(k)] * c->energy_factor;
            bool good = std::isfinite(energy);
            for (int64_t j=0; j<3*n; ++j) good = good && std::isfinite(c->force[offset+j] * ff);
            if (!good) {
                out->structure_status[k] = PYRPMD_STRUCTURE_NONFINITE_V1;
                if (out->structure_messages) {
                    pyrpmd_error_v1 msg{out->structure_messages+k*out->structure_message_stride, out->structure_message_stride, 0};
                    diagnostic(&msg, "nonfinite ONNX energy/force");
                }
                continue;
            }
            out->energies[k] = energy;
            for (int64_t j=0; j<3*n; ++j) out->forces[offset+j] = c->force[offset+j] * ff;
            out->structure_status[k] = PYRPMD_STRUCTURE_OK_V1;
        }
    });
}
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1* c) { delete c; }
}
