/* Custom float64 CUDA ScatterND kernel for libpyrpmd_onnx.
 * SPDX-License-Identifier: MIT
 */
#include "scatter_nd_cuda.h"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyrpmd_onnx {
namespace {

constexpr int32_t kMaxScatterRank = 16;

enum class ScatterReduction : int32_t {
    kNone = 0,
    kAdd = 1,
    kMul = 2,
    kMax = 3,
    kMin = 4,
};

struct ScatterNDParams {
    int64_t dims[kMaxScatterRank];
    int64_t prefix_strides[kMaxScatterRank];
    int64_t num_updates;
    int64_t slice_size;
    int64_t total_updates;
    int32_t k;
    volatile int32_t* oob_flag;
};

__device__ __forceinline__ void atomic_add_f64(double* addr, double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicAdd(addr, val);
#else
    auto* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_ull;
    unsigned long long assumed = 0;
    do {
        assumed = old;
        const double sum = __longlong_as_double(static_cast<long long>(assumed)) + val;
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(sum)));
    } while (assumed != old);
#endif
}

__device__ __forceinline__ void atomic_mul_f64(double* addr, double val) {
    auto* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_ull;
    unsigned long long assumed = 0;
    do {
        assumed = old;
        const double prod = __longlong_as_double(static_cast<long long>(assumed)) * val;
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(prod)));
    } while (assumed != old);
}

__device__ __forceinline__ void atomic_max_f64(double* addr, double val) {
    auto* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_ull;
    unsigned long long assumed = 0;
    do {
        assumed = old;
        const double cur = __longlong_as_double(static_cast<long long>(assumed));
        double next = cur;
        if (isnan(cur) || isnan(val)) {
            next = nan("");
        } else if (val > cur || (val == cur && !signbit(val))) {
            next = val;
        }
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(next)));
    } while (assumed != old);
}

__device__ __forceinline__ void atomic_min_f64(double* addr, double val) {
    auto* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_ull;
    unsigned long long assumed = 0;
    do {
        assumed = old;
        const double cur = __longlong_as_double(static_cast<long long>(assumed));
        double next = cur;
        if (isnan(cur) || isnan(val)) {
            next = nan("");
        } else if (val < cur || (val == cur && signbit(val))) {
            next = val;
        }
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(next)));
    } while (assumed != old);
}

template <ScatterReduction Mode>
__global__ void scatter_nd_f64_kernel(double* __restrict__ output,
                                      const int64_t* __restrict__ indices,
                                      const double* __restrict__ updates,
                                      ScatterNDParams params) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                + static_cast<int64_t>(threadIdx.x);
    for (; tid < params.total_updates; tid += stride) {
        const int64_t u = tid / params.slice_size;
        const int64_t s = tid % params.slice_size;
        const int64_t* idx_tuple = indices + u * static_cast<int64_t>(params.k);
        int64_t offset = s;
        bool oob = false;
        for (int32_t m = 0; m < params.k; ++m) {
            int64_t idx = idx_tuple[m];
            const int64_t dim = params.dims[m];
            if (idx < 0) idx += dim;
            if (idx < 0 || idx >= dim) {
                oob = true;
                break;
            }
            offset += idx * params.prefix_strides[m];
        }
        if (oob) {
            if (params.oob_flag != nullptr) {
                *params.oob_flag = 1;
                __threadfence_system();
            }
            output[s] = nan("");
            continue;
        }
        const double val = updates[tid];
        if constexpr (Mode == ScatterReduction::kNone) {
            output[offset] = val;
        } else if constexpr (Mode == ScatterReduction::kAdd) {
            atomic_add_f64(output + offset, val);
        } else if constexpr (Mode == ScatterReduction::kMul) {
            atomic_mul_f64(output + offset, val);
        } else if constexpr (Mode == ScatterReduction::kMax) {
            atomic_max_f64(output + offset, val);
        } else {
            atomic_min_f64(output + offset, val);
        }
    }
}

bool safe_mul_nonneg(int64_t a, int64_t b, int64_t* out) noexcept {
    if (a < 0 || b < 0) return false;
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > std::numeric_limits<int64_t>::max() / b) return false;
    *out = a * b;
    return true;
}

OrtStatusPtr get_tensor_shape_and_type(const OrtApi& api, const OrtValue* value,
                                       ONNXTensorElementDataType expected_type,
                                       std::vector<int64_t>* dims) noexcept {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatusPtr st = api.GetTensorTypeAndShape(value, &info)) return st;
    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (OrtStatusPtr st = api.GetTensorElementType(info, &elem_type)) {
        api.ReleaseTensorTypeAndShapeInfo(info);
        return st;
    }
    if (elem_type != expected_type) {
        api.ReleaseTensorTypeAndShapeInfo(info);
        return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND CUDA tensor element type mismatch");
    }
    size_t rank = 0;
    if (OrtStatusPtr st = api.GetDimensionsCount(info, &rank)) {
        api.ReleaseTensorTypeAndShapeInfo(info);
        return st;
    }
    try {
        dims->resize(rank);
    } catch (...) {
        api.ReleaseTensorTypeAndShapeInfo(info);
        return api.CreateStatus(ORT_RUNTIME_EXCEPTION, "ScatterND shape allocation failed");
    }
    if (rank > 0) {
        if (OrtStatusPtr st = api.GetDimensions(info, dims->data(), rank)) {
            api.ReleaseTensorTypeAndShapeInfo(info);
            return st;
        }
    }
    api.ReleaseTensorTypeAndShapeInfo(info);
    return nullptr;
}

struct ScatterNDCudaKernel {
    const OrtApi& api;
    ScatterReduction reduction;
    volatile int32_t* dev_oob_flag;

    ScatterNDCudaKernel(const OrtApi& ort_api, ScatterReduction red, volatile int32_t* oob_ptr) noexcept
        : api(ort_api), reduction(red), dev_oob_flag(oob_ptr) {}

    OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
        const OrtValue* data_val = nullptr;
        const OrtValue* indices_val = nullptr;
        const OrtValue* updates_val = nullptr;
        if (OrtStatusPtr st = api.KernelContext_GetInput(context, 0, &data_val)) return st;
        if (OrtStatusPtr st = api.KernelContext_GetInput(context, 1, &indices_val)) return st;
        if (OrtStatusPtr st = api.KernelContext_GetInput(context, 2, &updates_val)) return st;

        std::vector<int64_t> data_shape, indices_shape, updates_shape;
        if (OrtStatusPtr st = get_tensor_shape_and_type(
                api, data_val, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &data_shape)) return st;
        if (OrtStatusPtr st = get_tensor_shape_and_type(
                api, indices_val, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &indices_shape)) return st;
        if (OrtStatusPtr st = get_tensor_shape_and_type(
                api, updates_val, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &updates_shape)) return st;

        const size_t r = data_shape.size();
        const size_t q = indices_shape.size();
        if (r < 1 || r > static_cast<size_t>(kMaxScatterRank)) {
            return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND data rank must be in [1, 16]");
        }
        if (q < 1) {
            return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND indices rank must be >= 1");
        }
        const int64_t k64 = indices_shape[q - 1];
        if (k64 < 1 || static_cast<size_t>(k64) > r) {
            return api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    "ScatterND indices.shape[-1] must be in [1, rank(data)]");
        }
        const size_t k = static_cast<size_t>(k64);
        const size_t expected_updates_rank = (q - 1) + (r - k);
        if (updates_shape.size() != expected_updates_rank) {
            return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND updates rank mismatch");
        }
        for (size_t j = 0; j < q - 1; ++j) {
            if (updates_shape[j] != indices_shape[j]) {
                return api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "ScatterND updates leading dimensions must match indices");
            }
        }
        for (size_t m = 0; m < r - k; ++m) {
            if (updates_shape[(q - 1) + m] != data_shape[k + m]) {
                return api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "ScatterND updates trailing dimensions must match data slice");
            }
        }

        int64_t data_elements = 1;
        for (size_t i = 0; i < r; ++i) {
            if (!safe_mul_nonneg(data_elements, data_shape[i], &data_elements)) {
                return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND data shape overflow");
            }
        }
        int64_t num_updates = 1;
        for (size_t j = 0; j < q - 1; ++j) {
            if (!safe_mul_nonneg(num_updates, indices_shape[j], &num_updates)) {
                return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND indices shape overflow");
            }
        }
        int64_t slice_size = 1;
        for (size_t m = k; m < r; ++m) {
            if (!safe_mul_nonneg(slice_size, data_shape[m], &slice_size)) {
                return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND slice shape overflow");
            }
        }
        int64_t total_updates = 0;
        if (!safe_mul_nonneg(num_updates, slice_size, &total_updates)) {
            return api.CreateStatus(ORT_INVALID_ARGUMENT, "ScatterND updates count overflow");
        }

        OrtValue* output_val = nullptr;
        if (OrtStatusPtr st = api.KernelContext_GetOutput(
                context, 0, data_shape.data(), data_shape.size(), &output_val)) {
            return st;
        }
        if (data_elements == 0) {
            return nullptr;
        }

        const void* data_raw = nullptr;
        const void* indices_raw = nullptr;
        const void* updates_raw = nullptr;
        void* output_raw = nullptr;
        if (OrtStatusPtr st = api.GetTensorMutableData(const_cast<OrtValue*>(data_val),
                                                       const_cast<void**>(&data_raw))) return st;
        if (OrtStatusPtr st = api.GetTensorMutableData(OutputCast(output_val), &output_raw)) return st;

        void* raw_stream = nullptr;
        if (OrtStatusPtr st = api.KernelContext_GetGPUComputeStream(context, &raw_stream)) return st;
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_stream);

        const size_t data_bytes = static_cast<size_t>(data_elements) * sizeof(double);
        if (output_raw != data_raw) {
            const cudaError_t copy_err = cudaMemcpyAsync(
                output_raw, data_raw, data_bytes, cudaMemcpyDeviceToDevice, stream);
            if (copy_err != cudaSuccess) {
                return api.CreateStatus(ORT_RUNTIME_EXCEPTION, cudaGetErrorString(copy_err));
            }
        }
        if (total_updates == 0) {
            return nullptr;
        }

        if (OrtStatusPtr st = api.GetTensorMutableData(const_cast<OrtValue*>(indices_val),
                                                       const_cast<void**>(&indices_raw))) return st;
        if (OrtStatusPtr st = api.GetTensorMutableData(const_cast<OrtValue*>(updates_val),
                                                       const_cast<void**>(&updates_raw))) return st;

        ScatterNDParams params{};
        params.num_updates = num_updates;
        params.slice_size = slice_size;
        params.total_updates = total_updates;
        params.k = static_cast<int32_t>(k);
        params.oob_flag = dev_oob_flag;

        std::array<int64_t, kMaxScatterRank + 1> suffix_prod{};
        suffix_prod[r] = 1;
        for (size_t i = r; i > 0; --i) {
            suffix_prod[i - 1] = suffix_prod[i] * data_shape[i - 1];
        }
        for (size_t m = 0; m < k; ++m) {
            params.dims[m] = data_shape[m];
            params.prefix_strides[m] = suffix_prod[m + 1];
        }

        constexpr int kThreads = 256;
        const int64_t raw_blocks = (total_updates + kThreads - 1) / kThreads;
        const unsigned int blocks = static_cast<unsigned int>(
            raw_blocks > 65535 ? 65535 : raw_blocks);

        auto* out_ptr = static_cast<double*>(output_raw);
        const auto* idx_ptr = static_cast<const int64_t*>(indices_raw);
        const auto* upd_ptr = static_cast<const double*>(updates_raw);

        switch (reduction) {
            case ScatterReduction::kNone:
                scatter_nd_f64_kernel<ScatterReduction::kNone><<<blocks, kThreads, 0, stream>>>(
                    out_ptr, idx_ptr, upd_ptr, params);
                break;
            case ScatterReduction::kAdd:
                scatter_nd_f64_kernel<ScatterReduction::kAdd><<<blocks, kThreads, 0, stream>>>(
                    out_ptr, idx_ptr, upd_ptr, params);
                break;
            case ScatterReduction::kMul:
                scatter_nd_f64_kernel<ScatterReduction::kMul><<<blocks, kThreads, 0, stream>>>(
                    out_ptr, idx_ptr, upd_ptr, params);
                break;
            case ScatterReduction::kMax:
                scatter_nd_f64_kernel<ScatterReduction::kMax><<<blocks, kThreads, 0, stream>>>(
                    out_ptr, idx_ptr, upd_ptr, params);
                break;
            case ScatterReduction::kMin:
                scatter_nd_f64_kernel<ScatterReduction::kMin><<<blocks, kThreads, 0, stream>>>(
                    out_ptr, idx_ptr, upd_ptr, params);
                break;
        }
        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            return api.CreateStatus(ORT_RUNTIME_EXCEPTION, cudaGetErrorString(launch_err));
        }
        return nullptr;
    }

    static OrtValue* OutputCast(OrtValue* v) noexcept { return v; }
};

struct ScatterNDCudaOp : Ort::CustomOpBase<ScatterNDCudaOp, ScatterNDCudaKernel, true> {
    explicit ScatterNDCudaOp(volatile int32_t* dev_oob_flag) noexcept : dev_oob_flag_(dev_oob_flag) {}

    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** op_kernel) const noexcept {
        *op_kernel = nullptr;
        ScatterReduction mode = ScatterReduction::kNone;
        size_t attr_len = 0;
        OrtStatusPtr st = api.KernelInfoGetAttribute_string(info, "reduction", nullptr, &attr_len);
        if (st == nullptr && attr_len > 0) {
            std::string attr(attr_len, '\0');
            if (OrtStatusPtr st2 = api.KernelInfoGetAttribute_string(info, "reduction", attr.data(), &attr_len)) {
                return st2;
            }
            if (!attr.empty() && attr.back() == '\0') attr.pop_back();
            if (attr == "none") {
                mode = ScatterReduction::kNone;
            } else if (attr == "add") {
                mode = ScatterReduction::kAdd;
            } else if (attr == "mul") {
                mode = ScatterReduction::kMul;
            } else if (attr == "max") {
                mode = ScatterReduction::kMax;
            } else if (attr == "min") {
                mode = ScatterReduction::kMin;
            } else {
                return api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Unsupported ScatterND reduction mode for CUDA float64 kernel");
            }
        } else if (st != nullptr) {
            // Optional attribute omitted: ONNX ScatterND defaults to reduction = "none".
            api.ReleaseStatus(st);
        }

        auto* kernel = new (std::nothrow) ScatterNDCudaKernel(api, mode, dev_oob_flag_);
        if (kernel == nullptr) {
            return api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Failed to allocate ScatterNDCudaKernel");
        }
        *op_kernel = kernel;
        return nullptr;
    }

    const char* GetName() const noexcept { return "ScatterND"; }
    const char* GetExecutionProviderType() const noexcept { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const noexcept { return 3; }
    ONNXTensorElementDataType GetInputType(size_t index) const noexcept {
        return index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                          : ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    size_t GetOutputTypeCount() const noexcept { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }

private:
    volatile int32_t* dev_oob_flag_;
};

}  // namespace

struct ScatterNDCudaRegistration::Impl {
    int32_t* host_oob_flag = nullptr;
    volatile int32_t* dev_oob_flag = nullptr;
    std::unique_ptr<ScatterNDCudaOp> op;
    Ort::CustomOpDomain domain{kScatterNDCudaDomain};

    explicit Impl(int device_id) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA device selection failed: ") + cudaGetErrorString(err));
        }
        err = cudaHostAlloc(reinterpret_cast<void**>(&host_oob_flag), sizeof(int32_t),
                            cudaHostAllocPortable | cudaHostAllocMapped);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA mapped host allocation failed: ") + cudaGetErrorString(err));
        }
        *static_cast<volatile int32_t*>(host_oob_flag) = 0;
        int32_t* raw_dev_flag = nullptr;
        err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&raw_dev_flag), host_oob_flag, 0);
        if (err != cudaSuccess) {
            cudaFreeHost(host_oob_flag);
            host_oob_flag = nullptr;
            throw std::runtime_error(std::string("CUDA mapped device pointer lookup failed: ")
                                     + cudaGetErrorString(err));
        }
        dev_oob_flag = raw_dev_flag;
        op = std::make_unique<ScatterNDCudaOp>(dev_oob_flag);
        domain.Add(op.get());
    }

    ~Impl() {
        if (host_oob_flag != nullptr) {
            cudaFreeHost(host_oob_flag);
            host_oob_flag = nullptr;
        }
    }
};

ScatterNDCudaRegistration::ScatterNDCudaRegistration(int device_id)
    : impl_(std::make_unique<Impl>(device_id)) {}

ScatterNDCudaRegistration::~ScatterNDCudaRegistration() = default;

void ScatterNDCudaRegistration::Attach(Ort::SessionOptions& options) {
    options.Add(impl_->domain);
}

void ScatterNDCudaRegistration::ResetOutOfBoundsFlag() noexcept {
    if (impl_ && impl_->host_oob_flag) {
        *static_cast<volatile int32_t*>(impl_->host_oob_flag) = 0;
    }
}

bool ScatterNDCudaRegistration::ConsumeOutOfBoundsFlag() noexcept {
    if (!impl_ || !impl_->host_oob_flag) return false;
    auto* ptr = static_cast<volatile int32_t*>(impl_->host_oob_flag);
    const bool triggered = (*ptr != 0);
    *ptr = 0;
    return triggered;
}

}  // namespace pyrpmd_onnx
