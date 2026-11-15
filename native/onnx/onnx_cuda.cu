/* CUDA stream bridging, pointer validation, and unit-scaling/finiteness kernels
 * for zero-copy ONNX Runtime PES evaluation.
 * SPDX-License-Identifier: MIT
 */
#include "onnx_cuda.hpp"
#include "pyrpmd/pes_api_cuda_v1.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace pyrpmd_onnx_cuda {
namespace {

void check_cuda(cudaError_t err, const char* op, int32_t code = PYRPMD_CUDA_BACKEND_FAILURE_V1) {
    if (err != cudaSuccess) {
        cudaGetLastError();
        throw CudaError(code, std::string(op) + ": " + cudaGetErrorString(err));
    }
}

struct CudaDeviceGuard {
    int prev_device = 0;
    bool switched = false;

    explicit CudaDeviceGuard(int target_device) {
        check_cuda(cudaGetDevice(&prev_device), "cudaGetDevice");
        if (prev_device != target_device) {
            check_cuda(cudaSetDevice(target_device), "cudaSetDevice");
            switched = true;
        }
    }

    ~CudaDeviceGuard() noexcept {
        if (switched) {
            cudaSetDevice(prev_device);
            cudaGetLastError();
        }
    }
};

void validate_device_pointer(const void* ptr, int expected_device, const char* name) {
    if (!ptr) {
        throw CudaError(PYRPMD_CUDA_INVALID_ARGUMENT_V1, std::string("null ") + name + " device pointer");
    }
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        throw CudaError(
            PYRPMD_CUDA_INVALID_ARGUMENT_V1,
            std::string(name) + " is not a valid CUDA pointer: " + cudaGetErrorString(err));
    }
#if CUDART_VERSION >= 10000
    const bool is_device = (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
#else
    const bool is_device = (attr.memoryType == cudaMemoryTypeDevice);
#endif
    if (!is_device) {
        throw CudaError(
            PYRPMD_CUDA_INVALID_ARGUMENT_V1,
            std::string(name) + " must reside in CUDA device memory, not host memory");
    }
    if (attr.type == cudaMemoryTypeDevice && attr.device != expected_device) {
        throw CudaError(
            PYRPMD_CUDA_INVALID_ARGUMENT_V1,
            std::string(name) + " resides on cuda:" + std::to_string(attr.device)
                + " instead of context cuda:" + std::to_string(expected_device));
    }
}

bool ranges_overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) noexcept {
    if (!a || !b || a_bytes == 0 || b_bytes == 0) return false;
    const auto a0 = reinterpret_cast<uintptr_t>(a);
    const auto b0 = reinterpret_cast<uintptr_t>(b);
    const uintptr_t a1 = a0 + a_bytes;
    const uintptr_t b1 = b0 + b_bytes;
    return a0 < b1 && b0 < a1;
}

__global__ void scale_positions_kernel(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int64_t count,
    double factor) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        dst[idx] = src[idx] * factor;
    }
}

__global__ void finalize_outputs_kernel(
    const double* __restrict__ d_positions,
    double* __restrict__ d_energies,
    double* __restrict__ d_forces,
    int32_t* __restrict__ d_status,
    int64_t b,
    int64_t dof_per_frame,
    double coordinate_factor,
    double energy_factor,
    double force_factor) {
    __shared__ int s_all_finite[256];
    const double nan_val = nan("");

    for (int64_t k = blockIdx.x; k < b; k += gridDim.x) {
        const int64_t base = k * dof_per_frame;
        int local_ok = 1;

        for (int64_t j = threadIdx.x; j < dof_per_frame; j += blockDim.x) {
            const double q = d_positions[base + j];
            const double q_scaled = q * coordinate_factor;
            const double f_scaled = d_forces[base + j] * force_factor;
            if (!isfinite(q) || !isfinite(q_scaled) || !isfinite(f_scaled)) {
                local_ok = 0;
            }
            d_forces[base + j] = f_scaled;
        }

        double e_scaled = 0.0;
        if (threadIdx.x == 0) {
            e_scaled = d_energies[k] * energy_factor;
            if (!isfinite(e_scaled)) {
                local_ok = 0;
            }
        }

        s_all_finite[threadIdx.x] = local_ok;
        __syncthreads();

        for (unsigned int half = blockDim.x >> 1; half > 0; half >>= 1) {
            if (threadIdx.x < half) {
                s_all_finite[threadIdx.x] = s_all_finite[threadIdx.x] & s_all_finite[threadIdx.x + half];
            }
            __syncthreads();
        }

        const bool frame_ok = (s_all_finite[0] != 0);
        if (frame_ok) {
            if (threadIdx.x == 0) {
                d_energies[k] = e_scaled;
                if (d_status) {
                    d_status[k] = PYRPMD_CUDA_STRUCTURE_OK_V1;
                }
            }
        } else {
            if (threadIdx.x == 0) {
                d_energies[k] = nan_val;
                if (d_status) {
                    d_status[k] = PYRPMD_CUDA_STRUCTURE_NONFINITE_V1;
                }
            }
            for (int64_t j = threadIdx.x; j < dof_per_frame; j += blockDim.x) {
                d_forces[base + j] = nan_val;
            }
        }
        __syncthreads();
    }
}

}  // namespace

CudaExecState::CudaExecState(int device_id) : device_id_(device_id) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount", PYRPMD_CUDA_BACKEND_FAILURE_V1);
    if (device_id < 0 || device_id >= count) {
        throw CudaError(
            PYRPMD_CUDA_BACKEND_FAILURE_V1,
            "requested CUDA device index " + std::to_string(device_id)
                + " out of range (visible devices: " + std::to_string(count) + ")");
    }
    CudaDeviceGuard guard(device_id_);
    cudaStream_t stream = nullptr;
    cudaEvent_t ev_in = nullptr;
    cudaEvent_t ev_out = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    ort_stream_ = static_cast<void*>(stream);
    try {
        check_cuda(cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming), "cudaEventCreateWithFlags(in)");
        in_ready_event_ = static_cast<void*>(ev_in);
        check_cuda(cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming), "cudaEventCreateWithFlags(out)");
        out_ready_event_ = static_cast<void*>(ev_out);
    } catch (...) {
        if (ev_out) cudaEventDestroy(ev_out);
        if (ev_in) cudaEventDestroy(ev_in);
        if (stream) cudaStreamDestroy(stream);
        cudaGetLastError();
        throw;
    }
}

CudaExecState::~CudaExecState() noexcept {
    int prev = 0;
    const bool have_prev = (cudaGetDevice(&prev) == cudaSuccess);
    if (cudaSetDevice(device_id_) == cudaSuccess) {
        if (ort_stream_) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(ort_stream_));
        }
        if (d_scaled_positions_) {
            cudaFree(d_scaled_positions_);
            d_scaled_positions_ = nullptr;
        }
        if (out_ready_event_) {
            cudaEventDestroy(static_cast<cudaEvent_t>(out_ready_event_));
            out_ready_event_ = nullptr;
        }
        if (in_ready_event_) {
            cudaEventDestroy(static_cast<cudaEvent_t>(in_ready_event_));
            in_ready_event_ = nullptr;
        }
        if (ort_stream_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(ort_stream_));
            ort_stream_ = nullptr;
        }
        if (have_prev && prev != device_id_) {
            cudaSetDevice(prev);
        }
    }
    cudaGetLastError();
}

void CudaExecState::sync_ort_stream() {
    CudaDeviceGuard guard(device_id_);
    check_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(ort_stream_)), "cudaStreamSynchronize");
}

const double* CudaExecState::prepare_input_and_wait_ort(
    void* caller_stream_raw,
    const double* d_positions,
    double* d_energies,
    double* d_forces,
    int32_t* d_structure_status,
    int64_t b,
    int64_t count,
    double coordinate_factor) {
    CudaDeviceGuard guard(device_id_);

    validate_device_pointer(d_positions, device_id_, "d_positions");
    validate_device_pointer(d_energies, device_id_, "d_energies");
    validate_device_pointer(d_forces, device_id_, "d_forces");
    if (d_structure_status) {
        validate_device_pointer(d_structure_status, device_id_, "d_structure_status");
    }

    const size_t pos_bytes = static_cast<size_t>(count) * sizeof(double);
    const size_t ene_bytes = static_cast<size_t>(b) * sizeof(double);
    const size_t sta_bytes = d_structure_status ? static_cast<size_t>(b) * sizeof(int32_t) : 0;
    if (ranges_overlap(d_positions, pos_bytes, d_energies, ene_bytes)
        || ranges_overlap(d_positions, pos_bytes, d_forces, pos_bytes)
        || ranges_overlap(d_energies, ene_bytes, d_forces, pos_bytes)
        || ranges_overlap(d_positions, pos_bytes, d_structure_status, sta_bytes)
        || ranges_overlap(d_energies, ene_bytes, d_structure_status, sta_bytes)
        || ranges_overlap(d_forces, pos_bytes, d_structure_status, sta_bytes)) {
        throw CudaError(PYRPMD_CUDA_INVALID_ARGUMENT_V1, "CUDA input/output tensor buffers must not overlap");
    }

    const auto caller_stream = static_cast<cudaStream_t>(caller_stream_raw);
    const auto ort_stream = static_cast<cudaStream_t>(ort_stream_);
    const double* d_input = d_positions;

    if (caller_stream != ort_stream) {
        const auto ev_in = static_cast<cudaEvent_t>(in_ready_event_);
        check_cuda(cudaEventRecord(ev_in, caller_stream), "cudaEventRecord(in_ready)");
        check_cuda(cudaStreamWaitEvent(ort_stream, ev_in, 0), "cudaStreamWaitEvent(ort<-caller)");
    }

    if (coordinate_factor != 1.0) {
        const auto needed = static_cast<size_t>(count);
        if (needed > scaled_positions_cap_) {
            check_cuda(cudaStreamSynchronize(ort_stream), "cudaStreamSynchronize(resize)");
            if (d_scaled_positions_) {
                check_cuda(cudaFree(d_scaled_positions_), "cudaFree(d_scaled_positions)");
                d_scaled_positions_ = nullptr;
                scaled_positions_cap_ = 0;
            }
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_scaled_positions_), needed * sizeof(double)),
                "cudaMalloc(d_scaled_positions)");
            scaled_positions_cap_ = needed;
        }
        constexpr unsigned int threads = 256;
        const auto blocks = static_cast<unsigned int>(
            std::min<int64_t>((count + threads - 1) / threads, 65535));
        scale_positions_kernel<<<blocks, threads, 0, ort_stream>>>(
            d_positions, d_scaled_positions_, count, coordinate_factor);
        check_cuda(cudaGetLastError(), "scale_positions_kernel launch");
        d_input = d_scaled_positions_;
    }

    return d_input;
}

void CudaExecState::finalize_outputs_on_caller_stream(
    void* caller_stream_raw,
    const double* d_positions,
    double* d_energies,
    double* d_forces,
    int32_t* d_structure_status,
    int64_t b,
    int64_t natoms,
    double coordinate_factor,
    double energy_factor) {
    CudaDeviceGuard guard(device_id_);

    const auto caller_stream = static_cast<cudaStream_t>(caller_stream_raw);
    const auto ort_stream = static_cast<cudaStream_t>(ort_stream_);

    if (caller_stream != ort_stream) {
        const auto ev_out = static_cast<cudaEvent_t>(out_ready_event_);
        check_cuda(cudaEventRecord(ev_out, ort_stream), "cudaEventRecord(out_ready)");
        check_cuda(cudaStreamWaitEvent(caller_stream, ev_out, 0), "cudaStreamWaitEvent(caller<-ort)");
    }

    const int64_t dof = 3 * natoms;
    unsigned int threads = 32;
    while (threads < 256 && static_cast<int64_t>(threads) < dof) {
        threads <<= 1;
    }
    const auto blocks = static_cast<unsigned int>(std::min<int64_t>(b, 65535));
    const double force_factor = energy_factor * coordinate_factor;
    finalize_outputs_kernel<<<blocks, threads, 0, caller_stream>>>(
        d_positions, d_energies, d_forces, d_structure_status,
        b, dof, coordinate_factor, energy_factor, force_factor);
    check_cuda(cudaGetLastError(), "finalize_outputs_kernel launch");
}

}  // namespace pyrpmd_onnx_cuda
