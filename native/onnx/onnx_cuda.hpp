/* Optional CUDA stream/kernel helpers for zero-copy ONNX Runtime PES evaluation.
 * SPDX-License-Identifier: MIT
 * Pure C++17 header so onnx_pes.cpp does not depend on CUDA Toolkit headers.
 */
#ifndef PYRPMD_ONNX_CUDA_HPP
#define PYRPMD_ONNX_CUDA_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pyrpmd_onnx_cuda {

struct CudaError : std::runtime_error {
    int32_t code;
    CudaError(int32_t rc, const std::string& msg) : std::runtime_error(msg), code(rc) {}
};

class CudaExecState {
public:
    explicit CudaExecState(int device_id);
    ~CudaExecState() noexcept;

    CudaExecState(const CudaExecState&) = delete;
    CudaExecState& operator=(const CudaExecState&) = delete;

    int device_id() const noexcept { return device_id_; }
    void* ort_stream() const noexcept { return ort_stream_; }

    void sync_ort_stream();

    const double* prepare_input_and_wait_ort(
        void* caller_stream,
        const double* d_positions,
        double* d_energies,
        double* d_forces,
        int32_t* d_structure_status,
        int64_t b,
        int64_t count,
        double coordinate_factor);

    void finalize_outputs_on_caller_stream(
        void* caller_stream,
        const double* d_positions,
        double* d_energies,
        double* d_forces,
        int32_t* d_structure_status,
        int64_t b,
        int64_t natoms,
        double coordinate_factor,
        double energy_factor);

private:
    int device_id_ = 0;
    void* ort_stream_ = nullptr;
    void* in_ready_event_ = nullptr;
    void* out_ready_event_ = nullptr;
    double* d_scaled_positions_ = nullptr;
    size_t scaled_positions_cap_ = 0;
};

}  // namespace pyrpmd_onnx_cuda

#endif  // PYRPMD_ONNX_CUDA_HPP
