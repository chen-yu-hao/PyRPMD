/* Custom float64 CUDA ScatterND operator registration for libpyrpmd_onnx.
 * SPDX-License-Identifier: MIT
 *
 * Upstream ONNX Runtime's CUDAExecutionProvider registers ScatterND for float,
 * half and bfloat16, omitting double (ONNX element type 11). This header keeps
 * CUDA runtime headers out of onnx_pes.cpp so CPU-only builds compile with a
 * standard C++17 toolchain without CUDA headers or libraries.
 */
#ifndef PYRPMD_ONNX_SCATTER_ND_CUDA_H_
#define PYRPMD_ONNX_SCATTER_ND_CUDA_H_

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>

namespace pyrpmd_onnx {

inline constexpr const char* kScatterNDCudaDomain = "com.pyrpmd.cuda";

#if PYRPMD_ONNX_HAS_CUDA_KERNELS

class ScatterNDCudaRegistration {
public:
    explicit ScatterNDCudaRegistration(int device_id);
    ~ScatterNDCudaRegistration();
    ScatterNDCudaRegistration(const ScatterNDCudaRegistration&) = delete;
    ScatterNDCudaRegistration& operator=(const ScatterNDCudaRegistration&) = delete;

    void Attach(Ort::SessionOptions& options);
    void ResetOutOfBoundsFlag() noexcept;
    bool ConsumeOutOfBoundsFlag() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // PYRPMD_ONNX_HAS_CUDA_KERNELS

}  // namespace pyrpmd_onnx

#endif  // PYRPMD_ONNX_SCATTER_ND_CUDA_H_
