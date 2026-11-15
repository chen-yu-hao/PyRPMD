# PR201 Colab T4 验证记录

- 提交：`354523c8` (`Enable JaxPIP GPU trajectory batching`)
- 日期：2026-11-14
- 硬件：Google Colab Tesla T4，15 GiB，driver 580.82.07
- ONNX Runtime：1.24.1 GPU，CUDA Execution Provider 可用
- 原生插件：`native/onnx/libpyrpmd_onnx.so`，Release 构建

## PyRPMD 与 PR201 回归

在 T4 上构建真实 ONNX Runtime C ABI 插件后运行：

```text
60 passed, 1 skipped in 6.73s
```

覆盖 ABI 单位转换、动态 batch、独立性、生命周期、原生传播、OpenMP context 隔离和 PR201 新增的跨轨迹 batch。GPU 调度器实际创建的 `cabi` provider metadata 为 `execution_device = cuda:0`，即使输入配置中的嵌套 `params.device` 是 `cpu`，调度器仍会把设备路由到 CUDA context。

## T4 GPU 正确性与效率

使用带 `pyrpmd.onnx/1` metadata 的三原子 float64 harmonic ONNX 合约模型，设置 `device = cuda:0` 和 `allow_cpu_fallback = false`。此设置要求模型完全由 CUDA EP 执行；session 创建成功说明该模型没有落到 CPU EP。

- GPU 与 CPU 结果最大能量绝对误差：`2.17e-19 hartree`
- GPU 与 CPU 结果最大力绝对误差：`0`
- 与解析结果最大力绝对误差：`4.34e-19 hartree/bohr`
- 失败帧：`0`
- `K=8, P=8` 跨轨迹 batch：`64` 帧合并为 `1` 次 PES 调用，正确性通过

| 轨迹数 K | 珠子数 P | scalar PES calls | batched PES calls | scalar 中位时间 | batch 中位时间 | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1 | 1 | 0.199 ms | 0.196 ms | 1.01x |
| 2 | 8 | 2 | 1 | 0.402 ms | 0.202 ms | 1.99x |
| 4 | 8 | 4 | 1 | 0.757 ms | 0.217 ms | 3.48x |
| 8 | 8 | 8 | 1 | 1.644 ms | 0.305 ms | 5.39x |
| 16 | 8 | 16 | 1 | 3.280 ms | 0.277 ms | 11.84x |

这些时间测量的是小型接口模型的 host-to-device、CUDA 推理和结果拷贝，适合验证 batching 的调用开销趋势，不代表真实化学 PES 的生产速度。

## 真实 JaxPIP ONNX 模型

使用仓库锁定的 JaxPIP/JAX/jax2onnx 栈导出 linear 和 neural-network 两个真实导出流程模型。两者在 CPU 原生 C ABI 路径均通过 batch 1/3/32/128、JAX 对比、有限差分和 100 步 native/python propagation 对比；报告见 [`jaxpip-cpu-report.json`](jaxpip-cpu-report.json)。

在 T4 上：

1. `allow_cpu_fallback = false`：linear 模型在 session 初始化时被明确拒绝，因为部分图节点被分配给 CPU EP。该失败符合 PR201 的显式 GPU 校验语义，不会静默退回 CPU。
2. `allow_cpu_fallback = true`：模型继续初始化，但执行到 `ScatterND` 时失败：`element_type 11 not implemented for ScatterND operator`。因此当前锁定的 JaxPIP 导出图不能在此 T4/ORT 1.24.1 环境中完成 GPU 或混合 CPU/GPU force evaluation。
3. 这说明 PR201 的 CUDA device routing、原生 batch 和跨轨迹 batching 已通过；真实 JaxPIP 模型能否上 T4 还受其 ONNX 图中 float64 `ScatterND` 的 provider kernel 支持限制。

原始 GPU batching 数据见 [`gpu-harmonic-report.json`](gpu-harmonic-report.json)。

图示解释见 [`PR201_T4可视化报告.md`](PR201_T4可视化报告.md)。
