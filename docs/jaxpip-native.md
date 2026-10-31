# JaxPIP 原生批量势能路径

从 PySCF 参考数据开始的完整离线流程见
[数据集、线性/PIP-NN 拟合、导出与运行](zh/manual/jaxpip-generation.md)
（[English](manual/jaxpip-generation.md)），其中说明数据划分、单位和显式精度门槛。

本功能使用 `JaxPIP / Equinox → jax2onnx → ONNX Runtime C++ → PyRPMD PES C ABI v1
→ 现有 C++17 连续传播`。Python 只负责离线导出、初始化和任务调度，受支持的
原生传播 block 内不逐步调用 Python PES。

构建命令、导出工具、完整配置和接口约定见
[原生 ONNX 插件说明](../native/onnx/README.md)。默认使用 CPU；CUDA 需要匹配的
ONNX Runtime GPU SDK 与系统依赖，不能把 CPU CI 通过当作 GPU 验证完成。

## 已实现

每个 context 只加载一次模型；一次推理全部 `[B,N,3]` 坐标；缓存 host buffers；
float64 能量和负梯度；angstrom/eV 与 bohr/hartree 成套转换；严格检查原子顺序、
电子态、I/O 形状和逐结构失败状态。显式 CUDA 请求失败时不偷偷切回 CPU。

复用 `provider="cabi"` 与现有连续传播，不引入 Python 回调包装，也不复制积分器。
独立 context 可供 C++/OpenMP 调用方并发使用；同一 context 加锁保护。

## 导出兼容性

导出器先生成单构型的能量与力 JAXPR，再通过 `vmap` 导出动态批量，避免
转换器的 Equinox 插件干扰自动微分，也避免固定整个 batch 的中间维度。
对固定的 jax2onnx 0.17.1，另有仅在导出上下文内生效的 Pad 形状元数据修正：
将 JAX 符号维度转换为 ONNX 符号名，不改变 Pad 运算或数值计算。兼容层在
成功或异常退出时均恢复原插件引用，不修改已安装依赖文件；该信息写入模型
元数据。导出应使用独立进程，不支持同一解释器内多线程同时转换。

这些兼容措施不放宽 float64、动态形状、多 batch 能量/力一致性或有限差分
验证；任何检查失败时，不会发布待运行的 ONNX 文件。

## 不应由架构图推断为已经实现的功能

- 插件接受 `B=K*P`，但现有轨迹调度器不会自动把 K 条轨迹打包成一次推理。
- OpenMP 测试验证独立 context 的并发正确性，不是新增了一个多轨迹 RPMD 驱动。
- C ABI v1 仍使用 host memory，GPU 路径仍有主机/设备传输；不是全 GPU 积分器。
- 不包含 OpenVINO、周期性晶胞、virial 或非绝热耦合；也不附送真实化学体系的拟合势能面。

PIP/FI 基组不是训练好的势能面。生产计算仍须提供对应 checkpoint，并独立验证
长程渐近、反应通道、力的精度与轨迹能量守恒。测试报告区分初始化、批量 PES
调用与传播耗时，不把小模型结果宣称为实际反应体系的加速倍数。
