# JaxPIP T4 效率与不足报告

硬件：Tesla T4 15 GiB，JAX 0.11.1，PyTorch 2.14.1+cu130。

## 关键结果

- linear/NN GPU batch 能量和力最大误差：`0`
- DLPack device pointer 零拷贝：通过
- TorchDynamics + umbrella bias：通过
- US：2 samples；AUS：pilot 后 frozen；recrossing：2 pairs
- Torch regression：`389 passed`

## 效率结论

真实 JaxPIP smoke 测试使用 batch=4、3 个动力学步。linear 为 `230.195 ms`，NN 为 `10.656 ms`。这包含首次执行和编译影响，不能当作稳态生产吞吐。

此前 T4 harmonic contract 的跨轨迹 batching 在 K=1/2/4/8/16 时达到 `1.01×` 到 `11.84×` 加速；该数据只说明 batching 机制，不代表真实 JaxPIP 化学 PES。

## 不足

1. 没有用户实际 fitted chemical model 的 test-set 和 rate convergence 结果。
2. 没有长时间、多窗口、多 parent/child 的稳态吞吐与显存峰值。
3. T4 预装 JAX 0.11.1/CUDA 13 与仓库 requirements 的 JAX 0.9/CUDA 12 仍需统一。
4. 动力学仍由 Python orchestration 驱动每步 PES，尚未成为整段融合的 custom C++/CUDA runtime。
5. GPU rate 当前限制固定非周期系统、内置 CV 和 Andersen sampling。

完整图表请打开 [`t4-efficiency-report.html`](t4-efficiency-report.html)。
