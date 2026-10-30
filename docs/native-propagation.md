# C++17 分段传播与 Python 兼容路径

PyRPMD 在现有 `Cayley` 内核上加入了完整 BCB 时间步循环。势能、偏置、
约束与所需观测均有原生实现时，一个分段只进入 C++ 一次，直接调用 C ABI
或 Fortran `BIND(C)` 势能。自定义 Python PES/CV 不会被静默替换。

## 选择执行路径

```toml
[dynamics]
timestep = "0.1 fs"
loop_backend = "auto"  # auto / native / python
block_steps = 128      # 1..4096，最大分段长度，不是采样间隔
```

`auto` 在执行前检查全部能力；不支持时使用 Python 兼容路径，并记录原因。
`native` 要求融合路径，缺少任一能力立即报错。`python` 是数值对照路径。
已有 `native` 参数仍只控制 Cayley 小内核，并不等同于 `loop_backend`。
例如，`native="required"` 可以要求编译的 Cayley，但仍使用自定义 Python PES。

Python API：

```python
result = dynamics.propagate_block(
    state, 100, cv=my_cv, bias=my_bias,
    record_energy=True, record_states=False,
)
# state 原位更新；结果按实际观测时刻排列。
print(result.backend, result.completed, result.fallback_reason)
print(result.steps, result.energies, result.values)
```

`observe` 可传入长度等于步数的布尔 mask。默认每步记录能量、质心位置和
质心动量；`cv` 额外记录反应坐标。能量是总环哈密顿量 **H_P**，不是 H_P/P。
`record_states=True` 额外保留完整 q/p，供需要回退或检查的调用方使用。
每段观测缓冲区限制为 256 MiB；超过时要求减小分段，绝不自动降低记录频率。

## 已实现的原生能力

| 部件 | 原生范围 |
| --- | --- |
| 积分 | 原有周期 Cayley + 两次物理力 kick；真实质量、float64 |
| PES | PyRPMD C ABI v1；显式注册的 Fortran batch v1；实际 BKMP2 |
| CV | Distance、COMDistance、BondDifference、Cartesian、RPMDrateCoordinate、RateSurface、它们的 CVStack |
| estimator | 上述几何 CV 的 centroid/bead_average；速率坐标仍要求 centroid |
| 偏置 | 全矩阵 UmbrellaBias、FluxMetricBias、CompositeBias |
| 约束 | 与原 Python 实现一致的 centroid Cayley-RATTLE；片段质心位置/动量投影 |
| 观测 | 请求时刻的完整环能量、质心 q/p、CV；可选完整 q/p |

RPMDrate 等价通道仍采用 `min_bond(max_equivalent_TS)` 分支规则；分支
切换点的不可微性没有因编译而消失。Umbrella 的内部能量仍为 `P*W`，
flux metric 仍为 `-P*log(fs)/beta`，不改变统计测度。

当前 Angle、任意 PythonCV、用户自定义 bias、修改过方法的实例/子类、
普通 Python PES、ASE、i-PI、f2py、MPI/进程珠子代理、GPU provider 均保留
Python 路径，除非显式提供满足契约的原生绑定。不会跳过用户函数来获取加速。
BCOCB/PILE 的采样实现也保持原有 Python/NumPy 路径；本次融合的是 BCB。

## 工作流接入与边界

FES/PMF 在 Andersen 动量重采样、样本时刻、checkpoint 的原有时间轴上工作。
样本可在分段内部按 mask 记录；分段不会跨过热浴或 checkpoint 边界。
受约束 parent 使用原生 RATTLE，recrossing child 在原生层**逐步**记录分割面
符号，再按原来的通量权重汇总。

dRPMD 热 parent 保留 NumPy RNG 与原来的热化链、重采样顺序；碰撞传播每步
检查能量，分段结束于最近的分类/checkpoint 边界。任意 Python 分类器仍在
原来的检查时刻看到完整 q/p。不会提前传播超过终态分类时刻，再事后截断结果。
这里没有新增动态任务池、改变 MPI 拓扑或添加 GPU 积分器。

每次调用记录 `native_steps`、`python_steps`、`fallback_reasons`；工作流的
`propagation_this_call` / HDF5 stream 属性用于证明实际执行路径，不是历史累计
计数。仅打开已完成 checkpoint 时，不能据此把以前的步骤计为本次原生执行。

## 势能绑定契约

`pyrpmd.potentials.native.NativePESBinding` 持有动态库 owner、函数、上下文与
provider 的同一把锁。整段调用期间保持它们存活，并阻止其他线程调用/销毁
同一上下文。BKMP2 使用按动态库规范路径共享的锁，避免不同 Python 模块别名
同时进入旧 Fortran COMMON。没有擅自将旧 Fortran 势能改成可重入 OpenMP PES。

C ABI v1 provider 必须显式声明 `capabilities.stateful=false` 才自动提供绑定。
PythonPotential 只在单位和力符号完全恒等、delegate 主动提供绑定时透传。
Fortran batch v1 的显式 `BIND(C)` 函数原型如下：

```c
void evaluate(int32_t B, const double *q, double *energy,
              double *force, int32_t *status);
```

这里 B 按值传递；q/force 是 C 连续 `[B,N,3]`，对应 Fortran `(3,N,B)`；
坐标 bohr，能量 hartree，力 hartree/bohr；status 每帧 0 表示成功。
提供者保证固定 N、完整填写输出、确定性的物理能量/力，不使用 Python 回调，
不令异常或 STOP 穿过原生边界。任意旧 Fortran 签名仍需要 wrapper，不能猜测。
此接口不是不可信代码的安全沙箱。

每步仍有两次力调用，没有跨段隐含 force cache。记录终点能量时，原生路径
复用第二次力调用同时返回的能量；片段投影改变终点坐标时重新计算能量。
因此涉及能量观测的性能提升还包含**消除重复 PES 求值**，不能全部归因于
去掉 Python 循环。主工作缓冲区在分段内复用，但 CV 求导仍有局部临时对象。

## 故障与续跑

每个时间步及其请求的观测是一个提交单元。PES/status/NaN、偏置、CV 或约束
失败时，状态保留在最后成功步；`PropagationError.result` 保存有效前缀。
`raise_on_error=False` 返回同一有效前缀，由 dRPMD 记录失败。原生执行失败
后**不会重新用 Python 跑一遍**，避免重复更新或隐藏数值错误。

`loop_backend` 和 `block_steps` 不参与科学参数哈希，允许更改分段后续跑；
PES、温度、珠子数、步长及原有物理检查保持不变。在同一原生后端、环境和
初态下，测试要求不同分段长度及中断续跑逐值一致；原生/Python 之间由于
归约顺序可能有最后几位舍入差，不承诺长混沌轨迹逐位相同。文件系统异常
仍终止作业，不会被归类为化学反应失败。

## 可复现验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_native_propagation.py
python benchmarks/h3/build_pes.py
python benchmarks/h3/fused_propagation.py --output runs/native-fusion/report.json
```

新增测试覆盖数值映射、观测、P 缩放、RATTLE、RNG 边界、原生故障有效前缀、
Python 回退、C ABI 生命周期和分段一致性。真实 BKMP2 研究运行 NVE、二维
umbrella、rate-flux bias、约束 parent、逐步 recrossing，并运行实际 FES/PMF/
dRPMD 工作流与中断恢复。计时记录原始重复和实际后端，不设脆弱的速度验收阈值。

微基准用原先公开的 `Dynamics.step()` 作为对照；小工作流比较的是显式
`loop_backend="python"` 的兼容路径与 native，后者包含相同的检查点/输出。
两者不可混为同一种端到端加速比。短验证任务用于程序一致性，
`physical_convergence=false`；不把它当作新的收敛化学速率计算。
