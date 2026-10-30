# 分段传播 API：观测、RATTLE 与成功前缀

`Dynamics.propagate_block()` 在已有 C++17 Cayley 上执行完整 BCB 时间步、
显式原生 PES、支持的 CV/偏置及 centroid Cayley–RATTLE。它与
[`Dynamics.propagate()`](native-propagation.md) 同时保留：后者失败时整块
回滚，本接口逐步提交并在数值失败时保留成功前缀。

## 调用与记录

```python
result = dynamics.propagate_block(
    state, 100,
    cv=my_cv,
    bias=my_bias,
    observe=[True] * 100,
    record_energy=True,
    record_states=False,
    backend="auto",
)
print(result.backend, result.completed, result.fallback_reason)
print(result.steps, result.times, result.energies, result.values)
```

`state` 原位更新。单次 `steps` 为 0..4096，0 返回空结果；`observe` 是与
步数等长的布尔 mask，默认每步观测。`record_states=True` 额外保存观测
时刻的完整 q/p。结果还包含质心位置和质心动量；能量是总环哈密顿量
**H_P**，不是 H_P/P。观测缓存限制为 256 MiB，超过时要求缩小分段，
不会降低用户指定的记录频率。

输入文件仍使用 `block_steps=256`，合法范围 1..65536。使用本接口的
dRPMD 碰撞和受约束母轨迹会将实际分段限制在 4096 及相应的热浴、分类
或 checkpoint 边界。
直接 Python 调用超过 4096 则报错，调用者应自行分段。

`auto` 在执行前检查能力，不支持时进入 Python 兼容路径；`native` 缺少
任何必要能力即报错；`python` 强制参考路径。`native=False` 关闭编译的
Cayley 及融合原生执行，严格 `backend="native"` 不会绕过这个设置。

## 原生范围

| 部件 | 支持范围 |
| --- | --- |
| 积分 | 原有周期 Cayley 与两次物理力 kick，float64、真实质量 |
| PES | 显式 PyRPMD C ABI v1、Fortran batch BIND(C)、真实 BKMP2 |
| CV | Distance、COMDistance、BondDifference、Cartesian、RPMDrateCoordinate、RateSurface、CVStack |
| estimator | 几何 CV 的 centroid/bead_average；速率坐标与分界面为 centroid |
| 偏置 | 全矩阵 UmbrellaBias、支持内置 CV 的周期偏移、FluxMetricBias、CompositeBias |
| 约束 | centroid Cayley–RATTLE、片段质心位置/动量投影 |
| 观测 | 布尔 mask、完整环能量、质心 q/p、CV、可选完整 q/p |

等价通道保留原来的 `min_bond(max_equivalent_TS)` 分支规则，umbrella
内部能量为 `P*W`，flux metric 为 `-P*log(fs)/beta`。分支切换点的
不可微性和原本的采样测度均不因编译而改变。

自定义或覆写的 Python PES/CV/bias、Angle、BCOCB/PILE、ASE、
Torch/GPU、未显式绑定的 f2py、MPI/进程珠子代理保持原执行路径。原生
能力不会静默代替用户提供的数学定义。

## 失败、生命周期和计数

每个时间步及该步请求的观测是一个提交单元。PES 返回失败状态、非有限
能量/力、CV 或约束失败时，状态停留在最后成功步；
`PropagationError.result` 保存同一有效前缀：

```python
from pyrpmd.propagation import PropagationError

try:
    result = dynamics.propagate_block(state, 100, backend="native")
except PropagationError as error:
    result = error.result
    print(result.completed, result.error)
    # state already represents the final accepted step.
```

`raise_on_error=False` 返回同一前缀，供工作流明确处理。数值失败后不会
自动重试另一后端，也不撤销已经接受的步骤。与此不同，公开
`propagate()` 保留整块原子回滚语义。两者的 Python 回退都不承诺撤销
任意用户回调的外部副作用。

`pyrpmd.potentials.native.NativePESBinding` 与已有
`pyrpmd.propagation.NativePESBinding` 通过兼容桥保留相同 owner、上下文、
关闭检查和锁。整个原生调用持有 provider 的锁并保持动态库存活。
BKMP2 按动态库规范路径共享锁，普通求值、不同模块别名和两套传播接口
不能同时进入同一旧 Fortran COMMON。C ABI provider 必须明确声明
`capabilities.stateful=false`；PythonPotential 只在单位、符号完全恒等
且 delegate 主动提供绑定时透传。

Fortran batch v1 使用按值传递的 `int32_t B`，q/force 为 C 连续
`[B,N,3]`，对应 Fortran `(3,N,B)`。坐标 bohr、能量 hartree、力
hartree/bohr；status 每帧 0 表示成功。提供者需保证确定性的完整输出，
不能使用 Python 回调，也不能让异常或 STOP 穿越原生边界。

每步仍有两次物理力调用。能量观测复用最后一次力调用返回的能量；
片段投影改变终点坐标时重新求值。因此能量观测加速也包含减少重复
PES 调用，不能全部归因于去除 Python 循环。

`Dynamics.propagation_stats` 记录接受的 native/Python 步数和回退原因；
`loop_stats` 是其兼容别名。生产输出的 `propagation` 是相应轨迹或流的
累计计数，跨 checkpoint 延续；受约束母轨迹另有
`propagation_this_call`，用于区分本次调用与已完成 checkpoint 的旧工作。

## 工作流与复现

dRPMD 碰撞使用本接口，Python 分类器在原来的分类时刻读取完整 q/p；
块不跨越分类或 checkpoint。轨迹输出时刻可位于块内，由逐步记录保持
精确的输出 stride。受约束 recrossing 母轨迹也使用本接口，块停止于
Andersen 或 checkpoint 边界，观测 mask 保持块内的精确构型采样时刻。
FES/PMF、普通
Andersen 热化和 recrossing child 继续使用已验证的 `propagate()` 路径。
这次整合没有改变动态任务池、MPI 拓扑或添加 GPU 积分器。

`loop_backend` 与 `block_steps` 不参与科学参数哈希；温度、珠子数、
势能和步长等物理身份检查保留。相同后端、环境和初态下，不同分段和
中断续跑要求逐值一致；不同后端允许浮点归约顺序的末位差别。

```bash
python -m pytest -q tests/test_native_propagation.py
python benchmarks/h3/build_pes.py
python benchmarks/h3/native_propagation.py --output runs/fused-segmented/report.json
```

报告同时保存源码 commit/摘要、原始重复计时、数值误差、实际后端和
生产工作流结果；没有设定脆弱的速度阈值。`physical_convergence=false`
明确说明这些是实现一致性验证，不是收敛物理速率的结论。
