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

`check_energy=True` 额外检查初态及每步完整环能量，即使该步没有请求观测。
它与 `record_energy` 独立：可以检查能量而不把能量写入结果。默认 `False`
保留既有 API；请求该能力时，旧扩展缺少 `BLOCK_ENERGY_CHECK_API` 会在
严格原生模式拒绝，在 auto 模式进入相应的 Python 兼容实现。FES 使用
此选项保留原连续传播的每步能量检查，同时让采样点位于同一传播块内部。

`record_previous=True` 仅保留成功分段最后一步之前的完整 q/p 和精确时钟，
通过 `previous_q/previous_p/previous_step/previous_time` 返回，适合 Python
分类器拒绝最后一步时回滚。失败分段已经保留合法前缀，不提供这个额外快照。
它不依赖观测 mask，也不需要 `record_states=True` 的逐步完整历史。

输入文件仍使用 `block_steps=256`，合法范围 1..65536。使用本接口的
dRPMD 碰撞和受约束母轨迹会将实际分段限制在 4096 及相应的热浴、Python
分类或 checkpoint 边界。原生分类器可以在块内按原频率检查，并在确认
产物或非反应结局时提前返回。
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

默认每步有两次物理力调用。`reuse_forces=True` 在无偏置、约束和片段投影
的原生传播中启用块内终点力复用，L 步由 2L 次 PES 批调用降为 L+1 次。
普通原生 dRPMD 碰撞自动使用它；其他分支保持参考调用顺序。缓存不跨块、
不写入 checkpoint，恢复时重新求力。能量观测继续复用终点 PES 能量；
片段投影改变终点坐标时重新求值。这不会合并两次半步 kick 或更换积分器。

`Dynamics.propagation_stats` 记录接受的 native/Python 步数和回退原因；
`loop_stats` 是其兼容别名。生产输出的 `propagation` 是相应轨迹或流的
累计计数，跨 checkpoint 延续；受约束母轨迹另有
`propagation_this_call`，用于区分本次调用与已完成 checkpoint 的旧工作。

## 工作流与复现

dRPMD 碰撞使用本接口。精确内置 `ConnectivityClassifier` 的显式通道和
基准中的 `H3ExchangeClassifier` 支持原生判定：检查的绝对步号、候选标签、
持续时间和停止条件均保持原规则，块可以跨越检查时刻。接近成键阈值、
出口距离或朝外运动零点的状态会返回 Python 使用原始严格判据；这个
舍入保护区间只决定执行后端，不改变物理阈值。

Pilot 候选通道发现、任意自定义分类器及覆写的分类方法仍在 Python 中按
原检查时刻读取完整 q/p，此时块不跨越检查边界。轨迹输出时刻可位于块内，
由逐步记录保持精确的输出 stride。每个接受步都参与最大能量漂移统计。
终态 checkpoint 在状态未改变时不重复保存。受约束 recrossing 母轨迹也使用
本接口，块停止于
Andersen 或 checkpoint 边界，观测 mask 保持块内的精确构型采样时刻。
支持原生执行的 FES 及复用该采样器的 PMF 工作流使用本接口的多观测块，
在热浴、checkpoint 和最大块长处返回；原有不支持原生绑定的 Python FES
保持其势能调用节奏。失败时仅
提交并持久化成功前缀，若在刚重抽动量后的第一步失败，恢复到重抽前的
动量与 RNG，避免重启重复消费随机数。普通 Andersen 热化和
recrossing child 继续保留 `propagate()` 路径。

低层 `outcome` 参数包含 `descriptor`、`check_interval`、`persistence_steps`、
`start_step`、`candidate` 和 `candidate_since`。描述符使用显式通道编号，
不允许静默替换 Python 实现。`result.outcome` 返回新的候选状态、`resolved`
和本块原生 `checks`；`pending_check=True` 表示调用者必须在当前完整状态上
执行原 Python 判定并更新 persistence 后再继续。确认结局或待 Python 检查
都可能使 `completed < requested`，此时 `success=True` 且 `error=None`。
普通调用仍保留原来的完整块/失败前缀契约。内置 dRPMD 工作流负责这些
边界和恢复状态，常规输入无需手工构造此参数。
显式请求 `outcome` 而原生能力不足时，即使使用 `backend="auto"`，也会在
状态改变前报错，不会静默忽略分类器或进入不处理 outcome 的 Python 长块。

`loop_backend` 与 `block_steps` 不参与科学参数哈希；温度、珠子数、
势能和步长等物理身份检查保留。相同后端、环境和初态下，不同分段和
中断续跑要求逐值一致；不同后端允许浮点归约顺序的末位差别。

```bash
python -m pytest -q tests/test_native_propagation.py
python -m pytest -q tests/test_native_outcomes.py tests/test_drpmd_native_workflow.py
python benchmarks/h3/build_pes.py
python benchmarks/h3/native_propagation.py --output runs/fused-segmented/report.json
```

报告同时保存源码 commit/摘要、原始重复计时、数值误差、实际后端和
生产工作流结果；没有设定脆弱的速度阈值。`physical_convergence=false`
明确说明这些是实现一致性验证，不是收敛物理速率的结论。

dRPMD 原生分类、力复用及紧凑回滚的独立版本对照、完整工作流计时与
复现命令见 [2026-10-30 验证记录](validation/drpmd-optimization-2026-10-30/README.md)。
