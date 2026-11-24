# Python API 参考

本页签名由当前源码生成。高层 API 负责输入检查、运行身份、并行调度与续跑。低层 API 直接控制积分器，用于方法开发；
调用者自行管理采样、结果与对象生命周期。

## 高层入口

```python
from pyrpmd import JobSpec, prepare, run, resume_run, extend_run, analyze_run
```

| 对象/函数 | 输入与返回值 |
| --- | --- |
| `JobSpec.from_file(path)` | 读取 TOML 或 Python 输入，记录相对路径基准 |
| `JobSpec.from_mapping(mapping, base_directory=...)` | 由映射的副本构造配置 |
| `JobSpec.with_overrides(...)` | 返回应用覆盖后的新配置 |
| `JobSpec.resolve()` | 返回 `(config, identity)`；identity 是 `System` 或 PCET 模型 |
| `prepare(job, numerical=False)` | 返回准备信息；`numerical=True` 执行模型准备并调用 PES |
| `run(job, output=None, resume=False, init_from=None, init_index=None, init_bias=False, stage="all")` | 运行任务，在 root rank 返回 JSON 兼容的摘要；`init_from` 从旧运行的构型热启动新运行（见[热启动](../warm-start.md)） |
| `resume_run(directory, parallel=None, execution=None, adaptive_budget=None)` | 按保存的输入与检查点续跑；可覆盖执行资源或增加自适应 FES/rate pilot 的预算 |
| `extend_run(directory, *, max_time=None, production_steps=None)` | 必须且只能授权一种总预算：dRPMD 从发射计的时限，或固定窗口气相速率中每窗口、每副本的生产步数 |
| `analyze_run(directory, ...)` | 读取摘要，或对支持的任务离线重分析 |

`job` 可以是 `JobSpec`、字典或输入文件路径。非 root 的 MPI rank 返回 `None`，因此在 root 上处理返回值。
`resume_run` 保持保存的运行身份；新的科学设置需要新任务。

续跑时可以调整执行资源：

```python
if __name__ == "__main__":
    summary = resume_run("runs/example", parallel={"workers": 4})
```

请保存为 Python 脚本再运行，保证新建的工作进程能够导入脚本。
在这个覆盖参数中，`workers`、`trajectory_workers` 和 `trajectory_groups` 是同义字段。
指定其中一个会同时替换原先保存的 worker/group 数；同时指定多个时，数值必须一致。
未知字段和不合法的资源组合会在执行前抛出 `InputError`。覆盖不会改写保存的输入或科学设置指纹，
工作流仍会检查检查点兼容性。PCET 不接受并行或传播设置覆盖。
热初态 dRPMD 如需在续跑时改变 worker 数，请在原始输入中显式设置 `sampling.parents`；
旧运行的隐式母轨迹分配可能因执行布局改变而安全拒绝续跑。

扩展固定窗口气相速率的生产采样：

```python
if __name__ == "__main__":
    summary = extend_run("runs/rate", production_steps=200000)
```

`production_steps` 是每窗口、每副本的生产总步数，不含平衡阶段。
必须且只能提供 `production_steps` 或 `max_time`，后者保留 dRPMD 总时限语义。
原始输入与 manifest 保持不变。重复最近已授权的目标具有幂等性，`resume_run` 可识别中断或已完成的扩展。
更大的速率目标会生成新一代 PMF/回穿结果，并保留之前的结果。首版适用于使用本地伞抽样的一维固定窗口气相速率；
母链/子轨迹预算与研究层级汇总刷新不在扩展范围内。完整约定见
[扩展固定窗口速率生产采样](workflows.md#扩展固定窗口速率生产采样)。

```{autoclass} pyrpmd.input.JobSpec
:members: from_file, from_mapping, with_overrides, resolve
```

```{autofunction} pyrpmd.runner.prepare
```

```{autofunction} pyrpmd.runner.run
```

```{autofunction} pyrpmd.runner.resume_run
```

```{autofunction} pyrpmd.runner.extend_run
```

```{autofunction} pyrpmd.runner.analyze_run
```

## 其他科学计算 API

变温与同位素研究复用工作流与检查点。瞬子（instanton）与量子统计 API 直接调用 `Potential.evaluate`，保持实时动力学显式。
归一化约定与适用范围见英文专题指南：[速率研究](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-studies.md)、
[瞬子](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/instanton.md)与
[量子统计](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-statistics.md)。

```{autofunction} pyrpmd.studies.plan_rate_study
```

```{autofunction} pyrpmd.studies.run_rate_study
```

```{autofunction} pyrpmd.studies.arrhenius_fit
```

```{autofunction} pyrpmd.studies.kinetic_isotope_effect
```

```{autofunction} pyrpmd.instanton.optimize_instanton
```

```{autofunction} pyrpmd.instanton.instanton_rate_1d
```

```{autoclass} pyrpmd.path_integral.PathIntegralAction
:members: evaluate, evaluate_batch
```

```{autofunction} pyrpmd.path_integral.sample_path_integral
```

```{autoclass} pyrpmd.path_integral.OpenPathAction
```

```{autofunction} pyrpmd.path_integral.sample_open_path
```

```{autofunction} pyrpmd.quantum_statistics.energy_heat_capacity
```

```{autofunction} pyrpmd.quantum_statistics.quantum_kinetic_energy
```

```{autofunction} pyrpmd.quantum_statistics.isotope_substitution_free_energy
```

```{autofunction} pyrpmd.quantum_statistics.isotope_fractionation
```

```{autofunction} pyrpmd.quantum_statistics.momentum_distribution_from_open_paths
```

## System 与 State

`System` 保存原子、核质量、初始几何、片段、电子态以及可选的晶胞、PBC 和固定原子。其构造器使用原子单位：几何与晶胞
以 bohr 计，质量以电子质量计。`System.from_dict()` 读取保存的原子单位体系；带单位的输入经 `JobSpec` 解析。

`State.q` 与 `State.p` 的形状为 `[P,N,3]`。`p` 是动量，除以原子质量得到速度。`step` 为步数，`time` 以原子时间计。
构造器复制数组，`copy()` 返回独立的状态。

```{autoclass} pyrpmd.system.System
:members: natoms, atomic_numbers, to_dict, from_dict
```

```{autoclass} pyrpmd.dynamics.State
:members: copy
```

## 低层动力学程序

```{literalinclude} ../../../examples/manual/low_level.py
:language: python
```

运行：

```bash
python examples/manual/low_level.py
```

程序从零动量开始做确定性传播，并打印环 Hamiltonian 的变化；需要热平均时使用 `pimd` 任务或 `sample_step()`。
`beta` 以 `hartree^-1` 计，`dt` 以原子时间计；输入中的 `temperature="300 K"` 和 `timestep="0.1 fs"` 会自动转换。
最后一行打印 `loop_stats`：Python `HarmonicPotential` 没有原生 PES 绑定，因此每一步都在 Python 中运行，回退原因中会写明。

```{autoclass} pyrpmd.dynamics.Dynamics
:members: step, energy, sample_momenta, propagate, propagate_block, sample_step, trpmd_step, operator_state_dict, load_operator_state_dict, loop_stats
```

```{autofunction} pyrpmd.dynamics.native_capabilities
```

默认构造时，`step()` 执行一步无恒温的确定性传播。显式设置 `thermostat`、`barostat` 或 `force_splitting` 后，
同一步传播组合相应算符；见[统一传播栈](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/propagation-stack.md)。
`sample_step()` 用请求的恒温器执行一步采样。附加恒温器时须把 `operator_state_dict()` 与物理状态一同保存。
`energy()` 返回对珠子求和的环
Hamiltonian。`propagate()` 与 `propagate_block()` 在失败时提交的内容不同，见[执行与续跑](execution.md)。

## 批量轨迹、RPC/MTS 与谱分析

```python
from pyrpmd import BatchedState, BatchedDynamics, MTSDynamics
from pyrpmd import RingPolymerContraction, SplitPotential, DifferencePotential
from pyrpmd.analysis.spectroscopy import (
    velocity_autocorrelation, dipole_autocorrelation, flux_correlation,
    power_spectrum, relative_ir_spectrum,
)
```

`BatchedState` 保存 `[K,P,N,3]` 数组以及每条轨迹的独立时钟。
`MTSDynamics` 区分外层步长和内层步数；RPC 的慢势能必须是显式声明的修正。
`Dynamics.trpmd_step()` 保持真实核质量，只对环内部模施加摩擦。
示例和支持的组合见英文扩展指南：[批量传播](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-dynamics.md)、
[RPC/MTS](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rpc-mts.md)和
[TRPMD/谱](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/trpmd-spectroscopy.md)。

## 势能

```{autoclass} pyrpmd.potentials.base.PESResult
```

```{autofunction} pyrpmd.potentials.make_potential
```

```{autofunction} pyrpmd.potentials.base.validate_result
```

```{autofunction} pyrpmd.potentials.base.check_potential
```

适配器持有自己的模型，并在 `close()` 中释放。形状、符号、单位、周期绑定和可选 virial 见[势能章节](potentials.md)。

## 单位与检查点

```{autofunction} pyrpmd.units.parse_quantity
```

```{autofunction} pyrpmd.units.unit_factor
```

```{autofunction} pyrpmd.checkpoint.atomic_savez
```

```{autofunction} pyrpmd.checkpoint.load_checkpoint
```

```{autofunction} pyrpmd.checkpoint.inspect_checkpoint
```

`load_checkpoint()` 是 context manager，在暴露数据前校验容器，并且不读取 pickle 或 object 数组。自定义 payload 的
`kind` 与 schema 由调用者定义。

## PCET 内核

运行 PCET 通常使用 `JobSpec` 和 `run`。下列内核用于模型开发与独立检查；网格、势能、质量与谱以原子单位计。
能量零点、log 速率与截断诊断见 [PCET 教程](pcet.md)。

```{autofunction} pyrpmd.pcet.solve_proton
```

```{autofunction} pyrpmd.pcet.electrode_log_kernels
```

```{autofunction} pyrpmd.pcet.golden_rule_rates
```

## NRPMD 对象（实验性）

这些对象也可从 `pyrpmd` 导入，用于从指定入射 diabatic 态出发的碰撞，见[非绝热指南](nonadiabatic.md)。

`NRPMDState` 在核的 `q/p[P,N,3]` 之外保存 `mapping_q/mapping_p[P,K]`。`DiabaticResult` 接收能量矩阵 `[B,K,K]`
及其正梯度 `[B,K,K,N,3]`，两者须有限，并在电子指标上满足 Hermitian 条件；复数自旋–轨道耦合保留完整相位。

```{autoclass} pyrpmd.nonadiabatic.DiabaticResult
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDState
:members: copy, populations
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDDynamics
:members: step, energy, populations, asymptotic_coupling
```

## 增强采样 API

```python
from pyrpmd import (
    OPESBias, PlumedBias, TransitionInterfaceSampler, TISPath,
    estimate_effective_flux, combine_tis_rate,
)
```

`task="fes"` 配合 `[fes] method="opes"`（或旧版别名 `task="opes"`）使用标准的 `run`/`prepare`/`resume_run`/`analyze_run` 入口。
`OPESBias` 也支持通过显式 `update()` 事件直接集成到低层循环中；单纯的力求值不会改变概率估计。
固定带宽与冻结生产采样的约定见英文专题指南 [OPES](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/opes.md)。

`PlumedBias.step(dynamics, state)` 管理 PLUMED 的更新时钟并调用现有积分器。`evaluate()` 返回只读诊断而不触发历史更新。
可选的 PLUMED 运行时仅在构造桥接对象时加载。单位与外部重启文件约定见 [PLUMED](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md)。

`TransitionInterfaceSampler` 使用标量 CV 或可调用对象生成并采样无偏路径系综。`TISPath` 持有每条 `q/p` 路径；
可选的 sample 回调可以流式输出保留的路径（包括拒绝移动后的重复计数）。`estimate_effective_flux` 读取独立的无偏轨迹，
`combine_tis_rate` 返回候选一阶速率并显式标记其归一化与收敛状态。详见 [TIS](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-interface-sampling.md)。

## 电子摩擦与热非绝热模型

以下类同样可从 `pyrpmd` 直接导入。各方法的初始分布与适用限制见[非绝热章节](nonadiabatic.md)。
电子摩擦接口使用标量平均力势能面和物理摩擦张量；其他方法使用完整的 diabatic 矩阵势能及正梯度。

```{autoclass} pyrpmd.electronic_friction.ElectronicFrictionDynamics
:members: step, propagate
```

```{autoclass} pyrpmd.electronic_friction.ConstantElectronicFriction
:members: from_rates
```

```{autoclass} pyrpmd.electronic_friction.FrictionResult
```

```{autoclass} pyrpmd.thermal_nrpmd.ThermalNRPMDSampler
:members: sample, snapshot, restore
```

```{autoclass} pyrpmd.thermal_nrpmd.ThermalNRPMDDynamics
:members: step, energy, population_flux
```

```{autofunction} pyrpmd.thermal_nrpmd.thermal_flux_side
```

```{autofunction} pyrpmd.thermal_nrpmd.thermal_population_rate
```

```{autoclass} pyrpmd.surface_hopping.RPSHDynamics
:members: initialize, step, energy, populations, attempt_hop
```

```{autoclass} pyrpmd.mv_rpmd.MVRPMDDynamics
:members: step, energy, populations
```

```{autoclass} pyrpmd.kc_rpmd.KCRPMDDynamics
:members: step, energy
```

```{autoclass} pyrpmd.isomorphic.IsomorphicPotential
:members: evaluate
```

```{autoclass} pyrpmd.isomorphic.IsomorphicEhrenfestDynamics
:members: step, energy
```

## 非绝热系综 API

独立的非平衡模型工作流详见英文专题指南
[布居动力学](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-workflow.md)。可复用的 Python API 包括：

```{autofunction} pyrpmd.nonadiabatic_ensemble.run_nonadiabatic_ensemble
```

```{autofunction} pyrpmd.nonadiabatic_ensemble.density_matrix
```

```{autoclass} pyrpmd.nonadiabatic_ensemble.NonadiabaticEnsembleResult
:members: mean_populations, population_sem, mean_density, density_sem_real, density_sem_imag, active_surface_fractions
```

```{autofunction} pyrpmd.nonadiabatic_sampling.sample_harmonic_ring
```

模型类 `TullyModelOne`、`TullyModelTwo`、`TullyModelThree`、`LinearVibronicCoupling` 与 `DiscreteSpinBoson`
均从 `pyrpmd` 导出。其参数与坐标约定见
[模型指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-models.md)。

## 扩展非绝热方法与分子势能

[自旋映射指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-spin-methods.md)定义了二态 W-spin 与 CA-MASH 变体及其估计量；
[表面跳跃扩展指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-hopping-extensions.md)定义了 BA/DM、DBOC 与复数 SOC 约定。

```{autoclass} pyrpmd.spin_mapping.SMNRPMDDynamics
:members: initialize, step, energy, density_matrix, populations
```

```{autoclass} pyrpmd.rp_mash.RPMASHDynamics
:members: initialize, step, energy, populations, correlation
```

```{autoclass} pyrpmd.advanced_surface_hopping.AdvancedRPSHDynamics
:members: initialize, step, energy, density_matrix, populations, attempt_hop
```

```{autoclass} pyrpmd.batched_nonadiabatic.BatchedNRPMDDynamics
:members: step, energy, density_matrix, populations
```

```{autoclass} pyrpmd.batched_nonadiabatic.BatchedNRPMDState
:members: from_states, to_states
```

```{autofunction} pyrpmd.electronic_backends.electronic_mapping_flow
```

```{autofunction} pyrpmd.electronic_backends.torch_electronic_mapping_flow
```

```{autoclass} pyrpmd.potentials.nah2_leps.NaH2LEPS5G
:members: evaluate, to_dict
```

```{autoclass} pyrpmd.potentials.multistate_external.ChemPotPyDPEM
:members: evaluate
```

## 版本

包、输入 schema、运行 manifest、检查点容器和 PES ABI 各有独立版本；0.2.0 支持分子势能 C ABI v1 和带晶胞接口的 C ABI v2。以下划线开头的名称属于
内部实现。接口与能力协商见[公共接口约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)。
