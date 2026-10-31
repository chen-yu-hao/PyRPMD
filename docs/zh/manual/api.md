# Python API 参考

本页签名由当前源码生成。高层 API 负责输入检查、运行身份、并行调度与续跑。低层 API 直接控制积分器，用于方法开发；
调用者自行管理采样、结果与对象生命周期。

## 高层入口

```python
from pyrpmd import JobSpec, prepare, run, resume_run, analyze_run
```

| 对象/函数 | 输入与返回值 |
| --- | --- |
| `JobSpec.from_file(path)` | 读取 TOML 或 Python 输入，记录相对路径基准 |
| `JobSpec.from_mapping(mapping, base_directory=...)` | 由映射的副本构造配置 |
| `JobSpec.with_overrides(...)` | 返回应用覆盖后的新配置 |
| `JobSpec.resolve()` | 返回 `(config, identity)`；identity 是 `System` 或 PCET 模型 |
| `prepare(job, numerical=False)` | 返回准备信息；`numerical=True` 执行模型准备并调用 PES |
| `run(job, output=None, resume=False)` | 运行任务，在 root rank 返回 JSON 兼容的摘要 |
| `resume_run(directory, parallel=None, execution=None)` | 按保存的输入与检查点续跑；`execution` 接受分块大小和后端 |
| `analyze_run(directory, ...)` | 读取摘要，或对支持的任务离线重分析 |

`job` 可以是 `JobSpec`、字典或输入文件路径。非 root 的 MPI rank 返回 `None`，因此在 root 上处理返回值。
`resume_run` 保持保存的运行身份；新的科学设置需要新任务。

```{autoclass} pyrpmd.input.JobSpec
:members: from_file, from_mapping, with_overrides, resolve
```

```{autofunction} pyrpmd.runner.prepare
```

```{autofunction} pyrpmd.runner.run
```

```{autofunction} pyrpmd.runner.resume_run
```

```{autofunction} pyrpmd.runner.analyze_run
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
:members: step, energy, sample_momenta, propagate, propagate_block, sample_step, loop_stats
```

```{autofunction} pyrpmd.dynamics.native_capabilities
```

`step()` 执行一步无恒温的确定性传播。`sample_step()` 用匹配的恒温器执行一步采样。`energy()` 返回对珠子求和的环
Hamiltonian。`propagate()` 与 `propagate_block()` 在失败时提交的内容不同，见[执行与续跑](execution.md)。

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

## 电子摩擦与其他非绝热方法

以下对象可从 `pyrpmd` 直接导入。各方法的初始分布、适用范围和速率定义见[非绝热章节](nonadiabatic.md)。

| 功能 | Python 入口 |
| --- | --- |
| 质心电子摩擦 | `ElectronicFrictionDynamics`、`ConstantElectronicFriction`、`FrictionResult` |
| 电子热平衡 NRPMD | `ThermalNRPMDSampler`、`ThermalNRPMDDynamics`、`thermal_flux_side`、`thermal_population_rate` |
| 环聚合物表面跳跃 | `RPSHDynamics`、`RPSHState`、`HopResult` |
| MV-RPMD | `MVRPMDState`、`MVRPMDPotential`、`MVRPMDDynamics`、`MVRPMDSampler` |
| KC-RPMD | `KCCalibration`、`KCRPMDState`、`KCRPMDPotential`、`KCRPMDDynamics`、`KCRPMDSampler` |
| Isomorphic Hamiltonian | `IsomorphicPotential`、`IsomorphicEhrenfestState`、`IsomorphicEhrenfestDynamics` |

电子摩擦接口使用标量平均力势和物理摩擦张量；其他方法使用完整的 diabatic 矩阵势及正梯度。
热采样、符号重加权、QC 模型误差和数值收敛应分别检查。

## 版本

包、输入 schema、运行 manifest、检查点容器和 PES ABI 各有独立版本；0.2.0 包使用 PES ABI v1。以下划线开头的名称属于
内部实现。接口与能力协商见[公共接口约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)。
