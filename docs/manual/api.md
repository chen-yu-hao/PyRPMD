# Python API 参考

本页的函数签名从当前源码自动生成。高层 API 负责严格输入、运行身份、并行调度与恢复；
低层 API 适合方法开发，调用者需自己管理采样方案、结果与生命周期。

## 高层工作流入口

```python
from pyrpmd import JobSpec, prepare, run, resume_run, analyze_run
```

| 对象/函数 | 输入与返回值 |
| --- | --- |
| `JobSpec.from_file(path)` | 读取 TOML/Python 输入，记录相对路径基准 |
| `JobSpec.from_mapping(mapping, base_directory=...)` | 构造一个独立配置副本 |
| `JobSpec.with_overrides(...)` | 返回带覆盖的新配置，不就地修改原输入 |
| `JobSpec.resolve()` | 返回 `(config, identity)`；通常 identity 为 `System`，PCET 为其模型身份 |
| `prepare(job, numerical=False)` | 返回准备信息；数值模式会调用模型准备/PES |
| `run(job, output=None, resume=False)` | 执行任务并返回 root 的 JSON-compatible 摘要 |
| `resume_run(directory, parallel=None, execution=None)` | 读取保存输入与检查点；execution 仅接受块大小与后端 |
| `analyze_run(directory, ...)` | 默认读取摘要；支持的任务可离线重分析 |

`job` 可以是 `JobSpec`、字典或输入文件路径。MPI worker 可能返回 `None`，
因此仅在 root 处理聚合返回值。要改变科学参数，请创建新任务；`resume_run` 没有强行绕过身份校验的选项。

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

`System` 记录原子、核质量、初始几何、片段、电子态以及可选 cell/PBC/固定原子。
直接构造时使用原子单位：geometry/cell 为 bohr，masses 为电子质量。
`System.from_dict()` 读取保存的原子单位系统；用户带单位配置应经 `JobSpec` 解析。

`State.q` 和 `State.p` 的形状为 `[P,N,3]`。`p` 是动量；需要速度时除以每个原子的质量。
`step` 为非负步数，`time` 为原子时间。构造器复制数组，`copy()` 创建独立状态。

```{autoclass} pyrpmd.system.System
:members: natoms, atomic_numbers, to_dict, from_dict
```

```{autoclass} pyrpmd.dynamics.State
:members: copy
```

## 一个完整低层动力学程序

```{literalinclude} ../../examples/manual/low_level.py
:language: python
```

运行：

```bash
python examples/manual/low_level.py
```

这段示例使用零初始动量，演示确定性传播与环 Hamiltonian 变化。没有热化和生产采样，
不能从它估计 300 K 的量子热力学能量。变量 `beta` 的单位是 `hartree^-1`，
`dt` 是原子时间；高级输入中的 `temperature="300 K"` 会代为转换这些量。

```{autoclass} pyrpmd.dynamics.Dynamics
:members: step, energy, sample_momenta, propagate, propagate_block, sample_step, loop_stats
```

```{autofunction} pyrpmd.dynamics.native_capabilities
```

`step()` 是一次无恒温确定性步骤。`sample_step()` 需要匹配积分器与采样方法。
`energy()` 返回 sum-beads 环 Hamiltonian，不是物理量子能量估计器。
`propagate()` 与 `propagate_block()` 的失败提交边界不同，见[执行与续跑](execution.md)。

## 势能接口

```{autoclass} pyrpmd.potentials.base.PESResult
```

```{autofunction} pyrpmd.potentials.make_potential
```

```{autofunction} pyrpmd.potentials.base.validate_result
```

```{autofunction} pyrpmd.potentials.base.check_potential
```

适配器自身负责进程内模型与 `close()` 生命周期。使用[势能章节](potentials.md)约定的
形状、符号、单位、周期绑定与可选 virial，不能把一个不支持 PBC 的 provider 接到周期任务后继续忽略 cell。

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

`load_checkpoint()` 用作 context manager，在暴露数据前校验容器。
不读取 pickle/object array。用户自定义 payload 的 `kind` 与业务 schema 由调用者负责，
不能因为 NPZ 容器版本相同而忽略 payload 的物理意义。

## PCET 数值核

PCET 是独立的局域量子模型，主要用户入口仍是 `JobSpec` 和 `run`。
下列核函数用于模型开发与独立验证；网格、势能、质量和谱数组都采用原子单位。
电极能量零点、log-rate 与截断诊断见 [PCET 教程](pcet.md)。

```{autofunction} pyrpmd.pcet.solve_proton
```

```{autofunction} pyrpmd.pcet.electrode_log_kernels
```

```{autofunction} pyrpmd.pcet.golden_rule_rates
```

## 实验性 NRPMD 对象

以下对象也可从 `pyrpmd` 导入。它们服务于指定入射 diabatic 电子态的直接碰撞，
不构造电子热平衡 PMF；实验性接口与物理范围见[非绝热指南](nonadiabatic.md)。

`NRPMDState` 在核 `q/p[P,N,3]` 之外保存 `mapping_q/mapping_p[P,K]`。
`DiabaticResult` 接收能量矩阵 `[B,K,K]` 及正梯度 `[B,K,K,N,3]`。
所有矩阵元及其导数必须为实数、有限，并在电子指标上对称。

```{autoclass} pyrpmd.nonadiabatic.DiabaticResult
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDState
:members: copy, populations
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDDynamics
:members: step, energy, populations, asymptotic_coupling
```

## 接口稳定性

包版本、输入 schema、run manifest、checkpoint 容器和 PES ABI 各自版本化。
当前包版本 0.2.0 不意味着 PES ABI 升级为 v2。以下划线开头的辅助函数为内部实现，
不应作为用户扩展的长期接口。
更完整的接口与能力协商说明见
[公共接口约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)。

