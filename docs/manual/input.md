# 输入、体系与命令行

## 一个任务，两种写法

`.inp` / `.toml` 是 TOML 文件；`.py` 是普通 Python 模块，定义 `build()` 并返回
`JobSpec` 或字典。两者通过同一解析器转换为规范配置。Python 输入适合参数生成、
批量任务和可导入插件；TOML 适合清晰记录一个具体计算。

```python
from pathlib import Path
from pyrpmd import JobSpec


def build():
    return JobSpec.from_mapping(
        {
            "task": "pimd",
            "temperature": "300 K",
            "beads": 8,
            "system": {
                "symbols": ["H"],
                "geometry": [[0.0, 0.0, 0.0]],
                "geometry_unit": "bohr",
            },
            "potential": {"provider": "harmonic", "params": {"k": 0.002}},
            "dynamics": {"timestep": "0.1 fs"},
            "sampling": {
                "thermostat": "andersen",
                "equilibration_steps": 100,
                "production_steps": 500,
                "sample_interval": 5,
                "replicas": 2,
            },
        },
        base_directory=Path(__file__).resolve().parent,
    )
```

这段可保存为 `job.py` 并使用 `pyrpmd validate job.py`。数值是教学设置；正式 PIMD
应按[材料章节](periodic.md)检查平衡、估计量、步长和珠子数收敛。Python 输入会执行
普通 Python 代码；它不是声明式数据格式。

## 路径的基准

| 路径来源 | 相对路径基准 |
| --- | --- |
| 输入中的 XYZ、模型、`file.py:factory`、输出目录 | 输入文件所在目录 |
| `JobSpec.from_mapping(..., base_directory=...)` | 显式提供的目录；省略时为当前目录 |
| CLI 的 `--pes`、`--pes-manifest`、`--output` | 启动命令的当前目录 |
| `.o` 链接 manifest 中的源文件/依赖 | manifest 所在目录 |

例如，`examples/manual/quickstart.inp` 内的 `output.directory="runs/manual-quickstart"`
位于该输入旁边。快速入门显式使用 `--output runs/manual-first`，因而把结果放在仓库根目录
下的 `runs/`。迁移输入时同时迁移它引用的文件，并保留原子顺序。

## 通用字段

| 字段 | 约定与作用 |
| --- | --- |
| `schema_version` | 当前用户输入版本为整数 1 |
| `task` | 见[首页任务表](../index.md)；不同任务允许的 section 不完全相同 |
| `preset` | 常规动力学任务缺省 `pilot`；`production` 要求更多显式字段，仍不保证收敛 |
| `temperature` | 正温度，优先写 `"300 K"`；纯数值按 K |
| `beads` | 正整数；`auto` 是 pilot 中 P=32 的试算约定 |
| `seed` | 非负整数；独立任务使用稳定逻辑 ID 派生随机流 |
| `dynamics.timestep` | 正步长；`"0.1 fs"` 带单位，纯数值为原子时间 |
| `dynamics.integrator` | `cayley-bcb`、`normal-mode-bab`、`velocity-verlet` |
| `sampling` | 平衡、抽样间隔、生产长度、恒温器、副本或母链数 |
| `parallel` | 进程/MPI 拓扑、珠子分块与线程设置 |
| `output.checkpoint_steps` | 正整数检查点间隔；相应时间字段为 `checkpoint_interval` |

只有与任务有关的 section 才应出现。PCET 使用独立的 `[pcet]` 模型与扫描设置，
不要额外构造分子 `system` 或 RPMD 珠子，详见 [PCET](pcet.md)。
未知字段与无效组合会报错；拼写错误不会作为注释被忽略。

对于基于时间的字段，时间必须是步长的整数倍。需要精确控制计数时使用
`equilibration_steps`、`production_steps`、`interval_steps` 等该工作流支持的步数字段。
不要把一个任务里的间隔字段直接搬到另一个任务而不检查其含义：dRPMD 的母链发射间隔
与 FES 的保存采样间隔控制不同事件。

## 原子、同位素与几何

非 PCET 任务的体系通常用以下一种方式声明：

```toml
[system]
geometry = "molecule.xyz"
geometry_unit = "angstrom"
charge = 0
multiplicity = 1
isotopes = [{atom = 0, mass = "2.01410177812 amu"}]
```

也可使用内联 `symbols` 与 `geometry=[[x,y,z], ...]`；缺省内联长度为 bohr，
XYZ 缺省为 angstrom。`geometry_unit` 可显式覆盖。`masses` 的数值单位为电子质量，
`masses_amu` 为原子质量单位。同位素标记 H、D、T 保留不同核质量，向要求化学元素的
电子势接口传递时仍对应氢的核电荷。`charge` 和 `multiplicity` 必须与所选势能模型一致。

气相直接碰撞用 `reactants=["A.xyz", "B.xyz"]` 和显式 `separation`；
TS 入口可用 `transition_state="ts.xyz"`。体系不能仅凭 XYZ 自动获得全部反应通道或
统计简并度。所有原子索引均从 **0** 开始，fragments 应构成完整且不重叠的原子划分。

周期晶胞、PBC、固定原子和展开坐标见[材料与表面](periodic.md)。新输入建议始终明确
`geometry_unit`、`cell_unit` 和 `pbc`，避免依赖不同来源文件的缺省单位。

## 集体变量与偏置

`[[cvs]]` 给 CV 命名，`umbrella.cvs` 选择参与重建的坐标。
`centroid` 对质心求坐标，`bead_average` 先逐珠子计算再平均；对于非线性 CV 两者不同。
自定义 CV 必须返回完整 Jacobian，见[势能与插件](potentials.md)。

数值伞中心使用声明的 CV 输入单位；显式的带单位字符串携带自身单位。
数值 stiffness 是原子单位。普通 `distance`、`angle`、`bond_difference` 使用展开几何；
周期最小镜像必须选择 `periodic_distance`、`periodic_angle`、`periodic_bond_difference`。
连续 fractional 或 surface_lateral 坐标与折回主晶胞坐标也有不同含义，不能互换。

## 命令行参考

| 命令 | 用途 | 常用选项 |
| --- | --- | --- |
| `pyrpmd validate INPUT` | 解析配置并检查任务 | `--check-pes`、`--write-resolved FILE` |
| `pyrpmd prepare INPUT` | 查看准备信息 | `--pilot` 允许数值准备；`--output FILE` 写 JSON |
| `pyrpmd run INPUT` | 新运行或按输入恢复 | `--output DIR`、`--resume` |
| `pyrpmd resume DIR` | 从保存配置续跑 | `--workers N`、`--loop-backend auto`、`--block-steps N` |
| `pyrpmd analyze DIR` | 读结果或离线重分析 | `--recompute`、`--bins`、`--bootstrap`；扩散见下文 |
| `pyrpmd checkpoint FILE` | 验证检查点容器 | `--require-integrity` 拒绝无校验的旧容器 |
| `pyrpmd info` | 查看版本与原生能力 | 无需输入文件 |
| `pyrpmd pes build/inspect/check` | C ABI 势能构建与诊断 | [势能接口](potentials.md) |

`validate`、`prepare` 和 `run` 共同接受 `--pes`、`--pes-manifest`、`--temperature`、
`--beads`、`--workers`；覆盖值也必须适用于该任务。例如 PCET 不应接受一个无意义的珠子数。
使用 `pyrpmd COMMAND --help` 获取当前安装版本的精确选项。

diffusion 重分析支持 `--fit-range '10 fs' '100 fs'`、`--axes 0 1`、`--atoms 0` 和
`--max-lag N`，其中 max-lag 以保存帧数计。这些参数必须与保存时长及物理拟合区间相容，
具体输出解释见[结果与分析](results.md)。

更详尽的 section 规范保存在
[docs/input.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md)。

