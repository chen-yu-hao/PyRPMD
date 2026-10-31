# 快速入门：可续跑的自由能计算

本节使用内置谐振子势，无需外部模型，几秒即可完成。你将检查输入、运行伞抽样、读取结果、续跑并重新分析。

## 1. 运行第一个任务

在仓库根目录执行：

```bash
pyrpmd validate examples/manual/quickstart.inp --check-pes
pyrpmd run examples/manual/quickstart.inp --output runs/manual-first
pyrpmd analyze runs/manual-first
```

`validate` 检查输入、单位、任务限制和资源设置；`--check-pes` 还会构造势能并完成数值准备。`run`
执行计算、打印 JSON 摘要并写出运行目录。`analyze` 读取保存的摘要。

运行目录结构如下：

```text
runs/manual-first/
├── manifest.json          # 软件、依赖、原生能力与科学指纹
├── input/                 # original.inp、resolved.json（解析后配置）、system.json
├── checkpoints/           # window_00000_replica_0000.npz …（每窗口每副本一个）
├── samples/  logs/
├── fes.csv  fes.npz  fes.h5
├── summary.json           # 工作流写出的摘要
└── results/summary.json   # 公共摘要：另含软件版本、输出路径、preset 与输入提示
```

`analyze` 与 Python API 读取 `results/summary.json`。`run` 不写入已有的非空目录，每次重新开始时换一个新的 `--output`。

## 2. 输入文件

这个例子计算三维谐势中一个 H 核的 **x 方向质心概率密度**。三个伞窗口各运行两个独立副本：先平衡 20 步，
再生产 100 步、每 5 步保存一个样本，使用 4 个珠子和 0.1 fs 步长。

```{literalinclude} ../../../examples/manual/quickstart.inp
:language: toml
```

源文件：[quickstart.inp](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/manual/quickstart.inp)。

| 字段 | 含义 |
| --- | --- |
| `task="fes"` | 对保存的伞抽样样本做 MBAR 重建 |
| `beads=4` | 每个原子由 4 个珠子组成的环表示 |
| `potential.params.k=0.002` | 谐势 $V=\tfrac12 k\lvert q\rvert^2$ 的原子单位力常数 |
| `cvs.estimator="centroid"` | CV 在环质心上求值 |
| `umbrella.stiffness` | 物理伞势的力常数；珠子缩放由 PyRPMD 处理 |
| `sampling.replicas=2` | 每个窗口的独立随机流数 |
| `analysis.bootstrap=0` | 本次不计算 bootstrap 误差条 |

温度、长度、时间和力常数写成带单位的字符串。原子索引从 0 开始。完整规则见[输入](input.md)和[科学约定](science.md)。

## 3. 读取结果

```python
import json
from pathlib import Path

run_dir = Path("runs/manual-first")
summary = json.loads((run_dir / "results/summary.json").read_text())
for key in ("task", "dimension", "windows", "streams", "samples",
            "overlap_connected", "converged", "warnings", "files"):
    print(key, summary.get(key))
```

摘要给出 `task="fes"`、一维 CV、3 个窗口、6 条采样流和 120 个保存样本；`effective_samples_per_stream`
给出统计独立的样本数。`overlap_connected=true` 表示所有窗口共享一个全局归一化。`warnings` 列出有效样本不足
和未采样的网格。单次运行的 `converged` 总是 `false`；收敛由[科学约定](science.md)中的系列计算确定。

`summary["files"]` 给出数据文件路径。FES 表中 `free_energy_hartree` 是相对自由能，`raw_count` 是每格样本数，
`masked` 标记没有采样支持的格子，这些格子的值为 NaN。见[输出与分析](results.md)。

## 4. 续跑与重新分析

```bash
pyrpmd resume runs/manual-first
pyrpmd analyze runs/manual-first --recompute --bins 8 --bootstrap 10
```

`resume` 让中断的运行从最后提交的状态继续；已完成的运行保持不变。`--recompute` 用新的分箱或 bootstrap
设置从保存样本重建分析，不运行动力学。`--bins` 和 `--bootstrap` 隐含 `--recompute`。新的分析会覆盖
`fes.*` 和 `results/summary.json`，需要保留旧版本时先复制。

保存的输入定义运行身份。要改变珠子数、温度、势能、步长或采样，使用新的运行目录。

## 5. 同一任务的 Python 写法

保存为仓库根目录下的 `my_first_run.py`：

```python
from pyrpmd import JobSpec, prepare, run, analyze_run, resume_run


def main():
    spec = JobSpec.from_file("examples/manual/quickstart.inp")
    report = prepare(spec, numerical=True)
    print("ready:", report["ready"])
    summary = run(spec, output="runs/manual-python")
    print(summary["samples"])
    resume_run("runs/manual-python")
    print(analyze_run("runs/manual-python")["task"])


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__"` 保护使 PyRPMD 可以使用 spawn 方式启动的工作进程。Python 与命令行共用同一解析器
和工作流；高层入口是 `run`、`prepare`、`resume_run` 和 `analyze_run`。

## 下一步

需要逐步控制积分器时，见 [API](api.md) 中的低层示例。使用真实模型时，先[连接势能](potentials.md)，
再选择工作流：[气相](workflows.md)、[材料与表面](periodic.md)或 [PCET](pcet.md)。
