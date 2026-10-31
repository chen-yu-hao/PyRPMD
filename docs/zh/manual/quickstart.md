# 快速入门：可续跑的自由能计算

几秒内你将走完所有任务共用的 PyRPMD 流程：检查输入、运行伞抽样、读取结果、续跑，并在不重跑动力学的情况下重新分析。
本节使用内置谐振子势，无需外部模型。

## 1. 运行第一个任务

完成[安装](installation.md)后，在自己的工作目录中打开终端。入门输入随 PyRPMD 打包，wheel 安装与源码安装均可使用：

```bash
pyrpmd init my-first-run
pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
pyrpmd analyze my-first-run/run
```

`init` 在新目录或空目录中写入 `quickstart.inp` 并打印下一步命令；非空目录会被拒绝，开始另一个例子时请换一个名字。
若 shell 找不到 `pyrpmd`，可使用 `python -m pyrpmd` 加相同参数。

`validate` 检查输入、单位、任务限制和资源设置；`--check-pes` 还会构造势能并完成数值准备。`run`
执行计算、打印 JSON 摘要并写出运行目录。`analyze` 读取保存的摘要。
验证结果给出解析后的 `input`、`output` 和 `preset`；`validate --output` 预览运行目标路径，不创建目录。
入门输入中的 `output.directory="run"` 相对于输入文件解析，因此上面显式指定的 `--output my-first-run/run` 指向同一位置。

运行目录结构如下：

```text
my-first-run/run/
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

```{literalinclude} ../../../src/pyrpmd/templates/quickstart.inp
:language: toml
```

源文件：[随包提供的 quickstart.inp](https://github.com/chen-yu-hao/PyRPMD/blob/main/src/pyrpmd/templates/quickstart.inp)。

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

run_dir = Path("my-first-run/run")
summary = json.loads((run_dir / "results/summary.json").read_text())
for key in ("task", "dimension", "windows", "streams", "samples",
            "overlap_connected", "converged", "warnings", "files"):
    print(key, summary.get(key))
```

摘要给出 `task="fes"`、一维 CV、3 个窗口、6 条采样流和 120 个保存样本；`effective_samples_per_stream`
给出统计独立的样本数。`overlap_connected=true` 表示所有窗口共享一个全局归一化。`warnings` 列出有效样本不足
和未采样的网格。单次运行的 `converged` 按设计保持为 `false`：收敛由[科学约定](science.md)中的系列计算确定。

`summary["files"]` 给出数据文件路径。FES 表中 `free_energy_hartree` 是相对自由能，`raw_count` 是每格样本数，
`masked` 标记没有采样支持的格子，这些格子的值为 NaN。见[输出与分析](results.md)。

## 4. 续跑与重新分析

```bash
pyrpmd resume my-first-run/run
pyrpmd analyze my-first-run/run --recompute --bins 8 --bootstrap 10
```

`resume` 让中断的运行从最后提交的状态继续；已完成的运行保持不变。`--recompute` 用新的分箱或 bootstrap
设置从保存样本重建分析，不运行动力学。`--bins` 和 `--bootstrap` 隐含 `--recompute`。新的分析会覆盖
`fes.*` 和 `results/summary.json`，需要保留旧版本时先复制。

保存的输入定义运行身份。要改变珠子数、温度、势能、步长或采样，使用新的运行目录。

## 5. 同一任务的 Python 写法

保存为包含 `my-first-run/` 的工作目录下的 `my_first_run.py`：

```python
from pyrpmd import JobSpec, prepare, run, analyze_run, resume_run


def main():
    spec = JobSpec.from_file("my-first-run/quickstart.inp")
    report = prepare(spec, numerical=True)
    print("ready:", report["ready"])
    summary = run(spec, output="my-first-run/python-run")
    print(summary["samples"])
    resume_run("my-first-run/python-run")
    print(analyze_run("my-first-run/python-run")["task"])


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__"` 保护使 PyRPMD 可以使用 spawn 方式启动的工作进程。Python 与命令行共用同一解析器
和工作流；高层入口是 `run`、`prepare`、`resume_run` 和 `analyze_run`。

## 下一步

需要逐步控制积分器时，见 [API](api.md) 中的低层示例。使用真实模型时，先[连接势能](potentials.md)，
再选择工作流：[气相](workflows.md)、[材料与表面](periodic.md)或 [PCET](pcet.md)。
