# 快速入门：可续跑的自由能计算

几秒内你将走完所有任务共用的 PyRPMD 流程：检查输入、运行伞抽样、读取结果、与精确解对比、续跑，并在不重跑动力学的情况下重新分析。
本节使用内置谐振子势，无需外部模型；它的质心自由能有解析解，因此你可以直接检验算出的数值。

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

这个例子计算 300 K 下三维谐势中一个 H 核**沿环聚合物质心 x 分量的自由能**。五个伞窗口各运行两个独立副本：
先平衡 0.5 ps，再生产 25 ps、每 12.5 fs 保存一个样本，使用 4 个珠子和 0.5 fs 步长。整个计算在单核上约需 4 s。

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
| `sampling.andersen_interval=50` | 每 50 步（25 fs）重新抽取全部动量 |
| `sampling.replicas=2` | 每个窗口的独立随机流数 |
| `analysis.bootstrap=20` | 用于误差条的分块 bootstrap 抽样次数 |
| `analysis.min_count=20` | 样本数更少的格子会被屏蔽而不是给出数值 |

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

摘要给出 `task="fes"`、一维 CV、5 个窗口、10 条采样流和 20000 个保存样本；`effective_samples_per_stream`
给出统计独立的样本数（本例每条流有几百个）。`overlap_connected=true` 表示所有窗口共享一个全局归一化。`warnings` 列出有效样本不足
和未采样的网格，本例中为空。单次运行的 `converged` 按设计保持为 `false`：PyRPMD 不会凭一次运行认定珠子数、步长或采样已经收敛，
这些要由[科学约定](science.md)中的系列计算确定；下面第 4 步就是其中最简单的一种检验。

`summary["files"]` 给出数据文件路径。FES 表中 `free_energy_hartree` 是相对自由能，`raw_count` 是每格样本数，
`masked` 标记没有采样支持的格子，这些格子的值为 NaN。见[输出与分析](results.md)。

## 4. 与精确结果对比

谐势中环聚合物的质心与内部环模式解耦，所以无论珠子数是多少，质心自由能都等于经典结果：
$F(x)=\tfrac12 k x^2$ 加一个常数。因此这个入门例子实际检验了采样、伞势的珠子缩放和 MBAR 重建：

```python
import json
from pathlib import Path

import numpy as np

run_dir = Path("my-first-run/run")
beta = json.loads((run_dir / "results/summary.json").read_text())["beta"]  # 1/(kB T)，单位 hartree^-1
table = np.genfromtxt(run_dir / "fes.csv", delimiter=",", names=True)
x = table["cv0_bohr"]
exact = 0.5 * 0.002 * x**2
difference = table["free_energy_hartree"] - exact
difference -= difference.mean()            # 自由能的零点是任意的
print("x / bohr   F - F_exact (kBT)   1 SE (kBT)")
for row in zip(x, difference * beta, table["uncertainty_hartree"] * beta):
    print("%8.3f %14.3f %14.3f" % row)
print("largest deviation: %.2f kBT" % (np.abs(difference).max() * beta))
```

最大偏差约为 0.1 $k_BT$（300 K 下 0.06 kcal/mol），与给出的分块 bootstrap 标准误一致；越靠近边缘、样本越少的格子误差越大。
参考格（自由能零点）的不确定度按定义为 0。

下面三个小实验说明精度由什么决定；每次使用新的 `--output` 目录：

- **减少采样。** 设 `production_steps = 500`。曲线变得很粗糙，`warnings` 会指出独立样本少于 20 的采样流：
  此时可信的是误差条，而不是曲线本身。
- **增加珠子。** 用 `--beads 16` 运行。质心自由能在误差内不变，正如上面的理论所要求的。这是谐势质心的特殊性质：
  非谐势能或能量估计量需要做珠子数收敛系列。
- **加大步长。** 对这个很软的模型，`timestep = "2 fs"` 仍然可用；含刚性键的真实分子需要按[科学约定](science.md)做步长检验。

## 5. 续跑与重新分析

```bash
pyrpmd resume my-first-run/run
pyrpmd analyze my-first-run/run --recompute --bins 8 --bootstrap 10
```

`resume` 让中断的运行从最后提交的状态继续；已完成的运行保持不变。`--recompute` 用新的分箱或 bootstrap
设置从保存样本重建分析，不运行动力学。`--bins` 和 `--bootstrap` 隐含 `--recompute`。新的分析会覆盖
`fes.*` 和 `results/summary.json`，需要保留旧版本时先复制。

保存的输入定义运行身份。要改变珠子数、温度、势能、步长或采样，使用新的运行目录。

## 6. 同一任务的 Python 写法

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
