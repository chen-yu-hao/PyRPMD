# 快速入门：从输入到可恢复的自由能计算

这一节使用包内谐振子势能，不需要外部模型。目标是完成一次输入检查、采样、分析和续跑，
并学会判断输出。**短轨迹只演示程序流程，不能作为收敛的自由能结果。**

## 1. 运行第一个任务

在安装好的源码仓库根目录执行，`runs/manual-first` 必须是新的空目录：

```bash
pyrpmd validate examples/manual/quickstart.inp --check-pes
pyrpmd run examples/manual/quickstart.inp --output runs/manual-first
pyrpmd analyze runs/manual-first
```

`validate` 检查输入、单位、任务限制及资源设置；`--check-pes` 还会实例化势能并进行数值准备。
`run` 执行计算，终端显示 JSON 摘要，运行目录保存配置、样本和检查点。
`analyze` 默认读取已保存的摘要，不增加轨迹。

## 2. 阅读完整输入

例子研究一个 H 核在三维谐势中的 **x 方向质心概率密度**。三个伞窗口各有两个独立副本；
每个副本先平衡 20 步，再生产 100 步，每 5 步保存一个样本。四个珠子和 0.1 fs 步长
都是教学选择。

```{literalinclude} ../../examples/manual/quickstart.inp
:language: toml
```

源文件：[quickstart.inp](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/manual/quickstart.inp)。

| 字段 | 这个例子中的意义 |
| --- | --- |
| `task="fes"` | 对保存的伞抽样样本做 MBAR 重建 |
| `beads=4` | 每个原子以四个环珠子表示 |
| `potential.params.k=0.002` | 谐势 $V=\tfrac12 k\lvert q\rvert^2$ 的原子单位力常数 |
| `cvs.estimator="centroid"` | CV 作用于环质心，不能与珠子平均非线性 CV 混用 |
| `umbrella.stiffness` | 物理伞势的力常数；程序处理环 Hamiltonian 的珠子缩放 |
| `sampling.replicas=2` | 每个窗口的独立随机流数，不是珠子数 |
| `analysis.bootstrap=0` | 此例未估计 bootstrap 误差条；缺失误差不是零误差 |

温度、长度、时间和力常数尽量写成带单位的字符串。原子索引从 0 开始；
完整规则见[输入](input.md)与[科学约定](science.md)。

## 3. 检查结果，而不仅是退出码

```python
import json
from pathlib import Path

run_dir = Path("runs/manual-first")
summary = json.loads((run_dir / "results/summary.json").read_text())
for key in ("task", "dimension", "windows", "streams", "samples",
            "overlap_connected", "converged", "warnings", "files"):
    print(key, summary.get(key))
```

此例应报告 `task="fes"`、一维 CV、3 个窗口、6 条采样流、120 个保存样本。
样本总数不等于独立样本数。`overlap_connected` 检查窗口是否可以建立全局归一化；
`warnings` 可能包含有效样本不足和未采样网格。`converged` 保持 `false`，因为没有执行
珠子数、时间步、轨迹长度和窗口覆盖的收敛系列。

数据文件路径以 `summary["files"]` 为准。FES 表格的 `free_energy_hartree` 为相对自由能，
`masked` 标记未支持的区域，`raw_count` 是原始计数。被屏蔽的区域不能补成零势垒。
详见[输出与分析](results.md)。

## 4. 续跑与重新分析

```bash
pyrpmd resume runs/manual-first
pyrpmd analyze runs/manual-first --recompute --bins 8 --bootstrap 10
```

已完成运行的 `resume` 应保持终态，不再增加生产步。断点运行则从已提交状态继续。
`--recompute` 只重新分析保存样本，可修改 FES 分箱与 bootstrap 次数；它不会改善采样覆盖，
也不会延长动力学。新的分析结果会写回结果文件，重要版本应另外保存。

如要改变珠子数、温度、势能、时间步或采样方案，请创建新运行目录。为了运行更长的任务而
直接修改已保存 `resolved.json` 或检查点，会破坏运行身份；不要用这种方式延长任务。

## 5. 同一任务的 Python 入口

把下列程序保存为仓库根目录下的 `my_first_run.py`，使用一个新的输出目录：

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

`if __name__ == "__main__"` 保护执行入口，便于以后启用 spawn 进程并行。
Python 和命令行调用的是同一套输入解析与工作流。PyRPMD 的高层入口是 `run`、`prepare`、
`resume_run`、`analyze_run`；不要将 PySCF 的 `kernel()` 用法套到本项目。

## 下一步

需要自由控制积分步骤时，阅读 [API](api.md)中的完整低层示例。
需要真实模型时，先学习[连接势能](potentials.md)，然后选择[气相工作流](workflows.md)、
[材料与表面](periodic.md)或 [PCET](pcet.md)。
