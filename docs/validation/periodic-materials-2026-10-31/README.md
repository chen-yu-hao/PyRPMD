# 周期材料审查：本次验证证据

这组证据对应 [周期材料与表面催化审查](../../periodic-materials-audit.md) 的修复。
基线为 `c4fa743f552c1d71db35fab791a2e63409bc1c3e`，计算发生在提交前的修改工作树中；
`validation.json` 记录被测源码和新增测试的 SHA256，不把基线提交号误称为含修复的版本。

## 证据文件

| 文件 | 含义 |
| --- | --- |
| [validation.json](validation.json) | 环境、命令、测试结果、被测文件哈希和证据文件哈希 |
| [pytest-final.txt](pytest-final.txt) | 冻结源码后的完整 pytest 输出，包含逐项跳过原因 |
| [eam-validation.json](eam-validation.json) | 本次重新执行的四组 Ni–H/H–D EAM pilot 原始报告 |

新增的 80 个回归用例随全套测试执行：MIC 与周期 CV 14 项，ASE 能量及应变一致性 7 项，
能量约定/恢复身份 28 项，Python 失败绑定 3 项，表面面积 13 项，原生 PIMD 预检 15 项。
中英文指南各构建 HTML 与 singlehtml，并使用 `-W --keep-going` 将警告视为失败。

全套本地测试包含已编译的原生扩展、ASE 和固定来源 Ni–H EAM；未安装的可选 PyTorch、
PySCF、tblite、PLUMED，以及未配置的外部 PES/MPI 等检查按测试自身条件跳过。
精确原因保留在 pytest 日志中，不能把这些跳过项视为通过，也不能据此声称完成 GPU 或多节点验证。

## 复现

从包含本次修复的源码提交开始，使用支持原生扩展构建的 Python/C++ 环境：

```bash
python -m pip install -e '.[test,ase,docs]'

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python benchmarks/materials/ni_h_eam.py \
  --model /absolute/path/NiAlH_jea.eam.alloy --download \
  --output /absolute/path/new-ni-h-periodic-results

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYRPMD_NIALH_POTENTIAL=/absolute/path/NiAlH_jea.eam.alloy \
python -m pytest -q

python -m sphinx -W --keep-going -b html docs /absolute/path/manual-en
python -m sphinx -W --keep-going -b html docs/zh /absolute/path/manual-zh
python -m sphinx -W --keep-going -b singlehtml docs /absolute/path/manual-single-en
python -m sphinx -W --keep-going -b singlehtml docs/zh /absolute/path/manual-single-zh
```

`--download` 只在模型文件不存在时使用，下载器校验固定长度和 SHA256；已有模型同样需通过校验。
仓库仅归档精简 JSON，没有重新分发势文件、完整轨迹或检查点。报告中的 case 路径相对于新生成的
输出目录，重新运行命令才会产生这些文件。完整测试包含本机 socket 和子进程检查，运行环境需允许它们。

## 科学边界

EAM 示例为 27 Ni + 一个 H/D、600 K、0.1 fs 步长、64 步平衡和 160 步无恒温生产，
比较 P=1/4。它检查真实势接口、有限差分力、周期平移以及有限步长传播；
`source_files_changed_during_run=[]` 是计算期间相关源码稳定的必要记录。

四个短轨迹仍为 `ballistic_or_drifting`，没有可接受的扩散系数或采样误差；
`physical_converged=false`、`equilibration_converged=false`。这里的环尺寸或能量漂移不能
解释为已经测得平衡同位素效应，也没有复现 H/Ni(100) 论文中的模型或速率。
本次没有重跑 CO/O/Pt(111) 生产计算、GPAW、真实金属 DFT 或大规模 ML 势；
相关收敛与统计工作保留在审查报告的 TODO 中。
