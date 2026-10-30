# 二维采样方法与论文模型对照

本目录直接调用 PyRPMD 的 `run_fes`、`run_metadynamics` 和无偏 `Dynamics`，
保留独立种子、输入、原始样本、checkpoint、HILLS、支持 mask 和失败记录。
本次结果见 [2026-10-30 验证记录](../../docs/validation/fes-methods-2026-10-30/README.md)。

归档实验和优化数据来自 `6fde29d` 派生工作区；本 PR 的代码另移植至
`main` 的 `3dc82c3`，当前集成验证单独记录，不能将历史源码哈希视作当前提交的哈希。
Git 只保留紧凑汇总和关键 PNG。[完整证据 ZIP（22 MB）](https://drive.google.com/file/d/1ek7EqkyLH4ES1XT7Kd1Gu_fZOu76wdwG/view)
包含逐案例配置、数值数组和 PDF；完整 checkpoint 仍需从工作区读取或重新计算。

## 模型与文献

1. **二维谐振子**：`V=(x²+y²+z²)/2`，单位质量，β=4；不是实际氢原子模型。
   质心 FES 对任意 P 都是 `(x²+y²)/2+C`，珠子位置分布则有量子展宽。
   检查 P=1/8 的五种采样方法，P=1/4/8/16/32 的珠子二阶矩，以及
   P=1/8/32 的三种无偏积分器。RPMD 谐振子极限依据
   [Craig & Manolopoulos, JCP 2004](https://doi.org/10.1063/1.1777575)。
2. **OGRe 二维双阱、Ackley**：采用
   [Borgmans 等，JCTC 2023，19，9032–9048](https://doi.org/10.1021/acs.jctc.3c01028)
   的公式 13、14 和 300 K 温度。作者的
   [公开输入与数值记录](https://github.com/SanderBorgmans/SupportingInformation/tree/e7511c7dd48be80230c3cc4afc660696584712b1/2023_TBD_OGRe)
   固定到 commit `e7511c7dd48be80230c3cc4afc660696584712b1`。
   `ogre_models.c` 是公式的独立实现，能量单位 kJ/mol；额外 z 模式可分离。
   CV 数值映射到 bohr，质量取 1837.152647 a.u.，步长 0.1 fs。
   **仅 P=1 与裸势函数对照**，不可把非谐量子质心 FES 当成裸势。
3. **Metadynamics**：standard 与 well-tempered 的有限时间偏置估计，后者依据
   [Barducci、Bussi、Parrinello, PRL 2008](https://doi.org/10.1103/PhysRevLett.100.020603)。
   不进行时间依赖轨迹重加权或从有偏时间提取反应速率。
4. **MBAR**：对实际轨迹另用 PyMBAR 4.0.3 全样本核对；方法依据
   [Shirts & Chodera, JCP 2008](https://doi.org/10.1063/1.2978177)。

作者输入中的二维双阱为
`(x²+y²)² - 10 exp[-30(x-0.2)²-3(y-0.4)²] - 10 exp[-30(x+0.2)²-3(y+0.4)²]`。
Ackley 为 `-20 exp[-0.2 sqrt((x²+y²)/2)] - exp[(cos(2πx)+cos(2πy))/2] + e + 20`。
二者能量均为 kJ/mol。Ackley 在无限平面不限制粒子，因此本实现统一在
`|x|,|y|>8` 增加 `4[(|x|-8)_+^4+(|y|-8)_+^4]` kJ/mol 的墙。
论文比较区间 `[-6,6]²` 内的局部自由能差不变；墙是目标势的一部分。

基线 PyRPMD 自适应流程用 MBAR 重叠连通与 CV 有效样本数插入中点，保持刚度不变，
冻结后重新生产。OGRe 还检查 confinement、consistency 等并调整刚度。
这里对照相同模型的准确性与公开论文数值，没有执行 OGRe 本体或复现其整个参数扫描。
当前实现另支持显式覆盖网格和振幅相关性检查，配置见下方优化章节。

## 复现

依赖源码安装的 PyRPMD、C 编译器、NumPy、SciPy、h5py、matplotlib；
独立求解器检查另需 PyMBAR。无需 GNU Fortran。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
STUDY=/workspace/work/rpmd-fes-review

.venv/bin/python benchmarks/fes_methods/benchmark.py build --output "$STUDY/harmonic"
.venv/bin/python benchmarks/fes_methods/benchmark.py dynamics --output "$STUDY/harmonic"
.venv/bin/python benchmarks/fes_methods/benchmark.py quantum --output "$STUDY/harmonic"

for p in 1 8; do
  for seed in 3101 3102 3103; do
    for method in fixed adaptive unbiased standard well_tempered; do
      .venv/bin/python benchmarks/fes_methods/benchmark.py run \
        --method "$method" --beads "$p" --seed "$seed" --output "$STUDY/harmonic"
    done
  done
done

.venv/bin/python benchmarks/fes_methods/ogre_benchmark.py build --output "$STUDY/ogre"
for seed in 4101 4102 4103; do
  for method in fixed adaptive unbiased standard well_tempered; do
    .venv/bin/python benchmarks/fes_methods/ogre_benchmark.py run \
      --model double_well --method "$method" --seed "$seed" --output "$STUDY/ogre"
  done
  for method in fixed adaptive; do
    .venv/bin/python benchmarks/fes_methods/ogre_benchmark.py run \
      --model double_well --method "$method" --seed "$seed" --scale 5 --output "$STUDY/ogre"
    # 当前预算下 Ackley adaptive 预期返回非零并保存 failure.json；不应忽略该记录。
    .venv/bin/python benchmarks/fes_methods/ogre_benchmark.py run \
      --model ackley --method "$method" --seed "$seed" --output "$STUDY/ogre"
  done
done

.venv/bin/python benchmarks/fes_methods/solver_check.py "$STUDY/harmonic/fixed-p8-s3101-n1"
.venv/bin/python benchmarks/fes_methods/solver_check.py "$STUDY/ogre/double_well-fixed-s4101-n1"
.venv/bin/python -m pytest -q --junitxml="$STUDY/full-tests.xml"
.venv/bin/python benchmarks/fes_methods/summarize.py --base "$STUDY" --output "$STUDY/report"
```

新实验应使用新目录。已有目录会通过 PyRPMD 的 checkpoint 恢复，
`wall_seconds_this_call` 只是本次调用耗时，不可当作从头计算的性能指标。
`--scale 5` 是独立的新实验目录，仍从同一预先指定种子起始；不把短、长轨迹算作独立重复。

作者数值从 `benchmarking/2D/{double_well,ackley}/error_data/` 的三个
`grid_{data,error_data,info}.pkl` 取得，分别保存为 `{model}-grid_{...}.pkl`。
`extract_literature.py DIRECTORY` 仅允许 NumPy 数组和 dtype 构造器，转为 CSV/JSON。
论文图 12 的 0.29 对应三层条目；原始 `non_converged` 标记也会保留。
外部存档声明 CC BY-SA 4.0，本项目未将其 Python 实现或完整 pickle 数据并入源码。

## 统计定义与边界

- 每一正式配置使用三个独立种子，报告实际范围；这不是经校准的置信区间。
- 伞采样与弱偏置对照使用精确 **bin 平均密度**；metadynamics 使用点自由能参考。
  RMSE 在受支持格子上用最小二乘常数对齐，覆盖率必须同时阅读。
- `paper_point_min_gauge_rmse_kjmol` 另按作者脚本的点势和各自最小值归零计算。
  `summarize.py` 用同一组已验证的样本/MBAR 权重重分箱到公开输入的宽度：
  双阱 0.02、Ackley 0.05。没有重新采样，也没有填补空白区域。
- 几种方法的总预算不同，尤其 metadynamics 是一条生产流、伞方法是多窗口；
  本实验不能用于宣称等计算成本下某方法优于另一方法。
- `unbiased` 使用刚度 `1e-12` 的已知可移除偏置，以复用相同 MBAR 存储链路。
- 量子二阶矩的随机采样还用实际线性传播映射的离散 Lyapunov 方程独立核对，
  防止将有限样本波动误认成环聚合物温度或弹簧尺度错误。
- 有偏平衡采样、无偏 RPMD 动力学、经典论文模型比较是分别验证的三部分。
  本次没有提供非谐分子量子 FES 或化学反应速率的文献复现。

## 后续优化与相同预算对照

修复、参数与实际结果见[优化报告](../../docs/validation/fes-optimization-2026-10-30/README.md)。
`optimization.py` 接收 `ogre_benchmark.config` 的嵌套 JSON 覆盖项，并保存所有源码指纹。
每个标签使用独立目录；`evidence/configs/` 保存本次覆盖检查和恒温参数。

```bash
OPT=/tmp/rpmd-fes-optimization
python benchmarks/fes_methods/optimization.py --model double_well --method adaptive \
  --seed 4101 --label coverage-tuned --output "$OPT/production" \
  --library "$STUDY/ogre/build/double_well.so" \
  --override docs/validation/fes-optimization-2026-10-30/evidence/configs/double-adaptive-tuned.json
python benchmarks/fes_methods/budget_comparison.py \
  --output "$OPT" --baseline-data "$STUDY" --budget 4000000
python benchmarks/fes_methods/refresh_diagnostics.py --production "$OPT/production"
```

先对三个种子 4101/4102/4103 完成 `coverage-tuned`，再运行预算对照脚本。
它从基线 commit 加载原 adaptive 工作流，将已知先导成本计入总预算，
按最终窗口数分配剩余生产步数；不会使用解析参考选择窗口。
`refresh_diagnostics.py` 只重分析现有轨迹，并逐位检查自由能、概率、计数和 mask 不变。

`settings_scan.py --help` 给出控制实验参数；相同步数与相同物理时间实验分开报告。
MetaD 的复现入口为 `metadynamics_{parameters,performance,equivalence,compare}.py`，
分别用于参数对照、CPU 微基准、完整轨迹等价和共同支持区误差。
`performance` 和 `equivalence` 的 `--before-source` 接受基线 `bias.py` 快照：

```bash
git show 6fde29d2039c01921f00393bf17830b939af131b:src/pyrpmd/bias.py > "$OPT/bias_before.py"
python benchmarks/fes_methods/metadynamics_equivalence.py \
  --before-source "$OPT/bias_before.py" --output "$OPT/meta-equivalence"
```

`optimization_report.py --help` 可指定基线、优化、设置扫描和 MetaD 证据目录，
`--require-complete` 要求正式的三种子对照齐备。单种子探索和负结果单独保留，
不与三种子统计混算。图表同时给出原支持区覆盖率及前后共同支持区误差。
