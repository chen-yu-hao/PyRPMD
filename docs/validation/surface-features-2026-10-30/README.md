# 表面催化功能扩展验收（2026-10-30）

基准提交 `6fde29d2039c01921f00393bf17830b939af131b`，本次为未提交的工作区修改。
完整源码指纹见 [source_sha256.json](source_sha256.json)，结构化结果见 [results.json](results.json)。

## 已实现

- 1–3 维 standard/WT MetaD：周期晶胞、固定基底、Andersen/NHC/PILE-L、PLUMED 约定二次墙。
- q/p、RNG、热浴和 hills 一起原子保存；生产阶段热浴重置及墙启用时刻可指定。
- 实际非线性键长及安全最近像分割面的约束母轨迹与配对 NVE 子轨迹；质量度量只计活动端点，周期分割面严格限制在无像切换的保守半径内。
- 多维 FES 的条件 Boltzmann 投影，含非均匀 bin 体积；缺失积分支持保持 mask。
- 保存 FES → 条件 PMF → 同域母构型 → 配对子轨迹。温度、珠子数、PES、晶胞和文件哈希核对；同一静态墙哈密顿量贯穿 PMF/母/子体系。
- 显式面积、共反应物数、覆盖度和路径倍数换算，保留原始有限晶胞速率及其单位。
- 真实 CO/O/Pt(111) 模型下载、输入、运行/恢复/投影/速率示例和中文使用说明。

## 软件验证

CPU 全量：**1057 passed，153 subtests passed，11 skipped**。10 项缺可选 BKMP2，1 项缺可选 EAM。
新链路 MPI G=2/B=2 与独立串行的 **19 个 NPZ** 逐数组完全一致；原有表面速率的 **27 个 NPZ** 同样一致。
完成后的 resume 均不重新传播。新增 NHC 与 ASE 独立实现相符，曲面 RATTLE 时间可逆，沉积边界及热浴重置边界故障后可精确恢复。

## 真实金属实算

公开模型 DOI 10.5281/zenodo.21401971，论文 DOI 10.1038/s41467-026-77226-4。
39 原子中 21 个活动原子，共 63 个真实笛卡尔自由度；18 个固定底层 Pt。
600 K，P=1，0.2 fs；100 步热化、800 步生产，3 CV、2 hills、40 个样本。
run/resume/analyze/project 已执行；冻结位置和动量保持精确约束。
论文参数的三副本 40 ns 输入已校验，**未执行长生产采样**。

SP1 真实 EANN 势：每个下列算例先传播 20 fs 约束母轨迹，再传播两条各 20 fs 的无偏 ±p 子轨迹。

| 珠子数 | dt / fs | 最大约束残差 / bohr | 最大能量漂移 / eV/珠子 | 漂移 / kBT/珠子 |
| --- | --- | --- | --- | --- |
| 1 | 0.2 | 8.882e-16 | 5.41223e-05 | 0.0010468 |
| 4 | 0.2 | 1.776e-15 | 0.00933523 | 0.18055 |
| 4 | 0.1 | 2.220e-15 | 0.00232811 | 0.045028 |

所有算例固定坐标误差均为 0。四珠子 0.2 fs 的能量漂移超过现有速率诊断的 0.1 kBT 尺度；
缩到 0.1 fs 后约降为四分之一。这支持继续做步长收敛检查，不能将 0.2 fs 视为已验证的四珠子生产设置。

短 MetaD FES 没有覆盖氧化盆地/分割面，源速率工作流返回 `basin_lower_outside_histogram_centers`，
候选速率为空且没有启动子轨迹；没有把缺失数据补成论文速率。

## 可复现命令

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,mpi,analysis]'
python examples/surface/pt111/fetch_model.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
pyrpmd run examples/surface/pt111/pilot.py --output runs/pt111-pilot
pyrpmd resume runs/pt111-pilot
pyrpmd analyze runs/pt111-pilot --recompute
pyrpmd project runs/pt111-pilot --axes 2 --output runs/pt111-height
python examples/surface/pt111/check_dynamics.py
pyrpmd run examples/surface/pt111/rate_from_fes.py --output runs/pt111-rate
OMPI_MCA_rmaps_base_oversubscribe=1 mpiexec -n 4 python tests/mpi_surface_extensions_smoke.py runs/mpi-new
OMPI_MCA_rmaps_base_oversubscribe=1 mpiexec -n 4 python tests/mpi_surface_smoke.py runs/mpi-surface
python -m build
```

## 物理结论边界

这是功能与数值验收，不是论文催化速率的定量复现。直接 hill 求和没有网格加速；
长时间收敛采样尚未运行。源 PMF 的统计误差未校准，支持充分时仍只输出候选速率。
公开 double 模型与原文 float/旧 Fortran、NHC 自由度和墙处理存在明确差异，
先前记录的部分原文回穿轨迹不一致并未因接口实现而消失。
