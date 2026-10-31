# 可选 MBAR 速率 PMF 与 H₃ 独立重复研究

## 为什么增加这个后端

[先前真实 H₃ / BKMP2 测试](validation/h3-bkmp2-2026-10-29/README.md)
在同一分割面 ξ*=1.00925 上得到 Gaussian UI 势垒 15.3032 kcal/mol、
histogram MBAR 势垒 18.3085 kcal/mol。此前的 40 次配对重采样中，
差值为 3.0053 kcal/mol，条件 95% 区间为 [1.9222, 3.4011] kcal/mol。
这说明需要检查窗口高斯近似和有限采样，不能只因软件测试通过就选一个
数值当作收敛结果。旧的短/长数据使用相同随机种子、存在前缀关系，
不能用作两个独立重复。

新后端让速率工作流直接使用 MBAR，并保留默认 UI 的兼容行为。
输入和失败语义见 [输入规范](input.md#速率-pmf-后端与分割面)，
测度和速率公式见 [科学约定](science.md)。

## 数值实现与核验

求解器在固定自由能 gauge 后使用阻尼 Newton；完全相同的约化势能行
可以精确合并，连续时间序列仍分别重采样。原求解器保留为 fallback
和 `method="legacy"` 对照。最终验收使用原始固定点方程的残差。

`benchmarks/h3/solver_check.py` 对相同保存样本同时调用新求解器、
旧求解器和官方 [PyMBAR](https://pymbar.readthedocs.io/en/stable/mbar.html)。
独立实现得到一个显式无采样目标态 `u_target=0, N_target=0`，
通过公开的 `W_nk` 列比较目标权重，同时比较 gauge 对齐的窗口自由能。
stride 只控制核验成本，不宣称已去除时间相关性；不用 PyMBAR 的
渐近误差替代本项目的 block bootstrap。

数值测试还包括已知非高斯双阱的解析势垒、独立的端点协方差参考、
缺失中间 bin、断连、仅一次有效 bootstrap、重复状态和旧算法等价性。
这些测试分别检查统计估计、失败处理和非线性方程，不证明真实体系遍历性。

## 预先固定的四组计算

以下参数在读取本轮结果之前确定，完整机器可读计划由
`python benchmarks/h3/convergence.py plan` 生成。

| Case | 随机种子 | βK | 窗口中心范围 / 间距 | 窗口 × replica | 保存样本 |
| --- | ---: | ---: | --- | ---: | ---: |
| k400-a | 821071 | 400 | −0.05…1.10 / 0.025 | 47 × 2 | 188,000 |
| k400-b | 821072 | 400 | −0.05…1.10 / 0.025 | 47 × 2 | 188,000 |
| k800-a | 831071 | 800 | −0.06…1.10 / 0.020 | 59 × 2 | 236,000 |
| k800-b | 831072 | 800 | −0.06…1.10 / 0.020 | 59 × 2 | 236,000 |

共同设置：1500 K、16 珠子、0.1 fs、R∞=10 bohr；每条流 300 fs
平衡加 1000 fs 生产、每 0.5 fs 采样。主直方图范围 [−0.30, 1.40]、
170 bins、40 次 block bootstrap；85 和 340 bins 只作曲线敏感性，
不额外重复 40 次 bootstrap。UI 对照与 MBAR 使用相同 draw。

每组独立计算 2 条受约束母链，每条 32 个构型，每构型 2 对正反动量
子轨迹，共 256 条、每条 80 fs；κ 使用 100 次重采样。
分割面均固定为 ξ*=1.00925，避免把 PMF 算法差异与分割面移动混在一起。
两种设置同时改变刚度与间距；差异不能归因于刚度单一参数。

## 执行、保存和验收

```bash
python -m pip install -e '.[test,analysis]' matplotlib
python benchmarks/h3/build_pes.py
python benchmarks/h3/convergence.py run --case k400-a --output runs/k400-a
python benchmarks/h3/solver_check.py --run runs/k400-a/run \
  --stride 10 --reference-pymbar --output runs/k400-a/solver-check.json
# 用相同命令运行另三个 case，再汇总共同父目录。
python benchmarks/h3/convergence_report.py runs --output runs/PMF-CONVERGENCE.md
```

每个 case 保存实际输入、源文件/PES/样本哈希、全部 checkpoint、HDF5
原始样本与能量、PMF/κ 曲线、逐 draw 失败原因、端点支撑及窗口分布矩。
汇总器核对四组计划、独立种子、生产门槛与官方 PyMBAR 核验；不完整
或失败的研究仍生成诊断报告，但返回失败状态。
`--smoke` 缩短轨迹来检查程序连接，明确排除在正式研究汇总外。

验收保留有效样本数、真实子轨迹、父链约束、重叠、能量误差、
样本/checkpoint 一致性等门槛；完整记录全部 bootstrap draw。
当前输出只在全部请求的 PMF draw 均成功且相关性分块支持充分时报告主势垒误差与置信区间；
若存在拒绝 draw，主字段为 `null`，曲线主误差为 NaN，接受子集的统计只放在 `supported_draw_*` 中。
最终速率使用同一规则；条件统计不代表无条件置信区间。应一起检查请求、尝试、有效和拒绝次数。
本页与历史记录中的条件区间保留其原始限定，不应改称当前实现的无条件主误差。
报告区分同流 bootstrap 误差、独立 seed 差异和 bin 宽敏感性。
即使所有工程门槛通过，`converged` 仍为 `false`：只有两个独立种子，
也未改变温度、珠子数、步长、R∞，不足以给出正式收敛速率。

GitHub 工作流 **H3 PMF independent-seed study** 同时执行四组计算，
每组保存原始 artifact，最终汇总另存一个完整 artifact。
实际运行数字在完成后单独归档，历史测试记录不覆盖。

## 本次实际运行

[2026-10-29 四组独立研究记录](validation/h3-pmf-refinement-2026-10-29/README.md)
保存成功的 GitHub run `40921070882`：四组与汇总共 6 个 job 全部成功，
所有 160 次 PMF 配对重采样有效，保存样本总数 848,000。
UI/MBAR 配对差值为 −0.0022、+0.3646、+0.1162、+0.0406 kcal/mol，
四个条件 95% 区间均包含零。结果支持方法一致性改善；独立 seed 与窗口
配置之间仍有波动，最终报告保留 `converged=false`。
