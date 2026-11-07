# 修订验证：自动子空间、谐浴记忆与真实隧穿

## 旧三工况结论撤回

此前 `three_d_quantum_correction.py` 和 `three_regime_quantum_correction_report.py` 的物理精度解释不成立：全体系与子体系使用了不同观测量；二次观测量的经典高斯相关函数被误标为 RPMD；近简并线性谐振子的正常模混合不能验证量子 Fermi 共振恢复。旧报告的 366%、0.008%、33% 不应再作为方法优劣结论。

响应实部可以变号，negative_real_fraction 也不是谱正性检验。本轮使用 exp((iω−η)t) 约定下的耗散 Hermitian 部分 (χ−χ†)/(2i)。

## 新实现

`pyrpmd.analysis.quantum_subspace.select_subspace` 从独立先导坐标/力数据中发现非仿射力方向，利用三阶多项式力拟合识别跨子空间耦合，自动扩展至 1–3 维。目标观测量方向可显式保留。精确量子结果从不输入选择器。阈值是启发式，不是误差界；全三阶拟合目前只适合小体系。

`memory_embed` 对稳定线性谐性残余环境使用

```
K_A,eff = K_AA - K_AB K_BB^-1 K_BA
M_B(s) = K_AB [K_BB^-1 - (K_BB+s^2 I)^-1] K_BA
chi_embed = (I + chi_Q,A M_B)^-1 chi_Q,A
s = eta - i*omega
```

量子参考使用静态松弛后的势。剩余环境的线性 RPMD 响应以谐浴记忆核表示，不是冻结浴，也不是在线波函数与任意非谐 RPMD 环境联合传播。谐性极限恢复精确 Schur 补，非线性量子活跃部分中的闭合仍然是近似。可选 PIMD 静态协方差匹配单独报告，不默认认为它改善精度。

## 本轮实际计算

```
V = 0.4(q^2-1.4^2)^2 + sum_j omega_j^2/2*(b_j-lambda_j*q)^2
omega = (1.5, 2.3); hbar=m=kB=1
```

完整体系和低维参考均运行实际多珠 RPMD；所有比较均使用同一个 q 响应。量子全体系只用于审计。频率范围 [0.03,3.5]，700 点，eta=0.04。以下误差为复响应相对 L2 百分比，不是反应速率误差。

| 情形 | beta | lambda | 自动维数 | RPMD | 子空间+谐浴记忆 |
|---|---:|---|---:|---:|---:|
| 弱耦合 | 8 | (0.03,0.02) | 1 | 94.86% | 1.04% |
| 一个强伙伴 | 8 | (0.8,0.02) | 2 | 92.60% | 0.29% |
| 较强残余环境 | 8 | (0.8,0.05) | 2 | 92.38% | 1.48% |
| 较高温对照 | 2 | (0.8,0.02) | 2 | 93.03% | 0.97% |
| 两个强伙伴 | 8 | (0.8,0.5) | 3 | 89.90% | 全量子极限，不计降维成功 |

两个强伙伴时，强行使用一维、二维的记忆嵌入误差仍为 143.06%、26.02%。自动选择三维意味着没有压缩收益，不能把它宣传为低维方法成功。

独立耦合消融：相同二维量子空间，孤立参考与加入环境记忆的误差分别为 0.767%→0.292% 和 3.538%→1.478%。

最低双态宇称相反，扣除浴零点能后均低于势垒。孤立 q 劈裂约 0.290833；一个强伙伴时约 0.169408；两个强伙伴时约 0.135249。没有把任意振动峰当成隧穿。

## 数值设置和边界

- P=64；每个生产计算 2048 个初态，来自 64 条独立 HMC 链。链内快照和正反动量副本不是独立样本；按链组成 32 个块。
- 独立先导 2048 个快照；另以 128 条链、32768 个快照计算静态协方差。
- 实时传播无恒温器，tmax=200，输出间隔 0.04；beta=8 的 dt=0.02，beta=2 的 dt=0.005。
- 主量子基组 36×12×10、80 态；扩大至 44×16×14、120 态后，各案例响应变化均小于 0.015%。热尾本身不作为动态收敛证明。
- eta 从 0.04 减至 0.01 时，强伙伴误差升至 1.05%，较强残余环境升至 4.52%。不存在不依赖谱分辨率的固定精度保证。
- 独立先导重复和随机坐标旋转恢复相同子空间。20 项本地原型测试通过；这不等于全仓库 CI 已验证。
- 静态匹配对珠子数和采样有敏感性，不是总能改善。
- 所有结果 physical_converged=False。尚未验证一般非谐浴、严格后验误差界、反应通量/人口平台或真实分子体系。本轮没有重新完成 H+H2 混合反应速率或 Fermi 共振谱验证。

## 在仓库中复现

安装可选 numba 后，从仓库根目录执行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH=src python benchmarks/quantum_subspace/run.py --case weak
PYTHONPATH=src python benchmarks/quantum_subspace/run.py --case strong
PYTHONPATH=src python benchmarks/quantum_subspace/run.py --case residual
PYTHONPATH=src python benchmarks/quantum_subspace/run.py --case collective
PYTHONPATH=src python benchmarks/quantum_subspace/run.py --case warm
python -m pytest tests/test_quantum_subspace.py
```

输出位于 output/quantum-subspace，保留模型参数、随机种子、选模历程、采样记录、响应和原始链块。完整本轮计算快照、收敛脚本、图和中文 HTML 报告另作为会话附件提供。这里的数据属于这一特定模型族，不是通用方法精度承诺。
