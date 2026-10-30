# C++17 融合生产传播：真实 BKMP2 验证

这里验证的是已接入 `Dynamics.propagate`、umbrella/PMF、dRPMD 与 recrossing child 的生产代码，不是独立语言微基准。

## 可追溯证据

- 被测最终计算代码：`81a499e063635620b4378d4a337966ccbf8398c2`。
- 完整测试、BKMP2 对照与源码/wheel 构建：[GitHub run 41006610098](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41006610098)。工作流 checkout 的传输提交与实际被测源提交不同；`results.json.source_commit` 和产物内 `tested-commit.txt` 记录的是实际被测提交。
- 原始产物：artifact `12955786273`，ZIP SHA256 `7704005949223b3e290fd57adb3d89b2383eaae8e1c568715acf6cda0a8ec9bf`。下载后已独立核对完整 ZIP 摘要。
- 首轮核心实现 `37e3988a159d9122f596da990452f9f6f8f25781` 的真实 4-rank MPI、wheel 独立安装与运行验证：[run 41006136843](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41006136843)。后续只补充动态 Python 覆盖的回退保护与测试，未改动数值内核。
- 最终这一轮 CPU 测试：121 passed、2 skipped、87 subtests passed。两个跳过项为未安装 Torch；常规 CI 的独立 Torch CPU job 负责该适配器。
- 本目录的 `results.json` 保留所有重复计时和最大数值差，不只保留最佳值。

## 生产传播块吞吐量

H + H2/BKMP2，1500 K，dt=0.05 fs，1500 步；每项预热后重复5次，执行顺序随机化，报告中位数。OMP/OpenBLAS/MKL线程数均为1；runner允许CPU集合为0–3，没有将进程固定到单个CPU。Python 3.12.15，NumPy 2.5.4，GNU工具链13.3。

两条路径都计算逐步总能量、centroid与两个距离CV。Python对照仍使用已有C++ Cayley；新路径直接调用相同Fortran PES，复用最近一次PES返回的能量，避免观察量再调用一次PES。BCB本身仍保留每步两次势能力计算。计时包含传播和观测，不包含模型初始化与文件I/O。

| 工作量 | 珠子 | Python 步/秒 | 融合原生 步/秒 | 加速比 |
|---|---:|---:|---:|---:|
| NVE + 每步能量/CV | 16 | 5,104 | 51,698 | 10.13x |
| 二维 umbrella + 每步能量/CV | 16 | 3,127 | 50,702 | 16.22x |
| NVE + 每步能量/CV | 32 | 4,477 | 28,899 | 6.45x |
| 二维 umbrella + 每步能量/CV | 32 | 2,886 | 28,287 | 9.80x |

这些是特定runner/PES的短传播块基准，不能当作整个收敛速率计算的加速比，也不能与不同runner的旧语言基准直接相除。短workflow计时只执行一次且包含初始化/检查点；只用于回归诊断，不宣称稳定加速倍数。

## 正确性与工作流

1/2/8/16/32 beads的121步无偏传播中，终态q/p与旧循环逐值相同；能量最大绝对差1.4211e-14 hartree。二维umbrella窗口与rate-flux PMF窗口均实际传播253步、保存42个样本，含真实热浴、偏置/度量力和检查点；CV样本误差分别不超过1.78e-15和3.34e-16，能量误差不超过5.07e-14 hartree。受控中断后更换block大小恢复，与不中断的同后端结果逐值相同。

Recrossing对照执行一组成对child，每条500步；500个时间点的numerator全部逐值相同。dRPMD使用真实BKMP2和内置远区分类器，准确在第21步判定nonreactive，状态、停止步、轨迹与能量诊断匹配。这个测试不等于验证H交换自定义分类器的原生执行；自定义分类器仍走参考路径。

独立测试还覆盖：非对角二维伞矩阵、等价反应路径、flux metric、Andersen RNG顺序、PES失败整块回滚、dRPMD失败块重放到最后合法步、关闭context、状态型PES拒绝、Python PES/CV子类/实例覆写/嵌套覆写不被绕过，以及观测异常回滚。

## 当前边界

约束母轨迹Cayley–RATTLE、BCOCB/PILE、自定义Python PES/CV/分类器、未声明native binding的f2py、进程/MPI珠子服务和Torch/GPU仍保留原执行路径。并未将旧Fortran COMMON变为线程安全；普通调用和整块调用使用同一锁。这里没有证明珠子数、dt、采样长度或速率物理收敛，没有真实多物理节点/CUDA性能测试。

使用方法和完整分派规则见 [native-propagation.md](../../native-propagation.md)。永久CI为 `.github/workflows/fused-native.yml`；临时源传输工作流已删除。
