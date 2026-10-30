# 电极模型接口与平衡采样验证（2026-10-31）

本记录对应新增的 `pyrpmd.electrochemistry`。基线为
`3dc82c399a7701d19d636ee78b43f2e8ba7659f9`；实际验证的源文件 SHA256、环境
和示例结果保存在 [validation.json](validation.json)。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 最终完整测试套件 | **1562 passed, 14 skipped, 153 subtests passed** |
| 其中新增电化学测试 | **163 passed**，无新增功能测试跳过 |
| HTML / singlehtml 手册，警告作为错误 | 两种构建均通过 |
| 解析二次充电模型 | 电荷、巨势、核力、能量零点及电位符号检查通过 |
| 非线性电子数求解 | 独立标量根与有限差分力检查通过；局部极大值、域外根被拒绝 |
| 共享电子数分布 | P=1、P=4 耦合 Gaussian 质心/电子数协方差检查通过 |
| MC 边界与 HMC | 平坦有限区间均匀分布、偏置、固定原子、接受／拒绝和异常回滚检查通过 |
| Native / NumPy | 三种现有积分器路径一致性检查通过 |
| 5 步核 HMC 势能调用 | 12 → 6；另有一次独立的核提案后能量诊断调用 |
| 缓存对照 | 连续三轮改变 N 的采样中，坐标、N、接受结果、计数及 RNG 均与无缓存对照一致 |
| HER 后处理 | 两条解析析氢极限、详细平衡、独立 ODE 稳态、电荷与物料守恒检查通过 |
| 实际 CLI FES 小样例 | 4 beads，9 窗口，720 个保存样本；MBAR overlap connected |
| 完成后 resume | CLI summary 相同；集成测试逐数组检查 checkpoint/FES 保持相同 |

完整测试输出见 [pytest.txt](pytest.txt)。初次受限环境运行中的两个既有 i-PI
测试因不允许创建 localhost socket 而失败；允许本地 socket 后两项均通过，
并对最终源码重新运行了上表中的完整测试。最终运行没有失败。

14 个跳过包括 10 个未构建的可选 BKMP2 真实势测试、一个未指定外部 Ni–Al–H
EAM 数据的测试、一个可选独立 PMF 参考求解器测试、两个未安装 PyTorch 的测试。
没有将这些项目记作通过。仓库已有独立真实势和可选后端 CI 流程继续保留。

## 数值与物理范围

电子数求解的批次轴始终是独立核构型。绝热适配器构造每构型
\(\Omega(q)=\min_N[E(q,N)-\mu N]\)，不把共享电子数极小化伪装为普通可分批 PES。
共享 N 的采样通过另一个 HMC/MC API 实现，其电子数接受指数使用平均珠子能量，
核 HMC 使用总环 Hamiltonian 的 \(\beta/P\) 权重。

FES 示例为解析周期充电模型，横轴是横向位移、纵轴是表面高度；Pt 符号表示
固定参考几何，模型没有 Pt–水相互作用或成断键反应。720 样本仅验证接口流程，
`converged=false`；重叠连通不证明物理采样、bead 数或时间步收敛。

[共享 N 示例输出](shared-number-demo.json) 使用 500 个热化 sweep 和 2000 个
保存 sweep，并报告解析 Gaussian 的均值与协方差。短链样本有明显相关性：
本次质心高度方差约 0.04084 bohr²，解析值约 0.05278 bohr²，相差约 23%。
该示例明确保留 `physical_converged=false`，不能据其短链统计宣称收敛。
实现正确性的独立分布测试使用不同的解析模型和充分混合的专门采样设置。

[HER 扫描](her-demo.csv) 使用另行指定的教学正反向系数，在 U=0、单位活度下
给出覆盖度 0.5 和零电流。它未从此 FES 提取反应速率，电位也未标定到 SHE。
瞬时电流保留吸附氢积累项；只有稳态使用 \(j=-2F\Gamma\,\mathrm{TOF}_{H_2}\)。

## 复现

从仓库根目录，在有本地 socket 支持的标准环境中执行：

```bash
python -m pip install -e '.[test,ase,docs]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b singlehtml docs docs/_build/singlehtml
pyrpmd validate examples/electrochemistry/fes2d.inp --check-pes
pyrpmd run examples/electrochemistry/fes2d.inp --output runs/electrode-fes
pyrpmd resume runs/electrode-fes
python examples/electrochemistry/sample_grand_canonical.py
python examples/electrochemistry/her_scan.py
```

专门的 `.github/workflows/electrochemistry.yml` 在干净 runner 中执行新增测试、
CLI FES、resume、共享 N 与 HER 示例，并上传完整示例输出。实际远端结论请查看
对应提交的 Actions/PR Checks，不能由本地结果推断。

接口、公式与固定提交的参考源码见 [电化学文档](../../electrochemistry.md)。
