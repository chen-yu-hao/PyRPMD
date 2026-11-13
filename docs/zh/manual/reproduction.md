# 论文体系与复现计算清单

**建议先完成 H + H₂/BKMP2 的收敛链，并行做 NRPMD 2013 的两态谐振模型；再开展 q-TIP4P/F 水的平衡、扩散和光谱。**
这三条主线能分别验证速率、电子非绝热动力学和凝聚相核量子效应。下方列出全部 111 个条目的任务。
来源及源码映射见[原始论文目录](references.md)。

## 任务性质与优先级

“原论文体系”指所引原文实际计算的模型；“现有入口”可能只是替代回归或短流程。
**原论文复现**必须匹配势文件、参数、初态、动力学、估计量和后处理。
**替代模型回归**用于检查实现；**新体系应用**需要独立建立模型和参考。三者在记录中分别标识。
这里是计算计划和源码审计，未执行新的物理生产模拟。已有结果以固定提交下的
[literature-validation](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/literature-validation.md)
和 [campaign.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/literature/campaign.json) 为准。

| 优先级 | 含义 | 建议交付物 |
| --- | --- | --- |
| P0 | 解析/数值控制与接口校验 | 参考值、误差、时间步/网格/珠数检查 |
| P1 | 优先开展的论文模型或核心方法计算 | 原文目标图/表、独立种子区间、参数收敛记录 |
| P2 | 需要模型数据、较大采样或方法补全 | 先交付缺口清单和可验证的中间量，再计算目标曲线 |
| P3 | 复杂组合或目前明显不完整的原文流程 | 模型/采样/估计器补全后单独注册研究计划 |

优先级是本次建议，不是原论文给出的排序，也不是耗时保证。P0/P2 等表示先控制、后物理体系的两阶段任务。

## 推荐首先完成的八组计算

| 顺序 | 体系与方法 | 应计算的核心结果 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| 1 | 谐振子、四次非谐振子；PIMD、积分器、温控、估计器 | 位置方差、能量、热容、动量分布；P、dt、相关时间 | 解析/独立参考一致；热浴采样时间不当实时间 |
| 2 | H + H₂ / BKMP2；RPMD rate | PMF、k_QTST、κ(t)、k(T)、H/D KIE | 父链时间、独立伞窗精度、P256 动态收敛补齐 |
| 3 | Richardson–Thoss 2013 两态谐振模型 | Fig.1(c) 位置相关及电子相关；平均符号、有效样本 | 使用 lowest_adiabatic 分裂；beads/dt/样本收敛 |
| 4 | Tully I–III；FSSH/MASH/RPSH-CA/BA/DM | 反射/透射四通道、布居、相干、纯度、活动态比例 | 相同物理入射分布，正确宽度及终态定义 |
| 5 | q-TIP4P/F 水；PIMD、RPMD、PA-CMD/TRPMD | RDF、能量、MSD/VACF 的 D；偶极相关与 IR | 分开古典/量子与平衡/实时间生产；RPMD 用 NVE，PA-CMD/TRPMD 按内模温控定义；原 Fig.5 用 PA-CMD |
| 6 | OGRe 双阱/Ackley 与 OPES Müller 模型 | FES、势垒、RMSE、覆盖率、ESS、总力调用数 | 包含 pilot 成本；OPES 变体分别核验，固定历史对照 PLUMED |
| 7 | Eckart、束缚对称双阱；RPI、劈裂 | k(T)、路径、作用量、涨落行列式、Δ | 收敛珠数和 Hessian；分子体系先补平移/转动处理 |
| 8 | 2008 PCET 参数模型 → 2019 Au/TEAH⁺ | k(U,T)、正反速率、H/D KIE、振动态贡献 | 先核验局域模型；原文距离平均/双电层需另补 |

这些主线的出处分别见 [RPMD](references.md#rpm-rate)、[NRPMD](references.md#na-thermal-nrpmd)、
[Tully](references.md#na-tully-models)、[水模型](references.md#water09)、[采样](references.md#samp-adaptive)、
[瞬子](references.md#rpi)、[PCET](references.md#pcet)。

## 已有证据中仍待完成的环节

H + H₂ 的固定 P=128 四对扩展记录已在论文精度内一致，但补充父链半时长等效性未通过，
P=256 动态 campaign 尚未完成，独立伞窗精度也仍是单独条件。不要只凭最终速率接近论文就结束研究。
q-TIP4P/F 古典 OpenMM 参考已有完整八种子记录，但准备阶段能量检查、短时 MSD 窗口及步长等效性仍有问题，
量子与 PyRPMD 生产输运尚未运行。H + CH₄、OH + H₂、F + H₂、Ni 表面的状态各不相同，见上述原始记录，不能统一标“已复现”。

## 每一项都要保存什么

1. 原文 DOI/版本、目标图表和是否使用勘误；原文数字、数字化误差或独立参考方法。
2. PES/模型文件与 SHA256、软件提交、编译/设备信息、单位、质量、电子/核状态、边界与对称性因子。
3. 温度、珠数、dt、初态/平衡方法、独立随机种子、窗口/CV、父链与子轨迹长度、输出频率。
4. 与论文相同的估计量与归一化；PMF 只可移除任意加性常数，不能用拟合比例掩盖差异。
5. 独立链或相关块误差、有效样本、未决轨迹、平均符号；共享 walkers 和交换槽位不能当独立数据。
6. 预先确定的收敛范围与等效性容限；分别报告物理模型差异、统计误差和离散误差。没有统一的“误差小于5%即可”规则。

固定原始输入和全部失败/未决记录；通过条件由具体物理量决定，不能为贴合论文曲线而事后调整。

## 可直接定位的起步命令

以下是已有入口，默认预算是起点，不是收敛保证；先完成安装并在仓库根目录执行。
外部势、PLUMED、电子结构模型等依赖仍按各指南安装。命令本次仅核对入口，没有运行生产模拟。

```bash
# Paper-specific NRPMD setup; inspect its plan before production.
python examples/nonadiabatic/published_nrpmd_benchmark.py --plan

# Adaptive umbrella on a published analytic test model.
python benchmarks/fes_methods/ogre_benchmark.py build --output runs/ogre
python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method adaptive --seed 4101 --output runs/ogre

# Published OPES model and independent reference workflow.
python benchmarks/tier2_literature/sampling.py reference --output runs/tier2-sampling
python benchmarks/tier2_literature/sampling.py opes --output runs/tier2-sampling

# Fixed-full-ring committor: a small demonstration budget.
python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example
```

NRPMD 正式计算应按脚本说明保留 `--potential-split lowest_adiabatic` 并使用原文数字化数据。
H + H₂ 和水已有 campaign，不要绕过原记录另以默认短输入声称完成；从其未完成条件继续。

## 完整逐项清单

下列原体系描述只使用已核实信息。UI 原文具体生物模型、部分 constrained-dynamics 应用协议与 TREQ 六个反应的完整清单等
尚未从可得全文/补充信息恢复的内容均在“缺口”中保留，不以猜测的 alanine、水或 NH₃ 代替。


### 平衡统计、动力学、反应速率与光谱复现任务

#### task-pimd

**原始可区分粒子 PIMD — P0**

论文：[cw81](references.md#cw81), [pr84](references.md#pr84), [pile10](references.md#pile10).

**原文体系。** 理论源于量子—经典环聚合物同构；早期 PIMD 原文研究熔融 KCl 中的 F 中心，并非仓库自带的核动力学体系。

**应该算什么。** 先算谐振子位置方差、动能、势能、总能和珠数收敛；再用指定水模型算 RDF 与能量。

**现有入口。** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py); [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md)

**缺口与边界。** 谐振子回归不能称为 KCl 论文复现；需要原哈密顿量、参数和独立统计误差。

[实现与范围](references.md#pimd).

#### task-rpmd

**RPMD 实时间关联函数 — P0**

论文：[rpm04](references.md#rpm04), [kubo57](references.md#kubo57).

**原文体系。** Craig–Manolopoulos 的低维谐振/非谐模型关联函数。

**应该算什么。** 算 Cqq 的 Kubo 关联、频谱、短时误差以及 P 和时间步收敛，并与独立量子对角化比较。

**现有入口。** [benchmarks/trpmd/harmonic_spectrum.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/trpmd/harmonic_spectrum.py); [benchmarks/tier2_literature/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy.py)

**缺口与边界。** 现有四次势脚本是另一个文献模型，不能当成原图复现；非线性量与长时相干不保证精确。

[实现与范围](references.md#rpmd).

#### task-rpm-rate

**热平衡 RPMD 双分子反应速率 — P1**

论文：[bennett1977](references.md#bennett1977), [chandler1978](references.md#chandler1978), [rate05](references.md#rate05), [rate05b](references.md#rate05b), [bimol09](references.md#bimol09), [rpmrate13](references.md#rpmrate13), [bkmp21996](references.md#bkmp21996).

**原文体系。** 2005 年 Eckart 等模型；2009 年 H+H₂、Cl+HCl、F+H₂，须同时核对勘误。

**应该算什么。** 算 W(ξ)、势垒位置与高度、kQTST、κ(t) 平台和 k(T)；检查 P、dt、入口距离、窗口重叠、亲子轨迹时间和重复采样误差。

**现有入口。** [benchmarks/h3/rpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/rpmd.py); [benchmarks/h3/convergence.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/convergence.py); [docs/literature-validation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/literature-validation.md); [docs/science.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/science.md)

**缺口与边界。** 当前 H+H₂ 的 P128 定珠数比较已通过，但 P256 与伞形采样域仍是独立关卡；F+H₂ 必须匹配修正后的势能面及电子约定。

[实现与范围](references.md#rpm-rate).

#### task-direct-rpmd

**直接轨迹热平衡 dRPMD — P1/P2**

论文：[drpmd18](references.md#drpmd18).

**原文体系。** 原文是 H₂+H₃⁺，区分直接质子跳跃与经 H₅⁺ 络合物的交换机制。

**应该算什么。** 算 P反应(b,E)、积分截面、k(T)、机制分支比、碰撞寿命；检查 bmax、起始距离、轨迹终止和截尾误差。

**现有入口。** [benchmarks/h3/drpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/drpmd.py); [docs/science.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/science.md); [docs/validation/drpmd-optimization-2026-10-30/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/validation/drpmd-optimization-2026-10-30/README.md)

**缺口与边界。** 仓库 H+H₂ 是替代验证；原文机制复现还需 H₅⁺ 势能面、通道标记及原始条件。

[实现与范围](references.md#direct-rpmd).

#### task-state-rpmd

**谐振近似初态选择 NE-RPMD — P1**

论文：[nerp16](references.md#nerp16), [state20](references.md#state20), [state21](references.md#state21), [state22](references.md#state22).

**原文体系。** Mu/H/D+H₂(v=0,1)；多原子扩展为 H+CH₄/CHD₃ 和 F+CHD₃ 的 C–H 伸缩激发。

**应该算什么。** 算态分辨 σvj(E碰撞)、阈值、P(b)、初始能量分布和孤立反应物稳定性；收敛 β、P、dt、bmax 和轨迹数，与同能量定义的 QCT 对照。

**现有入口。** [benchmarks/h3/state_selected_reproduction.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/state_selected_reproduction.py); [benchmarks/h3/audit_state_selected_initial.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/audit_state_selected_initial.py); [benchmarks/h3/STATE_SELECTED_REPRODUCTION.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/STATE_SELECTED_REPRODUCTION.md); [examples/state_selected/ne_rpmd.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/state_selected/ne_rpmd.inp)

**缺口与边界。** 不是任意非谐本征态的精确制备；小规模试跑不等于原论文完整轨迹统计。

[实现与范围](references.md#state-rpmd).

#### task-qct

**谐振 QCT 与全势能面作用量匹配 QCT — P1**

论文：[qct65](references.md#qct65), [state20](references.md#state20), [state22](references.md#state22).

**原文体系。** QCT 奠基应用为简单势垒上的 H+H₂；现代 NE-RPMD 文献也使用 QCT 对照。

**应该算什么。** 算态分辨截面、作用量、孤立能量漂移；分别按固定碰撞能和固定总能比较初态分布与反应结果。

**现有入口。** [examples/state_selected/qct.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/state_selected/qct.inp); [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/state_preparation_study.py); [benchmarks/h3/STATE_PREPARATION_STUDY.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/STATE_PREPARATION_STUDY.md)

**缺口与边界。** 作用量量子化是半经典近似；全 PES 版本限 H₃ 双原子基准，不能宣称通用多原子算法或精确本征态。

[实现与范围](references.md#qct).

#### task-state-switch

**实验性平滑切换初态诊断 — P2**

论文：[state20](references.md#state20), [state22](references.md#state22).

**原文体系。** 没有找到此具体切换变体的原始论文；当前实验体系为 H+H₂。

**应该算什么。** 算切换做功、末态能量/作用量分布、孤立稳定性及截面对切换时长和曲线的敏感性。

**现有入口。** [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/state_preparation_study.py); [benchmarks/h3/STATE_PREPARATION_STUDY.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/STATE_PREPARATION_STUDY.md)

**缺口与边界。** 不得归为已有文献方法，也不能声称保证纯量子态或修正 NE-RPMD 的已知误差。

[实现与范围](references.md#state-switch).

#### task-integrators

**速度 Verlet、正常模 BAB 与 Cayley BCB — P0**

论文：[vv82](references.md#vv82), [pile10](references.md#pile10), [cayley19](references.md#cayley19).

**原文体系。** Verlet 参考为小水团簇；Cayley 原文使用谐振、弱非谐和四次振子。

**应该算什么。** 画稳定轨迹比例、能量误差/漂移及可逆性误差随 dt、P 的变化；核验 P=1 和自由环极限。

**现有入口。** [docs/integrators.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/integrators.md); [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/dynamics.py); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py)

**缺口与边界。** 稳定不等于无时间步偏差；论文图复现必须匹配原势能和 β。

[实现与范围](references.md#integrators).

#### task-stochastic-integrators

**BAOAB 与 Cayley BCOCB — P0/P1**

论文：[baoab13](references.md#baoab13), [bcocb20](references.md#bcocb20), [cayley19](references.md#cayley19).

**原文体系。** BAOAB 使用振子及丙氨酸二肽；BCOCB 包含 298 K、0.998 g/cm³ 的 32 分子 q-TIP4P/F 水。

**应该算什么。** 比较构型协方差、动能、相关时间、稳定性随 P/dt 的变化；原水体系再比动能珠数收敛和 IR 时间步敏感性。

**现有入口。** [docs/integrators.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/integrators.md); [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [benchmarks/trpmd/harmonic_spectrum.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/trpmd/harmonic_spectrum.py)

**缺口与边界。** Torch BCOCB 不支持 inner_steps>1 的 MTS；必须报告具体劈分与温控选择。

[实现与范围](references.md#stochastic-integrators).

#### task-classical-baths

**Andersen、Cartesian Langevin 与随机速度缩放温控 — P0**

论文：[and80](references.md#and80), [langevin07](references.md#langevin07), [svr07](references.md#svr07).

**原文体系。** 原文使用模型/Lennard-Jones 体系；SVR 包含 TIP4P 水的相态测试。

**应该算什么。** 算动能卡方分布、均值方差、谐振构型协方差及相关/平衡时间，检查耦合强度与 dt 敏感性。

**现有入口。** [docs/andersen-sampling.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/andersen-sampling.md); [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**缺口与边界。** 温控会改变动力学关联；全局离散 Andersen 碰撞时钟属于具体实现变体。

[实现与范围](references.md#classical-baths).

#### task-pile

**PILE-L 与 PILE-G 正常模温控 — P0/P2**

论文：[pile10](references.md#pile10), [svr07](references.md#svr07).

**原文体系。** 原文效率测试包括水和钯中的氢。

**应该算什么。** 算模协方差、量子动势能、积分相关时间、每次势力调用获得的独立样本数，扫描 P 和摩擦。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [docs/integrators.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/integrators.md)

**缺口与边界。** 分布检验不能代替原文的分子体系加速验证；需匹配水/Pd 势能与参数。

[实现与范围](references.md#pile).

#### task-nhc

**Nosé–Hoover 链温控 — P0**

论文：[nhc92](references.md#nhc92), [yoshida1990](references.md#yoshida1990).

**原文体系。** 原文谐振子和自由粒子的正则/遍历性检验。

**应该算什么。** 算联合相空间分布、动能分布、扩展能漂移，扫描链长、耦合和子步。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/thermostats.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/thermostats.py)

**缺口与边界。** 需核验遍历性与参数收敛；此路线不支持键约束。

[实现与范围](references.md#nhc).

#### task-gle

**Cartesian/正常模 GLE 热浴 — P0**

论文：[gle09](references.md#gle09), [pile10](references.md#pile10).

**原文体系。** 原文研究按谐振频率选择性调控热浴响应。

**应该算什么。** 对照 Lyapunov 目标算联合动量/辅助变量协方差，扫描频率下的位置与能量相关时间。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**缺口与边界。** 任意 GLE 矩阵不自动具备量子修正或最佳采样效率；必须提供参数来源。

[实现与范围](references.md#gle).

#### task-piglet

**外部标定矩阵驱动的 PIGLET — P2**

论文：[piglet12](references.md#piglet12).

**原文体系。** 原文为第一性原理液态水的结构、核动能和动量分布。

**应该算什么。** 用原标定热浴比较动能/静态量的珠数与成本收敛，并检验正常模协方差与质心经典温度。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**缺口与边界。** 需外部矩阵和原电子结构条件；不支持 PIGLET NPT，且未实现论文的瞬态各向异性 Gaussian 动量估计器。

[实现与范围](references.md#piglet).

#### task-ffl

**Fast-forward Langevin 的 rescale 变体 — P0/P2**

论文：[ffl18](references.md#ffl18).

**原文体系。** 原文包括一维谐振子、液态水和 Lennard-Jones 聚合物。

**应该算什么。** 比较正则分布及势能相关时间随阻尼的变化，报告强阻尼下相对普通 Langevin 的采样成本。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**缺口与边界。** 须注明 rescale 变体；采样加速不能作为真实输运系数的验证。

[实现与范围](references.md#ffl).

#### task-constraints

**逐珠 SHAKE/RATTLE 键约束 — P0**

论文：[shake77](references.md#shake77), [rattle83](references.md#rattle83).

**原文体系。** SHAKE 原文为受约束正烷烃分子动力学；RATTLE 为速度形式约束算法。

**应该算什么。** 算键长及速度约束残差、可逆性、能量漂移、dt/容差收敛；物理复现需原分子力场。

**现有入口。** [examples/materials/rigid_water.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/materials/rigid_water.inp); [docs/bond-constraints.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/bond-constraints.md)

**缺口与边界。** 只支持固定胞 VV、NVE/Andersen 的 Python 路线；零势刚性水样例仅验证运动学，不代表真实水或量子刚性转子。

[实现与范围](references.md#constraints).

#### task-rpc-mts

**环聚合物收缩 RPC 与可逆多时间步 — P1/P2**

论文：[rpc08](references.md#rpc08), [respa92](references.md#respa92), [mts16](references.md#mts16), [refrpc16](references.md#refrpc16).

**原文体系。** RPC 原文为柔性液态水的分项收缩；多级时空步原文包含以 DFT 加速 MP2 的 Zundel 阳离子。

**应该算什么。** 核验 Pc=P 的力能一致性，算观测量对 Pc 的偏差、内外 dt 能漂移及昂贵力调用节约。

**现有入口。** [docs/rpc-mts.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/rpc-mts.md); [src/pyrpmd/potentials/contraction.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/potentials/contraction.py); [src/pyrpmd/multitime.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/multitime.py)

**缺口与边界。** 是显式 API；需独立控制共振和收缩误差，谐振分项检查不等于原水/MP2 复现。

[实现与范围](references.md#rpc-mts).

#### task-pacmd

**部分绝热质心分子动力学 PA-CMD — P1**

论文：[cmd93](references.md#cmd93), [cmd94](references.md#cmd94), [pacmd06](references.md#pacmd06).

**原文体系。** 方法比较原文是液态 para-H₂/ortho-D₂；仓库基准另用 Cao–Voth 1993 的 V=q²/2+0.1q⁴、β=5 四次势。

**应该算什么。** 先算四次势质心 Kubo 关联、频峰/展宽和静态矩，与独立量子结果比较；扫描 P、Ω、dt、统计量。原分子目标再算速度关联及扩散。

**现有入口。** [benchmarks/tier2_literature/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy.py); [benchmarks/tier2_literature/spectroscopy_static.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy_static.py); [examples/spectroscopy/pacmd.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/pacmd.inp)

**缺口与边界。** 原四次势论文图是实部量子关联，仓库 Kubo 曲线不是原图；完整液体复现还需原相互作用和状态点。

[实现与范围](references.md#pacmd).

#### task-fcmd

**力匹配快速质心动力学 f-CMD — P1**

论文：[cmd94](references.md#cmd94), [fcmd05](references.md#fcmd05).

**原文体系。** 原文为液态 para-H₂ 和 ortho-D₂，比较结构、动力学与扩散。

**应该算什么。** 算留出集力误差、保守性、质心静态分布、关联/频谱或扩散误差，以及相对直接 PA-CMD 的加速。

**现有入口。** [benchmarks/tier2_literature/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy.py); [src/pyrpmd/cmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/cmd.py); [docs/tier2-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-spectroscopy.md)

**缺口与边界。** 局域二次基不足以代表通用分子势；非线性量需条件属性模型；未实现 QCMD、h-CMD 或 f-QCMD。

[实现与范围](references.md#fcmd).

#### task-trpmd

**TRPMD 与相对红外频谱 — P1**

论文：[trpmd14](references.md#trpmd14), [rpm04](references.md#rpm04), [kubo57](references.md#kubo57), [water09](references.md#water09).

**原文体系。** 原文含谐振/Morse OH、甲烷和 300 K 的 216 分子 q-TIP4P/F 水；可从 Morse OH 开始。

**应该算什么。** 算偶极关联、峰位/峰宽/强度比和摩擦依赖，对比 RPMD 共振与 TRPMD 展宽；收敛 P、dt、时长、窗函数和独立重复。

**现有入口。** [benchmarks/trpmd/harmonic_spectrum.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/trpmd/harmonic_spectrum.py); [benchmarks/water/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/water/README.md); [docs/trpmd-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/trpmd-spectroscopy.md)

**缺口与边界。** 输出缺绝对光谱归一化；TRPMD 与 2009 水模型论文中的 PA-CMD 不是同一方法目标。

[实现与范围](references.md#trpmd).

#### task-trpmd-rates

**TRPMD 速率对照模式 — P2**

论文：[trpmd14](references.md#trpmd14), [trprate15](references.md#trprate15).

**原文体系。** 原文 H+H₂、D+MuH、H+CH₄、F+H₂，比较交叉温度上下的速率。

**应该算什么。** 用一致 PMF 比较 κ(t) 与 k(T) 对摩擦的依赖，覆盖 Tc 两侧并分离统计误差。

**现有入口。** [docs/trpmd-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/trpmd-spectroscopy.md); [docs/science.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/science.md); existing RPMD rate workflow with explicitly selected TRPMD configuration

**缺口与边界。** 摩擦扫描不是精确量子速率的误差界；平衡采样检验不能验证速率。

[实现与范围](references.md#trpmd-rates).

#### task-raman

**各向同性/各向异性相对 Raman 光谱 — P0/P2**

论文：[kubo57](references.md#kubo57), [raman21](references.md#raman21).

**原文体系。** 原理引用 Kubo 线性响应；Dundas 等仅是当前不变量公式约定来源，其研究环境中的丙酮，并非 Raman 理论最早来源。

**应该算什么。** 先用解析谐振极化率模型检查峰位、旋转不变性、退偏比与温度/激光因子；再配原 PES、极化率模型和环境做分子目标。

**现有入口。** [examples/spectroscopy/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/README.md); [examples/spectroscopy/harmonic_properties.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/harmonic_properties.py); [docs/tier2-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-spectroscopy.md)

**缺口与边界。** 没有实现该公式来源论文的极化嵌入电子结构；相对谱不能直接作为绝对实验截面。

[实现与范围](references.md#raman).

#### task-sfg

**基于时间关联的相对 SFG 响应 — P0/P2**

论文：[sfg02](references.md#sfg02), [kubo57](references.md#kubo57).

**原文体系。** Morita–Hynes 原文为水—汽界面的时间相关 SFG 理论。

**应该算什么。** 先用解析信号核验 χijk、Imχ、|χ|² 的变换和符号；原体系需水界面、属性面及偏振方向一致。

**现有入口。** [examples/spectroscopy/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/README.md); [docs/tier2-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-spectroscopy.md); [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/analysis/spectroscopy.py)

**缺口与边界。** 只实现相对响应核；Fresnel/局域场、光路、归一化与电子背景由用户提供。

[实现与范围](references.md#sfg).

#### task-ti-pcn

**Takahashi–Imada 迹作用量与 pCN 采样 — P0**

论文：[ti84](references.md#ti84), [pcn13](references.md#pcn13), [cw81](references.md#cw81).

**原文体系。** TI 原文的谐振子迹收敛；pCN 来自函数空间 MCMC。

**应该算什么。** 算谐振子配分函数差、E/Cv 的珠数收敛，比较原始 P⁻² 和 TI P⁻⁴，并报告接受率和相关时间。

**现有入口。** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**缺口与边界。** TI 高阶性质限迹及相应热力学导数，不自动覆盖结构量或开路径；此 API 不支持 PBC、交换、冻结原子及质量依赖 PES。

[实现与范围](references.md#ti-pcn).

#### task-thermo-estimators

**量子动能、总能与热容估计器 — P0/P2**

论文：[yam05](references.md#yam05), [ti84](references.md#ti84), [cw81](references.md#cw81).

**原文体系。** Yamamoto 原文为 (H₂)₂₂ 与 Ne₁₃ 量子团簇。

**应该算什么。** 算 E(T)、Cv(T)、动势能分解，检查有限差分步长、P 外推、方差与相关时间；团簇需原势能和系综。

**现有入口。** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**缺口与边界。** 必须同时注明作用量与估计器；物理量子动能不能用虚拟环动量温度代替。

[实现与范围](references.md#thermo-estimators).

#### task-isotopes

**同位素自由能与分馏估计 — P1/P2**

论文：[iso13](references.md#iso13), [iso14](references.md#iso14).

**原文体系。** 2013 年为水/汽 H/D 和氧同位素；2014 年包括 Zundel 阳离子与简单烃。

**应该算什么。** 算 ΔF、lnα、质量积分；比较 FEP 与中间质量路线，报告有效样本数、对数权重方差和误差。

**现有入口。** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**缺口与边界。** 重叠差不能靠形式上的小误差补救；当前迹作用量限制阻碍直接周期液水复现。

[实现与范围](references.md#isotopes).

#### task-momentum

**原始开路径单分量动量分布 — P0**

论文：[cw81](references.md#cw81), [ceperley95](references.md#ceperley95), [mom10](references.md#mom10).

**原文体系。** 依据非对角密度矩阵关系；Lin 等的位移路径原文研究一维模型、水和冰，但其自由能算法未在此实现。

**应该算什么。** 对谐振子比较端点分布、n(p)、归一化、对称性和二阶矩，扫描 P、采样量、核宽和 Fourier 范围。

**现有入口。** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**缺口与边界。** 不是 Lin 位移路径法、完整三维动量分布或 PIGLET 的 TAG；不应裁掉负噪声来掩盖采样不足。

[实现与范围](references.md#momentum).

#### task-suzuki-chin

**有限差分 Suzuki–Chin 路径积分 — P0/P2**

论文：[suz95](references.md#suz95), [chin97](references.md#chin97), [sc16](references.md#sc16).

**原文体系。** 原文为色散修正混合 DFT 训练神经网络势的液态水与冰。

**应该算什么。** 先核验谐振子偶珠位置矩、物理势能和有限差分收敛；原体系再算 RDF、热力学量及成本随 P 的变化。

**现有入口。** [examples/path_integral/suzuki_chin.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/path_integral/suzuki_chin.inp); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py)

**缺口与边界。** 现路线非周期、串行，缺完整 SC 总能/Cv/压力估计器；水冰原文复现还需扩展和原 NN 势。

[实现与范围](references.md#suzuki-chin).

#### task-bosons

**二次复杂度玻色交换 PIMD — P1**

论文：[bose19](references.md#bose19), [bose23](references.md#bose23).

**原文体系。** 二维谐振阱理想/相互作用玻色子；2023 年另有大规模与周期氦测试。仓库已对接二维理想阱。

**应该算什么。** 算对称密度、势能及谐振 virial 总能随 N、β、P 的变化，对照精确正则递推并测量时间复杂度。

**现有入口。** [examples/path_integral/bosonic.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/path_integral/bosonic.inp); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py); [benchmarks/tier2_literature/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/README.md)

**缺口与边界。** 逐点势不必等于全排列和；只比较对称系综量。未实现周期绕数/超流估计，仓库温控与采样长度也不同于原文。

[实现与范围](references.md#bosons).

#### task-pigs

**正试探波函数的原始 PIGS — P0/P2**

论文：[pigs00](references.md#pigs00).

**原文体系。** 原文研究 He₂Cl₂ 团簇与液/固 ⁴He。

**应该算什么。** 扫描投影长度、虚时间间距、试探宽度，算中点位置矩与物理势能并对照谐振基态；氦体系需原势和试探函数。

**现有入口。** [examples/path_integral/pigs.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/path_integral/pigs.inp); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py)

**缺口与边界。** 辅助 β 不是物理温度；没有 PBC+玻色交换组合生产流程及完整原氦能量估计器。

[实现与范围](references.md#pigs).

#### task-rpi

**环聚合物瞬子热速率 — P1**

论文：[rpi09](references.md#rpi09), [rpi16](references.md#rpi16).

**原文体系。** 2009 年包含对称/非对称一维 Eckart 势垒；2016 年提供第一原理推导。

**应该算什么。** 算 Tc、路径/作用量、梯度残差、Hessian 符号、前因子及 k(T)，检查珠数/优化/Hessian 差分收敛，与一维精确透射积分比较。

**现有入口。** [examples/instanton/eckart.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/eckart.py); [examples/instanton/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/low_temperature.py); [examples/instanton/cross_checks.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/cross_checks.json); [docs/instanton.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/instanton.md)

**缺口与边界。** 优化收敛不代表物理收敛；分子目标还需独立核查势面、配分函数、对称/转动因子。

[实现与范围](references.md#rpi).

#### task-splitting

**对称束缚双阱的瞬子隧穿劈裂 — P1/P2**

论文：[split11](references.md#split11).

**原文体系。** 原文为全维 HO₂ 和丙二醛。

**应该算什么。** 先用一维对称双阱，对照量子对角化的基态双态差 Δ，核验作用量/前因子、投影时间和珠数收敛。

**现有入口。** [examples/instanton/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/low_temperature.py); [docs/instanton.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/instanton.md); [docs/low-temperature.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/low-temperature.md)

**缺口与边界。** 当前拒绝自由平移/转动模，不是气相转振劈裂或多阱多重态；原分子复现需扩展。

[实现与范围](references.md#splitting).

#### task-low-t-checks

**谐振 TST、Wigner 与 Eckart 对照 — P0**

论文：[eyr35](references.md#eyr35), [wig32](references.md#wig32), [eck30](references.md#eck30).

**原文体系。** 活化络合物、弱隧穿展开和解析势垒的理论原文；Eckart 本身不是特定分子反应。

**应该算什么。** 算 kTST、κWigner、κEckart 与修正 k(T)，核验高温极限、正反势垒、能量零点及 Tc 附近敏感性。

**现有入口。** [examples/instanton/cross_checks.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/cross_checks.json); [examples/instanton/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/low_temperature.py); [docs/low-temperature.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/low-temperature.md)

**缺口与边界。** Wigner 不适合深隧穿；一维拟合 Eckart 不证明多维准确性，也不是统一交叉区修正。

[实现与范围](references.md#low-t-checks).

### 自由能、增强采样与反应路径复现任务

#### task-samp-umbrella

**伞形采样与 1D–3D 自由能面 — P1**

论文：[samp-us1974](references.md#samp-us1974), [samp-us1977](references.md#samp-us1977).

**原文体系。** 原始 1974/1977 论文：Lennard–Jones 流体，包含气液共存区；1974 以 inverse-twelve / soft-sphere 流体为自由能参考。

**应该算什么。** 论文路线算不同温度、密度的自由能差；程序优先算解析谐振子 1D/2D/3D FES 与支持区覆盖率，再算 OGRe 的经典双阱。

**现有入口。** [examples/fes/harmonic2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/fes/harmonic2d.inp); [benchmarks/fes_methods/benchmark.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/benchmark.py)

**缺口与边界。** LJ 原论文的完整参数、路径与参考自由能尚需从原文恢复；现成谐振子例子是回归，不是 LJ 论文复现。

[实现与范围](references.md#samp-umbrella).

#### task-samp-mbar

**MBAR 自由能、直方图 FES 与重叠诊断 — P0**

论文：[samp-mbar2008](references.md#samp-mbar2008).

**原文体系。** 原论文：不同恒定拉力下的 DNA 发夹 optical-tweezer 测量，重构伸长坐标 PMF。

**应该算什么。** 算各状态自由能偏移、目标权重、PMF、重叠矩阵；先在同一批 umbrella 样本与 PyMBAR 做数值一致性，再检查样本和分箱收敛。

**现有入口。** [benchmarks/fes_methods/solver_check.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/solver_check.py); [benchmarks/h3/solver_check.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/solver_check.py)

**缺口与边界。** DNA 原始测量数据并未随包提供；应单独恢复数据、拉力和单位。

[实现与范围](references.md#samp-mbar).

#### task-samp-ui

**高斯窗口伞形积分 — P0**

论文：[samp-ui2005](references.md#samp-ui2005), [samp-uierr2006](references.md#samp-uierr2006).

**原文体系。** 2005 原论文含解析势函数和一个生物体系；本次可读原始摘要未给出生物体系名称，未将猜测写成事实。

**应该算什么。** 算平均力、PMF、势垒；在完全相同样本上比较 UI/MBAR 势垒差、Gaussian 窗口拟合和尾部外推支持。

**现有入口。** [benchmarks/h3/convergence.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/convergence.py); [benchmarks/h3/solver_check.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/solver_check.py); [benchmarks/fes_methods/benchmark.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/benchmark.py)

**缺口与边界。** 需取得 2005 全文恢复原解析势参数和生物模型。H+H2/BKMP2 是现有应用验证，并非 UI 方法论文原体系。

[实现与范围](references.md#samp-ui).

#### task-samp-adaptive

**自适应伞窗布置与覆盖规划 — P0**

论文：[samp-mbar2008](references.md#samp-mbar2008), [samp-us1977](references.md#samp-us1977), [samp-ogre2023](references.md#samp-ogre2023).

**原文体系。** 没有单一原创论文对应整个 PyRPMD planner。对照论文 OGRe：1D/2D 解析模型、沸石质子跳跃与 COF-5 层间堆垛。

**应该算什么。** 先算 OGRe Eq.13 双阱和 Eq.14 Ackley 的 P=1、300 K FES：PMF/RMSE、覆盖率、重叠、窗口数、总 PES 调用、独立种子误差，所有 pilot 成本计入总成本。

**现有入口。** `python benchmarks/fes_methods/ogre_benchmark.py build --output runs/ogre`; `python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method adaptive --seed 4101 --output runs/ogre`

**缺口与边界。** Ackley 需限定归一化域，仓库外墙属于模型定义；P>1 非谐 centroid FES 不等于裸势。不能据此声称复现 OGRe 完整算法。

[实现与范围](references.md#samp-adaptive).

#### task-samp-standard-meta

**标准元动力学 — P1**

论文：[samp-meta2002](references.md#samp-meta2002).

**原文体系。** 原论文：水中 NaCl 解离，以及水溶液 alanine dipeptide 构象变化。

**应该算什么。** NaCl 算离子距离/溶剂配位相关 FES；alanine 算 F(phi,psi) 与盆地自由能差、势垒及收敛历史。先用 OGRe 双阱验证偏置力和有限时间 FES。

**现有入口。** [examples/metadynamics/standard.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/metadynamics/standard.inp); [benchmarks/fes_methods/ogre_benchmark.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/ogre_benchmark.py)

**缺口与边界。** 需另外准备同原文 force field、水模型与 CV；普通有偏时间不能直接得到物理速率。

[实现与范围](references.md#samp-standard-meta).

#### task-samp-wt-meta

**良调元动力学 — P1**

论文：[samp-wt2008](references.md#samp-wt2008).

**原文体系。** 原论文：alanine dipeptide 自由能面。

**应该算什么。** 算 F(phi,psi)、盆地差和势垒误差随时间、bias factor 与沉积参数变化；独立种子检查，不把 hill 数量当收敛。

**现有入口。** [examples/metadynamics/well_tempered_2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/metadynamics/well_tempered_2d.inp); [benchmarks/fes_methods/metadynamics_compare.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/metadynamics_compare.py)

**缺口与边界。** 论文 molecular force field、温度、溶剂细节需按全文/SI 一致准备；仓库 harmonic2D 是方法回归。

[实现与范围](references.md#samp-wt-meta).

#### task-samp-walkers

**共享偏置的多行者元动力学 — P1**

论文：[samp-walkers2006](references.md#samp-walkers2006), [samp-wt2008](references.md#samp-wt2008).

**原文体系。** 原论文：乙腈中四价 cyclobis(paraquat-p-phenylene) 环芳烷与 1,5-dihydroxynaphthalene 的穿脱套过程；另有 1D/2D/3D 球形势阱 Langevin 模型。

**应该算什么。** 算两中心距离和溶剂配位数的二维 FES、脱套/回穿势垒、FES 误差及 wall time/总动力学时间随 walker 数变化；先用 OGRe 双阱做同预算回归，误差来自独立完整共享历史。

**现有入口。** [docs/enhanced-sampling.md#shared-multiple-walkers](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/enhanced-sampling.md#shared-multiple-walkers); [docs/enhanced-sampling.md#shared-bias-multiple-walkers](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/enhanced-sampling.md#shared-bias-multiple-walkers)

**缺口与边界。** 原文 Section 3 和 Fig.3 为 300 K、1–32 walkers，分子势详情指向其 ref.35，需补齐；原文 WHAM superposition 后处理未随共享偏置功能一并实现。多个 shared walkers 不等于多个独立重复。

[实现与范围](references.md#samp-walkers).

#### task-samp-grid-meta

**网格元动力学与自由能投影 — P1**

论文：[samp-meta2002](references.md#samp-meta2002), [samp-wt2008](references.md#samp-wt2008).

**原文体系。** 延用标准/WT-MetaD 论文体系；没有独立的 PyRPMD 网格工程原始论文。

**应该算什么。** 在同一轨迹参数下比较 direct/grid 偏置能量与力、FES，扫描网格间距；检查边际自由能的非均匀 bin 体积因子和 missing-support mask。

**现有入口。** [docs/grid-metadynamics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/grid-metadynamics.md); `pyrpmd project RUN --axes 2 --output NEW_DIRECTORY`

**缺口与边界。** 网格速度提升和投影数值一致性不建立独立物理准确性；不可把投影盆地权重用作动力学分支。

[实现与范围](references.md#samp-grid-meta).

#### task-samp-opes

**OPES_METAD 与冻结偏置重加权 — P0**

论文：[samp-opes2020](references.md#samp-opes2020).

**原文体系。** 原论文及 SI：double-well 模型、alanine dipeptide、alanine tetrapeptide。

**应该算什么。** 从 double well 做 KDE、bias、force 与归一化 Z 对照；算冻结偏置后的 FES、ESS、覆盖率和盆地自由能，再移至 alanine。

**现有入口。** [examples/opes/harmonic_2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/opes/harmonic_2d.inp); [benchmarks/tier2_literature/sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/sampling.py)

**缺口与边界。** 无 kernel compression/shared walkers；wrapped Gaussian 与 PLUMED 截断 kernel 不完全相同。跨码比较须统一初始化常数、实际 kernel 宽度、归一化和偏置 gauge。

[实现与范围](references.md#samp-opes).

#### task-samp-opes-explore

**OPES-explore 与可选自适应宽度 — P1**

论文：[samp-explore2022](references.md#samp-explore2022), [samp-opes2020](references.md#samp-opes2020).

**原文体系。** 原论文：Müller potential、alanine dipeptide 与 alanine tetrapeptide；用不充分 CV 研究探索速度与 FES 收敛之间的关系。

**应该算什么。** 算跨盆地次数、FES 误差、覆盖率随时间；分别比较 metad/explore 和 fixed/adaptive widths。探索更快不能当成 FES 已收敛。

**现有入口。** `pyrpmd run examples/opes/adaptive_explore.inp --output runs/adaptive-explore`; `python benchmarks/tier2_literature/sampling.py opes --output runs/tier2-sampling`

**缺口与边界。** tier2_literature/sampling.py 已包含论文 Müller 模型及 x/y CV 对照，但默认预算不等于原文 200M steps；自适应规则只能做本实现的对照，不能声称完整复现 PLUMED OPES。

[实现与范围](references.md#samp-opes-explore).

#### task-samp-reus

**同温伞形采样副本交换 — P1**

论文：[samp-reus2000](references.md#samp-reus2000).

**原文体系。** 原论文：N 端 acetyl、C 端 N-methyl 封端的 alanine trimer，AMBER 全原子势与 distance-dependent dielectric；反应坐标为 O1-H5 距离。

**应该算什么。** 算距离 PMF、helical/turn/extended 概率、交换接受率、walker round trips 与窗口联合自相关；与相同总预算普通 umbrella 比较。

**现有入口。** `pyrpmd run examples/fes/replica_exchange_2d.inp --output runs/reus2d`

**缺口与边界。** 现成例子为 harmonic 模型，原肽势/构型要另配；交换后的时间只用于采样，不能作真实 RPMD 相关时间。

[实现与范围](references.md#samp-reus).

#### task-samp-remd

**温度与哈密顿量副本交换 API — P1**

论文：[samp-remd1999](references.md#samp-remd1999), [samp-reus2000](references.md#samp-reus2000).

**原文体系。** 1999 REMD 原论文：Met-enkephalin 五肽；2000 多维扩展：blocked alanine trimer。

**应该算什么。** 算温度/哈密顿量槽位的能量分布、构象概率、交换 detailed balance、round trips；P>1 额外验证 spring-action 交换因子与精确 harmonic 分布。

**现有入口。** pyrpmd.replica_exchange.ReplicaExchange (Python API); [docs/tier2-sampling.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-sampling.md)

**缺口与边界。** P>1 的 ring 交换属于把经典交换原理用于本包测度；需另配论文分子势、温度梯和初态。

[实现与范围](references.md#samp-remd).

#### task-samp-biased-clock

**有偏时间加速与删失速率推断 — P1**

论文：[samp-imeta2013](references.md#samp-imeta2013), [samp-flooding2022](references.md#samp-flooding2022).

**原文体系。** OPES-flooding 原论文：二维双阱、气相 alanine dipeptide、显式水 chignolin 折叠/解折叠；infrequent-MetaD 论文可使用其 alanine 路线。

**应该算什么。** 先在双阱比较无偏 first-passage 与重标定时间分布、MLE 速率、删失比例及 KS 诊断；扫描沉积间隔、bias barrier 和 transition exclusion。

**现有入口。** pyrpmd.biased_kinetics.rescale_biased_time / estimate_biased_rate

**缺口与边界。** 原生 flooding exclusion 未实现；调用者必须提供每个积分区间的 bias、transition_bias 和无偏 TS 审核。稀疏 FES 保存帧不足以生成该时钟。

[实现与范围](references.md#samp-biased-clock).

#### task-samp-tis

**双向正则 shooting 的过渡界面采样 — P1**

论文：[samp-tis2003](references.md#samp-tis2003), [samp-tis2005](references.md#samp-tis2005), [samp-tps1998](references.md#samp-tps1998), [samp-shoot2025](references.md#samp-shoot2025).

**原文体系。** TIS 2003 原论文：简单流体中双原子分子的异构化。

**应该算什么。** 算各界面条件跨越概率、正向有效通量、乘积速率、path length/acceptance、不同路径链的机制混合；增大 path cutoff，验证与直接 MD/已知模型一致。

**现有入口。** `python examples/sampling/transition_interface.py --samples 200 --burn-in 50`

**缺口与边界。** 现成例子是可分离双阱，不是原流体模型；没有自动 flux run、随机 one-way shooting 或完整 molecular rate certification。

[实现与范围](references.md#samp-tis).

#### task-samp-retis

**正界面系综的交换式 TIS — P1**

论文：[samp-retis2007](references.md#samp-retis2007), [samp-tis2003](references.md#samp-tis2003).

**原文体系。** RETIS 2007 原论文：Peyrard–Bishop–Dauxois 模型的 DNA 变性。

**应该算什么。** 先在相同双阱比较 TIS/RETIS 的条件概率、自相关、walker labels 和总成本；论文路线需要实现 PBD 势，算 DNA 变性速率与采样效率。

**现有入口。** pyrpmd.retis.ReplicaExchangeTIS (Python API)

**缺口与边界。** 不能宣称完整 RETIS 原算法等价：minus-interface flux ensemble、持久 checkpoint 与 MPI 均缺失。

[实现与范围](references.md#samp-retis).

#### task-samp-committor

**固定完整环构型的 RPMD committor — P0**

论文：[samp-comm2000](references.md#samp-comm2000), [samp-cp1934](references.md#samp-cp1934).

**原文体系。** Bolhuis 等原始分子应用：真空与水溶液 alanine dipeptide，检验 phi/psi 是否足以描述反应坐标。该论文不提出本包的量子环算法。

**应该算什么。** 先算对称双阱障顶固定构型 p_B≈1/2 和盆地边界 0/1；扫描 shots、dt、horizon、beads；再对 alanine 的候选分界面算 committor 分布 P(p_B)。

**现有入口。** `python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example`

**缺口与边界。** P>1 必须区分固定完整 ring 与质心条件分布；折叠到同一质心的 ring 不是平衡条件 ensemble。未决 shots 不可丢弃；p_B_given_resolved 不是无条件 committor。

[实现与范围](references.md#samp-committor).

#### task-samp-pathcv

**对齐 RMSD 与参考路径 CV — P1**

论文：[samp-kabsch1976](references.md#samp-kabsch1976), [samp-path2007](references.md#samp-path2007).

**原文体系。** Path-CV 2007 原论文：alanine dipeptide 构象变化；Kabsch 1976 是几何最优旋转算法，没有对应动力学化学体系。

**应该算什么。** 算 s/z 及全 ring Jacobian、旋转平移不变性、F(s,z) 和沿路径 PMF；检查路径节点数和 lambda 稳健性。

**现有入口。** pyrpmd.advanced_cv.RMSD / PathCV; [docs/tier2-sampling.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-sampling.md)

**缺口与边界。** 需另配原文参考路径与分子势；只实现给定 path 的 CV 不等于完成原论文全局路径优化协议。

[实现与范围](references.md#samp-pathcv).

#### task-samp-string

**欧氏平均力与约束自适应 string — P1**

论文：[samp-string2002](references.md#samp-string2002), [samp-stringcv2006](references.md#samp-stringcv2006).

**原文体系。** 2002 string 原论文：Lennard–Jones cluster 重排和磁性薄膜翻转；2006 CV string：alanine dipeptide，比较 2 与 4 个二面角。

**应该算什么。** 先在解析弯曲谷检验法向力残差、节点数与重参数化；论文 2006 路线算 MFEP、自由能势垒及 committor 宽度随 CV 选择变化。

**现有入口。** pyrpmd.adaptive_string.adaptive_mean_force_string; [docs/manual/adaptive-nrpmd.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/adaptive-nrpmd.md)

**缺口与边界。** 用 PES 梯度只得到 MEP；本包不应直接声称实现任意 nonlinear CV 的完整 MFEP/等 committor 面。原论文 4-angle route 需补齐 metric 项。

[实现与范围](references.md#samp-string).

#### task-samp-prfo

**分区有理函数鞍点优化 — P0**

论文：[samp-prfo1985](references.md#samp-prfo1985).

**原文体系。** 1985 原论文的两个解析模型势面，包括 Cerjan–Miller 模型；论文 Fig.3 示 minimum 与 ±(1,0) saddles。

**应该算什么。** 算 saddle 坐标/能量、Hessian 负模数、梯度残差和初猜敏感性；扫描 Hessian 差分步长，再在 H+H2/BKMP2 算电子势垒和虚频。

**现有入口。** `python benchmarks/transition_state_search/analytic.py --output runs/ts/analytic.json`; [benchmarks/transition_state_search/h3.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/transition_state_search/h3.py)

**缺口与边界。** 需保持论文解析模型定义与本包能量缩放一致；优化迭代数受本实现控制策略影响，不可假设与原文完全相同。

[实现与范围](references.md#samp-prfo).

#### task-samp-cineb

**CI-NEB、改进切向与 FIRE 松弛 — P0**

论文：[samp-nebtangent2000](references.md#samp-nebtangent2000), [samp-cineb2000](references.md#samp-cineb2000), [samp-fire2006](references.md#samp-fire2006).

**原文体系。** CI-NEB 原论文：CH4 在 Ir(111) 的解离吸附及 H2/Si(100)；改进 tangent 论文还包括 Si 交换扩散和 Al(100) addimer。

**应该算什么。** 方法回归先算 Müller–Brown MEP、barrier、最高 image 真力、NEB force 与 image 数/初始路径收敛，再做 BKMP2；原 slab 体系需外部独立 DFT/ASE 路线。

**现有入口。** `python -m pyrpmd prepare examples/rate/from_endpoints.inp --pilot --output preparation.json`; [benchmarks/transition_state_search/analytic.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/transition_state_search/analytic.py)

**缺口与边界。** 论文 slab force field/DFT 与周期固定层限制不在当前 gas-phase prepare 支持范围；应明确原体系复现需要额外实现，不用 H3 冒充。

[实现与范围](references.md#samp-cineb).

#### task-samp-connectivity

**共价半径连通图与反应通道推断 — P0**

论文：[samp-radii2008](references.md#samp-radii2008).

**原文体系。** 原论文参数来自晶体结构统计，覆盖大部分原子序数至 96 的元素；不是需要跑 MD 的单一体系。

**应该算什么。** 核对所有 radii/units/isotopes 与原表；在已知 reactant/product 几何上验证图连通性和阈值敏感性。

**现有入口。** `pyrpmd prepare`; [src/pyrpmd/covalent.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/covalent.py)

**缺口与边界。** 分类阈值与两侧下降/等价通道组合属于包实现，不能由 Cordero 参数论文证明任意反应识别正确。

[实现与范围](references.md#samp-connectivity).

#### task-samp-metric

**一般非线性质心坐标的约束质量度量 — P0**

论文：[samp-bluemoon1989](references.md#samp-bluemoon1989), [samp-blue1998](references.md#samp-blue1998).

**原文体系。** 原论文为一般受约束反应坐标的 ensemble 关系；本次未恢复完整 1998 化学应用协议，不指定未经核实的分子。

**应该算什么。** 优先对同一解析势使用 s、a*s、非线性单调 f(s) 比较速率不变性，检查 mass metric 与 constrained-parent weights；再测一般键差坐标。

**现有入口。** [examples/rate/general_coordinate.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/rate/general_coordinate.inp); [tests/test_general_coordinate_rates.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/tests/test_general_coordinate_rates.py); [docs/general-coordinate-rates.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/general-coordinate-rates.md)

**缺口与边界。** 非线性 CV 的质量度量不能省略；相同投影 PMF 不保证相同无偏动力学速率。

[实现与范围](references.md#samp-metric).

#### task-samp-rate-postprocess

**二维通量与带符号多通道后处理 — P1**

论文：[samp-blue1998](references.md#samp-blue1998), [bennett1977](references.md#bennett1977), [chandler1978](references.md#chandler1978), [rate05](references.md#rate05), [bimol09](references.md#bimol09).

**原文体系。** 项目现有物理回归：H+H2/BKMP2 的两个带原子标签交换出口；不是独立新方法原始论文体系。

**应该算什么。** 算同样本一维/二维静态通量一致性、kappa_i(t)、k_i 与 branching_i 的共同父链 bootstrap；检查完整分界面支持和入口归一化。

**现有入口。** `python benchmarks/h3/rate_postprocess.py --profile quick --workers 2 --output runs/h3-rate-postprocess`

**缺口与边界。** 不能把产物盆地 Boltzmann 权重当动力学分支；不能无条件相加重叠 TS 通量；出口标签不提供额外倍数二。

[实现与范围](references.md#samp-rate-postprocess).

#### task-samp-statistics

**相关诊断与同步循环块 bootstrap — P0**

论文：[samp-cblock1992](references.md#samp-cblock1992), [samp-geyer1992](references.md#samp-geyer1992).

**原文体系。** 统计原论文不规定本包化学体系；可用已知协方差的 AR(1) 和解析 harmonic 时间序列作独立校准。

**应该算什么。** 比较标准误覆盖率、block 长度敏感性、独立种子方差；REUS 保持跨窗口同一个 resampling clock，channels 保持共同父链。

**现有入口。** [src/pyrpmd/analysis/fes_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/analysis/fes_statistics.py); [docs/replica-exchange.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/replica-exchange.md)

**缺口与边界。** 报告 g/ESS 不是已证实混合；不能把 5g、20 blocks 或固定 overlap threshold 写成论文保证。

[实现与范围](references.md#samp-statistics).

#### task-samp-plumed

**PLUMED 质心偏置接口 — P1**

论文：[samp-plumed2009](references.md#samp-plumed2009), [samp-plumed2014](references.md#samp-plumed2014).

**原文体系。** 原始软件论文包含多个示例；桥接验证应选择已确定算法/模型，不把软件名作为独立物理方法。

**应该算什么。** 同一模型比较本包与外部 PLUMED 的 CV、bias、forces 和单步轨迹；再按所选算法的原论文体系复现。

**现有入口。** `python examples/plumed/centroid_metadynamics.py --steps 200 --output runs/plumed-centroid`; `python benchmarks/tier2_literature/sampling.py plumed --output runs/tier2-sampling`

**缺口与边界。** 外部历史管理和自动 resume 有明确限制；尚未配置的 PLUMED action 不算 PyRPMD 已实现功能。

[实现与范围](references.md#samp-plumed).

### 非绝热方法与矩阵模型复现任务

#### task-na-nrpmd-focused

**Focused 非平衡 MMST-NRPMD — P1**

论文：[na-mm1979](references.md#na-mm1979), [na-stock1997](references.md#na-stock1997), [na-nrpmd2013](references.md#na-nrpmd2013), [na-focused2019](references.md#na-focused2019), [na-mint2018](references.md#na-mint2018).

**原文体系。** Chowdhury–Huo 2019：一维三态 Morse 光解离模型，以及单模两态 spin-boson 模型。2013 论文的热平衡耦合谐振子另列在 signed thermal NRPMD。

**应该算什么。** 各电子态布居 P_i(t)，匹配原文的垂直激发初始核分布与 focused MMST 估计量; 三态 Morse 解离分支与长时间布居；一模 spin-boson 相干振荡; 分别收敛时间步、beads、独立核轨迹数；保留带符号估计量与统计误差

**现有入口。** `pyrpmd run examples/nonadiabatic/population.inp --output runs/nrpmd`; `PYTHONPATH=src python benchmarks/nonadiabatic/population_dynamics.py --smoke --output population_smoke.json`

**缺口与边界。** 这两个现成入口是教学/合成 LVC 回归，不是 2019 原文参数集。需要录入原文 Table I 的三态 Morse 模型或对应一模模型及初始分布。注意论文中 Δ/2 与包 DiscreteSpinBoson.tunnelling=Δ 的矩阵元约定，不能直接复制同名参数。

[实现与范围](references.md#na-nrpmd-focused).

#### task-na-mint

**MInt 电子映射积分子步 — P0**

论文：[na-mint2018](references.md#na-mint2018).

**原文体系。** Church 等人 2018 提出混合量子/经典 IVR；此处适合复现积分子步的数学性质，不能把完整 IVR 数值结果列为已支持功能。

**应该算什么。** 固定核坐标下常矩阵解析电子旋转与动量冲量; 非对易矩阵梯度条件下时间反演和步长收敛；总能漂移是数值诊断，不是近似动力学精确性的证据

**现有入口。** NRPMDDynamics / SMNRPMDDynamics Python APIs; the normal NRPMD examples exercise this integrator.

**缺口与边界。** 无需把 MQC-IVR 的全部结果列为可复现；包没有对应 IVR 相位、振幅和前因子流程。

[实现与范围](references.md#na-mint).

#### task-na-thermal-nrpmd

**带符号 SEO 热平衡 NRPMD 与相关/速率估计量 — P1**

论文：[na-nrpmd2013](references.md#na-nrpmd2013), [na-stock1997](references.md#na-stock1997), [na-mint2018](references.md#na-mint2018).

**原文体系。** Richardson–Thoss 2013 图 1(c)：H=p²/2+x²/2+[[x,1],[1,-x]]，m=ℏ=ω=κ=β=Δ=1、α=0；仓库专用脚本明确转录了该模型。

**应该算什么。** Kubo 位置自相关 C_xx(t); 初始 SEO 电子投影与末态 MMST 布居构成的电子相关函数; 平均 sign、有效独立样本/链间误差、bead 与时间步收敛，和独立 sinc-DVR 结果比较; 若目标是速率，另外验证反应物归一化、相关函数平台和平台拟合区间

**现有入口。** `python examples/nonadiabatic/published_nrpmd_benchmark.py --plan`; `python examples/nonadiabatic/published_nrpmd_benchmark.py --workers 3 --potential-split lowest_adiabatic --output runs/published-nrpmd --published-data examples/nonadiabatic/data/richardson_thoss_2013_fig1c.csv`; `pyrpmd nonadiabatic-rate examples/nonadiabatic/thermal_rate.inp --output runs/thermal-nonadiabatic`

**缺口与边界。** 现有 Fig. 1(c) 入口是最直接的原文复现起点，仍须执行独立收敛研究。用默认 trace split 得到的曲线应标为不同分裂下的比较。thermal_rate.inp 的短时结果不能声称得到物理收敛速率；复杂矩阵的相位重加权未实现。

[实现与范围](references.md#na-thermal-nrpmd).

#### task-na-adaptive-thermal

**自适应约束路径与带符号 NRPMD 组合工作流 — P3**

论文：[na-nrpmd2013](references.md#na-nrpmd2013), [samp-stringcv2006](references.md#samp-stringcv2006), [samp-mbar2008](references.md#samp-mbar2008).

**原文体系。** 2013 耦合谐振子是 NRPMD 组成部分的原文基准；包内 confined 两态及 electrode 例子是组合验证体系，不能称为同一篇自适应 NRPMD 论文的体系。

**应该算什么。** 先复现 NRPMD2013 热平衡相关函数，再验证已知反应坐标下的约束自由能; 冻结路径后用独立生产样本计算自由能差/通量相关函数；报告 sign、窗口重叠与统计误差; 比较固定路径和自适应路径是否在不确定度内给出一致结果

**现有入口。** [docs/manual/adaptive-nrpmd.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/adaptive-nrpmd.md); [docs/adaptive-electrochemical-string-pilot.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/adaptive-electrochemical-string-pilot.md)

**缺口与边界。** 需要对每个组成部分分别引用；不能给组合工作流编造一篇原始论文，也不能用一条短适应轨迹宣称新的电化学速率已验证。

[实现与范围](references.md#na-adaptive-thermal).

#### task-na-sm-nrpmd

**Focused 两态自旋映射 NRPMD — P2**

论文：[na-sm2021](references.md#na-sm2021).

**原文体系。** Bossion 等人 2021：一个低温两态模型和七组单模谐振 spin-boson 模型。后者 H=p²/2+R²/2+[[R+ε,Δ],[Δ,-R-ε]]；m=ℏ=ω=β=1，各组 (Δ,ε) 为 (10,0)、(4,0)、(1,2)、(1,0.5)、(1,0)、(0.1,1.5)、(0.1,0)。

**应该算什么。** focused 初始条件下的电子布居与自旋长度守恒; 原文热平衡 C_RR(t) 与 C_nn(t) 必须同时使用原文 Gamma_W 初始分布、权重与估计量

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/sm_nrpmd.inp --output runs/sm-nrpmd`

**缺口与边界。** 现成例子只能验证 focused 动力学。原文主要平衡相关函数图不能由当前初始化直接复现；先补 Gamma_W 采样及对应估计量再列为原文定量复现。

[实现与范围](references.md#na-sm-nrpmd).

#### task-na-mash

**经典核 MASH 与实验性质心扩展 — P1**

论文：[na-mash2023](references.md#na-mash2023), [na-mashrates2024](references.md#na-mashrates2024).

**原文体系。** Mannouch–Richardson 2023：Tully 散射、spin-boson 和 pyrazine 模型。2024 速率后续论文使用 Brownian-oscillator 反应坐标加 Ohmic 浴的电子转移模型。

**应该算什么。** P=1 下 Tully I–III 的分态反射/透射几率随入射动量的变化; 电子布居与相干；严格匹配 MASH 的 |S_z| 布居采样和相应权重; 扩展到反应坐标 spin-boson 的电子转移速率与 Marcus/精确参考比较

**现有入口。** `python examples/nonadiabatic/adiabatic_on_the_fly.py --method mash --beads 1`

**缺口与边界。** 该入口是两态接口演示。需补原文散射/ET 参数和轨迹统计驱动；原文的多模 pyrazine 参数未随通用 LVC 类自动提供。不能把实验性 P>1 MASHDynamics 标为 CA-RP-MASH，亦不能声称实现了后续可逆事件积分器。

[实现与范围](references.md#na-mash).

#### task-na-carp-mash

**CA-RP-MASH（命令行 rp-mash） — P2**

论文：[na-carp2025](references.md#na-carp2025), [na-mash2023](references.md#na-mash2023), [na-mashrates2024](references.md#na-mashrates2024).

**原文体系。** Cao–Suarez–Kretchmer 2025 v1 预印本：Tully 模型、三模 pyrazine、spin-boson 电子转移。引用实际实现所依据的 v1；截至本次检索未核实期刊后继版本。

**应该算什么。** Tully 分态透射/反射以及布居时间序列，先比较 P=1 极限; 按原文预备与浴离散化计算 spin-boson 电子转移速率及核量子效应; 使用正确算符对权重：W_CC=3、W_CP=2、W_PP=2|S_z(0)|，并验证 event tolerance/max_step 收敛

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/rp_mash.inp --output runs/rp-mash`

**缺口与边界。** 现成输入是教学模型。需要原文参数化 pyrazine/bath、独立平衡或散射预备与速率分析；严禁把任意 harmonic-ring 初态布居曲线视作原文热速率复现。native=False 路径为当前实现。

[实现与范围](references.md#na-carp-mash).

#### task-na-rpsh-ca

**FSSH 与质心近似 RPSH — P1**

论文：[na-tully1990](references.md#na-tully1990), [na-rpsh2012](references.md#na-rpsh2012), [na-rpsh2017](references.md#na-rpsh2017), [na-rpshdm2023](references.md#na-rpshdm2023).

**原文体系。** Tully 1990 的三个一维两态散射模型；Shushkov–Li–Tully 2012 的两态热散射速率；Shakib–Huo 2017 的低动量 Tully 反射/透射研究。

**应该算什么。** 四个最终通道的反射/透射几率以及它们随 p0 的变化; active-surface 分数、振幅布居和电子相干分别保存，不能互相替代; 若复现 2012 热速率，还需入射通量和热平均，而不是直接将末态分数称为速率

**现有入口。** `pyrpmd run examples/nonadiabatic/rpsh_population.inp --output runs/rpsh`; `python examples/nonadiabatic/adiabatic_on_the_fly.py --method rpsh --beads 1`

**缺口与边界。** 现成输入是短时布居示例。完整原文散射需要动量扫描、入射波包或对应环聚合初始分布、终态通道判别与足够传播时间。Tully III 耦合在一侧不消失，不能以耦合趋零作为通用停止条件。

[实现与范围](references.md#na-rpsh-ca).

#### task-na-rpsh-ba

**Bead 平均表面跳跃 RPSH-BA — P1**

论文：[na-rpshdm2023](references.md#na-rpshdm2023), [na-rpsh2012](references.md#na-rpsh2012), [na-tully1990](references.md#na-tully1990).

**原文体系。** Shakya 等人 2023：Tully 扩展耦合模型 C/III，对比 CA、BA、DM；报告的条件包括 p0=20/30 au、质量 2000、初始质心 −15、宽度参数 2 和 64 beads。宽度的公式定义须与原文一致。

**应该算什么。** 与 CA/DM 使用同一物理初态，比较电子密度、相干和分态反射几率; 同时报告 active-surface 统计和电子振幅估计量

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/rpsh_ba.inp --output runs/rpsh-ba`

**缺口与边界。** 示例并非完整原文轨迹集。需从原文复制 width/温度/采样定义，不能把数字 2 直接混同任何 Gaussian sigma 约定；应按原文公式构造核初态。

[实现与范围](references.md#na-rpsh-ba).

#### task-na-rpsh-dm

**密度矩阵表面跳跃 RPSH-DM — P1**

论文：[na-rpshdm2023](references.md#na-rpshdm2023).

**原文体系。** 同一 Shakya 等人 2023 Tully III/模型 C 的 CA、BA、DM 直接比较，重点包括电子密度和去相干。

**应该算什么。** rho_ij(t)、Tr rho²、相干衰减，配合通道分辨反射/透射; 单独验证基底相位/幺正变换不改变可观测量，区别真实 bead 混合与数值相位错误

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/rpsh_dm.inp --output runs/rpsh-dm`

**缺口与边界。** 仅比较总布居不足以验证原文主张；要加入原文的密度矩阵与去相干量。原指南的参考文献作者 P. Shakya 应修正为 Y. Shakya。

[实现与范围](references.md#na-rpsh-dm).

#### task-na-edc

**能量型去相干修正 EDC — P2**

论文：[na-edc2007](references.md#na-edc2007).

**原文体系。** Granucci–Persico 2007：偶氮苯光异构化和两个专门设计的模型问题。Tully III 只能作为包内替代回归体系。

**应该算什么。** 电子振幅布居与 active-state 统计的内部一致性; 原文模型的去相干时间尺度与反应分支；可先以 Tully III 作包内回归替代体系

**现有入口。** RPSHDynamics(..., decoherence='edc', ...) and advanced CA/BA options documented in docs/nonadiabatic-hopping-extensions.md

**缺口与边界。** Tully III 是实用的替代回归体系，不能称为该 2007 论文原始体系。真实 azobenzene 复现还需一致的电子结构级别、初始态与势能数据。

[实现与范围](references.md#na-edc).

#### task-na-soc-rpsh

**复 Hermitian / SOC 表面跳跃与局域几何力 — P2**

论文：[na-soc2019](references.md#na-soc2019), [na-tully1990](references.md#na-tully1990), [na-rpsh2012](references.md#na-rpsh2012).

**原文体系。** Miao–Bellonzi–Subotnik 2019：二维两态复矩阵，θ(x)=π[erf(Bx)+1]/2、φ(y)=Wy，绝热面为常数 ±A；首个展示条件 A=0.1、B=3、W=0.3。

**应该算什么。** 先以 P=1 复现原文的横向动量偏转与态分辨散射概率; 比较有/无 Berry force，并用二维波包参考验证；检查常相位变换不变性; P>1 的环聚合扩展作为后续新比较，不能等同于 2019 原文

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/soc_rpsh.inp --output runs/soc-rpsh`

**缺口与边界。** 当前示例是合成 complex-LVC 模型，不是实际分子的 intersystem crossing。原文二维势与二维量子基准需要另行实现；包内 SplitOperator1D 无法提供这一二维参考。

[实现与范围](references.md#na-soc-rpsh).

#### task-na-dboc-rpsh

**投影对角 Born–Oppenheimer 修正 RPSH — P2**

论文：[na-dboc2025](references.md#na-dboc2025).

**原文体系。** Limbu 等人 2025：100 模离散 Debye spin-boson 浴，300 K 和 30 K，对称/非对称势以及重组强度和浴响应时间扫描，与 HEOM 比较。

**应该算什么。** 有/无 DBOC 时的 donor population P_D(t)，与 HEOM 参考比较; 使用论文的混合量子/经典布居估计量：active-state 对角项加振幅的非对角相干项; beads、时间步、浴离散化和 DBOC 梯度差分步长分别收敛

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/rpsh_dboc.inp --output runs/rpsh-dboc`

**缺口与边界。** 现成短时示例不是论文的 Debye bath/HEOM 实验。AdvancedRPSHDynamics.density_matrix 明确返回振幅密度而非论文的混合布居估计量，因此原文曲线比较必须补该观测量；不能直接用 |c|² 或一般 saved density 替代。

[实现与范围](references.md#na-dboc-rpsh).

#### task-na-qd-interface

**准 diabatic NRPMD 与即时绝热电子结构接口 — P2**

论文：[na-qd2018](references.md#na-qd2018), [na-qd2019](references.md#na-qd2019), [na-nrpmd2013](references.md#na-nrpmd2013), [na-mint2018](references.md#na-mint2018), [na-tully1990](references.md#na-tully1990).

**原文体系。** Mandal 等人 2018 的 QD 模型/凝聚相演示；Zhou 等人 2019 用 CASSCF 电子数据研究乙烯激发态失活，所用动力学为 QD-PLDM/SQC。

**应该算什么。** 先在完整两态解析模型上验证 QD 与固定 diabatic NRPMD 的等价、相位不变性、重叠奇异值及能量一致性; 真实分子阶段记录 S1 布居、基态恢复时间和 conical-intersection 途径分支，并与相同电子结构/初始化条件的参考比较

**现有入口。** `python examples/nonadiabatic/adiabatic_on_the_fly.py --method qd-nrpmd --beads 1`

**缺口与边界。** 当前是解析两态接口演示。真实 ethylene 原文复现需要相同 CASSCF 数据、态重叠/NAC、初始态和 PLDM/SQC 实现；当前 NRPMD 结果只能称为在同一体系上的方法比较。通用 Python provider 不自动提供可移植的电子结构重启。

[实现与范围](references.md#na-qd-interface).

#### task-na-mv-rpmd

**映射变量环聚合动力学 MV-RPMD — P2**

论文：[na-mv2013](references.md#na-mv2013).

**原文体系。** Ananth 2013 表 I 的四个一维耦合谐振两态模型：V0=R²/2、V11=R+c、V22=−R、V12=Δ、β=M=k=a=1；(c,Δ)=(0,0.1)、(0,1)、(0,10)、(3,0.1)。

**应该算什么。** 位置自相关 C_RR(t) 与精确 DVR 比较; 位置和电子平衡统计的稳定性、平均 sign 与独立样本误差; 检查 bead 收敛后不同有限 bead 电子离散方式对曲线的影响

**现有入口。** `python examples/nonadiabatic/method_models.py`; MVRPMDPotential, MVRPMDSampler, and MVRPMDDynamics APIs

**缺口与边界。** method_models.py 是教学入口；需补四组原文参数、热采样、相关函数与 DVR benchmark 驱动。不能将有限 bead 的不同电子短时近似误差误判为代码错误。

[实现与范围](references.md#na-mv-rpmd).

#### task-na-kc-rpmd

**动力学约束环聚合方法 KC-RPMD — P3**

论文：[na-kc2014](references.md#na-kc2014).

**原文体系。** Menzeleev–Bell–Miller 2014：模型 A 为一维两态非束缚散射的温度依赖速率；模型 B 为溶剂反应坐标加谐振浴的凝聚相电子转移，扫描耦合与正常/倒转区驱动力。

**应该算什么。** model A 的 k(T); model B 的速率随电子耦合 K 与驱动力的变化，比较非绝热和绝热极限; 分别计算 crossing calibration、auxiliary-coordinate 自由能与 flux-side 透射平台

**现有入口。** `python examples/nonadiabatic/method_models.py`; KCCalibration, KCRPMDPotential, KCRPMDSampler and KCRPMDDynamics APIs

**缺口与边界。** 现成演示不是论文的定量速率复现。用户必须提供物理一致的 crossing 平均、eta 和辅助质量；还缺对应原文参数化的标准化速率 benchmark。不能随意调 a/b 或辅助质量来贴合曲线。

[实现与范围](references.md#na-kc-rpmd).

#### task-na-isomorphic

**两态同构势与 Ehrenfest 动力学 — P0/P2**

论文：[na-iso2018](references.md#na-iso2018).

**原文体系。** Tao–Shushkov–Miller 2018：平移四次两态模型、三态 donor–bridge–acceptor 谐振模型、两态散射；论文动力学图使用 QCLE/CMD 或 SH-RP-iso。

**应该算什么。** 优先复现两态势的逐环电子 Boltzmann trace 恒等式、isomorphic coupling 和解析/有限差分梯度; 在独立实现对应 QCLE 或 iso-SH 后再复现论文的相关函数或态分辨速率

**现有入口。** `python examples/nonadiabatic/method_models.py`; IsomorphicPotential and IsomorphicEhrenfestDynamics APIs

**缺口与边界。** 当前 Ehrenfest 曲线不能标为原文 QCLE/iso-SH 曲线的复现；三态 donor–bridge–acceptor 也超出现有两态范围。没有隐含热初态采样器。

[实现与范围](references.md#na-isomorphic).

#### task-na-tully-models

**Tully I/II/III 解析矩阵势 — P0/P1**

论文：[na-tully1990](references.md#na-tully1990).

**原文体系。** Tully 1990 的三个一维两态散射模型，常用基准核质量为 2000 au。

**应该算什么。** 势能矩阵和导数先对原文方程核对，再做入射动量扫描和四通道散射概率; 把精确波包、P=1 FSSH/MASH、P>1 RPSH 等在同一物理初态下比较

**现有入口。** TullyModelOne/Two/Three APIs and documented saved-workflow model identities in docs/nonadiabatic-models.md

**缺口与边界。** 势函数存在不等于散射 benchmark 完整。需定义入射分布、宽度约定、停止区域和未分类/滞留轨迹。模型 III 的远区 offdiagonal coupling 一侧趋于 2b，不可要求所有区域耦合为零。

[实现与范围](references.md#na-tully-models).

#### task-na-generic-matrix-models

**通用 LVC / 离散 spin-boson 与谐振环初态 — P1**

论文：[na-focused2019](references.md#na-focused2019), [na-sm2021](references.md#na-sm2021).

**原文体系。** 可先选择 NRPMD 2019 或 SM-NRPMD 2021 的完整单模模型。任意 LVC 教学参数不能直接称为这些论文的模型；浴模型还需匹配谱密度离散方案。

**应该算什么。** 初始 harmonic ring 协方差对解析 finite-P 值；明确 active 坐标和零频 spectator 的处理; 选择某篇论文的完整矩阵参数后，再计算对应布居或热相关函数; 核查 tunnelling 参数就是矩阵 offdiagonal 元素，不带隐含 1/2 因子

**现有入口。** [examples/nonadiabatic/population.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/nonadiabatic/population.inp); [benchmarks/nonadiabatic/population_dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/nonadiabatic/population_dynamics.py); sample_harmonic_ring API

**缺口与边界。** 不能为所有任意 LVC 参数统一挂一篇原始模型论文。uncoupled harmonic initial density 是垂直开关/激发预备，不是 coupled equilibrium；自由 spectator 也没有归一化 canonical 位置分布。

[实现与范围](references.md#na-generic-matrix-models).

#### task-na-nah2-pes

**NaH₂ LEPS-5G 两态 diabatic 势 — P0/P3**

论文：[na-nah21992](references.md#na-nah21992), [na-nah2err1994](references.md#na-nah2err1994).

**原文体系。** Na(3p ²P)+H₂ 发生电子淬灭生成 Na(3s ²S)+H₂，或反应生成 NaH+H。

**应该算什么。** 先核验反应物/产物渐近能、绝热面切片、导数与耦合以及交叉区域; 再定义入射能与 H2 初始 rovibrational 状态，计算 quenching/reactive branching 或态分辨散射截面

**现有入口。** `pyrpmd run examples/nonadiabatic/extensions/nah2_sm_nrpmd.inp --output runs/nah2-sm-nrpmd`

**缺口与边界。** 当前分子例子只是短时传播，不是原始散射研究的重现。需同一 PES 参数/勘误、态预备和足够轨迹统计；非简并跳跃算法在交叉区域的适用性也要单独评估。

[实现与范围](references.md#na-nah2-pes).

#### task-na-quantum-reference

**有限区域矩阵分裂算符量子基准 — P0/P1**

论文：[na-feit1982](references.md#na-feit1982).

**原文体系。** Feit–Fleck–Steiger 1982 是谱传播方法来源。仓库当前合成单模两态 LVC、β=2 的垂直开关 benchmark 是包内回归体系，并非该 1982 论文某图的原始体系。

**应该算什么。** 解析 Rabi 振荡、自由 Gaussian 平移/展宽和独立稠密 Hamiltonian 指数的交叉验证; 目标一维矩阵模型的 P_i(t)；分别记录 dt、grid、box 改变后的敏感度以及边缘概率; NRPMD 方法偏差与其有限 beads、抽样误差分开报告；不把 approximate NRPMD=quantum 作为代码正确性要求

**现有入口。** `PYTHONPATH=src python benchmarks/nonadiabatic/population_dynamics.py --smoke --output population_smoke.json`

**缺口与边界。** --smoke 仅 16 trajectories/8 beads/短时，不能标为物理收敛。散射需避免周期边界回卷；二维 SOC 论文需要另外的二维量子参考。

[实现与范围](references.md#na-quantum-reference).

### 材料、电化学、势能与接口复现任务

#### task-bzp

**各向同性 BZP 恒压 PIMD — P1**

论文：[bzp2009](references.md#bzp2009), [ipi2014](references.md#ipi2014).

**原文体系。** BZP 原文：三相点附近的 Lennard-Jones 液体/晶体；i-PI 原文：750 K、10 GPa 超临界水。

**应该算什么。** 先做 P=1，比较体积均值/方差、压力与体积自相关，再做 PIMD；扫描步长及活塞时间常数。

**现有入口。** [examples/ensembles/native_npt.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ensembles/native_npt.toml); [docs/materials-npt.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/materials-npt.md)

**缺口与边界。** 已有理想气体控制算例是替代测试。须取得原文势函数及系综参数；恒压采样时间不能直接当扩散动力学时间。

[实现与范围](references.md#bzp).

#### task-mtk

**柔性晶胞 MTTK/MTK PIMD — P2**

论文：[mtk1994](references.md#mtk1994), [nptpimd1999](references.md#nptpimd1999).

**原文体系。** 恒压 PIMD 原文：周期势中的模型粒子、液态/固态 para-H2、液态丁烷。

**应该算什么。** 比较晶胞/体积分布、压力张量和平衡能量；对齐晶胞约定与参考实现。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [examples/ensembles/native_npt.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ensembles/native_npt.toml)

**缺口与边界。** 须补原文力场与输入；限制某些晶胞自由度不等于已验证的薄层恒压系综。

[实现与范围](references.md#mtk).

#### task-rgb

**RGB 外应力耦合 — P3**

论文：[rgb2011](references.md#rgb2011).

**原文体系。** 纯 BaZrO3 与 Y 掺杂 BaZrO3 中的质子扩散，使用 EVB 力场。

**应该算什么。** 先检查应力下晶胞/应变统计，再用原文 EVB 力场计算质子驻留时间和扩散。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md)

**缺口与边界。** 仓库未附该材料 EVB 势；只有外应力算法不能复现原文扩散结果。

[实现与范围](references.md#rgb).

#### task-scr

**随机晶胞重缩放 SCR — P1**

论文：[scr2020](references.md#scr2020).

**原文体系。** 原文包含 Lennard-Jones 流体及各向异性膜体系。

**应该算什么。** 在支持的各向同性液体上扫描步长/耦合时间，比较体积均值、方差及弛豫。

**现有入口。** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md)

**缺口与边界。** 不能用于 P>1，也不能称为原文的各向异性膜算法；须测量有限步长偏差。

[实现与范围](references.md#scr).

#### task-diffusion

**RPMD 扩散、MSD/VACF 与张量分析 — P2**

论文：[pd2024](references.md#pd2024), [ni2012](references.md#ni2012).

**原文体系。** 2012 年 Ni(100)/H 的 EAM 体系；2024 年柔性 Pd(111)/H 的 DFT 训练神经网络势。

**应该算什么。** 计算 D(T)、MSD、VACF、位点跃迁和多次跳跃相关性；比较冻结/可动基底及表面/亚表面通道。

**现有入口。** [examples/materials/diffusion.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/materials/diffusion.inp); [benchmarks/materials_reference/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/materials_reference/README.md); [benchmarks/water/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/water/README.md)

**缺口与边界。** Ni EAM4 目前是静态参考适配器；Pd 神经网络权重和参数仍须取得。NiAlH 试算使用另一势模型。

[实现与范围](references.md#diffusion).

#### task-site-surface

**盆地归一化位点/表面速率 — P2**

论文：[sds2025](references.md#sds2025), [pd2024](references.md#pd2024), [bazro2008](references.md#bazro2008).

**原文体系。** SDS 2025：H+H2、CH3NC 异构化、Pt(111) 上 H 重组脱附、Pd(111) 上 NO 脱附；BaZrO3 2008 仅作 centroid-TST 对照。

**应该算什么。** 计算盆地配分量、PMF 势垒、通量因子、k_QTST、kappa(t) 和有向速率；比较坐标与归一化选择。

**现有入口。** [examples/materials/site_rate.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/materials/site_rate.inp); [examples/surface/diffusion.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/surface/diffusion.inp); [docs/surface-catalysis.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/surface-catalysis.md)

**缺口与边界。** 须匹配表面势能面和反应物态约定；当前不自动生成多位点动力学 Monte Carlo 网络。

[实现与范围](references.md#site-surface).

#### task-nialh-model

**NiAlH EAM 试算势 — P2**

论文：[nialh1995](references.md#nialh1995).

**原文体系。** 氢在 Ni 位错及晶界处的俘获。

**应该算什么。** 先检查力/单位及缺陷结合能，再开展输运计算。

**现有入口。** [benchmarks/materials/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/materials/README.md)

**缺口与边界。** 小型 Ni(100) 试算不是原文缺陷体系，也不是 2012 年 EAM4 扩散模型。

[实现与范围](references.md#nialh-model).

#### task-pcet

**局域黄金规则电化学 PCET — P1**

论文：[pcet2008](references.md#pcet2008), [pypcet2025](references.md#pypcet2025), [gold2019](references.md#gold2019).

**原文体系。** 2008 年金属/溶液 PCET 模型；2019 年应用为乙腈中 TEAH+/TEAD+ 在 Au 电极上的放电。

**应该算什么。** 计算正逆 k(U,T)、H/D KIE、各振电通道贡献；收敛质子网格、能级数及电极能量求积。

**现有入口。** [examples/pcet/harmonic_scan.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pcet/harmonic_scan.inp); [docs/validation/pcet-model-2026-10-30/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/validation/pcet-model-2026-10-30/README.md)

**缺口与边界。** 与 pyPCET 一致是实现交叉验证；局域模型未覆盖真实体系的距离平均、双电层及更广的耦合区间。

[实现与范围](references.md#pcet).

#### task-gc

**共享电子数的巨正则 HMC/MC — P2**

论文：[hmc1987](references.md#hmc1987), [hastings1970](references.md#hastings1970), [gc2025](references.md#gc2025).

**原文体系。** HMC 起源于格点场论；相关电极应用是 5x5 四层 Pt(111)、1 ML H 与 36 个水分子的 GC-(PI)HMC Volmer/Heyrovsky 自由能。

**应该算什么。** 先算解析充电模型的 q,N 联合分布、N 均值/方差、接受率和巨势自由能，再比较界面经典/量子势垒。

**现有入口。** [examples/electrochemistry](https://github.com/chen-yu-hao/PyRPMD/tree/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/electrochemistry); [docs/electrochemistry.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/electrochemistry.md)

**缺口与边界。** 此采样器不自动提供 Sun2025 的完整受约束算法、DP-Ne 模型、电荷补偿板或混合量子化设置。

[实现与范围](references.md#gc).

#### task-adiabatic-electrode

**绝热恒电位标量与矩阵包装 — P1**

论文：[gc2025](references.md#gc2025).

**原文体系。** 这一 PyRPMD 组合没有独立原始论文；Sun2025 是物理背景，不能据此宣称算法相同。

**应该算什么。** 比较解析 N*(R)、求解残差、包络定理力，以及同位素/电势相关自由能。

**现有入口。** [examples/electrochemistry/adaptive_constant_potential.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/electrochemistry/adaptive_constant_potential.py); [docs/manual/adaptive-nrpmd.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/adaptive-nrpmd.md)

**缺口与边界。** 逐珠最优电子数不同于共享电子数涨落；共同充电项没有态依赖响应，也不是 TP-RPMD。

[实现与范围](references.md#adiabatic-electrode).

#### task-her

**Volmer–Heyrovsky–Tafel 平均场后处理 — P1**

论文：[gc2025](references.md#gc2025).

**原文体系。** 这些代数平衡没有唯一方法原始论文；电极论文只提供基元反应的物理背景。

**应该算什么。** 计算 theta(U)、各基元净通量、H2 TOF 和有符号电流；检验稳态计量守恒与详细平衡。

**现有入口。** [docs/electrochemistry.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/electrochemistry.md)

**缺口与边界。** 不从 PMF 或局域 PCET 自动推导基元速率，也不包含传质、侧向相互作用或双电层。

[实现与范围](references.md#her).

#### task-efrpmd

**Markov 质心电子摩擦 RPMD — P2**

论文：[efrpmd2024](references.md#efrpmd2024).

**原文体系。** Anderson–Holstein 化学吸附模型，对称/非对称电子转移势。

**应该算什么。** 计算图 2 核分布、图 3 布居弛豫、图 4 温度相关速率及图 5 非对称低温偏差；先匹配平均力势和摩擦。

**现有入口。** [examples/nonadiabatic/friction_surface.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/nonadiabatic/friction_surface.inp); [docs/electronic-friction.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/electronic-friction.md)

**缺口与边界。** 已有起伏表面示例是替代模型；须构造匹配平均力势和摩擦张量；尚无记忆效应或虚时间浴重整化。

[实现与范围](references.md#efrpmd).

#### task-pyscf

**PySCF 参考数据与即时受力接口 — P0**

论文：[pyscf2018](references.md#pyscf2018).

**原文体系。** 软件接口没有唯一物理复现体系；实际电子结构方法、基组和泛函应另外引用。

**应该算什么。** 同一构型与直接 PySCF 比较能量/梯度/Hessian，检查差分、自旋/电子态连续性与收敛。

**现有入口。** [examples/pesgen/water.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/water.toml); [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** RHF/STO-3G 水只是集成控制，不是已验证反应势。

[实现与范围](references.md#pyscf).

#### task-populations

**Hirshfeld 电荷与 Mayer 键级 — P1**

论文：[hirshfeld1977](references.md#hirshfeld1977), [mayer1983](references.md#mayer1983).

**原文体系。** 原文定义分子电荷/键级；当前数据桥接没有声称复现完整原文数表。

**应该算什么。** 验证电荷守恒及网格收敛；用闭壳层/开壳层分子对比独立同约定计算。

**现有入口。** [examples/pesgen/water.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/water.toml); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**缺口与边界。** 不同自由原子参考表会产生不同电荷，不能把 ORCA 默认值视为同一定义。

[实现与范围](references.md#populations).

#### task-qmdff

**QMDFF 势生成 — P1**

论文：[qmdff2014](references.md#qmdff2014), [caracal2023](references.md#caracal2023).

**原文体系。** 原始 QMDFF 的分子/凝聚相基准，基于 TPSS-D3/def-TZVP(-f) 数据；本地水分子算例是替代控制。

**应该算什么。** 先比较平衡结构、Hessian 频率、畸变构型能量/力及解离行为，再运行轨迹。

**现有入口。** [examples/pesgen/water.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/water.toml); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**缺口与边界。** 需要上游编译及合适参考理论；宣称复现前须取得选定原文基准的补充材料结构。

[实现与范围](references.md#qmdff).

#### task-deevb

**dE-EVB / EVB-QMDFF 耦合拟合 — P2**

论文：[deevb2015](references.md#deevb2015), [caracal2023](references.md#caracal2023).

**原文体系。** 原文示例包含 SN2：Cl−+CH3Br → CH3Cl+Br−，以及乙烯+环戊二烯 → 降冰片烯的 Diels–Alder 反应。

**应该算什么。** 比较端点极限、耦合、路径内外能量/力误差、势垒及相对参考势的 RPMD 速率。

**现有入口。** [examples/pesgen/nh3_inversion.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/nh3_inversion.py); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**缺口与边界。** 当前拟合工作流仅支持中性体系，不能直接运行带电 SN2；忠实原文体系应先恢复中性 Diels–Alder 协议。NH3 是独立仓库控制。

[实现与范围](references.md#deevb).

#### task-treq

**TREQ 势生成 — P2**

论文：[treq2019](references.md#treq2019), [caracal2023](references.md#caracal2023).

**原文体系。** 原文在六个具有解析全维参考势的反应上比较；选定体系的完整补充材料协议尚待转录。

**应该算什么。** 比较路径内外能量/梯度、差分步长收敛、能量漂移，再比较参考解析势上的 k(T)。

**现有入口。** [examples/pesgen/nh3_inversion.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/nh3_inversion.py); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**缺口与边界。** 桥接处于实验阶段；固定源码的解析力不一致已记录，不能把目前默认力直接等同于原文实现。

[实现与范围](references.md#treq).

#### task-jaxpip

**线性 PIP / PIP-NN 拟合与 JaxPIP 导出 — P2**

论文：[pip2010](references.md#pip2010), [pipnn2013](references.md#pipnn2013), [jaxpip2026](references.md#jaxpip2026).

**原文体系。** PIP 单项式对称化：H₃O⁺，约 62,000 个从头算能量点（Xie–Bowman 2010）。PIP-NN 2013：H+H2 与 Cl+H2；JaxPIP 2026：3–9 原子性能测试，以及 OH+CH3OH 的 QCT 截面/分支（SI 表 S7–S8）。

**应该算什么。** 计算留出集能量/力误差、置换不变性、Python/原生一致性及批量性能；再用相同训练势计算散射截面。

**现有入口。** [docs/manual/jaxpip-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/jaxpip-generation.md); [examples/pesgen](https://github.com/chen-yu-hao/PyRPMD/tree/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen)

**缺口与边界。** 小 H2 示例不能复现 OH+CH3OH；须取得准确基函数/权重、数据划分及 QCT 初态约定。

[实现与范围](references.md#jaxpip).

#### task-committee

**委员会不确定度与样本采集 — P2**

论文：[committee2020](references.md#committee2020).

**原文体系。** 跨液体状态点、冰相及气水界面的凝聚相水。

**应该算什么。** 比较分歧与真实留出误差、采样富集效果及学习曲线，固定各阶段模型版本。

**现有入口。** [examples/ml_lifecycle/committee.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ml_lifecycle/committee.inp); [docs/tier2-ml-lifecycle.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-ml-lifecycle.md)

**缺口与边界。** 解析势委员会示例只验证流程；须补外部标注数据与训练。

[实现与范围](references.md#committee).

#### task-delta

**加性 Δ 学习与 RPC/MTS — P2**

论文：[delta2015](references.md#delta2015).

**原文体系。** Δ 学习原文：C7H10O2 异构体及有机分子热化学数据，不是 RPMD 轨迹基准。

**应该算什么。** 检验校正后留出能量及力一致性；对 PyRPMD 扩展比较完整与收缩校正的同一观测量。

**现有入口。** [examples/ml_lifecycle/delta_rpc.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ml_lifecycle/delta_rpc.inp); [examples/ml_lifecycle/delta_rpc_mts.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ml_lifecycle/delta_rpc_mts.py)

**缺口与边界。** 原文分子性质基准和动力学扩展不同；仓库不提供训练好的校正势。

[实现与范围](references.md#delta).

#### task-ase

**ASE 计算器接口 — P0**

论文：[ase2017](references.md#ase2017).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#ase).

#### task-openmm

**OpenMM 力接口 — P0**

论文：[openmm2017](references.md#openmm2017).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#openmm).

#### task-lammps

**LAMMPS 库接口 — P0**

论文：[lammps1995](references.md#lammps1995).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#lammps).

#### task-ipi-socket

**i-PI socket 接口 — P0**

论文：[ipi2014](references.md#ipi2014).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#ipi-socket).

#### task-torch

**Torch/GPU 势及张量传播 — P0**

论文：[torch2019](references.md#torch2019).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#torch).

#### task-ml-adapters

**MACE / NequIP / DeePMD / fairchem 接口 — P0**

论文：[mace2022](references.md#mace2022), [nequip2022](references.md#nequip2022), [deepmd2018](references.md#deepmd2018), [uma2025](references.md#uma2025).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#ml-adapters).

#### task-tblite

**tblite 的 GFN1/GFN2/IPEA1-xTB — P0**

论文：[gfn12017](references.md#gfn12017), [gfn22019](references.md#gfn22019), [ipea2017](references.md#ipea2017).

**原文体系。** 外部软件/模型接口；软件论文用于归属引用，不对应唯一的 PyRPMD 物理基准。

**应该算什么。** 同一构型与上游直接比较能量、力、位力、单位、批处理和周期映像，再接已验证物理体系。

**现有入口。** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**缺口与边界。** 须提供可选引擎及兼容模型文件；接口不重做上游训练，也不复现其全部应用。

[实现与范围](references.md#tblite).

#### task-h5md-export

**H5MD、extxyz 与 ASE 轨迹导出 — P0**

论文：[h5md2014](references.md#h5md2014), [ase2017](references.md#ase2017).

**原文体系。** 数据格式论文，无独立物理基准。

**应该算什么。** 回读坐标、晶胞、同位素质量、时间和单位，比较重新分析的观测量。

**现有入口。** [docs/trajectory-export.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/trajectory-export.md)

**缺口与边界。** 须检查所选格式的读取支持及语义一致性。

[实现与范围](references.md#h5md-export).

#### task-engineering

**输入、续跑、并行、报告与运行组织 — P0**

论文：—.

**原文体系。** 这些 PyRPMD 专属机制没有独立原始方法论文；应引用准确 PyRPMD 版本及底层方法/势能面论文。

**应该算什么。** 用相同物理算例检查后端数值一致性、续跑一致性、单位/归一化、误差及归档回放。

**现有入口。** [docs/manual/execution.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/execution.md); [docs/checkpoints.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/checkpoints.md); [docs/manual/analysis-reports.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/analysis-reports.md); [docs/reaction-networks.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/reaction-networks.md)

**缺口与边界。** 软件测试通过不等于新的物理复现；不能为填满引用栏而附无关论文。

[实现与范围](references.md#engineering).

#### task-analytic-models

**解析自由粒子、谐振子、Morse、四次双阱与周期余弦模型 — P0**

论文：[morse1929](references.md#morse1929).

**原文体系。** Morse 原文研究双原子振动能级；其余通用测试势没有唯一指定的原始分子体系或方法论文。

**应该算什么。** 算能量/力有限差分、谐振子的解析平衡矩、Morse 束缚能级、积分自由能面和坐标周期性；统一能量零点与参数。

**现有入口。** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [examples/fes/harmonic2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/fes/harmonic2d.inp)

**缺口与边界。** 可分离余弦模型不能证明真实材料扩散正确；任意 Morse 参数也不等于复现 1929 原文的某个分子。

[实现与范围](references.md#analytic-models).

#### task-rate-studies

**温度/同位素扫描、Arrhenius 拟合与数值收敛研究 — P1**

论文：[bimol09](references.md#bimol09), [iso13](references.md#iso13).

**原文体系。** 速率扫描采用同一已确认势能面上的 H + H₂ 及同位素体系；平衡分馏另用原估计器论文规定的水模型。

**应该算什么。** 算 k(T)、k_H/k_D、活化参数与拟合残差，以及珠数、时间步、初始距离、最大碰撞参数改变后的差异；事先给定等效性容限。

**现有入口。** [docs/rate-studies.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/rate-studies.md)

**缺口与边界。** Arrhenius 拟合好不代表速率正确，也不排除隧穿曲率；差异不显著不能代替收敛或等效性证据。

[实现与范围](references.md#rate-studies).

#### task-bkmp2-model

**BKMP2 H₃ 论文基准与外部势能面溯源 — P1**

论文：[bkmp21996](references.md#bkmp21996), [bimol09](references.md#bimol09), [rpmrate13](references.md#rpmrate13).

**原文体系。** BKMP2 1996 原文研究 H + H₂ 转动激发和 H₃ 势能面验证；RPMD 速率论文研究 H + H₂ 交换，二者是同一势家族上的不同计算目标。

**应该算什么。** 先核对势面、鞍点、梯度和同位素质量，再算 PMF、k_QTST、κ(t) 和 k(T)；1996 原文态到态截面必须另用它的初态与散射协议。

**现有入口。** [benchmarks/literature/campaign.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/literature/campaign.json); [docs/literature-validation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/literature-validation.md)

**缺口与边界。** 固定势文件及校验和、通道与对称性约定；已有固定珠数一致性仍留下父链时间、伞窗精度和动态珠数收敛需要完成。

[实现与范围](references.md#bkmp2-model).
