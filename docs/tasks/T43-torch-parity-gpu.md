---
orphan: true
---

# T43 Torch 引擎功能对齐与 GPU 实测

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 2 |
| 前置 | T16b 合并；GPU 部分需要 NVIDIA 硬件 |
| 合并 | 原 T19（续）的 GPU 实测部分 |
| 建议分支 | `agent/t43-torch-parity` |

## 背景

Torch 引擎（`sampling.engine="torch"`）把所有副本放在一个设备上的 `[B,P,N,3]` 批次中，
支持 `pimd`/`diffusion` 和精确续算；但不支持 NPT、约束、交换、摩擦和 MPI 分组
（见 `docs/torch-materials-workflows.md`）。

T04 交付了 MACE、NequIP、fairchem、DeePMD、metatomic 的批量提供者；T19 写好了 CUDA 测量脚本
（`benchmarks/efficiency/t19_cuda.py` 等），但至今没有任何 GPU 测量。

i-PI 3 报告，在常见 ML 势下，直到数万原子 Python 开销都可以忽略；TorchSim 则在 GPU 上批量推进
多个体系，但它没有 PIMD。对 PyRPMD 来说，GPU 驻留的批量 PIMD/RPMD 是一个明显的机会。

## 目标

让 Torch 引擎支持最常用的生产组合：各向同性 NPT、质心 CV 偏置（伞形与 MetaD），以及
RPMD 速率中的约束父链与子轨迹。在真实 GPU 上测量正确性与吞吐量，并给出端到端的成本数据。

## 范围

**要做：**

1. **NPT。** 在设备上实现 BZP 各向同性压浴，需要提供者给出维里；与 legacy 引擎在同一输入下的
   体积分布一致。
2. **CV 与偏置。** 在设备上计算质心距离、角度、坐标数等常用 CV 及其雅可比，支持伞形与 MetaD
   偏置；与 CPU 实现逐帧对照。
3. **速率子任务。** 在设备上批量执行约束父链（质心 RATTLE）与配对子轨迹，并接入
   `task="rate"` 的父链和子轨迹阶段，可以按选项启用。
4. **GPU 实测。** 运行 T19 的脚本以及本卡新增的端到端基准：
   - 5 个 ML 提供者的单次调用；
   - 完整的 PIMD、扩散和速率子轨迹；
   - checkpoint 边界的开销与峰值显存；
   - 与 CPU 的结果一致性；
   - 吞吐量随 B、P、N 的标度。
   必须写明硬件、驱动和 CUDA/PyTorch 版本。
5. **同步点审计。** 找出标量读回等隐式设备同步，消除不必要的同步，
   并记录剩余同步点的位置与原因。

**不做：**

- 不实现交换、摩擦或 MPI 分组（可以在之后另行立卡）。
- 不做自动的设备或精度回退。

## 文件所有权

- **独占：** `src/pyrpmd/torch_backend.py`、`torch_normal_modes.py`、`workflows/torch_materials.py`、
  `potentials/_batched_ml_base.py` 及各 `_batched_*.py`（只限同步与性能改动）、
  `benchmarks/efficiency/`、相关测试。
- **共享注册点：** `input_tasks.py`（放开 torch 引擎的组合校验）、`workflows/rate*.py`
  （只追加 torch 执行分支）。

## 交付物

- 功能实现与英文指南（更新 `docs/torch-materials-workflows.md`）。
- 验证记录 `docs/validation/torch-gpu-<日期>/README.md`，包含硬件信息和原始测量数据的哈希。

## 验收标准

- 在 CPU 上：
  - NPT 的体积与焓分布与 legacy 引擎在统计误差内一致；
  - CV 与偏置和 CPU 实现逐帧一致（float64，相对误差 ≤ 1e-12）；
  - 速率子轨迹的 κ(t) 与 CPU 实现在统计误差内一致。
- 在 GPU 上（如有硬件）：
  - 与 CPU 结果在 float64 下的差异给出上限；
  - 吞吐量表覆盖至少 3 个体系规模，并与 CPU 原生引擎对比；
  - 报告“达到固定误差目标所需的成本”。
  没有硬件时，GPU 部分在 PR 中标注为“未完成”，不得用 CPU 数据代替。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "torch"
python benchmarks/efficiency/t19_cuda.py --help
```
