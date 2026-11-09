---
orphan: true
---

# NP8 全链路返回次数审计、基准与文档整合

| 字段 | 内容 |
| --- | --- |
| 规模 | S |
| 波次 | 3 |
| 前置 | NP0–NP7 中已合并的卡（未合并的卡在报告中标为"未接入"） |
| 等价性等级 | 不改数值代码 |
| 建议分支 | `agent/np8-native-audit` |

## 背景

本轮各卡分别在自己的链路上给出了返回次数和性能证据。需要一份在同一硬件、同一构建上、
覆盖全部链路的审计，以及一份统一的文档，说明"哪些配置可以全程在 C++ 中运行、哪些不行、为什么"。

## 目标

1. 审计脚本：对每条链路统计每 1 万步的原生调用次数、`python_steps`、`fallback_reasons` 与墙钟时间，
   在基线 `ca093eb` 与集成后的 `main` 上各跑一次。
2. 把各卡追加到 `docs/native-propagation.md` 的小节整理成连贯的文档，加一张"配置 → 是否全程原生 → 返回点"总表。
3. CI 中运行本轮新增测试。

## 范围

**要做：**

1. 新 `benchmarks/native_path/return_audit.py`：用内置势（谐振、Morse、双势阱）构造最小配置，覆盖
   伞形/PMF、FES、再穿越父链与子轨迹（RPMD、TRPMD）、dRPMD、REUS、thermalize、元动力学、committor、TIS、
   materials（Andersen、PILE-L）、diffusion（legacy 与 native）、原生 PIMD。输出 JSON 和 Markdown 表；
   能构建 BKMP2 时追加 H₃ 速率链的端到端墙钟时间。脚本记录源提交、扩展哈希、硬件、线程数与重复次数。
2. 整理 `docs/native-propagation.md`：合并各卡小节，保留契约细节；新增总表与"仍需 Python 的情况"清单
   （Python PES 类、自定义 CV/偏置/分类器、NRPMD/非绝热、bond constraints、PLUMED、force splitting/MTS、
   OPES/ABF/VES/steered、FFS `basin-andersen`、NPT 活塞旧引擎）。
3. 更新本目录 `README.md` 的卡片状态，以及开头"当前各链路返回 Python 的频率"表（追加"替换后"一列）。
4. 在 `.github/workflows/fused-native.yml` 追加运行本轮测试的步骤（只追加，遵守 G13 与 `tools/check_workflow_policy.py`）。

**不做：**

- 不改任何数值代码；发现缺陷时开 issue 或写进 PR"未完成事项"，交回对应卡的负责人。

## 文件所有权

- **独占：** 新 `benchmarks/native_path/`、`docs/native-propagation.md`（整理）、`docs/tasks/native-path/README.md`（状态更新）。
- **共享注册点：** `.github/workflows/fused-native.yml`（追加步骤）。

## 交付物

- 审计脚本与 `docs/validation/native-path-audit-<日期>/README.md`（含原始 JSON，≤ 1 MB）。
- 整理后的文档。

## 验收标准

- 审计表覆盖上述每条链路；每一行给出基线与集成后的原生调用次数、`python_steps`、墙钟时间（R9）。
- 声明"全程原生"的配置在审计中 `python_steps == 0`，且每 1 万步原生调用次数与该卡声明的上界一致。
- `python -m sphinx -W --keep-going -b html docs docs/_build/html` 通过；工作流策略测试通过。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
python benchmarks/native_path/return_audit.py --output runs/native-path-audit
python -m pytest -q tests/test_workflow_policy.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m pytest -q -m "not campaign"
```
