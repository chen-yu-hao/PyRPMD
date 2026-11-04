---
orphan: true
---

# T41 社区文件、可执行教程与插件入口

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t41-community-plugins` |

## 背景

2026-11-03 评审发现，仓库中还缺少以下内容：

- `CITATION.cff`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`；
- issue 模板和 PR 模板；
- Jupyter 教程（仓库中没有任何 `.ipynb`）；
- 第三方插件入口：`src/` 中没有任何 `entry_points` 发现逻辑。第三方的势、CV 或方法
  只能以 Python 输入中的 `factory="module.py:build"` 方式接入，无法作为可安装、
  可注册、带身份的插件分发。

`docs/publication-readiness.md` 和 JOSS 的审稿标准都要求上述社区文件。
i-PI 则通过 atomistic-cookbook 提供可执行教程。

## 目标

补齐社区与贡献基础设施；提供在 CI 中执行的教程；建立插件入口机制，
让第三方包可以注册势提供者和 CV，并且插件身份进入 run identity。

## 范围

**要做：**

1. **社区文件。**
   - `CITATION.cff`：作者信息留占位，必须由维护者填写，不得编造作者；
   - `CONTRIBUTING.md`：开发环境、测试命令、campaign marker、任务卡规则（链接到本目录）、
     提交规范、AI 协助披露；
   - `CODE_OF_CONDUCT.md`（采用 Contributor Covenant）；
   - `.github/ISSUE_TEMPLATE/`（bug、功能请求、科学问题）；
   - `.github/PULL_REQUEST_TEMPLATE.md`，各节与通用规则 G11 对应。
2. **可执行教程。** `docs/tutorials/` 下 4 本 notebook，每本在 CI 中运行不超过 5 分钟：
   - 快速开始的 FES；
   - H + H₂ 速率流程试点（使用内置或已有的小型 PES，不声称收敛）；
   - PIMD 与能量估计量；
   - Tully 模型上的非绝热方法比较。
   新增一个 CI job，用 `nbclient` 或 `nbmake` 执行这些 notebook；notebook 保存时不带输出。
3. **插件入口。** 新建 `src/pyrpmd/plugins.py`，读取 `importlib.metadata.entry_points` 中的两组：
   `pyrpmd.potentials`（`provider` 名称到工厂函数）和 `pyrpmd.cvs`（CV `kind` 到实现）。
   - 名称与内置提供者冲突时拒绝加载；
   - 插件的包名、版本和分发文件哈希（可得时）进入 run identity；
   - 提供 `pyrpmd plugins list` 命令；
   - 在文档中写明插件代码以完全信任的方式运行；
   - 附带一个最小的示例插件包（`examples/plugin_package/`），在测试中安装并使用。

**不做：**

- 不修改 `README.md`、`README.zh-CN.md`（归 T17）。
- 不迁移 `docs/validation` 中的大数据（需要维护者决定）。
- 不发布任何包，不配置 PyPI 或 conda-forge。

## 文件所有权

- **独占：** `CITATION.cff`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`.github/ISSUE_TEMPLATE/`、
  `.github/PULL_REQUEST_TEMPLATE.md`、新建 `.github/workflows/tutorials.yml`、`docs/tutorials/`、
  新建 `src/pyrpmd/plugins.py`、`examples/plugin_package/`、相关测试。
- **共享注册点：** `potentials/__init__.py`（提供者解析时查询插件）、CV 解析处（`cv.py` 或
  `input_tasks.py` 中解析 `[[cvs]] kind` 的位置，只追加查询插件的分支）、`cli.py`
  （`plugins` 子命令）、`pyproject.toml`（`docs` 或 `test` 可选依赖中追加 `nbclient`）。

## 交付物

- 社区文件、教程与 CI job、插件机制、示例插件包，以及英文指南 `docs/plugins.md`（`orphan: true`）。

## 验收标准

- 在干净的虚拟环境中安装示例插件包后，`pyrpmd plugins list` 能列出该插件；
  使用插件提供者的输入可以运行，其 run identity 包含插件版本；
  卸载或升级插件后续算会被拒绝，并给出明确提示。
- 与内置提供者重名的插件被拒绝加载，错误信息中指出冲突的分发包。
- 4 本教程在 CI 中全部执行通过，总耗时在 20 分钟以内。
- `CITATION.cff` 通过 `cffconvert --validate`（在 CI 或本地运行）。
- 满足通用规则 G5–G8、G10。

## 验证命令

```bash
python -m pytest -q tests -k "plugin"
python -m pip install ./examples/plugin_package && python -m pyrpmd plugins list
python -m pytest --nbmake docs/tutorials
```
