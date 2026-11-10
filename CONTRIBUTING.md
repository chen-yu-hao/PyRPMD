# Contributing to PyRPMD / 参与开发

Start from an existing Python 3.11+ environment. Read the
[stability and compatibility policy](docs/development/stability.md) before
changing public inputs, summary fields or checkpoints. Keep numerical fixes
focused, preserve unrelated edits, and include a regression that demonstrates
the failure and its scientific consequence.

## Development checks

```bash
python -m pip install -e '.[test]'
python setup.py build_ext --inplace
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -m 'not campaign'
```

Use an existing environment when installation is restricted. Rebuild native
extensions after modifying C++ and record the actual imported module path.
The fast library suite excludes campaign data imports. Run the affected tests
first; then use `python -m pytest -q -m campaign` for relevant archive-dependent
checks. Campaign tests run on demand or in their scheduled job. Optional
backend skips must be reported as missing coverage, not passes.

## Potentials and collective variables

A potential returns physical energies and forces in the documented units and
layout. Check analytic forces against finite differences, batch/scalar parity,
atom ordering, invalid geometry, and provider lifetime. Native bindings must
preserve Python overrides and demonstrate actual native execution. Start with
[potential adapters](docs/manual/potentials.md) and the
[C ABI example](native/examples/harmonic_pes.c).

A CV must provide its declared units, estimator and full bead Jacobian. Check
gradient/force signs, translation and relevant symmetries, singular domains,
and native/Python parity where supported. Rate CV changes also require
flux-metric and constrained-parent tests. See [CV input](docs/manual/input.md).

## Pull requests and scientific evidence

Describe the failing behavior, resulting behavior, changed scientific contract
and test commands. Include restart parity where state or random-stream behavior
changes. Preserve failed histories and null estimates. New methods enter beta
or research until their validation scope satisfies the stable criteria.

Add user-visible changes to [CHANGELOG.md](CHANGELOG.md). Release notes are
prepared from that file with `python tools/render_release_notes.py --version Unreleased`;
a source version is not proof of a published package.
Use [CITATION.cff](CITATION.cff) for software attribution and cite the actual
PES and method papers used in a calculation. Raw campaign migration follows
the [archive plan](docs/data-archive-plan.md); do not remove tracked evidence
before the replacement archive has been independently retrieved and checked.

中文：先运行相关回归，再运行 `-m 'not campaign'` 快速测试；改变原生源码后重建
扩展并确认实际导入路径。科学参数、采样系综、失败记录和 checkpoint 身份必须保留。
新增方法先标记 beta/research；说明验证范围，不用短测试通过代替科学收敛。
输入、结果字段或 checkpoint 的兼容性变化需在 CHANGELOG 中明确记录。
