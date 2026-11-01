# Distribution validation — 2026-11-01

This record covers distribution work based on `2f907ba96d22cb134198715e398b1d9f8d83d929`.
It records **actual local builds and installed-package tests**, not publication or a
completed cross-platform CI run. The artifacts were built while the shared worktree
was still under development; final combined-source package validation is recorded
separately in the enclosing ecosystem report.

## Final combined-source package gate

After the provider and workflow guards were frozen, a clean final sdist and native
Linux CPython 3.12 wheel were built. **All 142 recorded runtime/build input hashes
were unchanged through the build and installed smoke.** Every recorded input was
checked byte-for-byte against the source archive, and all **112 Python modules** in
the wheel match the source. Native `_native` and `_thermal` extensions are present.

Both artifacts pass `twine check --strict`. The wheel was installed into a new
environment outside the checkout with only its base runtime dependencies. Its smoke
test passed native/NumPy agreement for all three integrators, constrained H/D dimer
propagation, extxyz and H5MD exports, an H5MD coordinate roundtrip, lazy import of the
optional engine modules, and CLI init/validate/run/analyze/resume. `pip check` passed.
This gate validates the final combined code, independently of the earlier evolving-
worktree snapshots below.

- [Final package verification](final-package-verification.json) includes every source
  hash, the source digest, artifact checksums, installed dependency versions and scope.
- [Final package build](final-package-build.log) retains actual build output.
- [Final installed-wheel smoke](final-wheel-smoke.log) retains actual execution output.

Source digest: `4398e6e4652a8393cf1fa59a65dfcd06b7840798b6aa3bc66df06d3faff1bee2`.
The digest covers the listed runtime/build inputs, not documentation/evidence that
continued to be assembled after the build. Cross-platform portable wheels still need
their actual CI runners; optional engine physics is checked separately.

## Executed checks

| Check | Actual result |
| --- | --- |
| `python -m build` | Source distribution and native Linux CPython 3.12 wheel built |
| `twine check --strict` | Both artifacts passed |
| Isolated wheel installation outside the checkout | Native `_native`/`_thermal` imports, Linux OpenMP, all three integrators against NumPy, and CLI init/validate/run/analyze/resume passed |
| Rattler-build 0.80.0 | Actual Linux conda packages built and installed tests passed on Python **3.12.16**, **3.13.18** and **3.14.10** |
| Conda variants | Four variants rendered: Python 3.11, 3.12, 3.13, 3.14; 3.11 was not locally built in this check |
| Conda runtime linking | Explicit Linux `libgomp` dependency fixed an initial overlinking warning; the retained final three builds have no overlinking warning |
| cibuildwheel 3.4.0 selection | Exactly twelve identifiers: CPython 3.11–3.14 × manylinux x86_64, macOS arm64, Windows AMD64 |
| Workflow checks | All YAML parses; actionlint 1.7.12 passes wheels/conda/tests/packages workflows (external shellcheck/pyflakes disabled) |
| Documentation | English/Chinese HTML and singlehtml all pass Sphinx `-q -W --keep-going`; translated code blocks agree with originals and relative links resolve |
| Release-recipe generator | Actual sdist metadata/hash preserved, local path replaced, explicit maintainer retained; non-HTTPS/credential-bearing/ephemeral query URLs rejected |

The wheel smoke compares multi-step q/p for `cayley-bcb`, `normal-mode-bab` and
`velocity-verlet` against NumPy and exercises the installed quickstart. It does not
claim a full scientific regression suite on each interpreter. Conda tests also run
`pip check`. The test implementation is `tests/wheel_smoke.py`.

## Retained evidence

- [evidence.json](evidence.json): versions, outcomes, artifact SHA256 values and final
  distribution-source hashes.
- [conda-build-final.log](conda-build-final.log): complete final build/test output for
  the three actual conda packages, including `all tests passed` for each.
- [smoke.log](smoke.log): installed Linux CPython 3.12 wheel lifecycle.
- [conda-render.json](conda-render.json): rendered final four-variant recipe.

Reproduction commands:

```bash
python -m build
python -m twine check --strict dist/*
python -m cibuildwheel --platform linux --archs x86_64 --print-build-identifiers
python -m cibuildwheel --platform macos --archs arm64 --print-build-identifiers
python -m cibuildwheel --platform windows --archs AMD64 --print-build-identifiers
rattler-build build --recipe packaging/conda/recipe.yaml --variant python=3.12,3.13,3.14 \
  --output-dir conda-output -c conda-forge
```

The first conda pilot exposed the missing explicit OpenMP runtime dependency. It was
fixed before all three retained final builds. After those runs, a metadata-version
assertion was added to the recipe's build script; the final recipe was rendered and
parsed, but that extra assertion is not retroactively claimed as an executed command
in the retained logs.

The manylinux/macOS/Windows cibuildwheel matrix must still execute on its actual CI
runners. A local manylinux Docker build and macOS/Windows
hosts were unavailable in this environment. No package was uploaded to PyPI or TestPyPI, and no
conda-forge submission was made. Registry ownership, trusted-publisher/environment
configuration, a reviewed release tag, and consenting conda maintainers remain
external release prerequisites.
