# Conda packaging

`recipe.yaml` builds the local checkout, including both native C++ extensions.
`variants.yaml` supplies standalone compiler-library settings and CPython 3.11–3.14
variants. Run from the repository root:

```bash
rattler-build build --recipe packaging/conda/recipe.yaml -c conda-forge
```

For one interpreter, append `--variant python=3.12`. Rattler-build discovers the
adjacent variants automatically. The conda CI performs the complete Linux matrix.
Installed package tests include native/NumPy numerical agreement and the public
quickstart lifecycle. No channel upload is configured.

The local source path is intentionally unsuitable for submission to conda-forge.
After publishing an actual source release, use `prepare_release.py` with its exact
archive, public URL and consenting maintainers. The generated v1 recipe records the
archive's real SHA256. The feedstock will use conda-forge's compiler and Python
pinning; do not copy these standalone variant settings into staged-recipes blindly.
See [distribution and release instructions](../../docs/distribution.md).
