# Binary distributions and release procedure

The repository has build and validation automation for portable wheels and conda
packages. Adding these workflows does not publish a package. PyPI project ownership,
trusted publishers, protected release environments, and conda-forge review must be
configured by the maintainers before a public release.

## Wheel coverage

`.github/workflows/wheels.yml` builds a single source distribution, then uses
**cibuildwheel 3.4.0** to build every binary wheel from that same archive. The current
matrix is CPython **3.11, 3.12, 3.13 and 3.14** on:

| Platform | Architecture | Native threading |
| --- | --- | --- |
| Linux, manylinux 2.28 | x86_64 | OpenMP, repaired wheel includes required shared runtime |
| macOS 15 build runner | arm64 | Serial compiled kernels |
| Windows 2022 build runner | AMD64 | Serial compiled kernels |

The matrix does not certify free-threaded CPython, PyPy, Linux ARM, Intel macOS,
32-bit Windows, musllinux, or a GPU runtime. Optional engine packages are installed
separately. A wheel provides the native core; it does not bundle external PES models.

Each wheel is installed in cibuildwheel's isolated test environment. The test imports
both native extensions, compares all three compiled integrators against NumPy over
multiple steps, checks Linux OpenMP, and runs `init`, `validate`, `run`, `analyze` and
`resume` using the installed quickstart. A missing extension or unexpected source-tree
import fails the build. Existing `tests.yml` also runs the full CPU/MPI matrix on
Python 3.11–3.14; `packages.yml` separately rebuilds and exercises the sdist.

The final release check requires all twelve wheels and one sdist, validates metadata
with `twine check --strict`, and saves SHA256 checksums. It does not treat successful
compilation as evidence that scientific benchmark sampling has converged.

## PyPI and TestPyPI

1. Check project-name ownership and create or claim the intended **PyRPMD** project
   on PyPI and TestPyPI. No availability or ownership is assumed by the workflow.
2. Configure GitHub environments named `pypi` and `testpypi`, restrict them to release
   tags, and require a maintainer review. In each registry, register a trusted
   publisher for owner `chen-yu-hao`, repository `PyRPMD`, workflow `wheels.yml`, and
   the matching environment name. The identity must match exactly.
3. Set repository variable `PYRPMD_ENABLE_PUBLISH=true` only after that configuration
   is reviewed. The default workflow has no credential and performs no publication.
4. Set matching versions in `pyproject.toml` and `src/pyrpmd/__init__.py`, run CI and
   the appropriate benchmark validation, then create a `vVERSION` tag. A tag builds
   artifacts without publishing them. The workflow rejects a mismatched version tag.
5. Manually dispatch **Portable wheels and release** on that tag with
   `publish_target=testpypi`. Approve the environment only after reviewing the build.
   Inspect the resulting TestPyPI artifacts and install the exact version in a fresh
   environment; install dependencies from PyPI separately when TestPyPI lacks them.
6. Dispatch the same workflow on the same reviewed tag with `publish_target=pypi`.
   The job uses short-lived OIDC credentials through PyPA's publishing action and
   uploads only the distributions that passed the wheel coverage check. There is no
   `skip-existing` setting: a previously uploaded version must be investigated,
   not silently combined with a new build.

Each dispatch rebuilds its distributions. Checksums belong to that specific run;
do not assume two builds are byte-identical. Only the publishing job receives
`id-token: write`; build jobs have read-only repository permissions. Branch builds
and pull requests cannot enter a publishing job. PyPI publishing is an explicit
release action, separate from merging this implementation.

## Conda and conda-forge

`packaging/conda/recipe.yaml` is a v1 recipe with a **local source path**, C++ compiler
and C standard-library dependencies, runtime requirements, licenses, and installed
native/CLI tests. Build it from a checkout with:

```bash
rattler-build build --recipe packaging/conda/recipe.yaml \
  --variant-config packaging/conda/variants.yaml --output-dir conda-output -c conda-forge
```

The conda workflow runs that command on Linux with Python 3.11–3.14. It saves `.conda`
artifacts without uploading them to a public channel. The local-source recipe is
useful for build testing; it is not yet a conda-forge submission.

After a public sdist exists, download the exact released archive and generate a
submission recipe with its real checksum and the usernames of consenting maintainers:

```bash
python packaging/conda/prepare_release.py --sdist dist/pyrpmd-VERSION.tar.gz \
  --url https://files.pythonhosted.org/ACTUAL_RELEASE_ARCHIVE_PATH \
  --maintainer YOUR_GITHUB_USERNAME --output staged-recipes/recipes/pyrpmd/recipe.yaml
```

Replace every capitalized placeholder with the actual value. The helper reads the
archive metadata, computes SHA256 from its bytes, replaces the local source path,
and requires an explicit maintainer; it never invents a release hash. Verify that
the public URL serves those same bytes. Review dependencies and licenses, run the
staged-recipes linter/build, and submit the generated recipe to
`conda-forge/staged-recipes`. Acceptance creates the feedstock and its publishing
automation. This repository does not bypass that community review or claim that a
`conda install -c conda-forge pyrpmd` package already exists.

## Primary references

- [cibuildwheel 3.4.0 release notes](https://cibuildwheel.pypa.io/en/v3.4.0/changelog/)
  and [build/test options](https://cibuildwheel.pypa.io/en/stable/options/).
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
  and [creating a project through OIDC](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
- [conda-forge contribution process](https://conda-forge.org/docs/maintainer/adding_pkgs/)
  and [rattler-build recipe reference](https://rattler-build.prefix.dev/latest/reference/recipe_file/).

References checked on 2026-11-01. Availability of package names and completion of
external publication have not been asserted.
