# tsne-h2pack Packaging Guide

How releases of tsne-h2pack are built and published. The pipeline is CI-driven:
building wheels and uploading to PyPI happen in GitHub Actions on a version tag —
you don't build release artifacts by hand.

## Current Status

- **PyPI**: https://pypi.org/project/tsne-h2pack/ — every release since 3.1.0 is built and
  uploaded by CI from a `v*` tag (trusted publishing, no API token).
- **Artifacts per release** (11 files):
  - sdist (~300 KB — binaries are excluded by `MANIFEST.in`)
  - manylinux x86-64 wheels for cp39–cp313
  - macOS arm64 wheels (`macosx_15_0_arm64`) for cp39–cp313
- **H2Pack** is a submodule pinned to the `tsne-h2pack` branch of the fork
  `xinxing02/H2Pack`. The commit it points at must be pushed there before CI can check it out.

---

## The Release Pipeline

Two workflows, triggered automatically:

### `test.yml` — on every push/PR to `main`
Runs the pytest matrix on `ubuntu-latest` and `macos-15` for Python 3.10 and 3.12.
Installs build deps, verifies the native backend loads, then runs `pytest -m "not slow"`.

### `wheels.yml` — on every push/PR, **and publishes on `v*` tags**
- `build_wheels` — `cibuildwheel@v3.4.1` on `ubuntu-latest` (manylinux x86-64) and
  `macos-15` (arm64). All `[tool.cibuildwheel]` config lives in `pyproject.toml`.
  cibuildwheel automatically runs `auditwheel repair` (Linux) / `delocate` (macOS),
  which bundle `libopenblas`, `libgomp`/`libgfortran`, and `libH2Pack.so` into each wheel.
- `build_sdist` — `python -m build --sdist`.
- `publish_testpypi` — only when the workflow is started by hand from the Actions tab
  with the **"Upload ... to TestPyPI"** box ticked. Same artifacts, uploaded to
  https://test.pypi.org/legacy/ (environment `testpypi`, `skip-existing: true`).
- `publish` — **only runs on a `refs/tags/v*` push.** Downloads all wheel + sdist
  artifacts and uploads them to PyPI via **trusted publishing** (OIDC — no API token).
  Uses the `pypi` GitHub Environment and `id-token: write`.

**Implication:** to cut a release you push a `v*` tag. CI does the rest.

### `scripts/release_preflight.sh` — run locally before tagging

Checks that the version agrees everywhere (`pyproject.toml`, `__init__.py`, the first
dated CHANGELOG entry, the README footer), that the tree is clean and the H2Pack
submodule commit is on its remote, runs the full pytest suite, builds sdist + wheel,
runs `twine check`, verifies the sdist has no binaries and all compiled sources,
repairs the wheel with delocate (macOS), then installs the repaired wheel and the sdist
into two fresh venvs and runs `fit_transform` from each. `--quick` skips pytest and the
venv installs. It uploads nothing.

---

## One-Time Setup

Before the first tagged release can publish, register the GitHub repo as a
**trusted publisher** on PyPI (a manual web-UI step; done once):

1. Sign in at https://pypi.org/ and open the `tsne-h2pack` project → **Publishing**.
   (If the project doesn't exist yet, add a *pending publisher* under Account →
   Publishing instead — it creates the project on first upload.)
2. Add a GitHub Actions publisher with:
   - **Owner**: `xinxing02`
   - **Repository**: `tsne-h2pack-python`
   - **Workflow name**: `wheels.yml`
   - **Environment**: `pypi`
3. In the GitHub repo, confirm a `pypi` Environment exists
   (Settings → Environments). Optionally add required reviewers to gate publishes.

Do the same on https://test.pypi.org/ for the TestPyPI dry run, with **Environment**:
`testpypi` (the workflow's `publish_testpypi` job uses that name).

The `pypi` / `testpypi` GitHub Environments are created automatically the first time a
job references them. Adding a required reviewer to `pypi` (Settings → Environments)
turns every production upload into a click-to-approve step.

See https://docs.pypi.org/trusted-publishers/ for details.

---

## Cutting a Release

The version is set in **four** places that must agree (`scripts/release_preflight.sh` checks):
- `pyproject.toml` → `[project] version`
- `tsne_h2pack/__init__.py` → `__version__`
- `CHANGELOG.md` → first dated `## [x.y.z] - YYYY-MM-DD` entry
- `README.md` → `**Version**:` footer (also `docs/TROUBLESHOOTING.md`, cosmetic)

Steps:

```bash
# 0. The H2Pack submodule commit must be on the fork's remote, or CI cannot check it out.
git -C H2Pack push origin tsne-h2pack

# 1. Bump the version in the files above; move CHANGELOG's [Unreleased] into a dated entry.
# 2. Commit and run the preflight (full: ~5 min; it builds and installs the package twice).
git commit -am "release: vX.Y.Z"
scripts/release_preflight.sh

# 3. Push main first and let CI build the release commit (Test + Build wheels, ~5 min).
git push origin main
gh run watch                                        # or the Actions tab

# 4. Optional dry run: Actions -> "Build wheels" -> Run workflow -> tick "Upload ... to
#    TestPyPI" -> Run. Then install from TestPyPI on a clean venv (below).

# 5. Tag. The tag push is what triggers the PyPI upload.
git tag -a vX.Y.Z -m "tsne-h2pack vX.Y.Z"
git push origin vX.Y.Z
```

The `wheels.yml` run for the tag builds every wheel again and, once all matrix jobs are
green, the `publish` job uploads sdist + wheels to PyPI. Watch it under **Actions**;
then check https://pypi.org/project/tsne-h2pack/ lists 11 files (10 wheels + sdist).

Smoke test against a TestPyPI install (dependencies come from real PyPI):
```bash
python -m venv /tmp/tp && source /tmp/tp/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ tsne-h2pack
python -c "import numpy as np; from tsne_h2pack import TSNE; \
           Y = TSNE(knn_method='exact').fit_transform(np.random.randn(200, 50)); \
           print('OK', Y.shape)"
```

Same against production after the tag run: drop the two `--index-url` flags.

### If a release goes wrong

PyPI never lets a file name be re-uploaded, even after deletion. A broken `3.1.0` is
fixed by releasing `3.1.1` (or yanking `3.1.0` on the project page so `pip` skips it
unless pinned). Fix, bump, preflight, push, tag again.

---

## Local Builds (for debugging)

You normally don't build release artifacts locally — CI does. But to reproduce a
build or debug packaging:

**Prerequisites**
```bash
# macOS (Apple Silicon)
brew install gcc openblas
# Linux (Ubuntu/Debian)
sudo apt-get install gcc libopenblas-dev liblapack-dev liblapacke-dev
# Both
uv pip install build twine delocate   # delocate is macOS-only
```

**Build + repair**
```bash
python -m build                              # sdist + host-platform wheel → dist/
# macOS: bundle dylibs so the wheel is portable off your machine
delocate-wheel -w wheelhouse/ -v dist/*.whl
# Linux: use cibuildwheel (auditwheel repair needs the manylinux container)
```

A raw `dist/` wheel is **not** portable — it hardcodes your local library paths.
Only delocate/auditwheel-repaired wheels (as CI produces) are fit to upload.

**Validate**
```bash
twine check dist/*.tar.gz wheelhouse/*.whl
```

---

## Scope & Deliberate Limits

Documented here so they aren't mistaken for gaps:

| Target | Status | Reason |
|--------|--------|--------|
| Linux x86-64 (manylinux) | ✅ wheels | Primary platform |
| macOS arm64 (macOS 15+) | ✅ wheels | Floor forced by the Homebrew libs bundled on the macos-15 runners; macOS 14 builds from the sdist |
| Linux aarch64 | ⛔ sdist only | H2Pack's ASTER SVE intrinsics hit the AArch64 "no SVE type in fields" ABI rule; re-enable behind a `USE_SVE` guard |
| macOS Intel | ⛔ | Unsupported since v3.0 (`setup.py` guards arm64); use Linux |
| Windows | ⛔ | Use WSL2 |
| musllinux (Alpine) | ⛔ | Limited demand; OpenBLAS setup is more involved |
| CPython 3.8 | ⛔ | EOL since Oct 2024; `requires-python = ">=3.9"` after 3.3.0. On 3.8 the uv-managed interpreter's build flags break the GCC compile, and the pynndescent the resolver picks there is broken with its numba |
| PyPy | ⛔ | C-extension, no PyPy wheels |

---

## Resources

- PyPI Packaging Guide: https://packaging.python.org/
- Trusted publishing: https://docs.pypi.org/trusted-publishers/
- `cibuildwheel`: https://cibuildwheel.pypa.io/
- `delocate` (macOS): https://github.com/matthew-brett/delocate
- `manylinux`: https://github.com/pypa/manylinux

---

**Last Updated**: 2026-11-05
