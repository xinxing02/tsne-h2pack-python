#!/bin/bash
# Release preflight for tsne-h2pack.
#
# Runs every check that has to pass BEFORE a version tag is pushed (the tag is
# what makes CI build the wheels and upload them to PyPI; see
# docs/PACKAGING_GUIDE.md). Nothing here uploads anything.
#
#   scripts/release_preflight.sh              # everything (~5 min on an M1 Pro)
#   scripts/release_preflight.sh --quick      # skip pytest and the fresh-venv installs
#   SKIP_GIT_CHECKS=1 scripts/release_preflight.sh   # exercise the build with a dirty tree
#
# Checks, in order:
#   1. version agrees in pyproject.toml, tsne_h2pack/__init__.py, CHANGELOG.md, README.md
#   2. git: clean tree, on main, H2Pack submodule commit exists on its remote
#      (otherwise `actions/checkout --recursive` fails in CI)
#   3. pytest (full suite, slow tests included)
#   4. python -m build -> sdist + wheel; twine check; the sdist carries no binaries
#      and does carry every C source setup.py compiles
#   5. macOS: delocate-wheel bundles libgomp/OpenBLAS/libH2Pack into the wheel
#   6. fresh venv: install the (repaired) wheel, import, fit_transform
#   7. fresh venv: install the sdist (source build), import, fit_transform
set -euo pipefail

cd "$(dirname "$0")/.."   # repository root
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

PY=${PYTHON:-python}
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
fail()  { red "FAIL: $*"; exit 1; }
step()  { echo; echo "== $*"; }

# ---------------------------------------------------------------- 1. version
step "version consistency"
V_TOML=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
V_INIT=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' tsne_h2pack/__init__.py)
V_CHLOG=$(grep -m1 -E '^## \[[0-9]' CHANGELOG.md | sed -E 's/^## \[([^]]+)\] - ([0-9-]+).*/\1/')
D_CHLOG=$(grep -m1 -E '^## \[[0-9]' CHANGELOG.md | sed -E 's/^## \[([^]]+)\] - ([0-9-]+).*/\2/')
V_README=$(sed -n 's/^\*\*Version\*\*: \(.*\)/\1/p' README.md)
echo "pyproject.toml   $V_TOML"
echo "__init__.py      $V_INIT"
echo "CHANGELOG.md     $V_CHLOG (dated $D_CHLOG)"
echo "README.md        $V_README"
[ "$V_TOML" = "$V_INIT" ] || fail "pyproject.toml and __init__.py disagree"
[ "$V_TOML" = "$V_CHLOG" ] || fail "CHANGELOG.md's first dated entry is not $V_TOML"
[ "$V_TOML" = "$V_README" ] || fail "README.md footer is not $V_TOML"
VERSION=$V_TOML
green "version $VERSION"

# -------------------------------------------------------------------- 2. git
step "git state"
if [ "${SKIP_GIT_CHECKS:-0}" = "1" ]; then
    red "SKIP_GIT_CHECKS=1: not checking tree cleanliness / submodule reachability"
fi
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "main" ] || red "warning: on branch '$BRANCH', not main"
if [ -n "$(git status --porcelain --ignore-submodules=none)" ]; then
    git status --short | head -20
    [ "${SKIP_GIT_CHECKS:-0}" = "1" ] || fail "working tree is not clean (commit or stash first)"
fi
SUB=$(git -C H2Pack rev-parse HEAD)
echo "H2Pack submodule at $SUB"
if [ -z "$(git -C H2Pack branch -r --contains "$SUB" 2>/dev/null)" ]; then
    echo "  fetching H2Pack remote to double-check..."
    git -C H2Pack fetch -q origin || true
fi
if [ -z "$(git -C H2Pack branch -r --contains "$SUB" 2>/dev/null)" ]; then
    MSG="H2Pack commit $SUB is on no remote branch. CI cannot check it out.
      Push it first, e.g.:  git -C H2Pack push origin $(git -C H2Pack rev-parse --abbrev-ref HEAD)"
    [ "${SKIP_GIT_CHECKS:-0}" = "1" ] && red "$MSG" || fail "$MSG"
else
    green "submodule commit reachable on $(git -C H2Pack branch -r --contains "$SUB" | head -1 | xargs)"
fi
if git tag | grep -qx "v$VERSION"; then
    red "warning: tag v$VERSION already exists locally"
fi
AHEAD=$(git rev-list --count origin/main..main 2>/dev/null || echo "?")
echo "main is $AHEAD commit(s) ahead of origin/main"

# ----------------------------------------------------------------- 3. pytest
if [ $QUICK -eq 0 ]; then
    step "pytest (full suite)"
    $PY -m pytest tests/ -q -p no:cacheprovider
    green "pytest passed"
fi

# ------------------------------------------------------------------ 4. build
step "build sdist + wheel"
$PY -c "import build, twine" 2>/dev/null || fail "install build tools:  uv pip install build twine delocate"
rm -rf build/ dist/ wheelhouse/ ./*.egg-info
if [ "$(uname -s)" = "Darwin" ]; then
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
fi
$PY -m build > /tmp/tsne_h2pack_build.log 2>&1 || { tail -30 /tmp/tsne_h2pack_build.log; fail "python -m build failed (log: /tmp/tsne_h2pack_build.log)"; }
SDIST=$(ls dist/*.tar.gz)
WHEEL=$(ls dist/*.whl)
ls -l dist/
$PY -m twine check dist/*

# the sdist must contain no build products and every compiled source
if tar tzf "$SDIST" | grep -qE '\.(so|dylib|a|o)$'; then
    tar tzf "$SDIST" | grep -E '\.(so|dylib|a|o)$'
    fail "sdist contains binaries (check MANIFEST.in)"
fi
for f in $(sed -n "s/^ *'\(src\/[A-Za-z0-9_]*\.c\)',/\1/p" setup.py | sort -u) include/tsne_h2pack.h src/tsne_attr_2d.h \
         H2Pack/src/H2Pack_build.c H2Pack/src/H2Pack_partition.c H2Pack/src/utils.c H2Pack/src/GCC-OpenBLAS.make H2Pack/LICENSE LICENSE; do
    tar tzf "$SDIST" | grep -q "^tsne_h2pack-$VERSION/$f\$" || fail "sdist is missing $f"
done
green "sdist and wheel built; twine check passed; sdist contents OK"

# --------------------------------------------------------------- 5. delocate
INSTALL_WHEEL=$WHEEL
if [ "$(uname -s)" = "Darwin" ]; then
    step "delocate-wheel"
    $PY -c "import delocate" 2>/dev/null || fail "uv pip install delocate"
    mkdir -p wheelhouse
    $PY -m delocate.cmd.delocate_wheel -w wheelhouse/ "$WHEEL" > /dev/null
    INSTALL_WHEEL=$(ls wheelhouse/*.whl)
    $PY -m delocate.cmd.delocate_listdeps "$INSTALL_WHEEL"
    for lib in libgomp libopenblas libH2Pack; do
        $PY -m delocate.cmd.delocate_listdeps "$INSTALL_WHEEL" | grep -q "$lib" || fail "$lib not bundled in the repaired wheel"
    done
    green "repaired wheel: $INSTALL_WHEEL"
fi

# ------------------------------------------------------- 6./7. fresh installs
smoke() {  # $1 = venv python; run from outside the repo so the source tree cannot shadow the install
    (cd "$TMP" && "$1" -c "
import os, numpy as np, tsne_h2pack
from tsne_h2pack._backend import get_backend
b = get_backend(); assert b.is_available(), 'native backend not available'
assert tsne_h2pack.__version__ == '$VERSION', tsne_h2pack.__version__
assert '/site-packages/' in tsne_h2pack.__file__, tsne_h2pack.__file__
from tsne_h2pack import TSNE
Y = TSNE(perplexity=5, max_iter=100, knn_method='exact', random_state=0, n_jobs=2).fit_transform(np.random.RandomState(0).randn(500, 10))
assert Y.shape == (500, 2) and np.isfinite(Y).all()
print('  import + fit_transform OK from', os.path.dirname(tsne_h2pack.__file__))
")
}
if [ $QUICK -eq 0 ]; then
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT

    step "fresh venv: install the wheel"
    $PY -m venv "$TMP/whl"
    "$TMP/whl/bin/python" -m pip install -q --upgrade pip
    "$TMP/whl/bin/python" -m pip install -q "$INSTALL_WHEEL"
    smoke "$TMP/whl/bin/python"
    green "wheel installs and runs"

    step "fresh venv: install the sdist (source build; ~1 min)"
    $PY -m venv "$TMP/sdist"
    "$TMP/sdist/bin/python" -m pip install -q --upgrade pip
    "$TMP/sdist/bin/python" -m pip install -q "$SDIST" > /tmp/tsne_h2pack_sdist_install.log 2>&1 \
        || { tail -40 /tmp/tsne_h2pack_sdist_install.log; fail "sdist install failed (log: /tmp/tsne_h2pack_sdist_install.log)"; }
    smoke "$TMP/sdist/bin/python"
    green "sdist builds from source and runs"
fi

echo
green "PREFLIGHT PASSED for v$VERSION"
cat <<EOF

Next (see docs/PACKAGING_GUIDE.md, "Cutting a Release"):
  git push origin main                      # CI: Test + Build wheels on the release commit
  # wait for green, optionally TestPyPI dry-run via Actions -> Build wheels -> Run workflow
  git tag -a v$VERSION -m "tsne-h2pack v$VERSION" && git push origin v$VERSION
EOF
