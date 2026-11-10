#!/bin/bash
# Build script for tsne-h2pack package
# This script builds both source distribution and wheel

set -e  # Exit on error

cd "$(dirname "$0")/.."   # run from the repository root

echo "================================================"
echo "tsne-h2pack Package Build Script"
echo "================================================"
echo ""

# Check we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found. Run this script from the repository root."
    exit 1
fi

if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the repository root."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)
echo "✓ Platform: $PLATFORM ($ARCH)"

# Platform-specific checks
if [ "$PLATFORM" == "Darwin" ]; then
    echo ""
    echo "macOS build detected - checking prerequisites..."

    # Check for Homebrew GCC. Mirror setup.py's search list: 15, 14, 13, 12.
    GCC_FOUND=""
    for ver in 15 14 13 12; do
        if [ -f "/opt/homebrew/bin/gcc-${ver}" ]; then
            GCC_FOUND="/opt/homebrew/bin/gcc-${ver}"
            break
        fi
    done
    if [ -z "$GCC_FOUND" ]; then
        echo "❌ Error: Homebrew GCC not found (looked for gcc-15 down to gcc-12)"
        echo "   Install with: brew install gcc openblas"
        exit 1
    fi
    GCC_VERSION=$("$GCC_FOUND" --version | head -1)
    echo "✓ Found Homebrew GCC: $GCC_VERSION"

    # Check for OpenBLAS
    if [ -f "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" ]; then
        echo "✓ Found OpenBLAS"
    else
        echo "❌ Error: OpenBLAS not found"
        echo "   Install with: brew install openblas"
        exit 1
    fi

    # Check for Apple Silicon
    if [ "$ARCH" != "arm64" ]; then
        echo "⚠️  Warning: Native build requires Apple Silicon (arm64)"
        echo "   Current architecture: $ARCH"
        echo "   Continuing anyway, but build may fail..."
    fi
fi

echo ""
echo "Checking H2Pack submodule..."
if [ ! -d "H2Pack/src" ]; then
    echo "❌ Error: H2Pack submodule not initialized"
    echo "   Run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "H2Pack/src/H2Pack.h" ]; then
    echo "❌ Error: H2Pack source files missing"
    echo "   Run: git submodule update --init --recursive"
    exit 1
fi
echo "✓ H2Pack submodule present"

# Check if H2Pack is already built
if [ -f "H2Pack/lib/libH2Pack.a" ] && [ -f "H2Pack/lib/libH2Pack.so" ]; then
    echo "✓ H2Pack library already built"
else
    echo "⚠️  H2Pack library not built - setup.py will build it"
fi

echo ""
echo "================================================"
echo "Starting build process..."
echo "================================================"
echo ""

# Clean previous builds
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info
echo "✓ Cleaned"

echo ""
echo "Building sdist + wheel via 'python -m build'..."
if ! python -c "import build" >/dev/null 2>&1; then
    echo "  Installing 'build' package..."
    pip install --quiet build
fi

# On macOS, exporting MACOSX_DEPLOYMENT_TARGET *before* python is invoked
# is what controls the wheel-filename tag (sysconfig.get_platform reads
# this at interpreter startup). Setting it inside setup.py is too late
# for the wheel name (though it does affect the .so binary tags).
if [ "$PLATFORM" == "Darwin" ]; then
    # The floor depends on the macOS version of the build host: each
    # Homebrew lib (libgomp, libopenblas, libgfortran, libquadmath,
    # libomp) inherits the host's MACOSX_DEPLOYMENT_TARGET when built.
    # macos-14 CI runners produce libs requiring 14.0; earlier hosts
    # may be lower. We default to 14.0 to match CI; override locally
    # if you're building against older Homebrew libs.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    echo "  MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
fi

python -m build
if [ $? -eq 0 ]; then
    echo "✓ sdist and wheel built successfully"
else
    echo "❌ Build failed"
    exit 1
fi

# On macOS, run delocate-wheel to bundle libgomp/libopenblas/libH2Pack
# into the wheel so it works on machines without Homebrew.
if [ "$PLATFORM" == "Darwin" ]; then
    if ! python -c "import delocate" >/dev/null 2>&1; then
        echo ""
        echo "Installing delocate to repair the wheel..."
        pip install --quiet delocate
    fi
    echo ""
    echo "Repairing wheel with delocate-wheel (bundling Homebrew deps)..."
    mkdir -p wheelhouse
    delocate-wheel -w wheelhouse/ -v dist/*.whl
    if [ $? -eq 0 ]; then
        echo "✓ Wheel repaired and copied to wheelhouse/"
        ls -lh wheelhouse/
    else
        echo "❌ delocate-wheel failed"
        exit 1
    fi
fi

echo ""
echo "================================================"
echo "Build completed successfully!"
echo "================================================"
echo ""
echo "Distribution files created:"
ls -lh dist/
echo ""
echo "Next steps:"
echo "  1. Test installation: pip install dist/*.whl"
echo "  2. Run smoke tests: python -c 'import tsne_h2pack; print(tsne_h2pack.__version__)'"
echo "  3. Upload to Test PyPI: twine upload --repository testpypi dist/*"
echo "  4. Upload to PyPI: twine upload dist/*"
echo ""
