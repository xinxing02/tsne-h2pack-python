#!/bin/bash
# Test installation of tsne-h2pack in a clean environment
# This script creates a temporary venv and tests package installation

set -e  # Exit on error

cd "$(dirname "$0")/.."   # run from the repository root

echo "================================================"
echo "tsne-h2pack Installation Test"
echo "================================================"
echo ""

# Locate the most recent built wheel
WHEEL=$(ls -t dist/tsne_h2pack-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ] || [ ! -f "$WHEEL" ]; then
    echo "❌ Error: No wheel found in dist/. Run scripts/build_package.sh (or python -m build) first."
    exit 1
fi
echo "✓ Found wheel: $WHEEL"

# Create temporary test directory
TEST_DIR="test_install_$(date +%s)"
echo "Creating test environment: $TEST_DIR"
mkdir -p "$TEST_DIR"
WHEEL_ABS="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
cd "$TEST_DIR"

# Create clean virtual environment
echo "Creating clean virtual environment..."
python -m venv test_venv
source test_venv/bin/activate

echo "✓ Virtual environment created"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo ""

# Install the wheel
echo "Installing tsne-h2pack from wheel: $WHEEL_ABS"
pip install "$WHEEL_ABS"

if [ $? -eq 0 ]; then
    echo "✓ Installation completed"
else
    echo "❌ Installation failed"
    deactivate
    cd ..
    rm -rf "$TEST_DIR"
    exit 1
fi

echo ""
echo "Verifying installation..."

# Test 1: Import package
echo "Test 1: Import package"
python -c "import tsne_h2pack; print('✓ Package imported successfully')" || {
    echo "❌ Failed to import package"
    deactivate
    cd ..
    rm -rf "$TEST_DIR"
    exit 1
}

# Test 2: Check version
echo "Test 2: Check version"
VERSION=$(python -c "import tsne_h2pack; print(tsne_h2pack.__version__)")
echo "  Version: $VERSION"
echo "✓ Version reported"

# Test 3: Import main classes
echo "Test 3: Import main classes"
python -c "from tsne_h2pack import TSNE; print('✓ TSNE class imported successfully')" || {
    echo "❌ Failed to import TSNE class"
    deactivate
    cd ..
    rm -rf "$TEST_DIR"
    exit 1
}

# Test 4: Check backend
echo "Test 4: Check backend"
python -c "
from tsne_h2pack._backend import get_backend
backend = get_backend()
print(f'  Backend type: {backend.backend_type}')
print(f'  Available: {backend.is_available()}')
if backend.is_available():
    print('✓ Backend is available')
else:
    print('❌ Backend not available')
    exit(1)
" || {
    echo "❌ Backend check failed"
    deactivate
    cd ..
    rm -rf "$TEST_DIR"
    exit 1
}

# Test 5: Simple smoke test (12K dataset with H2Pack)
echo "Test 5: Simple smoke test (fit_transform on 12K dataset)"
python -c "
import numpy as np
from tsne_h2pack import TSNE
import time

# Create 12K random dataset
np.random.seed(42)
n_samples = 12000
n_features = 100

print(f'  Generating {n_samples:,} x {n_features} dataset...')
X = np.random.randn(n_samples, n_features)

print('  Creating TSNE instance with H2Pack acceleration...')
# Use 'exact' k-NN method (doesn't require optional dependencies)
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=250,
    random_state=42,
    verbose=1,
    method='h2pack',        # Use H2Pack for large dataset
    knn_method='exact',     # Use exact k-NN (sklearn-based, no optional deps)
    n_jobs=4                # Use 4 threads
)

print(f'  Running fit_transform on {n_samples:,} samples...')
print('  (This may take 1-2 minutes...)')
start_time = time.time()
Y = tsne.fit_transform(X)
elapsed_time = time.time() - start_time

print(f'  Input shape: {X.shape}')
print(f'  Output shape: {Y.shape}')
print(f'  Time elapsed: {elapsed_time:.1f}s')
print(f'  KL divergence: {tsne.kl_divergence_:.4f}')

if Y.shape == (n_samples, 2):
    print(f'✓ Smoke test passed ({n_samples:,} samples in {elapsed_time:.1f}s)')
else:
    print('❌ Unexpected output shape')
    exit(1)
" || {
    echo "❌ Smoke test failed"
    deactivate
    cd ..
    rm -rf "$TEST_DIR"
    exit 1
}

echo ""
echo "================================================"
echo "All tests passed! ✓"
echo "================================================"
echo ""

# Deactivate and clean up
deactivate
cd ..
echo "Cleaning up test environment: $TEST_DIR"
rm -rf "$TEST_DIR"

echo "Installation test completed successfully!"
echo ""
