# Troubleshooting Guide

This guide covers common issues and solutions when using tsne-h2pack.

---

## Installation Issues

### ImportError: No module named 'pynndescent'

**Cause**: PyNNDescent is an optional dependency.

**Solution**:
```bash
# Option 1: Install PyNNDescent
pip install tsne-h2pack[pynndescent]

# Option 2: Use exact k-NN instead
from tsne_h2pack import TSNE
tsne = TSNE(knn_method='exact')
```

### Module Not Found

**Cause**: Package not installed or virtual environment not activated.

**Solution**:
```bash
# Activate virtual environment
source /path/to/.venv/bin/activate

# Install package
pip install tsne-h2pack

# Or for development
pip install -e .
```

---

## macOS Issues

### Homebrew GCC not found

**Symptom**: `RuntimeError: Homebrew GCC not found`

**Solution**:
```bash
# Install Homebrew GCC and OpenBLAS
brew install gcc openblas

# Verify installation
/opt/homebrew/bin/gcc-15 --version
ls /opt/homebrew/bin/gcc-*  # Find installed version
```

### Build fails on Intel Mac

**Cause**: Native build requires Apple Silicon (ARM64).

**Options**:
1. Use Rosetta 2 (unsupported, may have issues)
2. Build on Linux VM
3. Use tsne-h2pack v2.x with Docker support

### H2Pack hangs or crashes with multiple threads

**Symptom**: Program hangs or crashes when `n_jobs > 1`

**Cause**: Build is using Apple Clang's `libomp` instead of GNU `libgomp`

**Verification**:
```bash
# Check which OpenMP library is linked
otool -L $(python -c "import tsne_h2pack._lib.libtsne_h2pack as l; print(l.__file__)") | grep omp

# BAD: libomp.dylib (LLVM OpenMP - causes race conditions)
# GOOD: libgomp.1.dylib (GNU OpenMP - works correctly)
```

**Solution**: Rebuild with Homebrew GCC:
```bash
pip install --force-reinstall --no-cache-dir tsne-h2pack
```

### GCC-15 header errors on macOS 14.4+

**Symptom**: `fatal error: _bounds.h: No such file or directory`

**Cause**: GCC-15's fixincludes doesn't handle macOS 14.4 SDK properly

**Solution**: The `setup.py` handles this automatically. For manual builds:
```bash
SDK_PATH=$(xcrun --show-sdk-path)
CPATH="$SDK_PATH/usr/include" LIBRARY_PATH="$SDK_PATH/usr/lib" \
  pip install --force-reinstall --no-cache-dir tsne-h2pack
```

### Library not found at runtime

**Solution**:
```bash
# Add library paths to environment
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:/opt/homebrew/opt/openblas/lib:$DYLD_LIBRARY_PATH
python -c "import tsne_h2pack"
```

---

## Linux Issues

### Library not found

**Solution**:
```bash
# Install OpenBLAS
sudo apt-get install libopenblas-dev  # Ubuntu/Debian
sudo yum install openblas-devel       # RHEL/CentOS
```

---

## Runtime Issues

### Backend Not Available

**Symptom**:
```python
from tsne_h2pack._backend import get_backend
backend = get_backend()
print(backend.is_available())  # False
```

**Solutions**:

1. **Verify native library was built:**
   ```bash
   ls build/lib*/tsne_h2pack/_lib/*.so
   # Should show libtsne_h2pack.so
   ```

2. **Check H2Pack library:**
   ```bash
   ls H2Pack/lib/libH2Pack.a
   # Should exist (1-2 MB)
   ```

3. **Rebuild from scratch:**
   ```bash
   pip uninstall -y tsne-h2pack
   rm -rf build/ dist/ *.egg-info
   pip install tsne-h2pack
   ```

### Script works from command line but not in virtual environment

**Cause**: Virtual environment not activated or package not installed in venv.

**Solution**:
```bash
# 1. Activate venv
source /path/to/.venv/bin/activate

# 2. Verify Python path
which python
# Should show: /path/to/.venv/bin/python

# 3. Verify package is installed
pip list | grep tsne-h2pack

# 4. If not found, reinstall
pip install tsne-h2pack
```

###  Import Error

**Symptom**: `ImportError: cannot import name 'TSNE'`

**Solution**:
```python
# Check backend status
from tsne_h2pack._backend import get_backend
backend = get_backend()
print(f"Backend type: {backend.backend_type}")  # Should be 'native'
print(f"Available: {backend.is_available()}")   # Should be True
```

---

## Performance Issues

### Slow on small datasets

**Cause**: H2Pack overhead outweighs benefits for small datasets.

**Solution**: Use exact method for datasets < 2000 samples:
```python
tsne = TSNE(method='exact')
```

### Poor clustering quality

**Try**:
1. Increase perplexity: `TSNE(perplexity=50)`
2. Increase iterations: `TSNE(max_iter=1500)`
3. Use exact k-NN: `TSNE(knn_method='exact')`
4. Lower H2Pack tolerance: `TSNE(h2_tol=1e-6)`

---

## k-NN Method Issues

### FAISS not found

```bash
pip install tsne-h2pack[faiss]
```

### Annoy not found

```bash
pip install tsne-h2pack[annoy]
```

### PyNNDescent slow or inaccurate

**Tune parameters**:
```python
tsne = TSNE(
    knn_method='pynndescent',
    knn_params={
        'n_trees': 25,  # More trees = more accurate
        'n_iters': 20   # More iterations = more accurate
    }
)
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check the documentation**:
   - [README.md](../README.md) - Quick start guide
   - [DEVELOPMENT.md](DEVELOPMENT.md) - Development guide with technical details
   - [CHANGELOG.md](../CHANGELOG.md) - Version history

2. **Search existing issues**:
   - https://github.com/xinxing02/tsne-h2pack-python/issues

3. **Report a bug**:
   - Open a new issue with:
     - Your platform (OS, Python version)
     - Package version: `python -c "import tsne_h2pack; print(tsne_h2pack.__version__)"`
     - Minimal reproduction code
     - Full error traceback

---

**Last Updated**: November 2026
**Version**: 3.3.1
