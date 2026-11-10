# Troubleshooting

## Installation

**`ImportError: No module named 'pynndescent'`**
:   Install the extra with `pip install 'tsne-h2pack[pynndescent]'`, or use `knn_method="exact"`.

**`RuntimeError: Homebrew GCC not found` on macOS**
:   The source build needs Homebrew GCC and OpenBLAS: `brew install gcc openblas`.

**`RuntimeError: ... Apple Clang ...`**
:   The `CC` environment variable points at Apple Clang. Run `unset CC` and rebuild.

**pip tries to compile tsne-h2pack and the build fails**
:   No wheel matches your platform or Python version, so pip fell back to the source distribution. Wheels exist for Linux x86-64 and Apple Silicon macOS with CPython 3.9–3.14. On those platforms with another Python version, install the build tools listed under [Building from source](installation.md#building-from-source).

**Linux on ARM, Intel Mac, or native Windows**
:   Not supported. Linux aarch64 does not build yet. Use Linux x86-64, an Apple Silicon Mac, or WSL2 on Windows.

## Runtime

**The backend reports it is unavailable**
:   The C extension did not build or cannot find its libraries. Reinstall with `pip install --force-reinstall --no-deps tsne-h2pack` and check:

    ```python
    from tsne_h2pack._backend import get_backend
    print(get_backend().is_available())
    ```

**Hangs or crashes with several threads on macOS (source builds)**
:   Check which OpenMP runtime the extension links. It must be GNU `libgomp`, not LLVM `libomp`:

    ```bash
    otool -L $(python -c "import tsne_h2pack, os; print(os.path.dirname(tsne_h2pack.__file__))")/_lib/*.so | grep omp
    ```

**Slow on every dataset**
:   `n_jobs` defaults to one thread. Set `n_jobs` to your core count, or `n_jobs=-1` for all cores.

**Slow on small datasets**
:   Below about 10K points the fixed costs dominate and any t-SNE library is fast. tsne-h2pack's advantage starts around 50K points.

**Clusters look poorly separated at large N**
:   Keep `learning_rate="auto"`. A fixed learning rate of 200 under-converges beyond about 100K points. Also check that the input is PCA-reduced and that the k-NN method is not a low-recall approximation.

**Results differ between runs with the same `random_state`**
:   Expected with `n_jobs > 1`. Use `n_jobs=1` for bit-exact reproduction. See [Reproducibility](technical/accuracy.md#reproducibility).
