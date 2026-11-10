#!/usr/bin/env python
"""
setup.py for tsne-h2pack

This script builds platform-specific wheels:
- Linux: Native C extension with OpenMP support
- macOS: Native C extension with Homebrew GCC and OpenMP support (Apple Silicon only)
- Windows: Not supported (use WSL2)
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy

# Determine platform
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'
IS_WINDOWS = platform.system() == 'Windows'

# Paths
HERE = Path(__file__).parent.resolve()
SRC_DIR = HERE / 'src'
INCLUDE_DIR = HERE / 'include'
H2PACK_DIR = HERE / 'H2Pack'
H2PACK_LIB_DIR = H2PACK_DIR / 'lib'
H2PACK_INCLUDE_DIR = H2PACK_DIR / 'include'


def find_homebrew_gcc():
    """Find Homebrew GCC installation on macOS

    Returns:
        str: Path to Homebrew GCC executable, or None if not found
    """
    gcc_versions = ['gcc-15', 'gcc-14', 'gcc-13', 'gcc-12']
    for gcc in gcc_versions:
        gcc_path = Path(f'/opt/homebrew/bin/{gcc}')
        if gcc_path.exists():
            print(f"Found Homebrew GCC: {gcc_path}")
            return str(gcc_path)
    return None


def verify_compiler_is_homebrew_gcc(gcc_path):
    """Verify the compiler is Homebrew GCC, not Apple Clang

    Args:
        gcc_path: Path to gcc executable

    Raises:
        RuntimeError: If compiler is not Homebrew GCC
    """
    result = subprocess.run(
        [gcc_path, '--version'],
        capture_output=True,
        text=True
    )

    if 'Apple clang' in result.stdout:
        raise RuntimeError(
            f"{gcc_path} is Apple Clang, not Homebrew GCC!\n"
            "Apple Clang has known OpenMP race conditions on Apple Silicon.\n"
            "Install Homebrew GCC with:\n"
            "  brew install gcc openblas\n"
        )

    if 'Homebrew GCC' not in result.stdout:
        raise RuntimeError(
            f"{gcc_path} is not Homebrew GCC!\n"
            "Expected 'Homebrew GCC' in version string.\n"
            f"Got: {result.stdout.split(chr(10))[0]}\n"
        )

    print(f"Verified Homebrew GCC: {result.stdout.split(chr(10))[0]}")


def verify_apple_silicon():
    """Verify we're building on Apple Silicon (ARM64)

    Raises:
        RuntimeError: If not on Apple Silicon
    """
    machine = platform.machine()
    if machine == 'arm64':
        return

    # cibuildwheel runs Python 3.8 under Rosetta on macos-14 runners
    # because CPython 3.8 has no native arm64 build, then cross-compiles
    # an arm64 wheel. In that case `platform.machine()` returns
    # 'x86_64' even though the target is arm64. Trust the cross-compile
    # signal from cibuildwheel / setuptools.
    archflags = os.environ.get('ARCHFLAGS', '')
    cibw_archs = os.environ.get('CIBW_ARCHS', '')
    host_platform = os.environ.get('_PYTHON_HOST_PLATFORM', '')
    if 'arm64' in archflags or 'arm64' in cibw_archs or 'arm64' in host_platform:
        print(f"Cross-compile detected (machine={machine}, "
              f"ARCHFLAGS={archflags!r}, target=arm64) — accepting build")
        return

    raise RuntimeError(
        f"Native macOS build only supports Apple Silicon (ARM64).\n"
        f"Current architecture: {machine}\n"
        f"Intel Mac users have these options:\n"
        f"  1. Run under Rosetta 2 (unsupported)\n"
        f"  2. Build on Linux VM\n"
        f"  3. Use tsne-h2pack v2.x with Docker support\n"
    )


def patch_h2pack_makefile():
    """Apply macOS compatibility patches to H2Pack Makefile

    Fixes:
    1. Add linker flags for shared library (-lopenblas -lgomp -lm)
    2. Add SDK library path for System library
    3. Remove GNU-specific 'cp -u' flag (not available on macOS BSD cp)
    """
    if not IS_MACOS:
        return

    common_make = H2PACK_DIR / 'src' / 'common.make'
    if not common_make.exists():
        return

    # Get SDK path for System library
    sdk_path = subprocess.run(['xcrun', '--show-sdk-path'], capture_output=True, text=True).stdout.strip()

    content = common_make.read_text()
    modified = False

    # Fix 1: Ensure SDK path is in the linker flags
    # We need to add -L{sdk_path}/usr/lib to the shared library link line
    if f'-L{sdk_path}/usr/lib' not in content:
        # Look for the shared library line with existing flags
        if '-L$(OPENBLAS_INSTALL_DIR)/lib -lopenblas -lgomp -lm' in content:
            old_pattern = '$(CC) -shared -o $@ $^ -L$(OPENBLAS_INSTALL_DIR)/lib -lopenblas -lgomp -lm'
            new_pattern = f'$(CC) -shared -o $@ $^ -L{sdk_path}/usr/lib -L$(OPENBLAS_INSTALL_DIR)/lib -lopenblas -lgomp -lm'
            content = content.replace(old_pattern, new_pattern)
            modified = True
            print(f"  - Added SDK library path: {sdk_path}/usr/lib")
        # Fallback: if the line doesn't have libraries yet, add them
        elif '$(CC) -shared -o $@ $^' in content:
            old_line = '$(CC) -shared -o $@ $^'
            new_line = f'$(CC) -shared -o $@ $^ -L{sdk_path}/usr/lib -L$(OPENBLAS_INSTALL_DIR)/lib -lopenblas -lgomp -lm'
            content = content.replace(old_line, new_line)
            modified = True
            print(f"  - Added linker flags with SDK path: {sdk_path}/usr/lib")

    # Fix 2: Remove 'cp -u' flag (GNU-specific, not available on macOS)
    if 'cp -u ' in content:
        content = content.replace('cp -u ', 'cp ')
        modified = True
        print("  - Removed GNU-specific 'cp -u' flag")

    if modified:
        common_make.write_text(content)
        print("Applied macOS compatibility patches to H2Pack Makefile")
    else:
        print("H2Pack Makefile already patched or doesn't need patching")


def fix_macos_rpath(extension_path):
    """Fix macOS rpath to load libH2Pack.so from @loader_path

    On macOS, the dynamic linker needs to know where to find libH2Pack.so.
    We copy it to the same directory as the extension and fix the rpath
    to use @loader_path (relative path).

    Args:
        extension_path: Path to the built extension .so file
    """
    if not IS_MACOS:
        return

    extension_path = Path(extension_path)
    if not extension_path.exists():
        print(f"Warning: Extension not found at {extension_path}")
        return

    lib_dir = extension_path.parent
    h2pack_so = H2PACK_LIB_DIR / 'libH2Pack.so'
    target_so = lib_dir / 'libH2Pack.so'

    # Step 1: Copy libH2Pack.so to same directory as extension
    if h2pack_so.exists():
        shutil.copy2(h2pack_so, target_so)
        print(f"Copied libH2Pack.so to {lib_dir}")
    else:
        print(f"Warning: libH2Pack.so not found at {h2pack_so}")
        return

    # Step 2: Fix rpath to use @loader_path (same directory)
    try:
        subprocess.run(
            ['install_name_tool', '-change', 'libH2Pack.so', '@loader_path/libH2Pack.so', str(extension_path)],
            check=True,
            capture_output=True
        )
        print(f"Fixed rpath for {extension_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fix rpath: {e}")
    except FileNotFoundError:
        print("Warning: install_name_tool not found (required on macOS)")


def _copy_h2pack_next_to_extension(extension_path):
    """Copy libH2Pack.so into the same directory as the built extension.

    Linux's dynamic loader resolves bare-soname dependencies via
    DT_RUNPATH / LD_LIBRARY_PATH at load time. Our extension is built
    with -Wl,-rpath,$ORIGIN-equivalent (effectively) only when packaged
    by auditwheel for a manylinux wheel. For plain editable installs
    (`pip install -e .`), neither rpath nor LD_LIBRARY_PATH points to
    H2Pack/lib/, so import fails with `cannot open shared object file:
    libH2Pack.so`. Copying it next to the extension makes ld.so find
    it via the cache lookup of the extension's load directory.
    """
    extension_path = Path(extension_path)
    target_so = extension_path.parent / 'libH2Pack.so'
    src_so = H2PACK_LIB_DIR / 'libH2Pack.so'
    if src_so.exists():
        shutil.copy2(src_so, target_so)
        print(f"Copied {src_so.name} -> {target_so}")
    else:
        print(f"Warning: {src_so} not found; runtime imports may fail")


class build_ext(_build_ext):
    """Custom build_ext command that applies macOS fixes after building"""

    def run(self):
        # Build H2Pack here (instead of at setup.py import time) so
        # that sdist creation (`python -m build --sdist`) doesn't try
        # to compile C code on machines that lack OpenBLAS / GCC.
        # macOS Makefile patches must be applied before the build.
        if IS_MACOS:
            print("Applying H2Pack Makefile patches for macOS...")
            patch_h2pack_makefile()
        build_h2pack()

        # Run the standard build
        super().run()

        # Copy libH2Pack.so next to the extension so dlopen can find it
        # via @loader_path / DT_RUNPATH. Without this, editable installs
        # on Linux fail at import with `cannot open shared object file:
        # libH2Pack.so`. macOS gets the same copy + an explicit rpath
        # rewrite via fix_macos_rpath().
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            if not Path(ext_path).exists():
                continue
            if IS_MACOS:
                fix_macos_rpath(ext_path)
            else:
                _copy_h2pack_next_to_extension(ext_path)


def build_h2pack():
    """Build H2Pack library for Linux and macOS"""
    # Set up build environment first (before checking if built)
    env = os.environ.copy()

    if IS_MACOS:
        # Find and verify Homebrew GCC
        gcc_path = find_homebrew_gcc()
        if not gcc_path:
            raise RuntimeError(
                "Homebrew GCC not found. Install with:\n"
                "  brew install gcc openblas\n"
                "\n"
                "Required for native macOS build with OpenMP support.\n"
            )
        verify_compiler_is_homebrew_gcc(gcc_path)
        env['CC'] = gcc_path
        print(f"Using compiler: {gcc_path}")
    else:
        # Linux: use system GCC or CC environment variable
        if 'CC' not in env:
            env['CC'] = 'gcc'
        print(f"Using compiler: {env['CC']}")
        # On RHEL/CentOS (i.e. cibuildwheel's manylinux container),
        # openblas-devel installs cblas.h into /usr/include/openblas/
        # rather than /usr/include/. The H2Pack Makefile only adds
        # -I/usr/include, so we need CPATH to bridge the gap.
        for extra_inc in ('/usr/include/openblas',):
            if Path(extra_inc).exists():
                env['CPATH'] = extra_inc + os.pathsep + env.get('CPATH', '')
                print(f"  Added to CPATH: {extra_inc}")

    # Check if already built
    libh2pack_a = H2PACK_LIB_DIR / 'libH2Pack.a'
    libh2pack_so = H2PACK_LIB_DIR / 'libH2Pack.so'
    if (libh2pack_a.exists() and libh2pack_a.stat().st_size > 1000000 and
        libh2pack_so.exists() and libh2pack_so.stat().st_size > 100000):  # At least 1MB for .a, 100KB for .so
        print(f"H2Pack already built:")
        print(f"  - {libh2pack_a} ({libh2pack_a.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  - {libh2pack_so} ({libh2pack_so.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    print("Building H2Pack library...")
    h2pack_src = H2PACK_DIR / 'src'

    if not h2pack_src.exists() or not (h2pack_src / 'ASTER').exists() or not (h2pack_src / 'ASTER' / 'include').exists():
        raise RuntimeError(
            "H2Pack source not found or incomplete (ASTER submodule missing).\n"
            "\n"
            "If you cloned without --recursive:\n"
            "  git submodule update --init --recursive\n"
            "\n"
            "If you downloaded as ZIP (submodules not included):\n"
            "  1. Delete the H2Pack directory\n"
            "  2. Run: git submodule update --init --recursive\n"
        )

    # Build H2Pack
    try:
        # H2Pack uses modular makefiles (GCC-OpenBLAS.make, ICC-MKL.make)
        subprocess.run(
            ['make', '-f', 'GCC-OpenBLAS.make'],
            cwd=h2pack_src,
            check=True,
            env=env
        )
        print("H2Pack built successfully")

        # Verify libraries exist
        if not libh2pack_a.exists():
            raise RuntimeError(f"H2Pack static library not found: {libh2pack_a}")
        if not libh2pack_so.exists():
            raise RuntimeError(f"H2Pack shared library not found: {libh2pack_so}")

        print(f"H2Pack libraries created:")
        print(f"  - {libh2pack_a} ({libh2pack_a.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  - {libh2pack_so} ({libh2pack_so.stat().st_size / 1024 / 1024:.1f} MB)")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build H2Pack: {e}")


def get_linux_extension_modules():
    """Build native C extension for Linux"""
    print("Platform Linux: Building native C extension")
    # NOTE: H2Pack build is deferred to build_ext.run() so that sdist
    # creation (`python -m build --sdist`) doesn't trigger a C build.

    # Source files
    sources = [
        'src/tsne_h2pack.c',
        'src/tsne_utils.c',
        'src/tsne_kernels.c',
        'src/tsne_h2_matmul.c',
        'src/tsne_reorder.c',
    ]

    # Include directories. /usr/include/openblas is where RHEL/CentOS
    # (i.e. the manylinux containers) put cblas.h via openblas-devel —
    # without it the H2Pack build fails with `cblas.h: No such file or
    # directory`. Harmless on Debian/Ubuntu where the path doesn't exist.
    include_dirs = [
        str(INCLUDE_DIR),
        str(H2PACK_INCLUDE_DIR),
        str(H2PACK_DIR / 'src'),  # For internal H2Pack headers
        '/usr/include/openblas',  # RHEL/CentOS layout
        numpy.get_include(),
    ]

    # Library directories. /usr/lib64 is where RHEL/CentOS puts the
    # 64-bit libraries (cibuildwheel's manylinux container is 64-bit
    # CentOS Stream); Debian-style /usr/lib/x86_64-linux-gnu is found
    # by the linker by default.
    library_dirs = [
        str(H2PACK_LIB_DIR),
        '/usr/lib64',
    ]

    # Libraries to link (must match reference Makefile exactly)
    # H2Pack requires: openblas, lapack, lapacke, m, pthread
    libraries = ['H2Pack', 'openblas', 'lapack', 'lapacke', 'm', 'pthread', 'gomp']

    # Compile flags. Note: NO `-march=native` — that bakes the build host's
    # CPU features into the wheel and causes "Illegal instruction" on older
    # CPUs. For PyPI wheels, allow override via $TSNE_H2PACK_MARCH (default:
    # x86-64-v3, which assumes AVX2 — safe for any CPU made since ~2013).
    _march = os.environ.get('TSNE_H2PACK_MARCH', 'x86-64-v3')
    extra_compile_args = [
        '-O3',
        f'-march={_march}',
        '-funroll-loops',
        '-ffast-math',
        '-fopenmp',
        '-std=gnu99',  # Use GNU extensions for M_LN2, M_LN10, strdup, etc.
        '-fPIC',
        '-DUSE_OPENBLAS',
        '-D_GNU_SOURCE',  # Required for strdup, M_LN2, M_LN10
        '-D_POSIX_C_SOURCE=200112L',  # For getaddrinfo, etc.
    ]

    # Link flags. -Wl,-rpath,$ORIGIN tells the dynamic loader to search
    # the directory containing the .so for its NEEDED libraries — that's
    # where build_ext copies libH2Pack.so. Without this, ld.so won't
    # find libH2Pack.so at import time even though it's right there.
    extra_link_args = [
        '-fopenmp',
        '-Wl,-rpath,$ORIGIN',
    ]

    # Create extension
    ext = Extension(
        'tsne_h2pack._lib.libtsne_h2pack',
        sources=sources,  # Use relative paths as-is
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c',
    )

    return [ext]


def get_macos_extension_modules():
    """Build native C extension for macOS (Apple Silicon only)"""
    print("Platform macOS: Building native C extension (Apple Silicon)")

    # Verify Apple Silicon
    verify_apple_silicon()

    # Find and verify Homebrew GCC
    gcc_path = find_homebrew_gcc()
    if not gcc_path:
        raise RuntimeError(
            "Homebrew GCC not found. Install with:\n"
            "  brew install gcc openblas\n"
            "\n"
            "Native macOS build requires Homebrew GCC for OpenMP support.\n"
            "Apple Clang has known race conditions with H2Pack on Apple Silicon.\n"
        )
    verify_compiler_is_homebrew_gcc(gcc_path)

    # Set CC environment variable for extension build
    os.environ['CC'] = gcc_path

    # NOTE: H2Pack build is deferred to build_ext.run() so that sdist
    # creation (`python -m build --sdist`) doesn't trigger a C build.

    # Source files
    sources = [
        'src/tsne_h2pack.c',
        'src/tsne_utils.c',
        'src/tsne_kernels.c',
        'src/tsne_h2_matmul.c',
        'src/tsne_reorder.c',
    ]

    # Include directories (Homebrew paths)
    include_dirs = [
        str(INCLUDE_DIR),
        str(H2PACK_INCLUDE_DIR),
        str(H2PACK_DIR / 'src'),  # For internal H2Pack headers
        '/opt/homebrew/opt/openblas/include',
        '/opt/homebrew/include',
        numpy.get_include(),
    ]

    # Library directories (Homebrew paths)
    library_dirs = [
        str(H2PACK_LIB_DIR),
        '/opt/homebrew/opt/openblas/lib',
        '/opt/homebrew/lib',
    ]

    # Libraries to link
    # Note: On macOS with OpenBLAS, LAPACK/LAPACKE are included in libopenblas
    libraries = ['H2Pack', 'openblas', 'm', 'pthread', 'gomp']

    # Compile flags for Apple Silicon
    # Workaround for Homebrew GCC include-fixed headers issue
    import subprocess
    sdk_path = subprocess.run(['xcrun', '--show-sdk-path'], capture_output=True, text=True).stdout.strip()
    gcc_include = subprocess.run([gcc_path, '-print-file-name=include'], capture_output=True, text=True).stdout.strip()

    # Set MACOSX_DEPLOYMENT_TARGET so the wheel is portable across macOS
    # versions. Default to 14.0 (Sonoma) so source builds work on macOS 14.
    # Release wheels set 15.0 instead (pyproject.toml / wheels.yml): the
    # Homebrew libs delocate bundles on the macos-15 CI runners require it.
    # Override via env var to match the libs you bundle.
    os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '14.0')

    # NOTE: do NOT use `-mcpu=native` — that targets the build host's
    # exact CPU and breaks the wheel for users on different Apple Silicon
    # generations. Default Apple Silicon ABI (-mcpu=apple-m1 equivalent)
    # is fine; allow override via $TSNE_H2PACK_MCPU.
    _mcpu_args = []
    _mcpu = os.environ.get('TSNE_H2PACK_MCPU')
    if _mcpu:
        _mcpu_args = [f'-mcpu={_mcpu}']

    extra_compile_args = [
        '-O3',
        *_mcpu_args,
        '-funroll-loops',
        '-ffast-math',
        '-fopenmp',
        '-std=gnu99',
        '-fPIC',
        '-DUSE_OPENBLAS',
        '-D_GNU_SOURCE',
        '-D_POSIX_C_SOURCE=200112L',
        # Fix for Homebrew GCC include-fixed headers
        '-nostdinc',
        f'-isystem{gcc_include}',
        f'-isystem{sdk_path}/usr/include',
    ]

    # Link flags with rpath for runtime library loading
    extra_link_args = [
        '-fopenmp',
        '-Wl,-rpath,/opt/homebrew/lib',
        '-Wl,-rpath,/opt/homebrew/opt/openblas/lib',
        # Add SDK library path for System library
        f'-L{sdk_path}/usr/lib',
    ]

    # Create extension
    ext = Extension(
        'tsne_h2pack._lib.libtsne_h2pack',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c',
    )

    return [ext]


def get_extension_modules():
    """Return Extension modules based on platform"""
    if IS_WINDOWS:
        raise RuntimeError(
            "Windows is not supported.\n"
            "Please use WSL2 (Windows Subsystem for Linux) instead:\n"
            "  1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install\n"
            "  2. Install Ubuntu from Microsoft Store\n"
            "  3. Inside WSL2: pip install tsne-h2pack\n"
        )

    if IS_MACOS:
        return get_macos_extension_modules()
    else:  # Linux
        return get_linux_extension_modules()


def detect_blas():
    """Detect available BLAS library"""
    # Check common locations for BLAS
    blas_options = [
        'openblas',  # Most common
        'blas',      # Generic
        'mkl_rt',    # Intel MKL
    ]

    for blas in blas_options:
        try:
            # Try to find library
            result = subprocess.run(
                ['pkg-config', '--exists', blas],
                capture_output=True
            )
            if result.returncode == 0:
                print(f"Found BLAS library: {blas}")
                return blas
        except FileNotFoundError:
            pass

    # Fallback: check if openblas is in standard locations
    openblas_paths = [
        '/usr/lib/x86_64-linux-gnu/',
        '/usr/lib64/',
        '/usr/lib/',
    ]

    for path in openblas_paths:
        if os.path.exists(os.path.join(path, 'libopenblas.so')):
            print(f"Found OpenBLAS at {path}")
            return 'openblas'

    return None


def main():
    # Get extension modules
    ext_modules = get_extension_modules()

    # Build package with custom build_ext command
    setup(
        ext_modules=ext_modules,
        cmdclass={'build_ext': build_ext},
    )


if __name__ == '__main__':
    main()
