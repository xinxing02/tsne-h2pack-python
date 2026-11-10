#!/usr/bin/env python3
"""
Basic MNIST Example - t-SNE with H2Pack acceleration

This example demonstrates:
- Loading MNIST dataset with adjustable size (via random subsampling)
- Using an H² matrix for O(N) repulsive force computation
- Using exact brute-force k-NN for nearest neighbors
- Tuning H2Pack parameters (tolerance and rebuild frequency)
- Creating clear visualizations of digit clusters

All key t-SNE parameters are documented and configurable. MNIST is downloaded from
OpenML on the first run (scikit-learn caches it afterwards).
"""

import numpy as np
from tsne_h2pack import TSNE
import time
import os
import argparse


def main(n_samples=60000, random_seed=42, max_iter=500, perplexity=30,
         h2_tol=1e-2, h2_rebuild_freq=3, n_jobs=4, verbose=1):
    """
    Run t-SNE on MNIST dataset with adjustable parameters.

    Args:
        n_samples: Number of samples to use (1-60000, default: 60000 full dataset)
        random_seed: Random seed for subsampling and t-SNE reproducibility
        max_iter: Maximum optimization iterations
        perplexity: Perplexity parameter (typically 5-50)
        h2_tol: H2Pack approximation tolerance (lower=more accurate)
        h2_rebuild_freq: How often to rebuild H2 matrix (lower=more accurate)
        n_jobs: Number of parallel threads for all operations
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    """
    print("=" * 70)
    print("t-SNE on MNIST: H2Pack + Exact k-NN")
    print("=" * 70)

    # ==========================================================================
    # 1. Load and Prepare Data
    # ==========================================================================
    print("\n[1/4] Loading MNIST dataset...")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # The 60,000 training images of MNIST (OpenML's mnist_784 lists them first)
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")  # no pandas needed
    X = mnist.data[:60000].astype(np.float64)
    y = mnist.target[:60000].astype(int)

    print(f"  Loaded full dataset: {X.shape[0]} samples × {X.shape[1]} features")

    # Random subsample if requested
    if n_samples < len(X):
        np.random.seed(random_seed)
        indices = np.random.choice(len(X), n_samples, replace=False)
        # Sort indices to maintain some locality (optional, for better cache)
        indices = np.sort(indices)
        X = X[indices]
        y = y[indices]
        print(f"  Subsampled to: {n_samples} samples (random_seed={random_seed})")

    # Normalize pixel values to [0, 1] range (CRITICAL for t-SNE)
    X = X / 255.0

    print(f"  Final data shape: {X.shape}")
    print(f"  Data range: [{X.min():.2f}, {X.max():.2f}] (normalized)")
    print(f"  Memory: ~{X.nbytes / 1024**2:.1f} MB")
    print(f"  Classes: 10 digits (0-9)")
    print(f"  Distribution: {np.bincount(y)}")

    # ==========================================================================
    # 2. Configure t-SNE with Well-Documented Parameters
    # ==========================================================================
    print("\n[2/4] Configuring t-SNE...")

    print(f"  Standard t-SNE parameters:")
    print(f"    perplexity:        {perplexity}")
    print(f"      └─ Controls effective number of nearest neighbors")
    print(f"      └─ Typical range: 5-50 (larger for bigger datasets)")
    print(f"    max_iter:          {max_iter}")
    print(f"      └─ Maximum optimization iterations")
    print(f"      └─ More iterations = better convergence (but slower)")
    print(f"    learning_rate:     auto (max(N/12, 200) = {max(X.shape[0] / 12, 200):.0f})")
    print(f"      └─ Gradient descent step size")
    print(f"      └─ N/12 is the rule FIt-SNE and openTSNE use; a fixed few hundred under-converges large datasets")

    print(f"\n  H2Pack parameters (accuracy vs speed trade-off):")
    print(f"    h2_tol:            {h2_tol}")
    print(f"      └─ Low-rank approximation tolerance")
    print(f"      └─ Lower (e.g., 1e-4) = more accurate, slower")
    print(f"      └─ Higher (e.g., 1e-1) = faster, slight accuracy loss")
    print(f"    h2_rebuild_freq:   {h2_rebuild_freq}")
    print(f"      └─ Rebuild H2 matrix every N iterations")
    print(f"      └─ Lower (e.g., 1-2) = more accurate, slower")
    print(f"      └─ Higher (e.g., 5-10) = faster, slight accuracy loss")

    print(f"\n  Parallelization:")
    print(f"    n_jobs:            {n_jobs}")
    print(f"      └─ Unified threading control (v2.0.0+):")
    print(f"      └─ • k-NN search (sklearn NearestNeighbors)")
    print(f"      └─ • H2Pack matvec operations (OpenMP native)")
    print(f"      └─ • P-matvec operations (OpenMP threading)")
    print(f"      └─ All parallel operations use n_jobs threads")

    print(f"\n  Other parameters:")
    print(f"    knn_method:        'exact'")
    print(f"      └─ Use exact brute-force k-NN (matches sklearn.manifold.TSNE)")
    print(f"      └─ Options: 'exact', 'balltree', 'pynndescent', 'faiss', 'annoy'")
    print(f"    method:            'h2pack'")
    print(f"      └─ Use H2Pack for O(N) repulsive forces")
    print(f"    random_state:      {random_seed}")
    print(f"      └─ Random seed for reproducibility")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        method='h2pack',           # Use H2Pack for O(N) acceleration
        knn_method='exact',        # Use exact brute-force k-NN
        learning_rate='auto',
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_seed,
        verbose=verbose,
        h2_tol=h2_tol,             # H2Pack accuracy parameter
        h2_rebuild_freq=h2_rebuild_freq,  # H2Pack rebuild frequency
    )

    # ==========================================================================
    # 3. Run t-SNE
    # ==========================================================================
    print("\n[3/4] Running t-SNE...")
    print("  Phase 1: Exact k-NN computation (brute-force nearest neighbors)")
    print("  Phase 2: H2Pack optimization (O(N) repulsive forces)")

    start_time = time.time()
    Y = tsne.fit_transform(X)
    elapsed = time.time() - start_time

    print(f"\n  ✓ Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final KL divergence: {tsne.kl_divergence_:.4f}")
    print(f"  Iterations: {tsne.n_iter_}")
    print(f"  Per sample: {elapsed/n_samples*1000:.2f} ms")

    # ==========================================================================
    # 4. Save Results and Visualize
    # ==========================================================================
    print("\n[4/4] Saving results and creating visualizations...")

    # Save embedding
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/basic_embedding.npy", Y)
    np.save(f"{output_dir}/basic_labels.npy", y)
    print(f"  ✓ Saved embedding: {output_dir}/basic_embedding.npy")
    print(f"  ✓ Saved labels: {output_dir}/basic_labels.npy")

    # Create visualization
    import matplotlib.pyplot as plt

    # Single comprehensive plot with all digits
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in range(10):
        mask = y == digit
        count = np.sum(mask)
        ax.scatter(Y[mask, 0], Y[mask, 1],
                  c=[colors[digit]],
                  label=f'{digit} (n={count})',
                  s=3,
                  alpha=0.6,
                  edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.set_title(f't-SNE: MNIST {n_samples} (H2Pack + Exact k-NN)\n'
                f'Time: {elapsed:.1f}s, KL: {tsne.kl_divergence_:.4f}, '
                f'h2_tol={h2_tol}, rebuild_freq={h2_rebuild_freq}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9,
             title='Digit')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/basic_visualization.png",
               dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {output_dir}/basic_visualization.png")

    plt.close()


    # Print summary
    print("\n" + "=" * 70)
    print("✓ Success!")
    print("=" * 70)
    print("\nResults:")
    print(f"  • Embedding: {output_dir}/basic_embedding.npy")
    print(f"  • Labels: {output_dir}/basic_labels.npy")
    print(f"  • Visualization: {output_dir}/basic_visualization.png")
    print(f"\nPerformance:")
    print(f"  • Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  • Per sample: {elapsed/n_samples*1000:.2f} ms")
    print(f"  • KL divergence: {tsne.kl_divergence_:.4f}")
    print(f"\nParameters used:")
    print(f"  • n_samples: {n_samples}")
    print(f"  • perplexity: {perplexity}")
    print(f"  • h2_tol: {h2_tol}")
    print(f"  • h2_rebuild_freq: {h2_rebuild_freq}")
    print(f"  • n_jobs: {n_jobs}")
    print("\nTo tune performance:")
    print("  • Faster: increase h2_tol (e.g., 1e-1) or h2_rebuild_freq (e.g., 10)")
    print("  • More accurate: decrease h2_tol (e.g., 1e-4) or h2_rebuild_freq (e.g., 1)")
    print("  • Smaller dataset: use --n_samples 10000")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Basic t-SNE example on MNIST with adjustable parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--n_samples', type=int, default=60000,
                       help='Number of MNIST samples to use (1-60000)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--max_iter', type=int, default=500,
                       help='Maximum optimization iterations')
    parser.add_argument('--perplexity', type=float, default=30,
                       help='Perplexity parameter (5-50 typical)')
    parser.add_argument('--h2_tol', type=float, default=1e-2,
                       help='H2Pack approximation tolerance')
    parser.add_argument('--h2_rebuild_freq', type=int, default=3,
                       help='H2 matrix rebuild frequency')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='Number of parallel threads')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0-2)')

    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        max_iter=args.max_iter,
        perplexity=args.perplexity,
        h2_tol=args.h2_tol,
        h2_rebuild_freq=args.h2_rebuild_freq,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )
