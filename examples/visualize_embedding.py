#!/usr/bin/env python3
"""
Visualize t-SNE Embedding Results

This script loads saved embeddings and labels from the output directory
and creates a comprehensive 2-panel visualization:
  - Panel 1: Full embedding with all points colored by digit class
  - Panel 2: Sampled points with distinct markers per digit

Usage:
    python visualize_embedding.py
    python visualize_embedding.py --embedding output/basic_embedding.npy --labels output/basic_labels.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def visualize_embedding(embedding_file, labels_file, output_file=None):
    """
    Visualize t-SNE embedding with labels.

    Args:
        embedding_file: Path to .npy file with shape (N, 2)
        labels_file: Path to .npy file with shape (N,)
        output_file: Optional output path for visualization
    """
    print(f"Loading embedding from: {embedding_file}")
    Y = np.load(embedding_file)

    print(f"Loading labels from: {labels_file}")
    labels = np.load(labels_file)

    print(f"Embedding shape: {Y.shape}")
    print(f"Labels shape: {labels.shape}")

    # Verify shapes match
    assert Y.shape[0] == len(labels), f"Mismatch: embedding {Y.shape[0]} vs labels {len(labels)}"
    assert Y.shape[1] == 2, f"Expected 2D embedding, got {Y.shape[1]}D"

    n_samples = Y.shape[0]
    n_classes = len(np.unique(labels))

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # =========================================================================
    # Plot 1: Full embedding with all points
    # =========================================================================
    ax1 = axes[0]

    # Use a colormap for the digit classes
    scatter = ax1.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10',
                         s=1, alpha=0.6, edgecolors='none')

    ax1.set_title(f't-SNE Embedding: Full Dataset ({n_samples:,} Points)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, ticks=range(n_classes))
    cbar.set_label('Digit Class', fontsize=11)

    ax1.grid(True, alpha=0.2)

    # =========================================================================
    # Plot 2: Sampled points with distinct markers
    # =========================================================================
    ax2 = axes[1]

    # Sample points for better visibility
    sample_size = min(5000, n_samples)
    indices = np.random.choice(len(Y), sample_size, replace=False)
    Y_sample = Y[indices]
    labels_sample = labels[indices]

    # Plot each digit class separately with different markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for digit in range(n_classes):
        mask = labels_sample == digit
        count = np.sum(mask)
        if count > 0:
            ax2.scatter(Y_sample[mask, 0], Y_sample[mask, 1],
                       c=[colors[digit]], marker=markers[digit % len(markers)],
                       s=20, alpha=0.7, label=f'{digit} (n={count})',
                       edgecolors='black', linewidths=0.5)

    ax2.set_title(f't-SNE Embedding: Random Sample ({sample_size:,} points)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    # =========================================================================
    # Add statistics box
    # =========================================================================
    stats_text = f"""
Embedding Statistics
━━━━━━━━━━━━━━━━━━━
Dataset: MNIST
Total points: {n_samples:,}
Dimensions: 784D → 2D
Classes: {n_classes}

Embedding Range:
• X: [{Y[:, 0].min():.2f}, {Y[:, 0].max():.2f}]
• Y: [{Y[:, 1].min():.2f}, {Y[:, 1].max():.2f}]
    """

    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle(f't-SNE Embedding Visualization ({n_samples:,} samples)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    if output_file is None:
        output_file = 'tsne_embedding_visualization.png'

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {output_file}")

    plt.close()


def find_latest_embedding(output_dir='output'):
    """Find the most recent embedding files in the output directory"""
    output_path = Path(output_dir)

    if not output_path.exists():
        return None, None

    # Look for embedding files
    embedding_files = list(output_path.glob('*embedding*.npy'))
    label_files = list(output_path.glob('*labels*.npy'))

    if not embedding_files:
        return None, None

    # Get most recent embedding
    latest_embedding = max(embedding_files, key=os.path.getmtime)

    # Try to find matching labels file
    latest_labels = None
    base_name = latest_embedding.stem.replace('_embedding', '_labels')
    for lf in label_files:
        if base_name in lf.stem:
            latest_labels = lf
            break

    if latest_labels is None and label_files:
        latest_labels = max(label_files, key=os.path.getmtime)

    return latest_embedding, latest_labels


def main():
    parser = argparse.ArgumentParser(
        description='Visualize t-SNE embedding results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--embedding', type=str, default=None,
                       help='Path to embedding .npy file')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels .npy file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to search for latest files')

    args = parser.parse_args()

    # Determine which files to use
    if args.embedding and args.labels:
        embedding_file = args.embedding
        labels_file = args.labels
    else:
        print("Searching for latest embedding in output directory...")
        embedding_file, labels_file = find_latest_embedding(args.output_dir)

        if embedding_file is None:
            print(f"\nError: No embedding files found in {args.output_dir}/")
            print("\nPlease run one of these scripts first:")
            print("  python basic_example.py")
            return

        if labels_file is None:
            print(f"\nError: No labels file found in {args.output_dir}/")
            print("Please specify --labels explicitly")
            return

        print(f"Found: {embedding_file}")
        print(f"Found: {labels_file}")

    # Verify files exist
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file not found: {embedding_file}")
        return

    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found: {labels_file}")
        return

    # Generate visualization
    visualize_embedding(embedding_file, labels_file, args.output)

    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
