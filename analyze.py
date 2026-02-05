"""Analysis and plotting for sparsity headroom experiments.

Generates:
- Per-layer sparsity trajectory plots
- Learning curves comparing conditions
- Correlation analysis between sparsity and loss
- Summary statistics

Usage:
    python analyze.py --phase 1 --input_dir results/phase1
    python analyze.py --phase 2 --input_dir results/phase2
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_sparsity_csv(path):
    """Load a sparsity CSV into a list of dicts."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key == 'layer_name':
                    continue
                try:
                    row[key] = float(row[key]) if '.' in str(row[key]) else int(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def load_training_csv(path):
    """Load training log CSV."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in row:
                try:
                    if row[key] and row[key] != 'None':
                        row[key] = float(row[key])
                    else:
                        row[key] = None
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def plot_phase1(input_dir, output_dir):
    """Generate Phase 1 analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all sparsity CSVs
    sparsity_files = sorted(glob.glob(os.path.join(input_dir, '*_sparsity.csv')))
    training_files = sorted(glob.glob(os.path.join(input_dir, '*_training.csv')))

    if not sparsity_files:
        print(f"No sparsity CSV files found in {input_dir}")
        return

    # Collect data across seeds
    all_sparsity = defaultdict(lambda: defaultdict(list))  # layer -> epoch -> [values]
    all_training = defaultdict(list)  # epoch -> [loss values]

    for sf in sparsity_files:
        rows = load_sparsity_csv(sf)
        for row in rows:
            layer = row['layer_name']
            epoch = row['epoch']
            all_sparsity[layer][epoch].append(row)

    for tf in training_files:
        rows = load_training_csv(tf)
        for row in rows:
            all_training[row['epoch']].append(row)

    # Get layer names (from first file)
    first_rows = load_sparsity_csv(sparsity_files[0])
    layers = []
    seen = set()
    for row in first_rows:
        if row['layer_name'] not in seen:
            seen.add(row['layer_name'])
            layers.append(row['layer_name'])

    # ---- Plot 1: Sparsity trajectories per layer (multiple epsilon thresholds) ----
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4 * len(layers)), squeeze=False)
    metrics = ['sparsity_exact', 'sparsity_1e5', 'sparsity_1e4', 'sparsity_1e3']
    labels = ['Exact zeros', '|w| < 1e-5', '|w| < 1e-4', '|w| < 1e-3']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, layer in enumerate(layers):
        ax = axes[i, 0]
        epochs_data = sorted(all_sparsity[layer].keys())

        for metric, label, color in zip(metrics, labels, colors):
            means = []
            stds = []
            for ep in epochs_data:
                vals = [r[metric] for r in all_sparsity[layer][ep]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(epochs_data, means, color=color, label=label, linewidth=2)
            ax.fill_between(epochs_data, means - stds, means + stds, color=color, alpha=0.15)

        ax.set_title(f'{layer}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sparsity Ratio')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 1: Sparsity Trajectories During Normal Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparsity_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sparsity_trajectories.png")

    # ---- Plot 2: All layers on one plot (exact zeros only) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    for i, layer in enumerate(layers):
        epochs_data = sorted(all_sparsity[layer].keys())
        means = []
        for ep in epochs_data:
            vals = [r['sparsity_exact'] for r in all_sparsity[layer][ep]]
            means.append(np.mean(vals))
        short_name = layer.split('.')[-1] if '.' in layer else layer
        ax.plot(epochs_data, means, color=cmap[i], label=f'{layer}', linewidth=2)

    ax.set_title('Exact Zero Sparsity by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sparsity (exact zeros)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparsity_all_layers.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sparsity_all_layers.png")

    # ---- Plot 3: Training curves with sparsity overlay ----
    if all_training:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        epochs_train = sorted(all_training.keys())
        train_losses = []
        val_losses = []
        val_accs = []
        for ep in epochs_train:
            tl = [r['train_loss'] for r in all_training[ep] if r.get('train_loss') is not None]
            vl = [r['val_loss'] for r in all_training[ep] if r.get('val_loss') is not None]
            va = [r['val_accuracy'] for r in all_training[ep] if r.get('val_accuracy') is not None]
            train_losses.append(np.mean(tl) if tl else None)
            val_losses.append(np.mean(vl) if vl else None)
            val_accs.append(np.mean(va) if va else None)

        ax1.plot(epochs_train, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_train, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Training Loss and Sparsity', fontsize=14, fontweight='bold')

        # Overlay mean sparsity across all layers
        ax1_twin = ax1.twinx()
        mean_sparsity = []
        for ep in epochs_train:
            ep_vals = []
            for layer in layers:
                if ep in all_sparsity[layer]:
                    ep_vals.extend([r['sparsity_exact'] for r in all_sparsity[layer][ep]])
            mean_sparsity.append(np.mean(ep_vals) if ep_vals else 0)
        ax1_twin.plot(epochs_train, mean_sparsity, 'g--', label='Mean Sparsity', linewidth=2, alpha=0.7)
        ax1_twin.set_ylabel('Mean Sparsity', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')

        ax2.plot(epochs_train, val_accs, 'g-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: training_curves.png")

    # ---- Plot 4: Weight distribution evolution ----
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3 * len(layers)), squeeze=False)
    for i, layer in enumerate(layers):
        ax = axes[i, 0]
        epochs_data = sorted(all_sparsity[layer].keys())
        # Plot mean weight magnitude and std over time
        means = [np.mean([r['weight_abs_mean'] for r in all_sparsity[layer][ep]]) for ep in epochs_data]
        stds = [np.mean([r['weight_std'] for r in all_sparsity[layer][ep]]) for ep in epochs_data]
        ax.plot(epochs_data, means, 'b-', label='Mean |weight|', linewidth=2)
        ax.plot(epochs_data, stds, 'r-', label='Weight std', linewidth=2)
        ax.set_title(f'{layer}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Weight Magnitude Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: weight_evolution.png")

    # ---- Summary statistics ----
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    for layer in layers:
        epochs_data = sorted(all_sparsity[layer].keys())
        if not epochs_data:
            continue
        initial = np.mean([r['sparsity_exact'] for r in all_sparsity[layer][epochs_data[0]]])
        final = np.mean([r['sparsity_exact'] for r in all_sparsity[layer][epochs_data[-1]]])
        min_sp = min(np.mean([r['sparsity_exact'] for r in all_sparsity[layer][ep]]) for ep in epochs_data)
        max_sp = max(np.mean([r['sparsity_exact'] for r in all_sparsity[layer][ep]]) for ep in epochs_data)
        print(f"  {layer}:")
        print(f"    Initial sparsity: {initial:.4f}")
        print(f"    Final sparsity:   {final:.4f}")
        print(f"    Min sparsity:     {min_sp:.4f}")
        print(f"    Max sparsity:     {max_sp:.4f}")
        print(f"    Change:           {final - initial:+.4f}")


def plot_phase2(input_dir, output_dir):
    """Generate Phase 2 analysis plots (capacity growth experiment)."""
    os.makedirs(output_dir, exist_ok=True)

    training_files = sorted(glob.glob(os.path.join(input_dir, '*_training.csv')))

    if not training_files:
        print(f"No training CSV files found in {input_dir}")
        return

    # Group by start_active level (filenames like "start60_seed0_training.csv")
    by_condition = defaultdict(list)
    for tf in training_files:
        basename = os.path.basename(tf)
        parts = basename.split('_')
        for part in parts:
            if part.startswith('start'):
                sa = int(part.replace('start', '')) / 100.0
                by_condition[sa].append(load_training_csv(tf))
                break

    if not by_condition:
        print("Could not parse start_active levels from filenames")
        return

    # ---- Plot 1: Learning curves comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(by_condition)))

    for idx, (sa, runs) in enumerate(sorted(by_condition.items())):
        color = colors[idx]

        epoch_data = defaultdict(lambda: {'train_loss': [], 'val_loss': [],
                                           'val_accuracy': [], 'active_fraction': []})
        for run in runs:
            for row in run:
                ep = row['epoch']
                for key in ['train_loss', 'val_loss', 'val_accuracy', 'active_fraction']:
                    if row.get(key) is not None:
                        epoch_data[ep][key].append(row[key])

        epochs = sorted(epoch_data.keys())
        label = f'Start {sa:.0%}'

        def mean_std(key):
            m = [np.mean(epoch_data[e][key]) if epoch_data[e][key] else 0 for e in epochs]
            s = [np.std(epoch_data[e][key]) if epoch_data[e][key] else 0 for e in epochs]
            return np.array(m), np.array(s)

        # Train loss
        m, s = mean_std('train_loss')
        axes[0, 0].plot(epochs, m, color=color, label=label, linewidth=2)
        axes[0, 0].fill_between(epochs, m - s, m + s, color=color, alpha=0.1)

        # Val loss
        m, s = mean_std('val_loss')
        axes[0, 1].plot(epochs, m, color=color, label=label, linewidth=2)
        axes[0, 1].fill_between(epochs, m - s, m + s, color=color, alpha=0.1)

        # Val accuracy
        m, s = mean_std('val_accuracy')
        axes[1, 0].plot(epochs, m, color=color, label=label, linewidth=2)
        axes[1, 0].fill_between(epochs, m - s, m + s, color=color, alpha=0.1)

        # Active fraction
        m, s = mean_std('active_fraction')
        axes[1, 1].plot(epochs, m, color=color, label=label, linewidth=2)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0, 0].set_title('Train Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 1].set_title('Active Weight Fraction')
    axes[1, 1].set_ylabel('Fraction')
    axes[1, 1].set_xlabel('Epoch')

    plt.suptitle('Phase 2: Capacity Growth Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'growth_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: growth_comparison.png")

    # ---- Plot 2: Final & best performance vs start_active ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    starts = []
    final_accs = []
    best_accs = []
    final_losses = []

    for sa, runs in sorted(by_condition.items()):
        starts.append(sa)
        accs = []
        bests = []
        losses = []
        for run in runs:
            if run:
                last = run[-1]
                if last.get('val_accuracy') is not None:
                    accs.append(last['val_accuracy'])
                if last.get('best_val_accuracy') is not None:
                    bests.append(last['best_val_accuracy'])
                if last.get('val_loss') is not None:
                    losses.append(last['val_loss'])
        final_accs.append((np.mean(accs), np.std(accs)) if accs else (0, 0))
        best_accs.append((np.mean(bests), np.std(bests)) if bests else (0, 0))
        final_losses.append((np.mean(losses), np.std(losses)) if losses else (0, 0))

    starts = np.array(starts)
    final_acc_m = np.array([a[0] for a in final_accs])
    final_acc_s = np.array([a[1] for a in final_accs])
    best_acc_m = np.array([a[0] for a in best_accs])
    best_acc_s = np.array([a[1] for a in best_accs])

    ax1.errorbar(starts * 100, final_acc_m, yerr=final_acc_s, fmt='o-',
                 linewidth=2, capsize=5, label='Final')
    ax1.errorbar(starts * 100, best_acc_m, yerr=best_acc_s, fmt='s--',
                 linewidth=2, capsize=5, label='Best')
    ax1.set_xlabel('Initial Active Fraction (%)')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Accuracy vs Starting Capacity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    loss_m = np.array([l[0] for l in final_losses])
    loss_s = np.array([l[1] for l in final_losses])
    ax2.errorbar(starts * 100, loss_m, yerr=loss_s, fmt='o-', linewidth=2,
                 capsize=5, color='red')
    ax2.set_xlabel('Initial Active Fraction (%)')
    ax2.set_ylabel('Final Validation Loss')
    ax2.set_title('Loss vs Starting Capacity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: final_performance.png")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"{'Start Active':>13} {'Best Acc':>10} {'Final Acc':>10} {'Final Loss':>11}")
    print("-" * 46)
    for i, sa in enumerate(starts):
        print(f"{sa:>12.0%} {best_acc_m[i]:>9.4f} {final_acc_m[i]:>9.4f} {loss_m[i]:>10.4f}")


def plot_phase4(input_dir, output_dir):
    """Generate Phase 4 analysis plots (teacher-student critical period)."""
    os.makedirs(output_dir, exist_ok=True)

    training_files = sorted(glob.glob(os.path.join(input_dir, '*_training.csv')))

    if not training_files:
        print(f"No training CSV files found in {input_dir}")
        return

    # Group by condition
    by_condition = defaultdict(list)
    for tf in training_files:
        basename = os.path.basename(tf)
        parts = basename.replace('_training.csv', '')
        tokens = parts.split('_seed')
        cond_name = tokens[0]
        by_condition[cond_name].append(load_training_csv(tf))

    CONDITION_COLORS = {
        'baseline':      '#333333',
        'early_zero':    '#2196F3',
        'early_kaiming': '#03A9F4',
        'late_zero':     '#F44336',
        'late_kaiming':  '#FF9800',
    }
    CONDITION_STYLES = {
        'baseline':      '-',
        'early_zero':    '--',
        'early_kaiming': '-',
        'late_zero':     '--',
        'late_kaiming':  '-',
    }

    def get_epoch_stats(runs, key):
        """Aggregate a metric across runs, returning (epochs, means, stds)."""
        epoch_vals = defaultdict(list)
        for run in runs:
            for row in run:
                ep = row.get('epoch')
                val = row.get(key)
                if ep is not None and val is not None:
                    epoch_vals[ep].append(val)
        epochs = sorted(epoch_vals.keys())
        means = np.array([np.mean(epoch_vals[e]) for e in epochs])
        stds = np.array([np.std(epoch_vals[e]) for e in epochs])
        return np.array(epochs), means, stds

    expansion_conds = [c for c in sorted(by_condition.keys()) if c != 'baseline']

    # ---- Plot 1: MSE over time (primary metric) ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond_name, runs in sorted(by_condition.items()):
        epochs, means, stds = get_epoch_stats(runs, 'val_mse')
        color = CONDITION_COLORS.get(cond_name, '#888888')
        style = CONDITION_STYLES.get(cond_name, '-')
        ax.plot(epochs, means, color=color, linestyle=style, label=cond_name, linewidth=2.5)
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MSE (lower = better)', fontsize=12)
    ax.set_title('Teacher-Student: Validation MSE', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_mse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: val_mse.png")

    # ---- Plot 2: Agreement accuracy over time ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond_name, runs in sorted(by_condition.items()):
        epochs, means, stds = get_epoch_stats(runs, 'val_agreement')
        color = CONDITION_COLORS.get(cond_name, '#888888')
        style = CONDITION_STYLES.get(cond_name, '-')
        ax.plot(epochs, means, color=color, linestyle=style, label=cond_name, linewidth=2.5)
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Agreement Accuracy', fontsize=12)
    ax.set_title('Teacher-Student: Agreement (argmax match)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: agreement.png")

    # ---- Plot 3: Training loss ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond_name, runs in sorted(by_condition.items()):
        epochs, means, stds = get_epoch_stats(runs, 'train_loss')
        color = CONDITION_COLORS.get(cond_name, '#888888')
        style = CONDITION_STYLES.get(cond_name, '-')
        ax.plot(epochs, means, color=color, linestyle=style, label=cond_name, linewidth=2)
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax.set_title('Teacher-Student: Training Loss', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: train_loss.png")

    # ---- Plot 4: Weight integration — new vs original weight magnitude ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, cond_name in enumerate(expansion_conds):
        ax = axes[idx // 2, idx % 2]
        runs = by_condition[cond_name]

        epochs_o, means_o, stds_o = get_epoch_stats(runs, 'agg_original_abs_mean')
        epochs_n, means_n, stds_n = get_epoch_stats(runs, 'agg_new_abs_mean')

        ax.plot(epochs_o, means_o, 'b-', label='Original weights', linewidth=2)
        ax.fill_between(epochs_o, means_o - stds_o, means_o + stds_o, color='blue', alpha=0.1)
        if len(means_n) > 0:
            ax.plot(epochs_n, means_n, 'r-', label='New weights', linewidth=2)
            ax.fill_between(epochs_n, means_n - stds_n, means_n + stds_n, color='red', alpha=0.1)

        ax.set_title(cond_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean |weight|')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Weight Integration: Original vs New Weight Magnitudes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_integration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: weight_integration.png")

    # ---- Plot 5: Gradient flow ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, cond_name in enumerate(expansion_conds):
        ax = axes[idx // 2, idx % 2]
        runs = by_condition[cond_name]

        epochs_o, means_o, stds_o = get_epoch_stats(runs, 'agg_original_grad_abs_mean')
        epochs_n, means_n, stds_n = get_epoch_stats(runs, 'agg_new_grad_abs_mean')

        ax.plot(epochs_o, means_o, 'b-', label='Original weights', linewidth=2)
        ax.fill_between(epochs_o, means_o - stds_o, means_o + stds_o, color='blue', alpha=0.1)
        if len(means_n) > 0:
            ax.plot(epochs_n, means_n, 'r-', label='New weights', linewidth=2)
            ax.fill_between(epochs_n, means_n - stds_n, means_n + stds_n, color='red', alpha=0.1)

        ax.set_title(cond_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean |gradient|')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Flow: Original vs New Weights',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gradient_flow.png")

    # ---- Plot 6: Integration ratio over time ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond_name in expansion_conds:
        runs = by_condition[cond_name]
        epoch_ratios = defaultdict(list)
        for run in runs:
            for row in run:
                ep = row.get('epoch')
                new_abs = row.get('agg_new_abs_mean')
                orig_abs = row.get('agg_original_abs_mean')
                if ep is not None and new_abs is not None and orig_abs is not None and orig_abs > 0:
                    epoch_ratios[ep].append(new_abs / orig_abs)

        if epoch_ratios:
            eps = sorted(epoch_ratios.keys())
            means = np.array([np.mean(epoch_ratios[e]) for e in eps])
            stds = np.array([np.std(epoch_ratios[e]) for e in eps])
            color = CONDITION_COLORS.get(cond_name, '#888')
            style = CONDITION_STYLES.get(cond_name, '-')
            ax.plot(eps, means, color=color, linestyle=style, label=cond_name, linewidth=2.5)
            ax.fill_between(eps, means - stds, means + stds, color=color, alpha=0.1)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Parity (1.0)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('New / Original Weight Magnitude Ratio', fontsize=12)
    ax.set_title('Weight Integration Ratio (1.0 = full integration)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'integration_ratio.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: integration_ratio.png")

    # ---- Plot 7: Final MSE bar chart ----
    summary_files = sorted(glob.glob(os.path.join(input_dir, '*_summary.json')))
    if summary_files:
        summaries = defaultdict(list)
        for sf in summary_files:
            with open(sf) as f:
                s = json.load(f)
            summaries[s['condition']].append(s)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        conds = sorted(summaries.keys())
        x = np.arange(len(conds))

        best_means = [np.mean([s['best_mse'] for s in summaries[c]]) for c in conds]
        best_stds = [np.std([s['best_mse'] for s in summaries[c]]) for c in conds]
        final_means = [np.mean([s['final_mse'] for s in summaries[c]]) for c in conds]
        final_stds = [np.std([s['final_mse'] for s in summaries[c]]) for c in conds]
        agree_means = [np.mean([s['final_agreement'] for s in summaries[c]]) for c in conds]
        agree_stds = [np.std([s['final_agreement'] for s in summaries[c]]) for c in conds]

        bar_colors = [CONDITION_COLORS.get(c, '#888') for c in conds]

        ax1.bar(x - 0.2, best_means, 0.35, yerr=best_stds, capsize=4,
                label='Best', color=bar_colors, alpha=0.8)
        ax1.bar(x + 0.2, final_means, 0.35, yerr=final_stds, capsize=4,
                label='Final', color=bar_colors, alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds, rotation=30, ha='right')
        ax1.set_ylabel('MSE (lower = better)')
        ax1.set_title('Best & Final MSE by Condition', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(x, agree_means, yerr=agree_stds, capsize=4, color=bar_colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(conds, rotation=30, ha='right')
        ax2.set_ylabel('Agreement Accuracy')
        ax2.set_title('Final Agreement (argmax match)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_performance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: final_performance.png")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY: TEACHER-STUDENT CRITICAL PERIOD")
    print("=" * 80)
    if summary_files:
        print(f"{'Condition':<18} {'Seeds':>5} {'Best MSE':>10} {'Final MSE':>10} "
              f"{'Agreement':>10} {'Pre-Exp MSE':>12}")
        print("-" * 70)
        for c in conds:
            runs = summaries[c]
            n = len(runs)
            best = np.mean([s['best_mse'] for s in runs])
            best_s = np.std([s['best_mse'] for s in runs])
            final = np.mean([s['final_mse'] for s in runs])
            agree = np.mean([s['final_agreement'] for s in runs])
            pre_exps = [s['pre_expand_mse'] for s in runs if s['pre_expand_mse'] is not None]
            pre = f"{np.mean(pre_exps):.6f}" if pre_exps else "       N/A"
            print(f"{c:<18} {n:>5} {best:>9.6f}+/-{best_s:.6f} {final:>9.6f} "
                  f"{agree:>9.4f} {pre:>12}")

        print("\nKey question: Does late_kaiming match early_kaiming?")
        if 'late_kaiming' in summaries and 'early_kaiming' in summaries:
            late_k = np.mean([s['best_mse'] for s in summaries['late_kaiming']])
            early_k = np.mean([s['best_mse'] for s in summaries['early_kaiming']])
            diff = late_k - early_k
            if abs(diff) < 0.001:
                print(f"  late_kaiming - early_kaiming = {diff:+.6f}  --> SIMILAR (no critical period)")
            elif diff > 0.001:
                print(f"  late_kaiming - early_kaiming = {diff:+.6f}  --> LATE IS WORSE (critical period!)")
            else:
                print(f"  late_kaiming - early_kaiming = {diff:+.6f}  --> LATE IS BETTER (unexpected)")

        if 'baseline' in summaries and 'late_kaiming' in summaries:
            base = np.mean([s['best_mse'] for s in summaries['baseline']])
            late_k = np.mean([s['best_mse'] for s in summaries['late_kaiming']])
            diff = late_k - base
            print(f"  late_kaiming - baseline = {diff:+.6f}  "
                  f"({'late worse' if diff > 0 else 'late better'})")


def main():
    parser = argparse.ArgumentParser(description="Analyze sparsity headroom experiments")
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir for plots (defaults to input_dir/plots)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'plots')

    if args.phase == 1:
        plot_phase1(args.input_dir, args.output_dir)
    elif args.phase == 2:
        plot_phase2(args.input_dir, args.output_dir)
    elif args.phase == 3:
        plot_phase3(args.input_dir, args.output_dir)
    elif args.phase == 4:
        plot_phase4(args.input_dir, args.output_dir)


def plot_phase3(input_dir, output_dir):
    """Generate Phase 3 analysis plots (critical period 2x2 experiment)."""
    os.makedirs(output_dir, exist_ok=True)

    training_files = sorted(glob.glob(os.path.join(input_dir, '*_training.csv')))
    integration_files = sorted(glob.glob(os.path.join(input_dir, '*_integration.csv')))

    if not training_files:
        print(f"No training CSV files found in {input_dir}")
        return

    # Group by condition
    by_condition = defaultdict(list)
    for tf in training_files:
        basename = os.path.basename(tf)
        # e.g. "early_zero_seed0_training.csv" or "baseline_seed0_training.csv"
        parts = basename.replace('_training.csv', '')
        # Extract seed number and condition name
        tokens = parts.split('_seed')
        cond_name = tokens[0]
        by_condition[cond_name].append(load_training_csv(tf))

    int_by_condition = defaultdict(list)
    for inf in integration_files:
        basename = os.path.basename(inf)
        parts = basename.replace('_integration.csv', '')
        tokens = parts.split('_seed')
        cond_name = tokens[0]
        int_by_condition[cond_name].append(load_training_csv(inf))

    CONDITION_COLORS = {
        'baseline':      '#333333',
        'early_zero':    '#2196F3',
        'early_kaiming': '#03A9F4',
        'late_zero':     '#F44336',
        'late_kaiming':  '#FF9800',
    }
    CONDITION_STYLES = {
        'baseline':      '-',
        'early_zero':    '--',
        'early_kaiming': '-',
        'late_zero':     '--',
        'late_kaiming':  '-',
    }

    def get_epoch_stats(runs, key):
        """Aggregate a metric across runs, returning (epochs, means, stds)."""
        epoch_vals = defaultdict(list)
        for run in runs:
            for row in run:
                ep = row.get('epoch')
                val = row.get(key)
                if ep is not None and val is not None:
                    epoch_vals[ep].append(val)
        epochs = sorted(epoch_vals.keys())
        means = np.array([np.mean(epoch_vals[e]) for e in epochs])
        stds = np.array([np.std(epoch_vals[e]) for e in epochs])
        return np.array(epochs), means, stds

    # ---- Plot 1: Val accuracy over time, all conditions ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond_name, runs in sorted(by_condition.items()):
        epochs, means, stds = get_epoch_stats(runs, 'val_acc')
        color = CONDITION_COLORS.get(cond_name, '#888888')
        style = CONDITION_STYLES.get(cond_name, '-')
        ax.plot(epochs, means, color=color, linestyle=style, label=cond_name,
                linewidth=2.5)
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.1)

    # Mark expansion points
    for cond_name, runs in sorted(by_condition.items()):
        if cond_name == 'baseline':
            continue
        # Find the expansion epoch (where active_fraction jumps)
        epochs, means, stds = get_epoch_stats(runs, 'active_fraction')
        for i in range(1, len(means)):
            if means[i] - means[i - 1] > 0.1:
                ax.axvline(x=epochs[i], color=CONDITION_COLORS.get(cond_name, '#888'),
                           linestyle=':', alpha=0.5, linewidth=1)
                break

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Critical Period: Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: val_accuracy.png")

    # ---- Plot 2: Train loss over time ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond_name, runs in sorted(by_condition.items()):
        epochs, means, stds = get_epoch_stats(runs, 'train_loss')
        color = CONDITION_COLORS.get(cond_name, '#888888')
        style = CONDITION_STYLES.get(cond_name, '-')
        ax.plot(epochs, means, color=color, linestyle=style, label=cond_name, linewidth=2)
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Critical Period: Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: train_loss.png")

    # ---- Plot 3: Weight integration — new vs original weight magnitude ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    expansion_conds = [c for c in sorted(by_condition.keys()) if c != 'baseline']

    for idx, cond_name in enumerate(expansion_conds):
        ax = axes[idx // 2, idx % 2]
        runs = by_condition[cond_name]

        epochs_o, means_o, stds_o = get_epoch_stats(runs, 'agg_original_abs_mean')
        epochs_n, means_n, stds_n = get_epoch_stats(runs, 'agg_new_abs_mean')

        ax.plot(epochs_o, means_o, 'b-', label='Original weights', linewidth=2)
        ax.fill_between(epochs_o, means_o - stds_o, means_o + stds_o, color='blue', alpha=0.1)
        ax.plot(epochs_n, means_n, 'r-', label='New weights', linewidth=2)
        ax.fill_between(epochs_n, means_n - stds_n, means_n + stds_n, color='red', alpha=0.1)

        ax.set_title(cond_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean |weight|')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Weight Integration: Original vs New Weight Magnitudes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_integration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: weight_integration.png")

    # ---- Plot 4: Gradient flow — new vs original ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, cond_name in enumerate(expansion_conds):
        ax = axes[idx // 2, idx % 2]
        runs = by_condition[cond_name]

        epochs_o, means_o, stds_o = get_epoch_stats(runs, 'agg_original_grad_abs_mean')
        epochs_n, means_n, stds_n = get_epoch_stats(runs, 'agg_new_grad_abs_mean')

        ax.plot(epochs_o, means_o, 'b-', label='Original weights', linewidth=2)
        ax.fill_between(epochs_o, means_o - stds_o, means_o + stds_o, color='blue', alpha=0.1)
        ax.plot(epochs_n, means_n, 'r-', label='New weights', linewidth=2)
        ax.fill_between(epochs_n, means_n - stds_n, means_n + stds_n, color='red', alpha=0.1)

        ax.set_title(cond_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean |gradient|')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Flow: Original vs New Weights',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gradient_flow.png")

    # ---- Plot 5: Final accuracy bar chart ----
    summary_files = sorted(glob.glob(os.path.join(input_dir, '*_summary.json')))
    if summary_files:
        summaries = defaultdict(list)
        for sf in summary_files:
            with open(sf) as f:
                s = json.load(f)
            summaries[s['condition']].append(s)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        conds = sorted(summaries.keys())
        x = np.arange(len(conds))

        best_means = [np.mean([s['best_val_acc'] for s in summaries[c]]) for c in conds]
        best_stds = [np.std([s['best_val_acc'] for s in summaries[c]]) for c in conds]
        final_means = [np.mean([s['final_val_acc'] for s in summaries[c]]) for c in conds]
        final_stds = [np.std([s['final_val_acc'] for s in summaries[c]]) for c in conds]
        gap_means = [np.mean([s['generalization_gap'] for s in summaries[c]]) for c in conds]
        gap_stds = [np.std([s['generalization_gap'] for s in summaries[c]]) for c in conds]

        bar_colors = [CONDITION_COLORS.get(c, '#888') for c in conds]

        bars1 = ax1.bar(x - 0.2, best_means, 0.35, yerr=best_stds, capsize=4,
                        label='Best', color=bar_colors, alpha=0.8)
        bars2 = ax1.bar(x + 0.2, final_means, 0.35, yerr=final_stds, capsize=4,
                        label='Final', color=bar_colors, alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds, rotation=30, ha='right')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Best & Final Accuracy by Condition', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(x, gap_means, yerr=gap_stds, capsize=4, color=bar_colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(conds, rotation=30, ha='right')
        ax2.set_ylabel('Train Acc - Val Acc')
        ax2.set_title('Generalization Gap (lower = better)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_accuracy.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: final_accuracy.png")

        # ---- Plot 6: New/Original weight ratio over time ----
        fig, ax = plt.subplots(figsize=(12, 7))
        for cond_name in expansion_conds:
            runs = by_condition[cond_name]
            # Compute ratio
            epoch_ratios = defaultdict(list)
            for run in runs:
                for row in run:
                    ep = row.get('epoch')
                    new_abs = row.get('agg_new_abs_mean')
                    orig_abs = row.get('agg_original_abs_mean')
                    if ep is not None and new_abs is not None and orig_abs is not None and orig_abs > 0:
                        epoch_ratios[ep].append(new_abs / orig_abs)

            if epoch_ratios:
                eps = sorted(epoch_ratios.keys())
                means = np.array([np.mean(epoch_ratios[e]) for e in eps])
                stds = np.array([np.std(epoch_ratios[e]) for e in eps])
                color = CONDITION_COLORS.get(cond_name, '#888')
                style = CONDITION_STYLES.get(cond_name, '-')
                ax.plot(eps, means, color=color, linestyle=style, label=cond_name, linewidth=2.5)
                ax.fill_between(eps, means - stds, means + stds, color=color, alpha=0.1)

        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Parity (1.0)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('New / Original Weight Magnitude Ratio', fontsize=12)
        ax.set_title('Weight Integration Ratio (1.0 = full integration)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'integration_ratio.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: integration_ratio.png")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY: CRITICAL PERIOD EXPERIMENT")
    print("=" * 80)
    if summary_files:
        print(f"{'Condition':<18} {'Seeds':>5} {'Best Acc':>10} {'Final Acc':>10} "
              f"{'Gen Gap':>8} {'Pre-Exp':>8}")
        print("-" * 70)
        for c in conds:
            runs = summaries[c]
            n = len(runs)
            best = np.mean([s['best_val_acc'] for s in runs])
            best_s = np.std([s['best_val_acc'] for s in runs])
            final = np.mean([s['final_val_acc'] for s in runs])
            gap = np.mean([s['generalization_gap'] for s in runs])
            pre_exps = [s['pre_expand_val_acc'] for s in runs if s['pre_expand_val_acc'] is not None]
            pre = f"{np.mean(pre_exps):.4f}" if pre_exps else "  N/A"
            print(f"{c:<18} {n:>5} {best:>9.4f}±{best_s:.4f} {final:>9.4f} "
                  f"{gap:>+7.4f} {pre:>8}")

        print("\nKey question: Does late_kaiming match early_kaiming?")
        if 'late_kaiming' in summaries and 'early_kaiming' in summaries:
            late_k = np.mean([s['best_val_acc'] for s in summaries['late_kaiming']])
            early_k = np.mean([s['best_val_acc'] for s in summaries['early_kaiming']])
            diff = late_k - early_k
            if abs(diff) < 0.01:
                print(f"  late_kaiming - early_kaiming = {diff:+.4f}  --> SIMILAR (no critical period)")
            elif diff < -0.01:
                print(f"  late_kaiming - early_kaiming = {diff:+.4f}  --> LATE IS WORSE (critical period!)")
            else:
                print(f"  late_kaiming - early_kaiming = {diff:+.4f}  --> LATE IS BETTER (unexpected)")

        if 'late_kaiming' in summaries and 'late_zero' in summaries:
            late_k = np.mean([s['best_val_acc'] for s in summaries['late_kaiming']])
            late_z = np.mean([s['best_val_acc'] for s in summaries['late_zero']])
            diff = late_k - late_z
            print(f"  late_kaiming - late_zero = {diff:+.4f}", end="")
            if diff > 0.01:
                print("  --> Kaiming helps (zero init is the problem)")
            elif abs(diff) < 0.01:
                print("  --> No difference (both fail or both succeed)")
            else:
                print("  --> Zero actually better? (surprising)")


if __name__ == '__main__':
    main()
