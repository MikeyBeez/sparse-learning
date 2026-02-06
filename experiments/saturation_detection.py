"""Experiment: Detecting First-Layer Saturation and Triggering Zero-Expansion.

This experiment investigates whether we can detect when a layer has saturated
its current capacity and automatically trigger zero-expansion at the right moment.

Phases:
1. Characterize saturation signals (gradient norm, Jacobian sensitivity, weight change)
2. Define trigger thresholds from Phase 1 data
3. Implement and test automatic expansion
4. Compare auto-expand vs fixed-timing baselines

Usage:
    # Phase 1: Characterize saturation signals
    python3 experiments/saturation_detection.py --phase 1 --epochs 200 --seeds 3

    # Phase 3: Run auto-expansion experiment
    python3 experiments/saturation_detection.py --phase 3 --epochs 200 --seeds 5
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SimpleMLP, MaskedMLP, MaskedLinear


DEFAULTS = {
    'input_dim': 100,
    'hidden_dims': (256, 128),
    'output_dim': 10,
    'epochs': 200,
    'train_samples': 50000,
    'val_samples': 10000,
    'batch_size': 256,
    'teacher_seed': 42,
    'lr': 1e-3,
    'start_active': 0.6,
    'hutchinson_samples': 5,  # Number of random vectors for Jacobian estimation
}


def create_teacher(cfg, device):
    """Create a frozen teacher network with fixed random weights."""
    torch.manual_seed(cfg['teacher_seed'])
    teacher = SimpleMLP(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        output_dim=cfg['output_dim']
    )
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def create_student(cfg, device, first_layer_active=None, other_layers_active=None):
    """Create a student network with configurable per-layer capacity.

    Args:
        cfg: config dict
        device: torch device
        first_layer_active: fraction of first layer active (default: cfg['start_active'])
        other_layers_active: fraction of other layers active (default: 1.0)
    """
    if first_layer_active is None:
        first_layer_active = cfg['start_active']
    if other_layers_active is None:
        other_layers_active = 1.0

    # Create with full capacity first
    student = MaskedMLP(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        output_dim=cfg['output_dim'],
        oversized_factor=1.0,
        initial_active_fraction=1.0  # Start full, then mask
    )

    # Now adjust capacity per layer
    for i, layer in enumerate(student.layers):
        if isinstance(layer, MaskedLinear):
            if i == 0:
                target_active = first_layer_active
            else:
                target_active = other_layers_active

            # Reset mask to desired fraction
            total = layer.mask.numel()
            num_active = int(target_active * total)
            layer.mask.fill_(0)
            flat_indices = torch.randperm(total)[:num_active]
            layer.mask.view(-1)[flat_indices] = 1.0
            layer.expansion_mask.fill_(0)

            # Zero out inactive weights
            with torch.no_grad():
                layer.weight.data *= layer.mask

    return student.to(device)


def generate_data(teacher, n_samples, input_dim, device):
    """Generate synthetic data: random inputs → teacher outputs."""
    with torch.no_grad():
        x = torch.randn(n_samples, input_dim, device=device)
        y = teacher(x)
    return x, y


@torch.no_grad()
def evaluate(student, val_x, val_y):
    """Evaluate student on fixed validation set."""
    student.eval()
    outputs = student(val_x)
    mse = ((outputs - val_y) ** 2).mean().item()
    agreement = (outputs.argmax(dim=1) == val_y.argmax(dim=1)).float().mean().item()
    return mse, agreement


def compute_per_layer_gradient_stats(model, x, y, criterion):
    """Compute gradient norm mean and variance per layer.

    Returns dict mapping layer name -> {grad_norm_mean, grad_norm_var, grad_norm_per_sample}
    """
    model.train()
    batch_size = x.size(0)

    # We need per-sample gradients to compute variance
    # Use a loop over samples (slower but accurate)
    layer_grads = defaultdict(list)

    for i in range(batch_size):
        model.zero_grad()
        out = model(x[i:i+1])
        loss = criterion(out, y[i:i+1])
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, MaskedLinear)):
                if module.weight.grad is not None:
                    grad_norm = module.weight.grad.norm().item()
                    layer_grads[name].append(grad_norm)

    stats = {}
    for name, grads in layer_grads.items():
        grads = np.array(grads)
        stats[name] = {
            'grad_norm_mean': grads.mean(),
            'grad_norm_var': grads.var(),
            'grad_norm_std': grads.std(),
        }

    return stats


def compute_jacobian_sensitivity_hutchinson(model, x, layer_name, n_vectors=5):
    """Estimate Jacobian Frobenius norm using Hutchinson estimator.

    For a layer with input h and output g(h), we estimate ||dg/dh||_F^2
    using: E[||J @ v||^2] where v ~ Rademacher.

    Returns mean and variance across the batch.
    """
    model.eval()
    batch_size = x.size(0)

    # Get the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    if target_layer is None:
        return {'jacobian_mean': 0.0, 'jacobian_var': 0.0}

    # Hook to capture layer input
    layer_input = [None]
    def input_hook(module, inp, out):
        layer_input[0] = inp[0]

    handle = target_layer.register_forward_hook(input_hook)

    # Forward pass to get layer input
    with torch.no_grad():
        _ = model(x)
    h = layer_input[0].detach().requires_grad_(True)

    handle.remove()

    # Now compute Jacobian-vector products
    jacobian_norms = []

    for sample_idx in range(min(batch_size, 100)):  # Limit to 100 samples for speed
        h_sample = h[sample_idx:sample_idx+1].clone().requires_grad_(True)

        # Recompute layer output for this sample
        if isinstance(target_layer, MaskedLinear):
            out = torch.nn.functional.linear(h_sample, target_layer.weight * target_layer.mask, target_layer.bias)
        else:
            out = target_layer(h_sample)

        # Hutchinson estimator
        jac_norm_sq_estimates = []
        for _ in range(n_vectors):
            # Rademacher random vector
            v = torch.randint(0, 2, out.shape, device=out.device).float() * 2 - 1

            # Compute J^T @ v via backward
            grad_h = torch.autograd.grad(out, h_sample, grad_outputs=v, retain_graph=True)[0]

            # ||J @ v||^2 ≈ ||J^T @ v||^2 for our estimation purposes
            jac_norm_sq_estimates.append((grad_h ** 2).sum().item())

        jacobian_norms.append(np.mean(jac_norm_sq_estimates) ** 0.5)

    jacobian_norms = np.array(jacobian_norms)
    return {
        'jacobian_mean': jacobian_norms.mean(),
        'jacobian_var': jacobian_norms.var(),
        'jacobian_std': jacobian_norms.std(),
    }


def compute_weight_change_rate(model, prev_weights):
    """Compute L2 norm of weight changes per layer.

    Args:
        model: current model
        prev_weights: dict mapping layer name -> previous weight tensor

    Returns dict mapping layer name -> weight_change_rate
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, MaskedLinear)):
            if name in prev_weights:
                with torch.no_grad():
                    if isinstance(module, MaskedLinear):
                        curr = module.weight.data * module.mask
                        prev = prev_weights[name] * module.mask
                    else:
                        curr = module.weight.data
                        prev = prev_weights[name]

                    delta = (curr - prev).norm().item()
                    stats[name] = {'weight_change_rate': delta}
    return stats


def get_layer_weights(model):
    """Get a copy of all layer weights."""
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, MaskedLinear)):
            weights[name] = module.weight.data.clone()
    return weights


def train_one_epoch_with_stats(student, teacher, optimizer, criterion, device, cfg,
                                compute_jacobian=True, prev_weights=None):
    """Train one epoch and compute saturation statistics.

    Returns:
        train_loss: average training loss
        layer_stats: dict of per-layer saturation metrics
    """
    student.train()
    n_samples = cfg['train_samples']
    batch_size = cfg['batch_size']
    input_dim = cfg['input_dim']

    # Generate training data
    with torch.no_grad():
        all_x = torch.randn(n_samples, input_dim, device=device)
        all_y = teacher(all_x)

    total_loss = 0.0
    n_batches = 0

    # Train
    perm = torch.randperm(n_samples, device=device)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i + batch_size]
        x = all_x[idx]
        y = all_y[idx]

        optimizer.zero_grad()
        output = student(x)
        loss = criterion(output, y)
        loss.backward()

        # Zero gradients for masked weights
        for module in student.modules():
            if isinstance(module, MaskedLinear):
                if module.weight.grad is not None:
                    module.weight.grad.data *= module.mask

        optimizer.step()

        # Re-zero masked weights
        with torch.no_grad():
            for module in student.modules():
                if isinstance(module, MaskedLinear):
                    module.weight.data *= module.mask

        total_loss += loss.item()
        n_batches += 1

    train_loss = total_loss / n_batches

    # Compute saturation statistics on a subset of data
    student.eval()
    stat_batch_size = min(256, n_samples)
    stat_x = all_x[:stat_batch_size]
    stat_y = all_y[:stat_batch_size]

    # Gradient statistics
    grad_stats = compute_per_layer_gradient_stats(student, stat_x, stat_y, criterion)

    # Jacobian statistics (expensive, so optional)
    jac_stats = {}
    if compute_jacobian:
        for name, module in student.named_modules():
            if isinstance(module, (nn.Linear, MaskedLinear)):
                jac_stats[name] = compute_jacobian_sensitivity_hutchinson(
                    student, stat_x, name, n_vectors=cfg.get('hutchinson_samples', 5)
                )

    # Weight change rate
    weight_stats = {}
    if prev_weights is not None:
        weight_stats = compute_weight_change_rate(student, prev_weights)

    # Combine all stats
    layer_stats = {}
    for name, module in student.named_modules():
        if isinstance(module, (nn.Linear, MaskedLinear)):
            layer_stats[name] = {}
            if name in grad_stats:
                layer_stats[name].update(grad_stats[name])
            if name in jac_stats:
                layer_stats[name].update(jac_stats[name])
            if name in weight_stats:
                layer_stats[name].update(weight_stats[name])

            # Add capacity info
            if isinstance(module, MaskedLinear):
                layer_stats[name]['active_fraction'] = module.active_fraction

    return train_loss, layer_stats


def expand_first_layer(model, init_method='zero'):
    """Expand only the first layer to 100% capacity."""
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            inactive = int((module.mask == 0).sum().item())
            if inactive > 0:
                n = module.unmask_weights(inactive, init_method=init_method)
                return {name: n}
            break  # Only first MaskedLinear
    return {}


def check_saturation_trigger(layer_stats, layer_name, trigger_cfg, window_stats=None):
    """Check if a layer has saturated based on the trigger configuration.

    Args:
        layer_stats: current epoch stats for all layers
        layer_name: which layer to check
        trigger_cfg: dict with threshold values
        window_stats: list of recent layer_stats for rolling window

    Returns:
        (triggered, reason_string)
    """
    if layer_name not in layer_stats:
        return False, "layer not found"

    stats = layer_stats[layer_name]

    # Use rolling window if available
    if window_stats and len(window_stats) >= trigger_cfg.get('window_size', 10):
        recent = window_stats[-trigger_cfg.get('window_size', 10):]
        grad_means = [s.get(layer_name, {}).get('grad_norm_mean', float('inf')) for s in recent]
        grad_vars = [s.get(layer_name, {}).get('grad_norm_var', float('inf')) for s in recent]
        avg_grad_mean = np.mean(grad_means)
        avg_grad_var = np.mean(grad_vars)
    else:
        avg_grad_mean = stats.get('grad_norm_mean', float('inf'))
        avg_grad_var = stats.get('grad_norm_var', float('inf'))

    # Gradient-based trigger
    if trigger_cfg.get('use_gradient', True):
        grad_threshold = trigger_cfg.get('grad_threshold', 0.01)
        var_threshold = trigger_cfg.get('var_threshold', 0.001)

        if avg_grad_mean < grad_threshold and avg_grad_var < var_threshold:
            return True, f"grad_mean={avg_grad_mean:.6f} < {grad_threshold}, var={avg_grad_var:.6f} < {var_threshold}"

    # Jacobian-based trigger
    if trigger_cfg.get('use_jacobian', False):
        jac_threshold = trigger_cfg.get('jac_threshold', 0.1)
        jac_mean = stats.get('jacobian_mean', float('inf'))

        if jac_mean < jac_threshold:
            return True, f"jacobian_mean={jac_mean:.6f} < {jac_threshold}"

    return False, "not saturated"


def run_phase1(cfg, device, output_dir):
    """Phase 1: Characterize saturation signals at 60% capacity (no expansion)."""
    os.makedirs(output_dir, exist_ok=True)

    teacher = create_teacher(cfg, device)

    # Fixed validation set
    torch.manual_seed(cfg['teacher_seed'] + 1000)
    val_x, val_y = generate_data(teacher, cfg['val_samples'], cfg['input_dim'], device)

    all_results = []

    for seed in range(cfg['seeds']):
        print(f"\n{'='*70}")
        print(f"Phase 1: Saturation characterization | Seed {seed}")
        print(f"{'='*70}")

        torch.manual_seed(seed)

        # Student at 60% first layer, 100% elsewhere
        student = create_student(cfg, device, first_layer_active=0.6, other_layers_active=1.0)
        optimizer = optim.Adam(student.parameters(), lr=cfg['lr'])
        criterion = nn.MSELoss()

        epoch_stats = []
        prev_weights = None

        for epoch in range(1, cfg['epochs'] + 1):
            t0 = time.time()

            train_loss, layer_stats = train_one_epoch_with_stats(
                student, teacher, optimizer, criterion, device, cfg,
                compute_jacobian=(epoch % 10 == 0),  # Jacobian every 10 epochs (expensive)
                prev_weights=prev_weights
            )

            prev_weights = get_layer_weights(student)

            val_mse, val_agree = evaluate(student, val_x, val_y)
            elapsed = time.time() - t0

            # Flatten layer stats for CSV
            row = {
                'seed': seed,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mse': val_mse,
                'val_agreement': val_agree,
                'elapsed': elapsed,
            }
            for layer_name, lstats in layer_stats.items():
                for key, val in lstats.items():
                    row[f'{layer_name}_{key}'] = val

            epoch_stats.append(row)

            if epoch % 20 == 0 or epoch == 1:
                first_layer = list(layer_stats.keys())[0]
                fl_stats = layer_stats[first_layer]
                print(f"  Epoch {epoch:3d} | loss={train_loss:.6f} | mse={val_mse:.6f} | "
                      f"L0_grad={fl_stats.get('grad_norm_mean', 0):.6f} | "
                      f"L0_grad_var={fl_stats.get('grad_norm_var', 0):.6f}")

        all_results.extend(epoch_stats)

        # Save per-seed results
        seed_path = os.path.join(output_dir, f'saturation_seed{seed}.csv')
        if epoch_stats:
            all_keys = set()
            for row in epoch_stats:
                all_keys.update(row.keys())
            fieldnames = sorted(all_keys, key=lambda x: (x != 'epoch', x != 'seed', x))
            with open(seed_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(epoch_stats)
        print(f"  Saved: {seed_path}")

    # Save combined results
    combined_path = os.path.join(output_dir, 'saturation_all.csv')
    if all_results:
        all_keys = set()
        for row in all_results:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys, key=lambda x: (x != 'epoch', x != 'seed', x))
        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nSaved combined results: {combined_path}")

    return all_results


def analyze_phase1_for_thresholds(results, target_mse=0.0005):
    """Analyze Phase 1 results to determine trigger thresholds.

    Look for the epoch where the first layer's gradient metrics plateau
    while loss is still above target.
    """
    import pandas as pd

    df = pd.DataFrame(results)

    # Find first layer name
    grad_cols = [c for c in df.columns if 'grad_norm_mean' in c]
    if not grad_cols:
        print("No gradient columns found!")
        return {}

    first_layer_col = sorted(grad_cols)[0]
    layer_name = first_layer_col.replace('_grad_norm_mean', '')

    print(f"\nAnalyzing saturation for layer: {layer_name}")

    # Group by epoch (across seeds)
    grouped = df.groupby('epoch').agg({
        'val_mse': ['mean', 'std'],
        first_layer_col: ['mean', 'std'],
        f'{layer_name}_grad_norm_var': ['mean', 'std'] if f'{layer_name}_grad_norm_var' in df.columns else None,
    })

    # Find where gradient norm stabilizes but MSE is above target
    stable_epochs = []
    for epoch in grouped.index:
        grad_mean = grouped.loc[epoch, (first_layer_col, 'mean')]
        mse = grouped.loc[epoch, ('val_mse', 'mean')]

        if mse > target_mse and epoch > 20:  # Still improving, past warmup
            stable_epochs.append({
                'epoch': epoch,
                'grad_mean': grad_mean,
                'mse': mse,
            })

    if stable_epochs:
        # Look for when gradient norm drops below a threshold
        early_grad = grouped.loc[20:40, (first_layer_col, 'mean')].mean()
        late_grad = grouped.loc[180:200, (first_layer_col, 'mean')].mean() if 180 in grouped.index else early_grad / 2

        suggested_threshold = (early_grad + late_grad) / 2

        print(f"  Early gradient norm (epoch 20-40): {early_grad:.6f}")
        print(f"  Late gradient norm (epoch 180-200): {late_grad:.6f}")
        print(f"  Suggested threshold: {suggested_threshold:.6f}")

        return {
            'layer_name': layer_name,
            'grad_threshold': suggested_threshold,
            'var_threshold': suggested_threshold / 10,  # Rough heuristic
            'early_grad': early_grad,
            'late_grad': late_grad,
        }

    return {}


def run_phase3(cfg, device, output_dir, trigger_cfg=None):
    """Phase 3: Run automatic expansion experiment."""
    os.makedirs(output_dir, exist_ok=True)

    if trigger_cfg is None:
        # Default trigger config (can be tuned from Phase 1/2)
        trigger_cfg = {
            'use_gradient': True,
            'grad_threshold': 0.005,
            'var_threshold': 0.0005,
            'window_size': 10,
            'min_epoch': 15,  # Don't expand before this
            'use_jacobian': False,
        }

    teacher = create_teacher(cfg, device)

    # Fixed validation set
    torch.manual_seed(cfg['teacher_seed'] + 1000)
    val_x, val_y = generate_data(teacher, cfg['val_samples'], cfg['input_dim'], device)

    # Baseline MSE (target to beat)
    baseline_mse = cfg.get('baseline_target_mse', 0.0005)

    CONDITIONS = {
        'baseline': {'first_active': 1.0, 'expand_epoch': None},
        'fixed_early': {'first_active': 0.6, 'expand_epoch': 20},
        'fixed_late': {'first_active': 0.6, 'expand_epoch': 100},
        'auto_expand': {'first_active': 0.6, 'expand_epoch': 'auto'},
    }

    all_summaries = []

    for cond_name, cond_cfg in CONDITIONS.items():
        for seed in range(cfg['seeds']):
            print(f"\n{'='*70}")
            print(f"Condition: {cond_name} | Seed: {seed}")
            print(f"{'='*70}")

            torch.manual_seed(seed)

            if cond_cfg['first_active'] == 1.0:
                student = create_student(cfg, device, first_layer_active=1.0, other_layers_active=1.0)
            else:
                student = create_student(cfg, device, first_layer_active=0.6, other_layers_active=1.0)

            optimizer = optim.Adam(student.parameters(), lr=cfg['lr'])
            criterion = nn.MSELoss()

            expanded = False
            expand_epoch = None
            prev_weights = None
            window_stats = []
            epoch_log = []

            best_mse = float('inf')

            for epoch in range(1, cfg['epochs'] + 1):
                t0 = time.time()

                # Check for expansion
                should_expand = False
                if not expanded and cond_cfg['expand_epoch'] is not None:
                    if cond_cfg['expand_epoch'] == 'auto':
                        # Check saturation trigger
                        if epoch >= trigger_cfg.get('min_epoch', 15) and len(window_stats) >= trigger_cfg.get('window_size', 10):
                            first_layer_name = None
                            for name, module in student.named_modules():
                                if isinstance(module, MaskedLinear):
                                    first_layer_name = name
                                    break
                            if first_layer_name:
                                triggered, reason = check_saturation_trigger(
                                    window_stats[-1], first_layer_name, trigger_cfg, window_stats
                                )
                                if triggered:
                                    should_expand = True
                                    print(f"  >>> AUTO TRIGGER at epoch {epoch}: {reason}")
                    elif epoch == cond_cfg['expand_epoch']:
                        should_expand = True

                if should_expand and not expanded:
                    pre_mse, pre_agree = evaluate(student, val_x, val_y)
                    print(f"  >>> PRE-EXPANSION mse={pre_mse:.6f}")
                    events = expand_first_layer(student, init_method='zero')
                    expanded = True
                    expand_epoch = epoch
                    print(f"  >>> EXPANDED first layer: {events}")

                # Train
                train_loss, layer_stats = train_one_epoch_with_stats(
                    student, teacher, optimizer, criterion, device, cfg,
                    compute_jacobian=False,  # Skip Jacobian for speed in Phase 3
                    prev_weights=prev_weights
                )
                prev_weights = get_layer_weights(student)
                window_stats.append(layer_stats)

                val_mse, val_agree = evaluate(student, val_x, val_y)
                best_mse = min(best_mse, val_mse)
                elapsed = time.time() - t0

                row = {
                    'condition': cond_name,
                    'seed': seed,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_mse': val_mse,
                    'val_agreement': val_agree,
                    'expanded': int(expanded),
                    'expand_epoch': expand_epoch,
                }
                epoch_log.append(row)

                if epoch % 20 == 0 or epoch == 1 or should_expand:
                    print(f"  Epoch {epoch:3d} | loss={train_loss:.6f} | mse={val_mse:.6f} | agree={val_agree:.4f}")

                if should_expand:
                    post_mse, _ = evaluate(student, val_x, val_y)
                    print(f"  >>> POST-EXPANSION mse={post_mse:.6f} delta={post_mse - pre_mse:+.6f}")

            summary = {
                'condition': cond_name,
                'seed': seed,
                'best_mse': best_mse,
                'final_mse': val_mse,
                'final_agreement': val_agree,
                'expand_epoch': expand_epoch,
            }
            all_summaries.append(summary)
            print(f"  Final: mse={val_mse:.6f} best={best_mse:.6f} expand_epoch={expand_epoch}")

            # Save epoch log
            log_path = os.path.join(output_dir, f'{cond_name}_seed{seed}_log.csv')
            if epoch_log:
                fieldnames = list(epoch_log[0].keys())
                with open(log_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(epoch_log)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<15} {'Seed':>4} {'Best MSE':>10} {'Final MSE':>10} {'Expand @':>10}")
    print("-" * 55)
    for s in all_summaries:
        exp = str(s['expand_epoch']) if s['expand_epoch'] else 'N/A'
        print(f"{s['condition']:<15} {s['seed']:>4} {s['best_mse']:>10.6f} {s['final_mse']:>10.6f} {exp:>10}")

    # Aggregate by condition
    print(f"\n{'='*80}")
    print("AGGREGATE BY CONDITION")
    print(f"{'='*80}")
    by_cond = defaultdict(list)
    for s in all_summaries:
        by_cond[s['condition']].append(s)

    for cond in CONDITIONS.keys():
        runs = by_cond[cond]
        best_mses = [r['best_mse'] for r in runs]
        expand_epochs = [r['expand_epoch'] for r in runs if r['expand_epoch'] is not None]

        mse_str = f"{np.mean(best_mses):.6f} +/- {np.std(best_mses):.6f}"
        if expand_epochs:
            exp_str = f"{np.mean(expand_epochs):.1f} +/- {np.std(expand_epochs):.1f}"
        else:
            exp_str = "N/A"
        print(f"  {cond:<15} best_mse={mse_str}  expand_epoch={exp_str}")

    # Save summaries
    summary_path = os.path.join(output_dir, 'summaries.json')
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summaries: {summary_path}")

    return all_summaries


def plot_phase1_results(input_dir, output_dir):
    """Generate plots from Phase 1 results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    csv_path = os.path.join(input_dir, 'saturation_all.csv')
    if not os.path.exists(csv_path):
        print(f"No data found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Find layer columns
    grad_mean_cols = [c for c in df.columns if 'grad_norm_mean' in c]
    grad_var_cols = [c for c in df.columns if 'grad_norm_var' in c]
    weight_change_cols = [c for c in df.columns if 'weight_change_rate' in c]

    layers = sorted(set(c.replace('_grad_norm_mean', '') for c in grad_mean_cols))

    # Plot 1: Gradient norm mean per layer over time
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4*len(layers)), squeeze=False)
    for i, layer in enumerate(layers):
        ax = axes[i, 0]
        col = f'{layer}_grad_norm_mean'
        if col in df.columns:
            grouped = df.groupby('epoch')[col].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], linewidth=2)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.2)
        ax.set_title(f'{layer}: Gradient Norm Mean', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norm_mean.png'), dpi=150)
    plt.close()
    print("Saved: gradient_norm_mean.png")

    # Plot 2: Gradient norm variance per layer
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4*len(layers)), squeeze=False)
    for i, layer in enumerate(layers):
        ax = axes[i, 0]
        col = f'{layer}_grad_norm_var'
        if col in df.columns:
            grouped = df.groupby('epoch')[col].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], linewidth=2, color='orange')
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.2, color='orange')
        ax.set_title(f'{layer}: Gradient Norm Variance', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norm_var.png'), dpi=150)
    plt.close()
    print("Saved: gradient_norm_var.png")

    # Plot 3: Weight change rate per layer
    if weight_change_cols:
        fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4*len(layers)), squeeze=False)
        for i, layer in enumerate(layers):
            ax = axes[i, 0]
            col = f'{layer}_weight_change_rate'
            if col in df.columns:
                grouped = df.groupby('epoch')[col].agg(['mean', 'std'])
                ax.plot(grouped.index, grouped['mean'], linewidth=2, color='green')
                ax.fill_between(grouped.index,
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               alpha=0.2, color='green')
            ax.set_title(f'{layer}: Weight Change Rate', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('L2 Delta')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weight_change_rate.png'), dpi=150)
        plt.close()
        print("Saved: weight_change_rate.png")

    # Plot 4: All layers gradient norm on one plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))
    for i, layer in enumerate(layers):
        col = f'{layer}_grad_norm_mean'
        if col in df.columns:
            grouped = df.groupby('epoch')[col].mean()
            ax.plot(grouped.index, grouped.values, linewidth=2, color=colors[i], label=layer)
    ax.set_title('Gradient Norm Mean: All Layers', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_layers_gradient.png'), dpi=150)
    plt.close()
    print("Saved: all_layers_gradient.png")

    # Plot 5: MSE with gradient overlay
    fig, ax1 = plt.subplots(figsize=(12, 6))

    grouped_mse = df.groupby('epoch')['val_mse'].agg(['mean', 'std'])
    ax1.plot(grouped_mse.index, grouped_mse['mean'], 'b-', linewidth=2, label='Val MSE')
    ax1.fill_between(grouped_mse.index,
                     grouped_mse['mean'] - grouped_mse['std'],
                     grouped_mse['mean'] + grouped_mse['std'],
                     alpha=0.2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation MSE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    first_layer = layers[0]
    col = f'{first_layer}_grad_norm_mean'
    if col in df.columns:
        grouped_grad = df.groupby('epoch')[col].mean()
        ax2.plot(grouped_grad.index, grouped_grad.values, 'r--', linewidth=2, label='L0 Grad Norm')
    ax2.set_ylabel(f'{first_layer} Gradient Norm', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_title('MSE and First Layer Gradient Dynamics', fontweight='bold', fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_gradient_overlay.png'), dpi=150)
    plt.close()
    print("Saved: mse_gradient_overlay.png")


def plot_phase3_results(input_dir, output_dir):
    """Generate plots from Phase 3 results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import glob

    os.makedirs(output_dir, exist_ok=True)

    # Load all log files
    log_files = glob.glob(os.path.join(input_dir, '*_log.csv'))
    if not log_files:
        print(f"No log files found in {input_dir}")
        return

    all_data = []
    for lf in log_files:
        df = pd.read_csv(lf)
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    conditions = df['condition'].unique()
    colors = {
        'baseline': '#333333',
        'fixed_early': '#2196F3',
        'fixed_late': '#F44336',
        'auto_expand': '#4CAF50',
    }

    # Plot 1: MSE over time by condition
    fig, ax = plt.subplots(figsize=(12, 7))
    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        grouped = cond_df.groupby('epoch')['val_mse'].agg(['mean', 'std'])
        color = colors.get(cond, '#888888')
        ax.plot(grouped.index, grouped['mean'], linewidth=2.5, color=color, label=cond)
        ax.fill_between(grouped.index,
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       alpha=0.15, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MSE', fontsize=12)
    ax.set_title('Auto-Expansion vs Fixed Timing', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_comparison.png'), dpi=150)
    plt.close()
    print("Saved: mse_comparison.png")

    # Plot 2: Expansion timing distribution for auto_expand
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load summaries
    summary_path = os.path.join(input_dir, 'summaries.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summaries = json.load(f)

        auto_epochs = [s['expand_epoch'] for s in summaries
                      if s['condition'] == 'auto_expand' and s['expand_epoch'] is not None]

        if auto_epochs:
            ax.hist(auto_epochs, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(auto_epochs), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(auto_epochs):.1f}')
            ax.axvline(20, color='#2196F3', linestyle=':', linewidth=2, label='fixed_early (20)')
            ax.axvline(100, color='#F44336', linestyle=':', linewidth=2, label='fixed_late (100)')

    ax.set_xlabel('Expansion Epoch', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Auto-Expansion Timing Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expansion_timing.png'), dpi=150)
    plt.close()
    print("Saved: expansion_timing.png")

    # Plot 3: Final MSE bar chart
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summaries = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 6))

        cond_order = ['baseline', 'fixed_early', 'fixed_late', 'auto_expand']
        x = np.arange(len(cond_order))

        means = []
        stds = []
        for cond in cond_order:
            mses = [s['best_mse'] for s in summaries if s['condition'] == cond]
            means.append(np.mean(mses))
            stds.append(np.std(mses))

        bar_colors = [colors.get(c, '#888888') for c in cond_order]
        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(cond_order)
        ax.set_ylabel('Best MSE', fontsize=12)
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_mse_comparison.png'), dpi=150)
        plt.close()
        print("Saved: final_mse_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Saturation Detection Experiment")
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3],
                       help='1=characterize, 2=analyze thresholds, 3=auto-expand')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--plot_only', action='store_true')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    cfg = dict(DEFAULTS)
    cfg['epochs'] = args.epochs
    cfg['seeds'] = args.seeds
    cfg['lr'] = args.lr

    if args.phase == 1:
        output_dir = args.output_dir or 'results/saturation_phase1'

        if args.plot_only:
            plot_phase1_results(output_dir, os.path.join(output_dir, 'plots'))
        else:
            print(f"Phase 1: Characterize saturation signals")
            print(f"Device: {device}")
            print(f"Epochs: {cfg['epochs']} | Seeds: {cfg['seeds']}")
            print(f"Output: {output_dir}")

            results = run_phase1(cfg, device, output_dir)
            plot_phase1_results(output_dir, os.path.join(output_dir, 'plots'))

            # Analyze for thresholds
            thresholds = analyze_phase1_for_thresholds(results)
            if thresholds:
                thresh_path = os.path.join(output_dir, 'suggested_thresholds.json')
                with open(thresh_path, 'w') as f:
                    json.dump(thresholds, f, indent=2)
                print(f"\nSaved thresholds: {thresh_path}")

    elif args.phase == 2:
        input_dir = args.input_dir or 'results/saturation_phase1'
        output_dir = args.output_dir or input_dir

        # Load Phase 1 results and analyze
        csv_path = os.path.join(input_dir, 'saturation_all.csv')
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            results = df.to_dict('records')
            thresholds = analyze_phase1_for_thresholds(results)
            print(f"\nSuggested thresholds: {thresholds}")
        else:
            print(f"No Phase 1 data found at {csv_path}")

    elif args.phase == 3:
        output_dir = args.output_dir or 'results/saturation_phase3'

        if args.plot_only:
            plot_phase3_results(output_dir, os.path.join(output_dir, 'plots'))
        else:
            # Load trigger config from Phase 1 if available
            trigger_cfg = None
            phase1_thresh = 'results/saturation_phase1/suggested_thresholds.json'
            if os.path.exists(phase1_thresh):
                with open(phase1_thresh) as f:
                    thresh_data = json.load(f)
                trigger_cfg = {
                    'use_gradient': True,
                    'grad_threshold': thresh_data.get('grad_threshold', 0.005),
                    'var_threshold': thresh_data.get('var_threshold', 0.0005),
                    'window_size': 10,
                    'min_epoch': 15,
                }
                print(f"Loaded trigger config from Phase 1: {trigger_cfg}")

            print(f"\nPhase 3: Auto-expansion experiment")
            print(f"Device: {device}")
            print(f"Epochs: {cfg['epochs']} | Seeds: {cfg['seeds']}")
            print(f"Output: {output_dir}")

            run_phase3(cfg, device, output_dir, trigger_cfg)
            plot_phase3_results(output_dir, os.path.join(output_dir, 'plots'))


if __name__ == '__main__':
    main()
