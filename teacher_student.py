"""Phase 4: Critical Period with Teacher-Student Task.

Uses a synthetic teacher-student setup where capacity is provably
the bottleneck. The teacher is a random MLP; the student tries to
match its outputs. At 60% capacity, the student literally cannot
represent the teacher's function. At 100%, it can.

This eliminates the overfitting confound from Phase 3: fresh random
inputs each epoch = infinite data = zero overfitting by construction.
Any performance gap between conditions is purely about capacity and
weight integration.

2x2 Design (same as Phase 3):
  - Factor 1: Timing — Early (epoch 20) vs Late (epoch 100)
  - Factor 2: Init — Zero vs Kaiming for newly activated weights
  - Plus baseline: 100% active from start

Usage:
    python3 teacher_student.py --epochs 200 --seeds 3
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from models import SimpleMLP, MaskedMLP, MaskedLinear


CONDITIONS = {
    'baseline':      {'start_active': 1.0, 'timing': None,    'init_method': None},
    'early_zero':    {'start_active': 0.6, 'timing': 'early', 'init_method': 'zero'},
    'early_kaiming': {'start_active': 0.6, 'timing': 'early', 'init_method': 'kaiming'},
    'late_zero':     {'start_active': 0.6, 'timing': 'late',  'init_method': 'zero'},
    'late_kaiming':  {'start_active': 0.6, 'timing': 'late',  'init_method': 'kaiming'},
}

DEFAULTS = {
    'input_dim': 100,
    'hidden_dims': (256, 128),
    'output_dim': 10,
    'epochs': 200,
    'early_epoch': 20,
    'late_epoch': 100,
    'train_samples': 50000,
    'val_samples': 10000,
    'batch_size': 256,
    'teacher_seed': 42,
}


def create_teacher(input_dim, hidden_dims, output_dim, seed, device):
    """Create a frozen teacher network with fixed random weights."""
    torch.manual_seed(seed)
    teacher = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


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
    # Agreement: fraction where argmax matches
    agreement = (outputs.argmax(dim=1) == val_y.argmax(dim=1)).float().mean().item()
    return mse, agreement


def train_one_epoch(student, teacher, optimizer, criterion, device, cfg):
    """Train one epoch on fresh random data."""
    student.train()
    n_samples = cfg['train_samples']
    batch_size = cfg['batch_size']
    input_dim = cfg['input_dim']

    total_loss = 0.0
    n_batches = 0

    # Generate all training data at once (faster than per-batch)
    with torch.no_grad():
        all_x = torch.randn(n_samples, input_dim, device=device)
        all_y = teacher(all_x)

    # Shuffle and batch
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

    return total_loss / n_batches


def expand_all(model, init_method='zero'):
    """Expand all masked layers to 100% active at once."""
    events = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            inactive = int((module.mask == 0).sum().item())
            if inactive > 0:
                n = module.unmask_weights(inactive, init_method=init_method)
                if n > 0:
                    events[name] = n
    return events


def get_integration_stats(model):
    """Get weight and gradient stats for original vs new weights."""
    all_orig_w, all_new_w = [], []
    all_orig_g, all_new_g = [], []
    layer_stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, MaskedLinear):
            continue
        w_stats = module.get_weight_stats()
        g_stats = module.get_gradient_stats()
        layer_stats[name] = {**w_stats}
        if g_stats:
            layer_stats[name].update(g_stats)

        with torch.no_grad():
            original_mask = module.mask * (1 - module.expansion_mask)
            new_mask = module.expansion_mask
            w = module.weight.data
            if original_mask.sum() > 0:
                all_orig_w.append(w[original_mask.bool()])
            if new_mask.sum() > 0:
                all_new_w.append(w[new_mask.bool()])
            if module.weight.grad is not None:
                g = module.weight.grad.data
                if original_mask.sum() > 0:
                    all_orig_g.append(g[original_mask.bool()])
                if new_mask.sum() > 0:
                    all_new_g.append(g[new_mask.bool()])

    agg = {}
    if all_orig_w:
        ow = torch.cat(all_orig_w)
        agg['agg_original_abs_mean'] = ow.abs().mean().item()
        agg['agg_original_std'] = ow.std().item()
    if all_new_w:
        nw = torch.cat(all_new_w)
        agg['agg_new_abs_mean'] = nw.abs().mean().item()
        agg['agg_new_std'] = nw.std().item()
    if all_orig_g:
        og = torch.cat(all_orig_g)
        agg['agg_original_grad_abs_mean'] = og.abs().mean().item()
    if all_new_g:
        ng = torch.cat(all_new_g)
        agg['agg_new_grad_abs_mean'] = ng.abs().mean().item()

    return layer_stats, agg


def count_active_params(model):
    total = 0
    active = 0
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            total += int(module.total_params)
            active += int(module.active_params)
            if module.bias is not None:
                total += module.bias.numel()
                active += module.bias.numel()
        elif isinstance(module, nn.Linear) and not isinstance(module, MaskedLinear):
            n = module.weight.numel()
            total += n
            active += n
            if module.bias is not None:
                total += module.bias.numel()
                active += module.bias.numel()
    return active, total


def run_condition(condition_name, condition_cfg, seed, epochs, output_dir, device,
                  cfg, teacher):
    """Run one condition of the teacher-student experiment."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    start_active = condition_cfg['start_active']
    timing = condition_cfg['timing']
    init_method = condition_cfg['init_method']

    if timing == 'early':
        expand_epoch = cfg['early_epoch']
    elif timing == 'late':
        expand_epoch = cfg['late_epoch']
    else:
        expand_epoch = None

    student = MaskedMLP(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        output_dim=cfg['output_dim'],
        oversized_factor=1.0,  # No oversizing — match teacher architecture
        initial_active_fraction=start_active,
    ).to(device)

    optimizer = optim.Adam(student.parameters(), lr=cfg.get('lr', 1e-3))
    criterion = nn.MSELoss()

    # Fixed validation set (same across all epochs)
    torch.manual_seed(cfg['teacher_seed'] + 1000)  # Deterministic but different from teacher
    val_x, val_y = generate_data(teacher, cfg['val_samples'], cfg['input_dim'], device)

    exp_name = f"{condition_name}_seed{seed}"
    training_log = []
    integration_log = []
    expanded = False
    pre_expand_mse = None
    pre_expand_agreement = None

    active_params, total_params = count_active_params(student)

    print(f"\n{'='*70}")
    print(f"Condition: {condition_name} | Seed: {seed}")
    if expand_epoch:
        print(f"  Start: {start_active:.0%} active ({active_params:,}/{total_params:,})")
        print(f"  Expand to 100% at epoch {expand_epoch} ({init_method} init)")
    else:
        print(f"  Baseline: 100% active from start ({active_params:,} params)")
    print(f"{'='*70}")

    best_mse = float('inf')

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Check for expansion
        expansion_events = {}
        if expand_epoch and epoch == expand_epoch and not expanded:
            pre_expand_mse, pre_expand_agreement = evaluate(student, val_x, val_y)
            print(f"  >>> PRE-EXPANSION  mse={pre_expand_mse:.6f}  agreement={pre_expand_agreement:.4f}")

            expansion_events = expand_all(student, init_method=init_method)
            expanded = True
            total_expanded = sum(expansion_events.values())
            print(f"  >>> EXPANDED {total_expanded:,} weights ({init_method} init)")

        # Train on fresh random data
        train_loss = train_one_epoch(student, teacher, optimizer, criterion, device, cfg)

        # Evaluate on fixed validation set
        val_mse, val_agreement = evaluate(student, val_x, val_y)
        best_mse = min(best_mse, val_mse)

        elapsed = time.time() - t0
        active_params, total_params = count_active_params(student)

        # Integration stats
        layer_stats, agg_stats = get_integration_stats(student)

        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_mse,
            'val_agreement': val_agreement,
            'best_mse': best_mse,
            'active_fraction': active_params / total_params if total_params > 0 else 1.0,
            'expanded': int(expanded),
            'elapsed': elapsed,
        }
        for k, v in agg_stats.items():
            entry[k] = v
        training_log.append(entry)

        int_entry = {'epoch': epoch}
        for layer_name, stats in layer_stats.items():
            for key, val in stats.items():
                int_entry[f"{layer_name}_{key}"] = val
        integration_log.append(int_entry)

        # Print
        if epoch % 20 == 0 or epoch == 1 or expansion_events:
            new_info = ""
            if agg_stats.get('agg_new_abs_mean') and agg_stats.get('agg_original_abs_mean'):
                ratio = agg_stats['agg_new_abs_mean'] / max(agg_stats['agg_original_abs_mean'], 1e-8)
                new_info = f" | new/orig={ratio:.3f}"
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.6f} "
                  f"| val_mse={val_mse:.6f} | agree={val_agreement:.4f}{new_info}")

        if expansion_events:
            print(f"  >>> POST-EXPANSION mse={val_mse:.6f}  "
                  f"delta={val_mse - pre_expand_mse:+.6f}")

    print(f"  Final: mse={val_mse:.6f}  best={best_mse:.6f}  agree={val_agreement:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"{exp_name}_training.csv")
    if training_log:
        all_keys = set()
        for row in training_log:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys, key=lambda x: (x != 'epoch', x))
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(training_log)

    int_path = os.path.join(output_dir, f"{exp_name}_integration.csv")
    if integration_log:
        all_keys = set()
        for row in integration_log:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys, key=lambda x: (x != 'epoch', x))
        with open(int_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(integration_log)

    summary = {
        'condition': condition_name,
        'seed': seed,
        'epochs': epochs,
        'start_active': start_active,
        'expand_epoch': expand_epoch,
        'init_method': init_method,
        'best_mse': best_mse,
        'final_mse': val_mse,
        'final_agreement': val_agreement,
        'pre_expand_mse': pre_expand_mse,
        'pre_expand_agreement': pre_expand_agreement,
    }
    summary_path = os.path.join(output_dir, f"{exp_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    ckpt_path = os.path.join(output_dir, f"model_{exp_name}.pt")
    torch.save({
        'model_state_dict': student.state_dict(),
        'condition': condition_name,
        'seed': seed,
        'best_mse': best_mse,
    }, ckpt_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Teacher-Student Critical Period")
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--conditions', type=str, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--start_active', type=float, default=0.6)
    parser.add_argument('--early_epoch', type=int, default=DEFAULTS['early_epoch'])
    parser.add_argument('--late_epoch', type=int, default=DEFAULTS['late_epoch'])
    parser.add_argument('--train_samples', type=int, default=DEFAULTS['train_samples'])
    parser.add_argument('--val_samples', type=int, default=DEFAULTS['val_samples'])
    parser.add_argument('--output_dir', type=str, default='results/phase4_teacher_student')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    cfg = dict(DEFAULTS)
    cfg['lr'] = args.lr
    cfg['early_epoch'] = args.early_epoch
    cfg['late_epoch'] = args.late_epoch
    cfg['train_samples'] = args.train_samples
    cfg['val_samples'] = args.val_samples

    # Update start_active for non-baseline conditions
    for name, cond in CONDITIONS.items():
        if name != 'baseline' and cond['start_active'] != 1.0:
            cond['start_active'] = args.start_active

    conditions_to_run = args.conditions or list(CONDITIONS.keys())

    # Create teacher (fixed across all conditions)
    teacher = create_teacher(
        cfg['input_dim'], cfg['hidden_dims'], cfg['output_dim'],
        cfg['teacher_seed'], device
    )
    teacher_params = sum(p.numel() for p in teacher.parameters())

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Phase 4: Teacher-Student Critical Period")
    print(f"Teacher: {cfg['input_dim']} -> {cfg['hidden_dims']} -> {cfg['output_dim']} "
          f"({teacher_params:,} params, frozen)")
    print(f"Student: same architecture, oversized_factor=1.0")
    print(f"Data: {cfg['train_samples']:,} train (fresh/epoch), {cfg['val_samples']:,} val (fixed)")
    print(f"Epochs: {args.epochs} | Seeds: {args.seeds} | LR: {args.lr}")
    print(f"Start active: {args.start_active:.0%}")
    print(f"Early expansion: epoch {cfg['early_epoch']} | Late expansion: epoch {cfg['late_epoch']}")
    print(f"Conditions: {conditions_to_run}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    all_summaries = []
    for cond_name in conditions_to_run:
        cond_cfg = CONDITIONS[cond_name]
        for seed in range(args.seeds):
            summary = run_condition(
                condition_name=cond_name,
                condition_cfg=cond_cfg,
                seed=seed,
                epochs=args.epochs,
                output_dir=args.output_dir,
                device=device,
                cfg=cfg,
                teacher=teacher,
            )
            all_summaries.append(summary)

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<18} {'Seed':>4} {'Best MSE':>10} {'Final MSE':>10} "
          f"{'Agreement':>10} {'Pre-Exp MSE':>12}")
    print(f"{'-'*80}")
    for s in all_summaries:
        pre = f"{s['pre_expand_mse']:.6f}" if s['pre_expand_mse'] is not None else "      N/A"
        print(f"{s['condition']:<18} {s['seed']:>4} {s['best_mse']:>10.6f} "
              f"{s['final_mse']:>10.6f} {s['final_agreement']:>10.4f} {pre:>12}")

    # Aggregate by condition
    print(f"\n{'='*80}")
    print(f"AGGREGATE")
    print(f"{'='*80}")
    by_cond = defaultdict(list)
    for s in all_summaries:
        by_cond[s['condition']].append(s)

    import numpy as np
    for cond_name in conditions_to_run:
        runs = by_cond[cond_name]
        best_mses = [r['best_mse'] for r in runs]
        agreements = [r['final_agreement'] for r in runs]
        print(f"  {cond_name:<18} best_mse={np.mean(best_mses):.6f}±{np.std(best_mses):.6f}  "
              f"agreement={np.mean(agreements):.4f}±{np.std(agreements):.4f}")

    # Save combined summary
    combined_path = os.path.join(args.output_dir, 'all_summaries.json')
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll experiments complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
