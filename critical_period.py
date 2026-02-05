"""Phase 3: Critical Period Experiment (2x2 Design).

Tests whether network resistance to structural change is a real critical period
or just an artifact of zero initialization.

2x2 Design:
  - Factor 1: Timing — Early (epoch 10) vs Late (epoch 50)
  - Factor 2: Init — Zero vs Kaiming for newly activated weights
  - Plus baseline: 100% active from start

Key question: If late+Kaiming integrates fine, it's just bad init (boring).
If late+Kaiming ALSO fails, real critical period (interesting).

Tracks:
  - Weight integration: magnitude of new vs old weights over time
  - Gradient flow: gradient magnitude of new vs old weights
  - Performance: does expansion help or hurt generalization?
  - Recovery: how quickly does performance recover after expansion?

Usage:
    python3 critical_period.py --dataset cifar10 [--epochs 100] [--seeds 3]
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MaskedMLP, MaskedLinear


DATASET_CONFIGS = {
    'mnist': {
        'input_dim': 784,
        'hidden_dims': (256, 128),
        'output_dim': 10,
        'default_epochs': 50,
        'early_epoch': 5,
        'late_epoch': 25,
    },
    'cifar10': {
        'input_dim': 3072,
        'hidden_dims': (1024, 512, 256),
        'output_dim': 10,
        'default_epochs': 100,
        'early_epoch': 10,
        'late_epoch': 50,
    },
}

# 2x2 conditions + baseline
# expand_epoch is set per-dataset below
CONDITIONS = {
    'baseline':      {'start_active': 1.0, 'timing': None,    'init_method': None},
    'early_zero':    {'start_active': 0.6, 'timing': 'early', 'init_method': 'zero'},
    'early_kaiming': {'start_active': 0.6, 'timing': 'early', 'init_method': 'kaiming'},
    'late_zero':     {'start_active': 0.6, 'timing': 'late',  'init_method': 'zero'},
    'late_kaiming':  {'start_active': 0.6, 'timing': 'late',  'init_method': 'kaiming'},
}


def get_data_loaders(dataset_name, batch_size=128):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('data', train=False, transform=transform)
    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10('data', train=False, transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # Zero out gradients for masked weights
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                if module.weight.grad is not None:
                    module.weight.grad.data *= module.mask
        optimizer.step()
        # Re-zero masked weights after optimizer step
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, MaskedLinear):
                    module.weight.data *= module.mask
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


def expand_all(model, init_method='zero'):
    """Expand all masked layers to 100% active at once.

    Returns dict of expansion events per layer.
    """
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
    """Get weight and gradient stats for original vs new weights across all layers."""
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

        # Accumulate for aggregate stats
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

    # Aggregate across all layers
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
                  dataset_name='cifar10', lr=1e-3, oversized_factor=1.5):
    """Run one condition of the critical period experiment."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    cfg = DATASET_CONFIGS[dataset_name]
    start_active = condition_cfg['start_active']
    timing = condition_cfg['timing']
    init_method = condition_cfg['init_method']

    # Resolve expand_epoch from timing
    if timing == 'early':
        expand_epoch = cfg['early_epoch']
    elif timing == 'late':
        expand_epoch = cfg['late_epoch']
    else:
        expand_epoch = None

    model = MaskedMLP(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        output_dim=cfg['output_dim'],
        oversized_factor=oversized_factor,
        initial_active_fraction=start_active,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = get_data_loaders(dataset_name)

    exp_name = f"{condition_name}_seed{seed}"

    # Tracking
    training_log = []
    integration_log = []
    expanded = False
    pre_expand_val_acc = None
    pre_expand_val_loss = None

    active_params, total_params = count_active_params(model)

    print(f"\n{'='*70}")
    print(f"Condition: {condition_name} | Seed: {seed}")
    if expand_epoch:
        print(f"  Start: {start_active:.0%} active ({active_params:,}/{total_params:,})")
        print(f"  Expand to 100% at epoch {expand_epoch} ({init_method} init)")
    else:
        print(f"  Baseline: 100% active from start ({active_params:,} params)")
    print(f"{'='*70}")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Check for expansion
        expansion_events = {}
        if expand_epoch and epoch == expand_epoch and not expanded:
            # Record pre-expansion accuracy
            pre_expand_val_loss, pre_expand_val_acc = evaluate(model, val_loader, criterion, device)
            print(f"  >>> PRE-EXPANSION  val_acc={pre_expand_val_acc:.4f}  val_loss={pre_expand_val_loss:.4f}")

            expansion_events = expand_all(model, init_method=init_method)
            expanded = True
            total_expanded = sum(expansion_events.values())
            print(f"  >>> EXPANDED {total_expanded:,} weights ({init_method} init)")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_acc)

        elapsed = time.time() - t0
        active_params, total_params = count_active_params(model)

        # Integration stats (collected after training step so gradients are available)
        layer_stats, agg_stats = get_integration_stats(model)

        # Training log entry
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'active_fraction': active_params / total_params if total_params > 0 else 1.0,
            'generalization_gap': train_acc - val_acc,
            'expanded': int(expanded),
            'elapsed': elapsed,
        }
        # Add aggregate integration stats
        for k, v in agg_stats.items():
            entry[k] = v
        training_log.append(entry)

        # Per-layer integration log
        int_entry = {'epoch': epoch}
        for layer_name, stats in layer_stats.items():
            for key, val in stats.items():
                int_entry[f"{layer_name}_{key}"] = val
        integration_log.append(int_entry)

        # Print
        if epoch % 10 == 0 or epoch == 1 or expansion_events:
            new_info = ""
            if agg_stats.get('agg_new_abs_mean'):
                ratio = agg_stats['agg_new_abs_mean'] / max(agg_stats.get('agg_original_abs_mean', 1e-8), 1e-8)
                new_info = f" | new/orig={ratio:.3f}"
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f}/{train_acc:.4f} "
                  f"| val={val_loss:.4f}/{val_acc:.4f} "
                  f"| gap={train_acc - val_acc:+.4f}{new_info}")

        if expansion_events:
            # Print post-expansion accuracy
            print(f"  >>> POST-EXPANSION val_acc={val_acc:.4f}  "
                  f"delta={val_acc - pre_expand_val_acc:+.4f}")

    print(f"  Final: val_acc={val_acc:.4f}  best={best_val_acc:.4f}  "
          f"gap={train_acc - val_acc:+.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Training log CSV - collect all keys across all rows
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

    # Integration log CSV - collect all keys across all rows
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

    # Summary JSON
    summary = {
        'condition': condition_name,
        'seed': seed,
        'dataset': dataset_name,
        'epochs': epochs,
        'start_active': start_active,
        'expand_epoch': expand_epoch,
        'init_method': init_method,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'final_train_acc': train_acc,
        'final_val_loss': val_loss,
        'pre_expand_val_acc': pre_expand_val_acc,
        'generalization_gap': train_acc - val_acc,
    }
    summary_path = os.path.join(output_dir, f"{exp_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Model checkpoint
    ckpt_path = os.path.join(output_dir, f"model_{exp_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'condition': condition_name,
        'seed': seed,
        'best_val_accuracy': best_val_acc,
    }, ckpt_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Critical Period Experiment (2x2)")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='Which conditions to run (default: all)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--oversized_factor', type=float, default=1.5)
    parser.add_argument('--start_active', type=float, default=0.6,
                        help='Initial active fraction for non-baseline conditions')
    parser.add_argument('--early_epoch', type=int, default=None,
                        help='Override early expansion epoch')
    parser.add_argument('--late_epoch', type=int, default=None,
                        help='Override late expansion epoch')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    cfg = DATASET_CONFIGS[args.dataset]
    epochs = args.epochs or cfg['default_epochs']
    output_dir = args.output_dir or f'results/phase3_{args.dataset}'

    # Apply overrides
    if args.early_epoch is not None:
        cfg['early_epoch'] = args.early_epoch
    if args.late_epoch is not None:
        cfg['late_epoch'] = args.late_epoch

    # Update start_active for non-baseline conditions
    for name, cond in CONDITIONS.items():
        if name != 'baseline' and cond['start_active'] != 1.0:
            cond['start_active'] = args.start_active

    conditions_to_run = args.conditions or list(CONDITIONS.keys())

    os.makedirs(output_dir, exist_ok=True)
    print(f"Phase 3: Critical Period Experiment ({args.dataset.upper()})")
    print(f"Architecture: {cfg['input_dim']} -> {cfg['hidden_dims']} -> {cfg['output_dim']}")
    print(f"Oversized: {args.oversized_factor}x | LR: {args.lr}")
    print(f"Epochs: {epochs} | Seeds: {args.seeds}")
    print(f"Start active: {args.start_active:.0%}")
    print(f"Early expansion: epoch {cfg['early_epoch']} | Late expansion: epoch {cfg['late_epoch']}")
    print(f"Conditions: {conditions_to_run}")
    print(f"Output: {output_dir}")

    all_summaries = []
    for cond_name in conditions_to_run:
        cond_cfg = CONDITIONS[cond_name]
        for seed in range(args.seeds):
            summary = run_condition(
                condition_name=cond_name,
                condition_cfg=cond_cfg,
                seed=seed,
                epochs=epochs,
                output_dir=output_dir,
                device=device,
                dataset_name=args.dataset,
                lr=args.lr,
                oversized_factor=args.oversized_factor,
            )
            all_summaries.append(summary)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<18} {'Seed':>4} {'Best Acc':>9} {'Final Acc':>10} "
          f"{'Pre-Exp Acc':>11} {'Gen Gap':>8}")
    print(f"{'-'*70}")
    for s in all_summaries:
        pre = f"{s['pre_expand_val_acc']:.4f}" if s['pre_expand_val_acc'] is not None else "   N/A"
        print(f"{s['condition']:<18} {s['seed']:>4} {s['best_val_acc']:>9.4f} "
              f"{s['final_val_acc']:>10.4f} {pre:>11} {s['generalization_gap']:>+8.4f}")

    # Save combined summary
    combined_path = os.path.join(output_dir, 'all_summaries.json')
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll experiments complete. Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
