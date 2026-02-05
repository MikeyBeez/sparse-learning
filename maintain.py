"""Phase 2: Headroom / capacity growth experiments.

Tests whether the trajectory of capacity utilization matters:
does starting with fewer active weights and gradually growing
outperform using all weights from the start?

All conditions use the SAME total architecture. The only variable is the
growth schedule: what fraction starts active, growing linearly to 100%.

Usage:
    python maintain.py --dataset cifar10 [--epochs 100] [--seeds 3] \
        --start_active 0.6 0.7 0.8 0.9 1.0
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MaskedMLP, MaskedLinear
from utils import SparsityTracker


DATASET_CONFIGS = {
    'mnist': {
        'input_dim': 784,
        'hidden_dims': (256, 128),
        'output_dim': 10,
        'default_epochs': 50,
    },
    'cifar10': {
        'input_dim': 3072,
        'hidden_dims': (1024, 512, 256),
        'output_dim': 10,
        'default_epochs': 100,
    },
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
        # Re-zero masked weights
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, MaskedLinear):
                    module.weight.data *= module.mask
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


def grow_to_target(model, target_active_fraction):
    """Unmask weights in each MaskedLinear to reach target_active_fraction.

    Returns dict of growth events (layer_name -> n_unmasked).
    """
    events = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            current_frac = module.active_fraction
            if current_frac >= target_active_fraction:
                continue
            # How many more weights to unmask
            total = module.total_params
            target_active = int(target_active_fraction * total)
            current_active = int(module.active_params)
            n_to_unmask = target_active - current_active
            if n_to_unmask > 0:
                actually_unmasked = module.unmask_weights(n_to_unmask)
                if actually_unmasked > 0:
                    events[name] = actually_unmasked
    return events


def count_active_params(model):
    """Count active (unmasked) parameters."""
    total = 0
    active = 0
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            total += module.total_params
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


def run_growth_experiment(seed, epochs, start_active, output_dir, device,
                          dataset_name='cifar10', lr=1e-3, oversized_factor=1.5,
                          growth_end_fraction=0.75):
    """Run one growth experiment.

    Args:
        start_active: fraction of weights active at epoch 0 (1.0 = baseline)
        growth_end_fraction: fraction of training at which growth reaches 100%.
            E.g., 0.75 means fully grown by 75% of training, last 25% at full capacity.
    """
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    cfg = DATASET_CONFIGS[dataset_name]

    model = MaskedMLP(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        output_dim=cfg['output_dim'],
        oversized_factor=oversized_factor,
        initial_active_fraction=start_active,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = get_data_loaders(dataset_name, batch_size=128)

    exp_name = f"start{int(start_active*100)}_seed{seed}"
    tracker = SparsityTracker(output_dir, experiment_name=exp_name)

    growth_events = []
    active_params, total_params = count_active_params(model)

    tracker.record(model, epoch=0, step=0)

    # Growth schedule: linearly from start_active to 1.0 over growth_end_fraction of training
    growth_end_epoch = int(epochs * growth_end_fraction)

    print(f"\n{'='*60}")
    print(f"Start={start_active:.0%} | Seed={seed} | "
          f"Active: {active_params:,} / {total_params:,} ({active_params/total_params:.1%})")
    print(f"Growth: {start_active:.0%} -> 100% over epochs 1-{growth_end_epoch}")
    print(f"{'='*60}")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Scheduled growth
        events = {}
        if start_active < 1.0 and epoch <= growth_end_epoch:
            # Linear interpolation from start_active to 1.0
            progress = epoch / growth_end_epoch
            target_frac = start_active + (1.0 - start_active) * progress
            target_frac = min(target_frac, 1.0)
            events = grow_to_target(model, target_frac)
            if events:
                growth_events.append({'epoch': epoch, 'target_frac': target_frac, **events})

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_acc)

        elapsed = time.time() - t0
        snaps = tracker.record(model, epoch=epoch, step=epoch)
        active_params, total_params = count_active_params(model)

        tracker.log_training(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            best_val_accuracy=best_val_acc,
            active_params=active_params,
            total_params=total_params,
            active_fraction=active_params / total_params if total_params > 0 else 1.0,
            n_growth_events=len(events),
            elapsed_seconds=elapsed,
        )

        if epoch % 10 == 0 or epoch == 1 or events:
            grow_str = f" [grew {sum(events.values())} weights]" if events else ""
            print(f"Epoch {epoch:3d} | train={train_loss:.4f}/{train_acc:.4f} "
                  f"| val={val_loss:.4f}/{val_acc:.4f} "
                  f"| active={active_params:,} ({active_params/total_params:.1%}){grow_str}")

    print(f"  Best val acc: {best_val_acc:.4f}")

    tracker.summary()
    tracker.save()

    if growth_events:
        events_path = os.path.join(output_dir, f"{exp_name}_growth.json")
        with open(events_path, 'w') as f:
            json.dump(growth_events, f, indent=2)

    ckpt_path = os.path.join(output_dir, f"model_{exp_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'seed': seed,
        'epochs': epochs,
        'start_active': start_active,
        'best_val_accuracy': best_val_acc,
    }, ckpt_path)

    return tracker


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Capacity growth experiments")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--start_active', type=float, nargs='+',
                        default=[0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Initial active fractions to test (1.0 = baseline)')
    parser.add_argument('--growth_end', type=float, default=0.75,
                        help='Fraction of training when growth reaches 100%%')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--oversized_factor', type=float, default=1.5)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    cfg = DATASET_CONFIGS[args.dataset]
    epochs = args.epochs or cfg['default_epochs']
    output_dir = args.output_dir or f'results/phase2_{args.dataset}'

    os.makedirs(output_dir, exist_ok=True)
    print(f"Phase 2: Capacity Growth ({args.dataset.upper()})")
    print(f"Architecture: {cfg['input_dim']} -> {cfg['hidden_dims']} -> {cfg['output_dim']}")
    print(f"Oversized factor: {args.oversized_factor}x | LR: {args.lr}")
    print(f"Epochs: {epochs} | Seeds: {args.seeds}")
    print(f"Start active fractions: {args.start_active}")
    print(f"Growth completes at: {args.growth_end:.0%} of training (epoch {int(epochs * args.growth_end)})")
    print(f"Output: {output_dir}")

    for start in args.start_active:
        for seed in range(args.seeds):
            run_growth_experiment(
                seed=seed,
                epochs=epochs,
                start_active=start,
                output_dir=output_dir,
                device=device,
                dataset_name=args.dataset,
                lr=args.lr,
                oversized_factor=args.oversized_factor,
                growth_end_fraction=args.growth_end,
            )

    print(f"\nAll experiments complete. Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
