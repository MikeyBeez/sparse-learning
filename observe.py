"""Phase 1: Observe sparsity dynamics during normal training.

Train a simple MLP on MNIST and track per-layer sparsity at each epoch.
No intervention -- just observation.

Usage:
    python observe.py [--epochs 50] [--seeds 5] [--output_dir results/phase1]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SimpleMLP
from utils import SparsityTracker


def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
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
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


def run_observation(seed, epochs, output_dir, device, lr=1e-3):
    """Run a single observation experiment with given seed."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    model = SimpleMLP(input_dim=784, hidden_dims=(256, 128), output_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = get_mnist_loaders(batch_size=128)

    tracker = SparsityTracker(output_dir, experiment_name=f"observe_seed{seed}")

    # Record initial sparsity (before any training)
    tracker.record(model, epoch=0, step=0)

    print(f"\n{'='*60}")
    print(f"Seed {seed} | Device: {device}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # Record sparsity after this epoch (gradients are populated from training)
        snaps = tracker.record(model, epoch=epoch, step=epoch)

        tracker.log_training(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            elapsed_seconds=elapsed,
        )

        if epoch % 5 == 0 or epoch == 1:
            # Print compact summary
            layer_sparsities = {s.layer_name: s.sparsity_exact for s in snaps}
            sp_str = " | ".join(f"{k}: {v:.4f}" for k, v in layer_sparsities.items()
                                if 'weight' in k.lower() or '.' in k)
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
                  f"| val_loss={val_loss:.4f} acc={val_acc:.4f} | {elapsed:.1f}s")
            print(f"          Sparsity(exact): {sp_str}")

    tracker.summary()
    tracker.save()

    # Save model checkpoint
    ckpt_path = os.path.join(output_dir, f"model_seed{seed}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'seed': seed,
        'epochs': epochs,
    }, ckpt_path)

    return tracker


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Observe sparsity during training")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='results/phase1')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Phase 1: Sparsity Observation")
    print(f"Epochs: {args.epochs} | Seeds: {args.seeds} | LR: {args.lr}")
    print(f"Output: {args.output_dir}")

    trackers = []
    for seed in range(args.seeds):
        tracker = run_observation(
            seed=seed,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=device,
            lr=args.lr,
        )
        trackers.append(tracker)

    print(f"\nAll seeds complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
