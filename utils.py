"""Sparsity calculation, logging, and tracking utilities."""

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn


EPSILONS = [0.0, 1e-5, 1e-4, 1e-3]


@dataclass
class LayerSparsitySnapshot:
    """Sparsity measurements for a single layer at a single point in time."""
    layer_name: str
    epoch: int
    step: int
    total_weights: int
    # Sparsity at different thresholds
    exact_zeros: int
    near_zero_1e5: int  # |w| < 1e-5
    near_zero_1e4: int  # |w| < 1e-4
    near_zero_1e3: int  # |w| < 1e-3
    # Derived ratios
    sparsity_exact: float
    sparsity_1e5: float
    sparsity_1e4: float
    sparsity_1e3: float
    # Weight statistics
    weight_mean: float
    weight_std: float
    weight_abs_mean: float
    weight_max: float
    weight_min: float
    # Gradient sparsity (if available)
    grad_exact_zeros: Optional[int] = None
    grad_sparsity_exact: Optional[float] = None
    grad_abs_mean: Optional[float] = None


def calculate_sparsity(tensor, epsilon=0.0):
    """Count zeros (or near-zeros) in a tensor and return (count, ratio)."""
    total = tensor.numel()
    if epsilon == 0.0:
        zeros = (tensor == 0).sum().item()
    else:
        zeros = (tensor.abs() < epsilon).sum().item()
    return zeros, zeros / total if total > 0 else 0.0


def snapshot_layer(layer_name, weight, epoch, step, grad=None):
    """Take a full sparsity snapshot of a layer's weights."""
    w = weight.data
    total = w.numel()

    exact_zeros, sparsity_exact = calculate_sparsity(w, 0.0)
    nz_1e5, sp_1e5 = calculate_sparsity(w, 1e-5)
    nz_1e4, sp_1e4 = calculate_sparsity(w, 1e-4)
    nz_1e3, sp_1e3 = calculate_sparsity(w, 1e-3)

    snap = LayerSparsitySnapshot(
        layer_name=layer_name,
        epoch=epoch,
        step=step,
        total_weights=total,
        exact_zeros=exact_zeros,
        near_zero_1e5=nz_1e5,
        near_zero_1e4=nz_1e4,
        near_zero_1e3=nz_1e3,
        sparsity_exact=sparsity_exact,
        sparsity_1e5=sp_1e5,
        sparsity_1e4=sp_1e4,
        sparsity_1e3=sp_1e3,
        weight_mean=w.mean().item(),
        weight_std=w.std().item(),
        weight_abs_mean=w.abs().mean().item(),
        weight_max=w.max().item(),
        weight_min=w.min().item(),
    )

    if grad is not None:
        g = grad.data
        grad_zeros, grad_sp = calculate_sparsity(g, 0.0)
        snap.grad_exact_zeros = grad_zeros
        snap.grad_sparsity_exact = grad_sp
        snap.grad_abs_mean = g.abs().mean().item()

    return snap


def snapshot_model(model, epoch, step):
    """Take sparsity snapshots of all weight-bearing layers in a model."""
    snapshots = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            grad = module.weight.grad if module.weight.grad is not None else None
            snap = snapshot_layer(name, module.weight, epoch, step, grad=grad)
            snapshots.append(snap)
    return snapshots


class SparsityTracker:
    """Accumulates sparsity snapshots over training and writes to disk."""

    def __init__(self, output_dir, experiment_name="experiment"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.snapshots: List[LayerSparsitySnapshot] = []
        self.training_log: List[Dict] = []  # loss, accuracy per epoch
        os.makedirs(output_dir, exist_ok=True)

    def record(self, model, epoch, step):
        """Record sparsity snapshot for all layers."""
        snaps = snapshot_model(model, epoch, step)
        self.snapshots.extend(snaps)
        return snaps

    def log_training(self, epoch, train_loss, val_loss=None, val_accuracy=None, **kwargs):
        """Log training metrics for an epoch."""
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }
        entry.update(kwargs)
        self.training_log.append(entry)

    def save_csv(self):
        """Save all snapshots to CSV."""
        if not self.snapshots:
            return
        path = os.path.join(self.output_dir, f"{self.experiment_name}_sparsity.csv")
        fieldnames = list(asdict(self.snapshots[0]).keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snap in self.snapshots:
                writer.writerow(asdict(snap))

        # Also save training log
        if self.training_log:
            log_path = os.path.join(self.output_dir, f"{self.experiment_name}_training.csv")
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.training_log[0].keys())
                writer.writeheader()
                writer.writerows(self.training_log)

    def save_json(self):
        """Save all data to JSON."""
        path = os.path.join(self.output_dir, f"{self.experiment_name}_data.json")
        data = {
            'experiment': self.experiment_name,
            'sparsity_snapshots': [asdict(s) for s in self.snapshots],
            'training_log': self.training_log,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def save(self):
        """Save both CSV and JSON."""
        self.save_csv()
        self.save_json()

    def get_layer_trajectory(self, layer_name, metric='sparsity_exact'):
        """Get the trajectory of a metric for a specific layer over epochs."""
        epochs = []
        values = []
        for snap in self.snapshots:
            if snap.layer_name == layer_name:
                epochs.append(snap.epoch)
                values.append(getattr(snap, metric))
        return epochs, values

    def get_layer_names(self):
        """Get unique layer names."""
        seen = set()
        names = []
        for snap in self.snapshots:
            if snap.layer_name not in seen:
                seen.add(snap.layer_name)
                names.append(snap.layer_name)
        return names

    def summary(self):
        """Print a summary of latest sparsity per layer."""
        if not self.snapshots:
            print("No snapshots recorded.")
            return
        last_epoch = max(s.epoch for s in self.snapshots)
        print(f"\nSparsity Summary (epoch {last_epoch}):")
        print(f"{'Layer':<40} {'Exact':>8} {'<1e-5':>8} {'<1e-4':>8} {'<1e-3':>8}")
        print("-" * 76)
        for snap in self.snapshots:
            if snap.epoch == last_epoch:
                print(f"{snap.layer_name:<40} {snap.sparsity_exact:>8.4f} "
                      f"{snap.sparsity_1e5:>8.4f} {snap.sparsity_1e4:>8.4f} "
                      f"{snap.sparsity_1e3:>8.4f}")
