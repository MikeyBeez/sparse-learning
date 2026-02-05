"""Simple models for sparsity headroom experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """3-layer MLP for MNIST classification."""

    def __init__(self, input_dim=784, hidden_dims=(256, 128), output_dim=10):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation after final layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class MaskedLinear(nn.Linear):
    """Linear layer with a binary mask for headroom maintenance (Option A).

    Allocates an oversized weight matrix. A binary mask controls which weights
    are "active". Inactive weights are zero and don't receive gradients.
    When headroom is needed, additional weights are unmasked (initialized to zero).
    """

    def __init__(self, in_features, out_features, initial_active_fraction=1.0, bias=True):
        # Allocate the full oversized layer
        super().__init__(in_features, out_features, bias=bias)
        # Mask: 1 = active, 0 = frozen/unavailable
        mask = torch.zeros(out_features, in_features)
        # Activate a fraction of weights
        num_active = int(initial_active_fraction * out_features * in_features)
        # Randomly choose which weights are initially active
        flat_indices = torch.randperm(out_features * in_features)[:num_active]
        mask.view(-1)[flat_indices] = 1.0
        self.register_buffer('mask', mask)
        # Track which weights were added post-init (for integration analysis)
        self.register_buffer('expansion_mask', torch.zeros_like(mask))
        # Zero out inactive weights
        with torch.no_grad():
            self.weight.data *= self.mask

    def forward(self, x):
        # Apply mask to ensure inactive weights stay zero
        return F.linear(x, self.weight * self.mask, self.bias)

    @property
    def active_params(self):
        return self.mask.sum().item()

    @property
    def total_params(self):
        return self.mask.numel()

    @property
    def active_fraction(self):
        return self.active_params / self.total_params

    def unmask_weights(self, n_to_unmask, init_method='zero'):
        """Unmask n additional weights.

        Args:
            n_to_unmask: number of weights to activate
            init_method: 'zero' or 'kaiming'

        Returns the number actually unmasked (may be less if not enough inactive weights).
        """
        inactive = (self.mask == 0).nonzero(as_tuple=False)
        n_available = len(inactive)
        n_to_unmask = min(n_to_unmask, n_available)
        if n_to_unmask == 0:
            return 0
        # Randomly select which inactive weights to unmask
        perm = torch.randperm(n_available, device=self.mask.device)[:n_to_unmask]
        selected = inactive[perm]
        self.mask[selected[:, 0], selected[:, 1]] = 1.0
        self.expansion_mask[selected[:, 0], selected[:, 1]] = 1.0
        with torch.no_grad():
            if init_method == 'zero':
                self.weight.data[selected[:, 0], selected[:, 1]] = 0.0
            elif init_method == 'kaiming':
                # Kaiming normal for ReLU: std = sqrt(2 / fan_in)
                std = (2.0 / self.in_features) ** 0.5
                new_vals = torch.randn(n_to_unmask, device=self.weight.device) * std
                self.weight.data[selected[:, 0], selected[:, 1]] = new_vals
            else:
                raise ValueError(f"Unknown init_method: {init_method}")
        return n_to_unmask

    def get_weight_stats(self):
        """Get weight statistics separated by original vs newly expanded weights."""
        with torch.no_grad():
            original_mask = self.mask * (1 - self.expansion_mask)
            new_mask = self.expansion_mask
            w = self.weight.data

            orig_w = w[original_mask.bool()]
            stats = {
                'n_original': int(original_mask.sum().item()),
                'n_new': int(new_mask.sum().item()),
                'original_abs_mean': orig_w.abs().mean().item() if orig_w.numel() > 0 else 0.0,
                'original_std': orig_w.std().item() if orig_w.numel() > 0 else 0.0,
            }
            if new_mask.sum() > 0:
                new_w = w[new_mask.bool()]
                stats['new_abs_mean'] = new_w.abs().mean().item()
                stats['new_std'] = new_w.std().item()
            else:
                stats['new_abs_mean'] = 0.0
                stats['new_std'] = 0.0
            return stats

    def get_gradient_stats(self):
        """Get gradient statistics separated by original vs newly expanded weights."""
        if self.weight.grad is None:
            return None
        with torch.no_grad():
            original_mask = self.mask * (1 - self.expansion_mask)
            new_mask = self.expansion_mask
            g = self.weight.grad.data

            orig_g = g[original_mask.bool()]
            stats = {
                'original_grad_abs_mean': orig_g.abs().mean().item() if orig_g.numel() > 0 else 0.0,
            }
            if new_mask.sum() > 0:
                new_g = g[new_mask.bool()]
                stats['new_grad_abs_mean'] = new_g.abs().mean().item()
            else:
                stats['new_grad_abs_mean'] = 0.0
            return stats


class MaskedMLP(nn.Module):
    """MLP with MaskedLinear layers for headroom experiments.

    The oversized_factor controls how much extra capacity is allocated.
    initial_active_fraction controls what fraction starts active.
    """

    def __init__(self, input_dim=784, hidden_dims=(256, 128), output_dim=10,
                 oversized_factor=1.5, initial_active_fraction=0.6):
        super().__init__()
        # Scale hidden dims by oversized_factor
        scaled_hidden = [int(d * oversized_factor) for d in hidden_dims]
        dims = [input_dim] + scaled_hidden + [output_dim]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(len(dims) - 1):
            if i == len(dims) - 2:
                # Output layer is normal (small, maps to class logits)
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                # All hidden layers (including first) are masked
                self.layers.append(
                    MaskedLinear(dims[i], dims[i + 1],
                                 initial_active_fraction=initial_active_fraction)
                )
            if i < len(dims) - 2:
                self.activations.append(nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x
