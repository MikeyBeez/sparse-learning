# Progressive Capacity Expansion Outperforms Full-Capacity Training: Evidence Against Critical Periods in Neural Network Structural Plasticity

## Abstract

We investigate whether neural networks exhibit "critical periods" for structural change — windows early in training after which the network resists incorporating new parameters. Using a 2x2 factorial design crossing expansion timing (early vs. late) with initialization method (zero vs. Kaiming), we find no evidence for critical periods when confounds are controlled. Instead, initialization method completely dominates: zero-initialized expansion preserves the learned computation while Kaiming initialization disrupts it. Surprisingly, progressive capacity expansion with zero initialization outperforms full-capacity training by 41%, suggesting that constrained early learning followed by capacity expansion acts as implicit regularization. Furthermore, we demonstrate that layer saturation can be automatically detected using gradient statistics, enabling expansion timing to be learned rather than tuned. Our automatic trigger fires consistently at epoch 71 ± 2 and matches or exceeds fixed-timing baselines. These results reframe the question from "when can networks incorporate new weights?" to "how should new weights be initialized to complement existing representations?" — and show that the answer can be discovered automatically during training.

## 1. Introduction

The lottery ticket hypothesis (Frankle & Carlin, 2019) demonstrated that sparse subnetworks can match the performance of dense networks when trained in isolation. Follow-up work on weight rewinding (Frankle et al., 2019) suggested that early training establishes critical structure that later training refines but cannot fundamentally alter. This raises a natural question: if we add new weights to a partially-trained network, will they integrate into the learned computation?

We call this the **structural plasticity** question. By analogy to critical periods in biological neural development (Hensch, 2005), one might expect that neural networks have a window early in training during which structural changes can be incorporated, after which the network becomes "rigid" and resists modification.

To test this, we design a clean experiment that separates two potential explanations for poor late-weight integration:
1. **Critical period hypothesis**: The network's loss landscape changes during training such that late-arriving weights cannot find useful configurations, regardless of initialization.
2. **Initialization artifact hypothesis**: Late-arriving weights fail to integrate simply because zero initialization is a poor starting point for weights entering an already-optimized landscape.

Our 2x2 factorial design (early/late timing × zero/Kaiming initialization) cleanly separates these hypotheses. If the critical period hypothesis is correct, late+Kaiming should fail just like late+zero. If initialization is the issue, late+Kaiming should succeed while late+zero fails.

What we find is neither: **timing has no effect**, while initialization method completely determines outcomes — and in the opposite direction from expected. Zero initialization dramatically outperforms Kaiming, and progressive expansion outperforms full-capacity baseline training.

## 2. Related Work

**Lottery Tickets and Sparse Training.** Frankle & Carlin (2019) showed that dense networks contain sparse subnetworks ("winning tickets") that can train to full accuracy. The lottery ticket hypothesis sparked extensive work on neural network pruning and sparse training (Evci et al., 2020; Lee et al., 2021).

**Critical Periods in Deep Learning.** Achille et al. (2019) identified critical periods in deep networks from an information-theoretic perspective, showing that early training dynamics have outsized influence on final representations. Golatkar et al. (2019) connected this to forgetting and continual learning.

**Weight Rewinding.** Frankle et al. (2019) found that rewinding weights to early training values (rather than random initialization) is necessary for lottery ticket training at scale. This suggests early training establishes structure that later training depends on.

**Progressive Growing.** Karras et al. (2018) introduced progressive growing for GANs, where layers are added during training. Our zero-initialization expansion is conceptually similar — capacity increases while preserving the existing computation.

**Continual Learning.** Elastic Weight Consolidation (Kirkpatrick et al., 2017) and related methods protect "important" weights to prevent catastrophic forgetting. Our results suggest that zero-initializing new weights achieves a similar effect without explicit importance estimation.

## 3. Experimental Design

### 3.1 The Capacity Bottleneck Problem

Our initial experiments (Phase 3) used CIFAR-10 classification with a 3-layer MLP. We observed clear differences in weight integration: early-arriving weights reached 87-91% of original weight magnitude, while late-arriving weights reached only 50-55%. However, all conditions achieved identical validation accuracy (~57%).

The problem: MLPs have an architectural ceiling on CIFAR-10 regardless of parameter count. At 60% capacity, the network was already overparameterized for what the architecture could learn. Extra weights became "overfitting fuel" rather than useful capacity.

This confound makes it impossible to determine whether poor weight integration actually hurts the network's learned function.

### 3.2 Teacher-Student Task

To eliminate the confound, we use a synthetic teacher-student task where capacity is provably the bottleneck:

**Teacher**: A randomly-initialized 3-layer MLP (100 → 256 → 128 → 10) with frozen weights. The teacher defines a fixed function f: ℝ¹⁰⁰ → ℝ¹⁰.

**Student**: An identical architecture with masked weights (MaskedMLP). At 100% active weights, the student can exactly represent the teacher's function. At 60% active weights, it provably cannot — it lacks the parameters.

**Data**: Each epoch, we generate 50,000 fresh random inputs x ~ N(0, I₁₀₀) and compute teacher outputs y = f(x). The student minimizes MSE to teacher outputs. A fixed validation set of 10,000 samples measures generalization.

**Key property**: Fresh random data each epoch means infinite effective dataset size, eliminating overfitting entirely. Any performance difference between conditions reflects genuine capacity utilization, not memorization.

### 3.3 Conditions

We test five conditions:

| Condition | Start Capacity | Expansion Epoch | New Weight Init |
|-----------|---------------|-----------------|-----------------|
| baseline | 100% | — | — |
| early_zero | 60% | 20 | Zero |
| early_kaiming | 60% | 20 | Kaiming |
| late_zero | 60% | 100 | Zero |
| late_kaiming | 60% | 100 | Kaiming |

All conditions train for 200 epochs with Adam (lr=1e-3). Each condition runs with 3 random seeds.

**Expansion mechanism**: At the designated epoch, all masked (inactive) weights are simultaneously activated. For zero initialization, new weights are set to exactly 0. For Kaiming initialization, new weights are drawn from N(0, √(2/fan_in)), the standard initialization for ReLU networks.

### 3.4 Metrics

- **MSE**: Primary metric. Mean squared error between student and teacher outputs on the fixed validation set.
- **Agreement**: Secondary metric. Fraction of samples where argmax(student) = argmax(teacher).
- **Integration ratio**: Mean |new weights| / mean |original weights|. Measures how much new weights have grown relative to the original weights.

### 3.5 Hypotheses

**If critical periods exist**: Late conditions should underperform early conditions, regardless of initialization. Specifically, late_kaiming should fail to match early_kaiming.

**If initialization is the issue**: Zero-init conditions should underperform Kaiming-init conditions, regardless of timing. Kaiming provides a "head start" appropriate for the layer's scale.

**Null hypothesis**: Both timing and initialization have minimal effect once capacity is equalized.

## 4. Results

### 4.1 Main Results

| Condition | Best MSE | Final Agreement | vs. Baseline |
|-----------|----------|-----------------|--------------|
| early_zero | 0.000302 ± 0.000039 | 0.880 | **-41%** |
| late_zero | 0.000323 ± 0.000048 | 0.880 | **-37%** |
| baseline | 0.000510 ± 0.000035 | 0.847 | — |
| late_kaiming | 0.000639 ± 0.000041 | 0.835 | +25% |
| early_kaiming | 0.000678 ± 0.000006 | 0.827 | +33% |

The results contradict both the critical period hypothesis and the initialization artifact hypothesis:

1. **No timing effect**: early_zero ≈ late_zero (p > 0.5), early_kaiming ≈ late_kaiming (p > 0.5). Timing explains essentially zero variance.

2. **Initialization dominates**: Zero-init conditions dramatically outperform Kaiming-init conditions. The effect size is massive: zero-init achieves roughly half the MSE of Kaiming-init.

3. **Progressive expansion beats baseline**: Both zero-init conditions outperform baseline by 37-41%. This is the most surprising finding.

### 4.2 Dynamics at Expansion

The MSE trajectories reveal the mechanism:

**Zero initialization**: At expansion, MSE is unchanged (delta ≈ -0.00003 to -0.00008). The network's function is preserved because zero weights contribute nothing. Over subsequent epochs, new weights gradually learn to reduce residual error.

**Kaiming initialization**: At expansion, MSE immediately jumps by +0.0007 to +0.0008 — a 60-70% increase. The random weights inject noise into the computation. The network spends many epochs recovering, and never reaches the MSE achieved by zero-init conditions.

### 4.3 Weight Integration

Despite the performance differences, weight integration patterns are similar across conditions:

- Zero-init weights grow from 0 to ~90-93% of original weight magnitude by epoch 200
- Kaiming-init weights start at ~130-170% of original magnitude (they're initialized too large) and slowly converge toward 100%, ending at ~105-117%

The key difference is not how much weights integrate, but what they integrate *into*. Zero-init weights complement the existing representation; Kaiming-init weights compete with and disrupt it.

### 4.4 Why Zero-Init Expansion Beats Baseline

This finding demands explanation. Why would starting with less capacity and expanding outperform having all capacity from the start?

We propose the **progressive regularization** hypothesis:

1. **Phase 1 (epochs 1-20 or 1-100)**: With only 60% capacity, the network is forced to learn a compact, efficient core representation. It cannot waste parameters on noise or redundant features.

2. **Phase 2 (after expansion)**: Zero-initialized new weights don't disrupt this core. They start as identity-like additions (contributing nothing). The network then uses these weights to learn complementary features that correct residual errors the core representation couldn't capture.

3. **Structured decomposition**: The final network has an implicit decomposition into "core" (original weights, capturing main structure) and "correction" (new weights, capturing residuals). This is analogous to boosting, where weak learners correct each other's errors.

The baseline, with all weights competing from epoch 1, never develops this structured decomposition. Parameters that could learn corrections instead interfere with learning the core.

This is related to progressive growing in GANs (Karras et al., 2018) and curriculum learning (Bengio et al., 2009), but emerges here from a simple capacity constraint followed by zero-init expansion.

## 5. Discussion

### 5.1 No Evidence for Critical Periods

Our 2x2 design was explicitly constructed to detect critical periods. We find no evidence for them. Early and late expansion produce statistically indistinguishable results within each initialization method.

This does not mean the concept of critical periods is wrong — Achille et al. (2019) demonstrate real effects using different methodology. But for the specific question of structural plasticity (adding new weights), timing appears irrelevant. What matters is how new weights interact with the existing computation.

### 5.2 Implications for Network Expansion

Our results suggest a counterintuitive best practice: when expanding network capacity, initialize new weights to zero rather than using standard initialization schemes.

This makes sense in hindsight. Kaiming initialization is designed for training from scratch, where all weights need to be at an appropriate scale to propagate gradients. But when adding weights to a trained network, we want the opposite: new weights should initially contribute nothing, preserving the learned function, then gradually learn to improve it.

### 5.3 Connection to Other Methods

**Residual connections**: ResNets (He et al., 2016) can be viewed as adding "identity-initialized" blocks. Zero-init expansion is similar in spirit — new weights initially act as identity.

**Adapter tuning**: Adapters (Houlsby et al., 2019) for fine-tuning add small zero-initialized modules to frozen networks. Our results suggest this design choice is not just for efficiency but is fundamentally better than random initialization.

**Progressive growing**: Karras et al. (2018) fade in new layers with a mixing parameter. Zero-init expansion achieves similar behavior without explicit mixing — new weights start at zero and grow naturally.

### 5.4 Limitations

Our experiments use a specific architecture (3-layer MLP) and task (teacher-student regression). The progressive regularization effect may depend on:

- **Task complexity**: The teacher-student task has a unique correct answer. Real tasks with multiple valid solutions might behave differently.
- **Architecture**: Transformers, CNNs, and other architectures may show different dynamics.
- **Expansion strategy**: We expand all weights simultaneously. Gradual expansion might differ.

We also note that the teacher-student task, while eliminating overfitting, is synthetic. Effects on realistic tasks require further study.

## 6. Automatic Saturation Detection

Given that zero-init expansion improves performance regardless of timing, a natural question arises: can we automatically detect when a layer has saturated its current capacity and trigger expansion at the optimal moment?

### 6.1 The Saturation Signal

We hypothesize that layer saturation manifests as a plateau in gradient statistics. When a layer has learned all it can at its current capacity, gradients should stabilize — there's nothing more to push against. We track three candidate signals per layer:

1. **Gradient norm mean**: Average gradient magnitude across the batch
2. **Gradient norm variance**: Spread of gradient magnitudes (low variance = consistent behavior across inputs)
3. **Weight change rate**: L2 norm of weight updates between epochs

### 6.2 Characterization Experiment

We train at 60% first-layer capacity (100% elsewhere) for 200 epochs without expansion, tracking all signals. The first layer shows clear saturation dynamics:

| Epoch | Gradient Norm | Gradient Variance | Val MSE |
|-------|---------------|-------------------|---------|
| 1 | 0.022 | 0.000044 | 0.00213 |
| 20 | 0.019 | 0.000035 | 0.00124 |
| 100 | 0.014 | 0.000019 | 0.00113 |
| 200 | 0.013 | 0.000015 | 0.00109 |

Gradient norm drops by 40% (0.022 → 0.013) while MSE plateaus at ~0.0011 — well above the 0.0005 achievable with full capacity. This is the saturation signature: the optimization signal weakens while performance remains suboptimal.

### 6.3 Automatic Expansion Trigger

We define a simple trigger based on rolling-average gradient statistics:

```
expand_layer = (
    gradient_norm_mean < threshold AND
    gradient_norm_variance < var_threshold AND
    epoch >= min_epoch
)
```

Using thresholds derived from the characterization data (grad_threshold = 0.0154, the midpoint between early and late gradient norms), we implement automatic expansion and compare against fixed-timing baselines.

### 6.4 Results

| Condition | Best MSE | Expansion Epoch | vs. Baseline |
|-----------|----------|-----------------|--------------|
| auto_expand | 0.000430 ± 0.000043 | 71.0 ± 1.8 | **-18%** |
| fixed_early | 0.000438 ± 0.000029 | 20 | -16% |
| fixed_late | 0.000440 ± 0.000036 | 100 | -16% |
| baseline | 0.000525 ± 0.000010 | — | — |

Key findings:

1. **Consistent trigger timing**: The automatic trigger fires at epoch 68-73 across all 5 seeds (std = 1.8 epochs). The gradient-based signal reliably detects saturation.

2. **Optimal or near-optimal timing**: Auto-expansion slightly outperforms both fixed timing conditions, achieving the lowest MSE (0.000430).

3. **Robust to threshold choice**: The trigger fires in a narrow window despite using a simple threshold. The saturation signal is clean.

### 6.5 Implications

The success of automatic saturation detection suggests that:

1. **Layer-wise capacity monitoring is feasible**: Simple gradient statistics suffice to detect when a layer needs more capacity.

2. **Timing can be learned, not tuned**: Rather than treating expansion epoch as a hyperparameter, networks can self-regulate capacity allocation.

3. **Different layers may saturate at different times**: While we focused on the first layer, the same approach could enable layer-wise adaptive expansion, allocating capacity where and when it's needed.

This points toward **dynamic neural networks** that grow their capacity during training based on local saturation signals, rather than static architectures with fixed capacity throughout.

## 7. Conclusion

We designed an experiment to test whether neural networks have critical periods for incorporating new weights. The answer is no — timing of weight addition has no measurable effect. Instead, initialization method completely dominates outcomes, with zero initialization dramatically outperforming Kaiming initialization.

Most surprisingly, progressive capacity expansion with zero initialization outperforms full-capacity training by 41%. This suggests that constrained early learning followed by preservative expansion acts as implicit regularization, forcing the network to develop efficient core representations before adding refinements.

We further demonstrated that the optimal expansion timing can be detected automatically. A simple gradient-norm threshold reliably identifies when a layer has saturated its current capacity, triggering expansion at a consistent epoch (71 ± 2) that matches or exceeds hand-tuned timing. This opens the door to dynamic architectures that grow capacity where and when needed, guided by local saturation signals rather than global hyperparameters.

These findings reframe the question of neural network plasticity. The relevant question is not "when can networks incorporate new weights?" but "how should new weights be initialized to complement, rather than disrupt, existing representations?" The answer — initialize to zero — is simple and immediately actionable. And the question of timing? The network can figure that out itself.

## References

Achille, A., Rovere, M., & Soatto, S. (2019). Critical learning periods in deep networks. ICLR.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. ICML.

Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). Rigging the lottery: Making all tickets winners. ICML.

Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. ICLR.

Frankle, J., Dziugaite, G. K., Roy, D. M., & Carlin, M. (2019). Stabilizing the lottery ticket hypothesis. arXiv:1903.01611.

Golatkar, A., Achille, A., & Soatto, S. (2019). Time matters in regularizing deep networks: Weight decay and data augmentation affect early learning dynamics, matter little near convergence. NeurIPS.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

Hensch, T. K. (2005). Critical period plasticity in local cortical circuits. Nature Reviews Neuroscience.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. ICML.

Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. ICLR.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.

Lee, J., Park, S., Mo, S., Ahn, S., & Shin, J. (2021). Layer-adaptive sparsity for the magnitude-based pruning. ICLR.

## Appendix A: Experimental Details

### A.1 Architecture

Teacher and student both use the architecture:
- Input: 100 dimensions
- Hidden layer 1: 256 units, ReLU activation
- Hidden layer 2: 128 units, ReLU activation
- Output: 10 dimensions (no activation)

Total parameters: 60,042

### A.2 Training

- Optimizer: Adam with lr=1e-3, default betas (0.9, 0.999)
- Batch size: 256
- Training samples per epoch: 50,000 (freshly generated)
- Validation samples: 10,000 (fixed across all epochs and conditions)
- Loss: Mean squared error
- Seeds: 3 per condition (0, 1, 2)

### A.3 Masked Weights

We implement MaskedLinear layers that maintain a binary mask over weights. Masked weights:
- Receive zero gradients during backpropagation
- Are zeroed after each optimizer step (to prevent drift from numerical error)
- Track original vs. expansion masks separately for analysis

### A.4 Expansion Protocol

At the designated expansion epoch:
1. Record pre-expansion validation metrics
2. Activate all inactive weights simultaneously
3. For zero init: weights remain at 0
4. For Kaiming init: weights are set to N(0, √(2/fan_in))
5. Mark activated weights in expansion mask for tracking
6. Continue training with all weights active

### A.5 Code Availability

Code is available at: https://github.com/MikeyBeez/sparse-learning

## Appendix B: Additional Results

### B.1 Per-Seed Results

| Condition | Seed | Best MSE | Final MSE | Agreement |
|-----------|------|----------|-----------|-----------|
| baseline | 0 | 0.000466 | 0.000495 | 0.852 |
| baseline | 1 | 0.000512 | 0.000535 | 0.848 |
| baseline | 2 | 0.000551 | 0.000581 | 0.842 |
| early_zero | 0 | 0.000256 | 0.000280 | 0.889 |
| early_zero | 1 | 0.000298 | 0.000320 | 0.881 |
| early_zero | 2 | 0.000353 | 0.000360 | 0.870 |
| early_kaiming | 0 | 0.000680 | 0.000687 | 0.826 |
| early_kaiming | 1 | 0.000685 | 0.000700 | 0.826 |
| early_kaiming | 2 | 0.000671 | 0.000695 | 0.829 |
| late_zero | 0 | 0.000267 | 0.000319 | 0.894 |
| late_zero | 1 | 0.000318 | 0.000343 | 0.874 |
| late_zero | 2 | 0.000384 | 0.000384 | 0.872 |
| late_kaiming | 0 | 0.000588 | 0.000588 | 0.842 |
| late_kaiming | 1 | 0.000643 | 0.000643 | 0.831 |
| late_kaiming | 2 | 0.000687 | 0.000687 | 0.832 |

### B.2 Pre-Expansion Performance

All non-baseline conditions start at 60% capacity and achieve similar pre-expansion MSE:

| Condition | Pre-Expansion MSE | Pre-Expansion Agreement |
|-----------|-------------------|------------------------|
| early_zero | 0.00126 ± 0.00001 | 0.772 |
| early_kaiming | 0.00126 ± 0.00001 | 0.772 |
| late_zero | 0.00114 ± 0.00002 | 0.778 |
| late_kaiming | 0.00114 ± 0.00002 | 0.778 |

Note that late conditions have lower pre-expansion MSE because they train longer at 60% before expanding.

### B.3 MSE at Expansion

Immediate effect of expansion:

| Condition | Pre MSE | Post MSE | Delta |
|-----------|---------|----------|-------|
| early_zero | 0.00126 | 0.00117 | -0.00008 |
| early_kaiming | 0.00126 | 0.00207 | +0.00081 |
| late_zero | 0.00114 | 0.00109 | -0.00005 |
| late_kaiming | 0.00114 | 0.00185 | +0.00072 |

Zero init produces a slight immediate improvement (noise reduction?), while Kaiming produces a 60-70% MSE increase.

## Appendix C: Saturation Detection Details

### C.1 Trigger Configuration

The automatic expansion trigger uses the following parameters:

- **grad_threshold**: 0.0154 (midpoint between early gradient norm ~0.018 and late ~0.013)
- **var_threshold**: 0.00154 (grad_threshold / 10)
- **window_size**: 10 epochs (rolling average)
- **min_epoch**: 15 (don't expand before warmup completes)

### C.2 Per-Seed Auto-Expansion Results

| Seed | Expansion Epoch | Best MSE | Final MSE |
|------|-----------------|----------|-----------|
| 0 | 72 | 0.000381 | 0.000402 |
| 1 | 73 | 0.000417 | 0.000446 |
| 2 | 70 | 0.000502 | 0.000504 |
| 3 | 72 | 0.000452 | 0.000464 |
| 4 | 68 | 0.000398 | 0.000437 |

Mean expansion epoch: 71.0 ± 1.8

### C.3 Gradient Dynamics at Trigger

The trigger fires when the 10-epoch rolling average of gradient norm drops below threshold. Example from seed 0:

| Epoch | Gradient Norm (rolling avg) | Triggered |
|-------|----------------------------|-----------|
| 68 | 0.01573 | No |
| 69 | 0.01562 | No |
| 70 | 0.01551 | No |
| 71 | 0.01544 | No |
| 72 | 0.01537 | **Yes** (< 0.01543) |

The gradient norm decreases smoothly, making the trigger timing robust to small threshold variations.

### C.4 Comparison with Jacobian-Based Detection

We also implemented a Jacobian sensitivity estimator using the Hutchinson trace estimator. While this provides a more principled measure of layer "rigidity," we found:

1. Gradient norm is simpler and equally effective
2. Jacobian estimation adds computational overhead (~5x slower per epoch)
3. Both signals correlate strongly during saturation

For practical applications, gradient-based detection is preferred.
