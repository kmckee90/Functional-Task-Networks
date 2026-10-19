Implement a new reconfiguration option alongside AnnealSTE. Suggested name: `trajlin`.

The method is successive/local trajectory linearization with a trust region. The idea is to run the real recurrent FTN at a current sequence of hard masks, linearize the recurrent dynamics around that execution trace, optimize all step masks cheaply against that fixed linear surrogate, make only a limited number of hard mask swaps, evaluate the real model again, and relinearize.

Assume the recurrent FTN currently has the form

```text
h[0] = x

for t in 0..T-1:
    z[t] = f(h[t])                       # [batch, H]
    a[t] = z[t] * m[t]                  # m[t]: [H], k-hot
    logits += W_y(a[t])                 # skip/output path
    h[t+1] = W_h(a[t])                  # recurrent path
```

Use the actual repo implementation if there are minor differences. If `W_y` or `W_h` are step-specific, substitute the appropriate weights at each step.

The important distinction from AnnealSTE is that `trajlin` does not differentiate through TopK during its main optimization. Hard masks define the anchor trajectory. Around that trajectory, masks are temporarily treated as continuous variables constrained to `[0,1]` with exactly `k` total mass.

For a current hard mask sequence `m_star[t]`, first run one exact rollout on the fixed support set and store

```text
h_star[t]
z_star[t] = f(h_star[t])
logits_star
```

for every processing depth.

Then compute the Jacobian of the population output with respect to the recurrent state at each depth:

```text
J[t] = d f(h) / d h evaluated at h_star[t]
```

For a batch,

```text
J[t].shape = [batch, H, state_dim]
```

The preferred implementation is forward-mode differentiation (`torch.func.jvp`, `jacfwd`, optionally `vmap`) because the current recurrent FTN is expected to have a relatively small recurrent-state dimension and a very large `H`. Do not compute one reverse-mode gradient per neuron.

If `state_dim = D`, a straightforward implementation is to perform `D` JVPs for each processing step, one for each state-space basis direction, then stack the resulting `[batch,H]` tangents into `[batch,H,D]`.

Freeze all normal model parameters during this procedure. The only objects optimized later are temporary relaxed masks.

Given candidate relaxed masks `mu[t]`, define

```text
delta_m[t] = mu[t] - m_star[t]
```

and evaluate the local surrogate without calling `f` again:

```text
delta_h = zeros_like(h_star[0])
delta_logits = 0

for t in 0..T-1:
    delta_z = J[t] @ delta_h

    delta_a =
        m_star[t] * delta_z
        + z_star[t] * delta_m[t]

    delta_logits += linear_without_bias(W_y, delta_a)
    delta_h = linear_without_bias(W_h, delta_a)

logits_linearized = logits_star + delta_logits
```

For batched tensors the `J[t] @ delta_h` operation is, for example,

```python
delta_z = torch.einsum("bhd,bd->bh", J[t], delta_h)
```

The equation for `delta_a` comes from the first-order expansion

```text
(z_star + delta_z) * (m_star + delta_m)

≈ z_star*m_star
  + m_star*delta_z
  + z_star*delta_m
```

and deliberately drops the second-order term

```text
delta_z * delta_m.
```

That omitted term is part of the nonlinear/compositional interaction. It is recovered progressively by rerunning the real model and relinearizing after mask updates.

Biases in `W_h` and `W_y` must NOT be added in the delta rollout. They are already present in the anchor trajectory and cancel when computing perturbations. Use only the linear weight matrices for `delta_h` and `delta_logits`.

Optimize the exact existing support loss on

```text
logits_linearized
```

rather than linearizing the loss itself. For cross-entropy this means that the local logits are affine in the relaxed masks while cross-entropy remains the real cross-entropy function. This is preferable to using only a first-order loss gradient.

The relaxed mask constraint at every processing depth is

```text
0 <= mu[t,i] <= 1
sum_i mu[t,i] = k
```

Initialize

```text
mu[t] = float(m_star[t])
```

and optimize all `T` masks jointly.

Use Adam for the first implementation. After every optimizer step, project each `mu[t]` back onto the capped simplex

```text
{u : 0 <= u_i <= 1, sum(u) = k}.
```

Projection can be implemented by finding a scalar threshold `tau` such that

```text
u = clamp(v - tau, 0, 1)
sum(u) = k.
```

A 32-iteration bisection is sufficient:

```python
def project_capped_simplex(v, k, n_iter=32):
    lo = v.min(dim=-1, keepdim=True).values - 1.0
    hi = v.max(dim=-1, keepdim=True).values

    for _ in range(n_iter):
        tau = (lo + hi) * 0.5
        u = (v - tau).clamp(0.0, 1.0)
        s = u.sum(dim=-1, keepdim=True)

        # sum is too large -> increase tau
        lo = torch.where(s > k, tau, lo)
        hi = torch.where(s > k, hi, tau)

    return (v - hi).clamp(0.0, 1.0)
```

Run several cheap inner optimization steps against the fixed local surrogate. These inner steps must not call `f` and must not rerun the true recurrent network. `J`, `z_star`, `m_star`, `W_h`, and `W_y` should all be detached constants during the inner solve.

At the end of the inner solve, convert each relaxed mask to an intended hard target:

```text
target[t] = TopK(mu[t], k)
```

Do NOT immediately replace the current mask with the entire target. The linearization is only reliable locally.

Instead apply a hard swap trust region. If the current mask has `k` active neurons, allow at most

```text
max_swaps = max(1, round(k * swap_frac))
```

active neurons to be replaced at a given depth during one outer iteration.

For each step `t`:

```text
add_candidates  = target active AND m_star inactive
drop_candidates = m_star active AND target inactive
```

Rank additions by highest `mu[t]` value and drops by lowest `mu[t]` value. Exchange at most `max_swaps` matched pairs. The resulting candidate remains exactly k-hot.

Run the REAL recurrent FTN with this candidate hard mask trajectory on the same support set and compute the true support loss.

If true loss improves, accept the candidate:

```text
m_star = candidate
```

and begin another outer iteration by rerunning the true trajectory and recomputing the Jacobians around the new execution trace.

If true loss becomes worse, backtrack the trust region. Halve the number of permitted swaps and reevaluate. Repeat until either an improving candidate is found or the swap count reaches zero. If no improving hard update can be found, terminate reconfiguration early and return the best hard mask trajectory seen so far.

Keep the support set fixed for the entire reconfiguration call. Do not resample it between outer iterations, because the accept/reject logic assumes that predicted and actual losses refer to the same objective.

Configuration also needs to be deterministic while constructing and testing a local model. If the population contains dropout or any other stochastic operation, either disable it during reconfiguration or guarantee that exactly the same stochastic realization is used for the anchor, Jacobian construction, and candidate evaluation. Otherwise the trust-region comparison becomes noisy and the local Taylor model is not modeling the function that is actually being evaluated.

The high-level routine should therefore look approximately like this:

```python
def reconfigure_trajlin(model, support_x, support_y, init_masks, cfg):
    masks = init_masks.clone()
    best_masks = masks.clone()

    with torch.no_grad():
        best_loss = true_support_loss(model, support_x, support_y, masks)

    swap_frac = cfg.TRAJLIN_SWAP_FRAC

    for outer in range(cfg.TRAJLIN_OUTER_STEPS):

        # Expensive stage.
        trajectory = rollout_and_store(
            model,
            support_x,
            masks,
        )
        # contains h_star, z_star, logits_star

        J = compute_population_state_jacobians(
            model,
            trajectory.h_star,
            mode=cfg.TRAJLIN_JACOBIAN_MODE,
        )

        # Cheap local optimization.
        mu = masks.float().detach().clone().requires_grad_(True)
        optimizer = Adam([mu], lr=cfg.TRAJLIN_INNER_LR)

        for inner in range(cfg.TRAJLIN_INNER_STEPS):
            logits_lin = linearized_rollout(
                model,
                trajectory,
                J,
                masks,
                mu,
            )

            loss_lin = criterion(logits_lin, support_y)

            optimizer.zero_grad()
            loss_lin.backward()
            optimizer.step()

            with torch.no_grad():
                mu.copy_(
                    project_capped_simplex(
                        mu,
                        k=model.k,
                        n_iter=cfg.TRAJLIN_PROJECTION_ITERS,
                    )
                )

        target = hard_topk(mu.detach(), model.k)

        # Trust-region update.
        candidate = move_toward_target_by_swaps(
            masks,
            target,
            mu.detach(),
            max_swaps=max(
                1,
                round(model.k * swap_frac),
            ),
        )

        candidate_loss = true_support_loss(
            model,
            support_x,
            support_y,
            candidate,
        )

        # Backtracking if necessary.
        swaps = max(1, round(model.k * swap_frac))

        while candidate_loss >= best_loss and swaps > 1:
            swaps = max(1, swaps // 2)

            candidate = move_toward_target_by_swaps(
                masks,
                target,
                mu.detach(),
                max_swaps=swaps,
            )

            candidate_loss = true_support_loss(
                model,
                support_x,
                support_y,
                candidate,
            )

        if candidate_loss < best_loss:
            masks = candidate
            best_masks = candidate.clone()
            best_loss = candidate_loss
        else:
            break

    return best_masks
```

Reuse the existing AnnealSTE initialization/warm-start logic when possible so comparisons do not confound the optimizer with different starting masks. The first implementation should not call AnnealSTE internally; initialize `trajlin` from the same initial hard mask sequence AnnealSTE would start from.

Add the following configuration parameters to `config.py`:

```python
# Existing selector or equivalent:
RECONFIG_METHOD = "annealste"   # alternative: "trajlin"

# Number of expensive relinearization iterations.
TRAJLIN_OUTER_STEPS = 6

# Number of cheap optimization steps against each fixed linear surrogate.
TRAJLIN_INNER_STEPS = 32

# Adam learning rate for relaxed mask variables.
TRAJLIN_INNER_LR = 0.1

# Maximum fraction of the k active neurons that may be replaced
# at one processing depth during one outer update.
TRAJLIN_SWAP_FRAC = 0.05

# Iterations used to project relaxed masks onto
# 0 <= m <= 1, sum(m) = k.
TRAJLIN_PROJECTION_ITERS = 32

# Jacobian implementation. "forward" means JVP/jacfwd with
# respect to recurrent state h.
TRAJLIN_JACOBIAN_MODE = "forward"
```

These should be the only exposed parameters initially. Hard-code backtracking to halve the number of swaps, terminate when a one-swap candidate still fails to improve, and require strict true-loss improvement for acceptance. Add more trust-region parameters only if experiments show they are necessary.

The defaults are intended for the current roughly 8-processing-step recurrent FTN:

```text
outer steps       = 6
inner steps       = 32
inner LR          = 0.1
swap fraction     = 0.05
projection iters  = 32
Jacobian mode     = forward
```

The most important sweep initially is

```text
TRAJLIN_OUTER_STEPS:  3, 6, 10
TRAJLIN_SWAP_FRAC:    0.02, 0.05, 0.10
TRAJLIN_INNER_STEPS:  16, 32, 64
```

Do not interpret `TRAJLIN_INNER_STEPS` as equivalent to AnnealSTE steps. A trajectory-linearization inner step should contain no calls to the expensive population network `f`; it is only tensor arithmetic using the cached trajectory and Jacobians.

Instrument the implementation carefully. At minimum log, per outer iteration:

```text
true anchor loss
linearized loss at anchor
linearized predicted candidate loss
true candidate loss
number of swaps at each depth
whether backtracking occurred
wall-clock time spent building Jacobians
wall-clock time spent in the cheap inner solver
wall-clock time spent in true candidate rollouts
```

Also log

```text
predicted improvement = anchor_true_loss - candidate_linearized_loss
actual improvement    = anchor_true_loss - candidate_true_loss
```

This is the main diagnostic for whether the local approximation is useful. If predicted and actual improvements agree reasonably well for 5% mask swaps, trajectory linearization is working as intended. If predicted improvement is consistently positive while actual improvement is negative, reduce `TRAJLIN_SWAP_FRAC`. If they agree very closely and progress is slow, increase it.

The method is likely to be computationally attractive only while recurrent `state_dim` is small. Storing the population Jacobians costs approximately

```text
batch_size * T * H * state_dim
```

elements. Forward-mode Jacobian construction also scales roughly with `state_dim`. This is favorable for the current synthetic recurrent FTN if the recurrent state is low-dimensional, but it may become inappropriate if this architecture is later scaled to a large hidden state. Add an assertion or warning when `state_dim` becomes large enough that the Jacobian tensor would be unreasonable.

The skip-to-output path needs no special treatment. It appears directly in the linear surrogate through

```text
delta_logits += W_y(delta_a)
```

and gives every step an immediate first-order route to the loss. The recurrent path

```text
delta_h = W_h(delta_a)
```

simultaneously propagates the effect of an early mask through all later processing stages. Thus the surrogate contains both the direct linear contribution of a mask decision and its first-order compositional consequences.

Conceptually, this is not another STE. AnnealSTE evaluates the actual nonlinear recurrent model after every mask-logit update and uses an approximate derivative through hard selection. `trajlin` instead evaluates the nonlinear model relatively infrequently, builds a first-order model of the entire execution trajectory, and performs many mask updates against that cached model before executing the nonlinear network again.

A useful name in comments/documentation is:

```text
trust-region successive trajectory linearization
```

or

```text
sequential convex programming (SCP) over recurrent FTN masks
```

It is closer to SCP/successive linearization than to classical iLQR because the controls are sparse masks, the final candidate is projected to hard top-k selections, and the local objective can use the existing cross-entropy loss directly rather than a quadratic cost.