# FTN Compositional Program Induction on an Abelian Task Grid

**Run:** `results/main2` (seed 0; seed-replication table below) · **Hardware:** actl pod `nar`, 1×H100 · **Wall clock:** ~20 min train+eval per seed

---

## 1. Question

Can a Functional Task Network — a fixed bank of neurons specialized per task by a
binary top-k mask, trained **block-sequentially** with **no task identity ever given
as input** — *compose* solutions to data-generating algorithms it was never trained
on, with full out-of-distribution correctness verified against the generator?

This is a program-induction framing: each task is an algorithm (a program), the mask
is the program's address in the bank, and the test is whether never-trained programs
are reachable by **spatial selection alone** — no weight updates at any point.

## 2. The task family

![task explainer](results/main2/fig1_task.png)

Sequences of L=4 tokens over an alphabet of p=10 values, one-hot encoded (a 4×10
binary matrix; a 40-dim input). One task cell (a, r) applies

```
y_i = x_{(i−r) mod L} + a   (mod p)
```

i.e. a **value shift** T_a (columns of the one-hot matrix rotate by a) composed with
a **cyclic rotation** R_r (rows rotate by r). Properties that make this the right
minimal testbed:

- Every cell is a **fixed permutation of the 40 input bits** — trivially expressible
  by the backbone, so every result is about the *mask mechanism*, never expressivity.
- The operator set is the **abelian group Z₁₀ × Z₄**: all cells commute, so
  "compose these operators" is well-defined *without an order* — exactly what
  order-free spatial selection can express, and the property that licenses studying
  composition without temporal selection.
- **Task identity is never in the input.** The same x maps to different y under
  different cells; the mask is the entire program.

**Grid split.** 20 of 40 cells are trained (block-sequentially, shuffled, identity
excluded); 20 are held out, always including the identity (0,0). Coverage
constraints guarantee every a-level and r-level appears in training (a held-out cell
with a never-seen factor level is unidentifiable in principle). **Input split:** all
10⁴ sequences enumerated; each cell trains on a fixed random 50%, all accuracies
reported on the other 50% — so trained-cell numbers are already input-OOD
(systematic generalization to unseen token combinations), and held-out-cell numbers
are OOD in both senses at once.

Chance = 0.10 per-position, 10⁻⁴ exact-match.

## 3. Model: FTN², two canonical FTN stages in series

Each stage is exactly the FTN of the paper: a bank of H independent deep neuron-MLPs
with scalar output z_j = f_j(·), and a linear readout of `Dropout(z) ⊙ m` under a
binary per-bank top-k mask. Stage 1's readout produces a 64-dim interface code;
stage 2 reads it and produces the 40 output logits (4 softmax heads):

```
h = W_mid( Dropout(bank₁(x)) ⊙ m₁ )        bank₁: H=2048 neurons, k=256 active
ŷ = W_out( Dropout(bank₂(h)) ⊙ m₂ )        bank₂: H=2048 neurons, k=256 active
```

Neurons are depth-4, width-8 MLPs (the paper's "small but deep" pyramidal-neuron
abstraction); dropout p=0.2; 7.4M parameters total; the mask is the concatenation
(m₁, m₂) ∈ {0,1}⁴⁰⁹⁶ with per-bank KWTA.

**The one deliberate deviation from single-bank FTN, and why it is forced.** A
single additive readout Σᵢ zᵢ(x) wᵢ cannot express these composites: if the composite
solution decomposed as h(x; a, r) = f(x; a) + g(x; r), then h(x; a, r) − h(x; 0, r)
would be independent of r, but it equals (S^r ⊗ (S^a − I))x, which depends on r. So
composition must happen **through depth** while selection stays purely spatial.
FTN² is the minimal such extension: two unchanged FTN stages stacked, mask semantics
untouched. Everything else — the AnnealSTE support-conditioned configurer, per-batch
reconfiguration, mask-overlap-filtered replay (ρ ∈ (0.2, 0.8]), optimizer reset at
block boundaries, dropout-as-unreliability — is the canonical pipeline (Algorithm 1
of the FTN paper).

**What a neuron ends up being here:** a selectable piece of routing. The ideal
factored circuit is "bank-1 assembly relabels values, bank-2 assembly relocates
positions"; whether training *finds* that factored organization is one of the
experiment's questions (§6 answers: it does not need to for composition to work,
and in fact it doesn't).

## 4. Training protocol

Block-sequential stream over the 20 training cells, **3 cycles** × 300 Adam steps
per visit (lr 1e-3, batch 256, optimizer reset every block), primitives-first
curriculum within each cycle (axis cells (a,0)/(0,r) before composites). Every
minibatch: draw a 128-example support batch from the current cell, run the
configurer (S=15 inner steps, weights frozen) to get the batch's mask, one gradient
step through the active mask on current loss + replay loss (up to 8 mask-overlap-
filtered slots per step from a 200-slot buffer).

Iteration history that got here (all runs preserved under `results/`):

| run | change | trained (reconfig) | held-out (reconfig) |
|---|---|---|---|
| `main0` | single pass, lr 3e-3, 800 steps/cell, 4 replay slots | 0.651 | 0.558 |
| `main1` | lr 1e-3, 600 steps, 8 replay slots, post-stream consensus masks | 0.719 | 0.601 |
| `main2` | **3 cycles × 300 steps + primitives-first curriculum** | **0.986** | **0.979** |

Two findings from the single-cell sweep that fed the design, both consistent with
the FTN conjectures: **large k is the decisive lever** (k=25% of H took one-cell
accuracy from 0.62 to 1.00; capacity/distributedness, not depth or width, was the
bottleneck), and per-batch reconfiguration produces highly variable assemblies
(within-block mask overlap ~0.2–0.3, barely above the k/H chance rate) — solutions
are *populations*, which is why block-end mask snapshots go stale under later-block
drift (0.70 stored vs 0.99 consensus below) and why masks must be read out against
the final weights.

![training](results/main2/fig2_training.png)

## 5. Results

### 5.1 Retention (trained cells, held-out input sequences)

After the full stream, on the final frozen weights:

| protocol | per-position acc | exact match |
|---|---|---|
| block-end mask snapshot (stale pointer) | 0.704 | — |
| post-stream consensus mask (5 configurer restarts, averaged, top-k) | **0.991** | 0.965 |
| reconfig protocol (fresh support-conditioned search) | 0.986 | — |

No recency gradient remains; every one of the 20 cells is at 0.95+.

### 5.2 Held-out cells: composition by spatial selection (the headline)

![grid accuracy](results/main2/fig3_grid_accuracy.png)

**Reconfig protocol: 0.979 per-position, 0.918 exact-match, across all 20
never-trained cells** (the never-trained identity cell: 0.996/0.984). Weights are
frozen; the configurer sees 128 labeled examples of the new algorithm and takes 60
gradient steps *on mask logits only*. The composed program is then verified against
the data-generating algorithm on 5,000 input sequences the network never saw during
any training — full OOD on both axes at once.

![methods](results/main2/fig4_methods.png)

### 5.3 The controls that make it a result

![controls](results/main2/fig7_controls.png)

- **Random-bank control (decisive):** the same configurer, same budget, on an
  identical but untrained FTN² scores **0.115 ≈ chance** (best cell 0.14). The
  composition lives in what block-sequential training built into the bank, not in
  the search procedure's expressivity.
- **Joint task-conditioned MLP — the "this is already solved by representation
  learning" baseline — fails completely: 0.997 on trained cells, 0.086 on held-out
  cells.** Given task identity as a one-hot input and i.i.d. joint training (both
  denied to FTN), a standard MLP memorizes the 20 training cells and transfers
  nothing: a permutation-valued task grid under one-hot task codes gives it nothing
  to interpolate. Combinatorial generalization on this family is *not* free.
- **Program induction from few examples:** support of 4 examples → 0.57; 8 → 0.77;
  16 → 0.88; 32 → 0.94; 128 → 0.98.
- **Almost no search needed:** S=5 configurer steps already reach 0.91 (S=60:
  0.98). The frozen bank is so close to containing the composite that the
  configurer's job is nearly retrieval — the paper's "head start" made extreme.

### 5.4 Algorithm vs. generic function approximation: the off-group probe

Held-out-cell accuracy alone cannot distinguish "the bank learned the operator
family" from "the bank learned a generic one-hot-routing basis that lets the
configurer fit any structured relabeling" (compositional function approximation,
as in the original FTN's transfer results). The discriminating test is
**reachability of off-group tasks at matched complexity**: bijections on the same
input space, same support (128) and search budget (S=60), graded by algebraic
distance from the trained family (`scripts/offgroup_probe.py`,
`results/main2/offgroup_probe.json`):

| tier | probe class | trained bank | random bank |
|---|---|---|---|
| 0 | held-out cells of the trained group Z₁₀×Z₄ | **0.982** (0.96–1.00) | 0.11 |
| 1 | product maps S₄×S₁₀ outside the abelian subgroup | **0.733** (0.51–0.93) | 0.11 |
| 2 | non-product bit-permutations (position-dependent value maps) | **0.497** (0.46–0.55) | 0.12 |

Reachability falls in a clean staircase tracking the group → product-schema →
unstructured hierarchy. This rules out both extremes: a generic routing basis
would make all bit-permutations comparably reachable (they are not), and a
lookup library of the 40 trained cells would put tier 1 at the random-bank floor
(it is not — never-trained operator *types* such as value negation, 0.93, and
position reversal, 0.85, are highly reachable, including reversal composed with
a trained-type shift, 0.87). The supported reading: the bank learned the task
family's **operator schema** — one uniform value bijection applied at every
position, composed with one position bijection — with fidelity highest on the
trained subgroup, partial competence across the schema's wider orbit, and
breakdown precisely where the schema is violated (tier 2 requires
position-dependent value maps; accuracy halves).

Caveats, stated plainly: (i) tier 1's internal spread suggests proximity effects
alongside schema abstraction — its easiest probes are near-identity (val-swap
fixes 8/10 values: 0.89) or reflection-like, while full-support random product
maps sit at 0.51–0.60, so part of tier-1 competence is likely partial overlap
with trained routing sub-circuits (disentangleable by regressing probe accuracy
on agreement with the nearest trained cell); (ii) tier 2 at 0.50 still far
exceeds the random bank's 0.11, so training also built broadly reusable routing
competence — the schema-specific and generic components coexist, and this probe
quantifies the mix; (iii) none of this implies a symbolic representation of the
algorithm (§5.6) — the claim "learned the algorithm" holds extensionally and
structurally, not intensionally.

### 5.5 Two routes to the same held-out algorithm

Because the group is cyclic, iterating cell (a, r) k times must equal the one-shot
cell (ka mod 10, kr mod 4) — which we arranged to be held out:

![consistency](results/main2/fig6_consistency.png)

The **temporal route** — apply a *trained* cell's mask autoregressively k times,
re-encoding the argmax between steps — reaches the held-out composite at **0.99
(k=2) to 0.965 (k=4)** per-position accuracy, verified step-by-step against the
generator. This is fully **zero-shot**: no data from the composite cell, no support
set, no search. Composites in the cyclic subgroup of any trained cell come for free,
with mild error compounding as the only cost.

### 5.6 The honest negative: data-free mask *algebra* fails, and why

The strongest version of the claim — synthesize a held-out cell's mask from grid
structure alone (additive factor completion μ + α(a) + β(r), majority-vote unions,
or hard bank-to-factor assignment), zero data — sits at chance (0.04–0.09), for
every synthesizer variant, in every run.

![mask structure](results/main2/fig5_mask_structure.png)

The diagnostics say why, and the explanation is informative rather than merely
disappointing:

- **Solutions are populations, not canonical circuits.** Five configurer restarts on
  the *same* cell against the *same* weights agree only partially; within-block
  per-batch masks overlap near chance. There is no "the" mask for a cell to do
  algebra on — averaging incompatible members of a solution population does not
  yield a member.
- **No depth factorization emerged.** ANOVA of consensus mask scores across the grid:
  the additive component is substantial (~0.62) but **both banks are a-dominated**
  (a ≈ 0.65, r ≈ 0.21–0.26 in *each* bank) — training never specialized bank 1 to
  values and bank 2 to positions, and nothing in the objective asked it to.
  Same-a / same-r / neither pairwise overlaps are nearly identical (0.42/0.36/0.39).
  The factor structure that exists in mask statistics is generic task similarity;
  the functional circuit lives in the interaction residual, which is exactly the
  part grid-averaging destroys.

So: **the compositional basis demonstrably exists in the frozen weights** (§5.2's
0.98 with 16–128 examples; §5.5's zero-shot iteration), but it is stored as a
distributed, non-unique population — addressable by support-conditioned search,
not by closed-form mask arithmetic. This cleanly separates two hypotheses that
held-out accuracy alone would have conflated.

### 5.7 Seed replication

Headline numbers across 3 seeds (mean ± sd; each seed redraws the grid split, input
split, task order, and init):

| metric | mean ± sd (3 seeds) |
|---|---|
| trained cells, consensus mask | 0.990 ± 0.001 (exact 0.963 ± 0.004) |
| trained cells, reconfig protocol | 0.986 ± 0.001 |
| **held-out cells, reconfig protocol** | **0.973 ± 0.005** (exact 0.901 ± 0.015) |
| held-out composites, zero-shot temporal iteration | 0.977 ± 0.003 |
| held-out cells, additive (data-free) synthesis | 0.086 ± 0.005 (chance) |
| joint task-conditioned MLP, held-out | 0.096 ± 0.010 (chance) |

## 6. What this says, and what it doesn't

**Established.** A canonical-pipeline FTN (two stages deep), trained block-
sequentially on 20 cells of an abelian operator grid with no task identity in the
input, holds a **compositional basis** in its frozen weights: all 20 never-trained
cells — new data-generating algorithms, including compositions never assembled
anywhere in training — are solved at 0.98 per-position / 0.92 exact-match by
spatial mask selection alone, from as few as ~16 labeled examples, with ~5 mask-
gradient steps, at full OOD verified against the generator on never-seen inputs.
A matched random bank is at chance (composition was *learned*), and a jointly
trained task-conditioned MLP is at chance on the same cells (composition here is
*not* what standard representation learning gives you for free — the objection that
this testbed is already solved is empirically false in its strongest available
form). Cyclic-subgroup composites are additionally reachable **zero-shot** by
temporal iteration of a trained mask. And what the bank holds is not an
undifferentiated function-approximation basis: the off-group probe (§5.4) shows
reachability ordered by algebraic distance from the trained family — near-perfect
on the group orbit, partial across the family's product schema (including
never-trained operator types like negation and reversal), collapsing when the
schema is violated.

**Not established.** Masks did not become factored symbols: data-free mask algebra
fails because per-cell solutions are non-unique populations and no bank-to-factor
specialization emerged. "FTN discovers reusable *addresses* whose algebra mirrors
the task algebra" remains open — the measured obstacle is solution non-uniqueness,
which suggests concrete next steps: a canonicalization pressure during training
(e.g. mask-consistency regularization across a block, or configurer warm-starting
from the cell's running consensus), or synthesis in the space of *unit scores*
under the trained configurer prior rather than raw binary masks.

**Scope.** By design this testbed removes NAR's usual concerns (size generalization,
algorithmic alignment): fixed-shape tasks, an MLP-sufficient function class. What it
isolates is the mechanism NAR-style composition would need from a continual learner:
recombination of frozen competence. The natural escalation is non-commutative
operator sets (e.g. add mod p vs multiply mod p), where order matters and spatial
selection alone provably cannot suffice — the concrete motivation for temporal
selection in FTN.

## 7. Reproduction

```bash
# on the actl pod `nar`, from /home/dev/workspace
uv run python run_experiment.py --config full --run main2 --seed 0   # ~20 min on 1xH100
PYTHONPATH=. uv run python scripts/controls.py --run main2           # random-net + sweeps
PYTHONPATH=. uv run python scripts/offgroup_probe.py --run main2     # algorithm-vs-approximation probe
PYTHONPATH=. uv run python scripts/make_fig7.py main2
PYTHONPATH=. uv run python scripts/refigure.py main2                 # regenerate figs 1-6
```

Layout: `ftn_nar/{config,data,model,configurer,train,synthesis,baselines,evals,viz}.py`;
artifacts per run under `results/<run>/` (checkpoint, consensus masks, results.json,
controls.json, train_log.txt, figures). `main0`/`main1` are the preserved iteration
history; `main2_s1`/`main2_s2` the seed replications.
