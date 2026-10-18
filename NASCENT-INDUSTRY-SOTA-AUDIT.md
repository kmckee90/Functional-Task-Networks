# Nascent: FTN and M2E industry/SOTA audit

Audit date: October 17, 2026 (Pacific). Includes uncommitted and ignored local artifacts and saved Globo histories inspected through `globo1`. No training, model evaluation, or tests were launched. Numerical aggregation of existing pod histories ran on the pod.

## Executive assessment

**Nascent is a credible research project in continual adaptation and autonomous exploration, with meaningful results beyond toy classification, but it has not established general SOTA superiority or a convincing primary mechanism for AGI.** Its strongest integrated evidence is repeated visual exploration in Globo: at 341,000 observations, FTN–M2E covers 2.35 times as many cells as the local Plan2Explore implementation and 17% more than M2E alone. However, the longer saved histories materially qualify the proposal: at 825,000 observations, FTN–M2E and M2E have essentially equal mean coverage, while FTN retains an advantage in the combined achievement/discovery count. FTN also shows positive backward transfer in a five-seed language-classification experiment, useful retention in small language-model pretraining, and strong supported recall across eleven image-generation domains. Those results justify further work. They do not yet establish inexpensive, indefinite learning, reliable acquisition of broadly useful skills, or autonomous scientific reasoning. My assessment is that FTN and M2E could become valuable components of an increasingly capable agent; the evidence and mechanisms currently favor that interpretation over their being the principal breakthrough that produces AGI.

## Scope and provenance

The local checkout is `master`, commit `f479efa521a5ad2eab67d4aa55123c126d549b60`; locally known `origin/master` points to the same commit. This audit did not treat that commit as the project boundary. It inspected the modified arXiv manuscript, untracked proposal and Globo code, current QA evaluations, robotics results, and ignored `e2e/artifacts` and SiT artifacts. No remote branch refresh was performed. A commit alone cannot reproduce this snapshot.

The proposal source is [presentation/proposal.md](presentation/proposal.md). The current FTN manuscript is [manuscripts/arxiv/main.tex](manuscripts/arxiv/main.tex); the external M2E reference is [Meta-Learning to Explore via Memory Density Feedback](https://arxiv.org/html/2503.02831v2). The proposal, older READMEs, cached progress files, and completed artifacts sometimes describe different stages. Where they conflict, this report prefers identifiable result artifacts over prose about intended or ongoing work.

On inspection, `globo1` was running as a workspace, but its experiment records said the user had stopped the explorers. I did not restart them. Shared published histories from other Globo pods were readable through `globo1` under `/mnt/obelisk-shared/kmckee90/globo_phase{4,5}`. These are saved monitor publications, not freshly rerun evaluations. The extracted per-run values, timestamps, source paths, and SHA-256 hashes are preserved in [NASCENT-AUDIT-GLOBO-EVIDENCE.json](NASCENT-AUDIT-GLOBO-EVIDENCE.json). Capture time was October 18, 03:17 UTC, still October 17 Pacific.

This is a research-evidence audit, not exhaustive verification of every experiment directory, checkpoint, rendered video, or software path. Literature comparisons below identify relevant established and contemporary methods; they do not claim an exhaustive October 2026 leaderboard survey.

## What the flagship results actually establish

### 1. Integrated FTN–M2E: repeated Globo exploration

The cleanest repeated comparison uses Phase 4 M2E and convolutional FTN–M2E, and Phase 5 Plan2Explore runs 1–6. All use **world 19**, the same spawn, and learner seeds 1901–1906. These are six learner replicates per method, not six independent worlds. Phase 4 starts fresh; later continuations restore learned state and replay but restart the physical game instance. See [Phase 3](globo_dreamer/PHASE3.md), [Phase 4](globo_dreamer/PHASE4.md), and [Phase 5](globo_dreamer/PHASE5.md).

Means recomputed from saved histories, using the latest recorded sample at or before each horizon:

| Observations per run | M2E explored cells | FTN–M2E explored cells | P2E explored cells | FTN / M2E | FTN / P2E |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 1,003.7 | 1,022.8 | 434.7 | 1.02× | 2.35× |
| 200,000 | 1,217.2 | 1,429.2 | 554.2 | 1.17× | 2.58× |
| 341,000 | 1,449.7 | 1,691.7 | 720.3 | 1.17× | 2.35× |
| 825,000 | 2,395.7 | 2,372.2 | 1,278.0 | 0.99× | 1.86× |

Coverage counts unique **4 m × 4 m horizontal cells**, not navigational task success or learned skills. The 341k numbers agree with the proposal’s saved chart evidence. The 825k horizon is shared by all eighteen selected runs and extends the comparison beyond the proposal cutoff. The FTN–M2E cell counts there are `[1792, 2422, 2526, 2365, 2495, 2633]`; M2E’s are `[2771, 2299, 2218, 2330, 2287, 2469]`. FTN wins five of six paired comparisons but loses one by a large margin; its mean is slightly lower. This is an unstable aggregate advantage, not evidence of sustained mean coverage superiority.

| Horizon | M2E checked achievements/discoveries | FTN–M2E | P2E |
|---:|---:|---:|---:|
| 341,000 | 12.83 | 17.00 | 8.50 |
| 825,000 | 15.83 | 19.67 | 12.67 |

The second table uses the monitor’s checklist definition: unlocked event categories **plus individual discovery entries**. It does not mean twenty independently learned skills. At 825k, M2E and FTN each average **7.17 unlocked categories**, versus 7.00 for P2E; FTN’s remaining aggregate advantage is in discovery entries. That is useful breadth of experience, but materially narrower than a demonstrated larger repertoire of reliable behaviors. See [achievement definitions](globo_dreamer/ACHIEVEMENTS.md) and [counting code](globo_dreamer/achievement_charts.py).

**Verdict:** strong evidence that this M2E-based system collects wider experience than this P2E baseline on this world. Evidence for FTN’s additional contribution is mixed: an intermediate coverage advantage and persistent discovery-count advantage, without a sustained mean coverage advantage. The plots’ shaded ranges are not confidence intervals. Six seeds on a tuned world cannot establish broad environmental generalization.

The proposal’s “6x faster and further than DreamerV3 and Plan2Explore” should not be presented as the replicated result. Its preliminary headline chart contains one early flat-decoder FTN run per comparison. At its saved 311,475-observation endpoint it shows FTN 2,217 cells, M2E 1,160, P2E 734, and Dreamer 195: approximately 3.02× P2E and 11.37× Dreamer at that endpoint, not one uniform 6× effect. Earlier two-world M2E examples also give different ratios. “Faster” on an observation axis means sample efficiency, not measured wall-clock acceleration. Sources: [chart evidence](<presentation/proposal assets/results/globo-report/evidence.json>) and [export code](globo_dreamer/proposal_globo_report.py).

### 2. FTN’s best controlled mechanism evidence

The revised manuscript has useful ablations and is more careful than the proposal. On correlated synthetic tasks, FTN reaches **91.9% ± 2.0%** final accuracy, versus **77.4% ± 8.9%** with random per-batch masks and **78.0% ± 3.6%** without replay. Fixed masks with retrieval reach **94.8% ± 2.1%**, but FTN has much better pre-weight-update head-start loss: **0.399 versus 0.702**. The contribution is a retention/transfer tradeoff, not dominance on every metric. On Permuted MNIST, ordinary replay achieves 96.0% accuracy versus FTN’s 95.4%. [Manuscript comparison table](manuscripts/arxiv/tables/revision_main.tex).

On the six-task Qwen2.5-1.5B classification stream, five seeds give:

| Method | Final accuracy | Mean forgetting | Backward transfer |
|---|---:|---:|---:|
| Sequential LoRA | .417 ± .131 | .550 ± .197 | −.465 ± .174 |
| Sequential LoRA + replay | .750 ± .132 | .145 ± .112 | −.090 ± .129 |
| FTN, random per-batch masks | **.854 ± .118** | .120 ± .074 | −.075 ± .115 |
| FTN-LoRA | .833 ± .101 | **.085 ± .086** | **+.190 ± .101** |

The positive backward transfer is a worthwhile result. But random masks slightly exceed FTN’s final accuracy, and the manuscript attributes substantial retention to the bank/replay pipeline. The strongest distinctive claim is **support-conditioned transfer and recoverability**, not that optimized masks alone solve forgetting. These are six classification tasks, not general language reasoning. Active parameter budgets are matched; the bank has 128 atoms with 16 active. [LoRA table](manuscripts/arxiv/tables/lora_main.tex).

The newer Bayesian router also has a genuine five-seed diagnostic: 84.06% recall versus 77.78% for matched two-step cold LoRA, at 13.7 versus 11.5 seconds per task block. Its improvement is reliability at somewhat higher cost, not a speedup over the equally short-search baseline. [BayesLoRA findings](base/projects/bayesian_adv/FINDINGS.md).

### 3. Continual language-model pretraining: stronger local evidence, still limited

The short replay-free, repaired-mask run retains WikiText with loss **4.932 → 5.218**, after 5,000 WikiText then 5,000 CodeParrot updates. The matched MoE goes **4.712 → 9.261**. That is a substantial retention result. However, FTN ends CodeParrot at **4.639**, versus MoE **4.390**: the short experiment has a plasticity/retention tradeoff. Both use the same validation queries, and the saved MoE audit documents active and trainable parameter matching to within 0.1%. [Short FTN curves](e2e/artifacts/short_sd10/noreplay/curve_data.json), [MoE audit](e2e/artifacts/short_moe/comparison_audit.json).

The longer **corrected-optimizer** FTN run is complete, despite stale prose saying it is ongoing. Its completion artifact records 60,000 updates, zero replay hits, and an unchanged frozen backbone. The saved comparison curves give:

| Method | WikiText loss at 20k boundary | WikiText at 40k | CodeParrot at 40k |
|---|---:|---:|---:|
| Transformer | 4.372 | 7.191 | 5.191 |
| TTT | **4.098** | 6.172 | 4.760 |
| MoE | 4.244 | 7.602 | 5.162 |
| FTN, repaired/refined recall | 4.555 | **5.630** | **4.308** |

At 60k, FTN’s WikiText loss is **5.956** and CodeParrot **4.265**, versus TTT **6.682 / 4.460**. This is encouraging. Nonetheless FTN’s WikiText loss has risen **1.401 nats** from its boundary value, equivalent to roughly a **4.1× perplexity increase**. “Forgetting nearly eliminated” is not an accurate description of this longer result. Do not compare unequal final endpoints for the Transformer and MoE; the table deliberately uses the common 40k checkpoint.

These are small, two-layer, sequence-length-128 models and principally a single-seed comparison, not frontier-scale pretraining. The FTN artifact records **218.37M trainable bank parameters**, **54.59M active**, plus **52.79M frozen random backbone parameters**. “53m” is a backbone nickname, not the entire FTN system’s storage footprint. Repair uses stored training examples and extra optimization. Zero gradient replay during weight training does **not** mean zero retained data or zero later adaptation work. Sources: [comparison curves](outputs/e2e_status/long_noreplay_sd10_isolated/partial_repair_refine4_moe_curve_data.json), [parameters](outputs/e2e_status/long_noreplay_sd10_isolated/parameters.json), [completed run](e2e/artifacts/long_noreplay_sd10_isolated/full_h1024_k256_r5_e1_noreplay_isolated_s123/complete.json), [correction protocol](e2e/artifacts/long_noreplay_sd10_isolated/protocol.json), [support/repair protocol](e2e/BAYES_FULL_MODEL.md).

### 4. Large-model QA: real integration, weak generalization evidence

The October 15 local evaluation uses a configured **Qwen3.6-27B**, with 2,304 rank-one atoms, 256 active, on the last quarter of layers. This is materially beyond the 1.5B manuscript experiment. Input-conditioned routing lowers answer-token cross-entropy from **1.233 to .597**, **.827 to .456**, and **1.190 to .689** across the three project-document topics, approaching oracle routing.

But the evaluation explicitly samples **cached training Q&A**, excluding routing exemplars rather than bank-training examples. It is retained fit, not held-out QA accuracy or a measured forgetting trajectory. The small qualitative generation review gives fully correct answers on **9/27 scorable questions** for input-conditioned FTN versus **6/27** for the base, with no improvement on two of the three topics. It uses training questions and an unblinded assistant reviewer. This demonstrates that routing and adaptation work technically; it does not establish a capable general continual-learning assistant. The artifact also acknowledges incomplete historical configuration provenance. [Protocol](nlp/ftn_chat/evaluation_results/qa_ic_20261015/protocol.json), [metrics](nlp/ftn_chat/evaluation_results/qa_ic_20261015/summary.json), [interpretation](nlp/ftn_chat/evaluation_results/qa_ic_20261015/interpretation.json), [generation review](nlp/ftn_chat/evaluation_results/qa_ic_20261015/generation/correctness_summary.json).

### 5. Diffusion: promising supported recall, a revealing failure mode

The eleven-domain SiT experiment is substantive. At 5% retained memory, the saved direct four-image-support recall report gives **99.2% face acceptance**, **93.0% object acceptance**, and **0.0129 filter KID**. These are evaluator-specific measurements on a small set of identities, objects, and filters; not universal generation-quality scores. The protocol includes retained-memory replay, mask repair, and a readout rebuilt from retained features.

The same frozen bank’s additional mask refinement can severely regress generation: ten refinement steps reduce face acceptance to **9.4%**, and thirty-step variants give **12.0–22.9%**. This directly demonstrates that a support reconstruction objective can improve the wrong thing for identity-preserving generation. The report uses one fixed support draw and one training checkpoint for this comparison. Multiple search seeds are not multiple training replicates. The result is promising for engineered recall, but does not show an automatically reliable inference-time optimizer or superiority over modern continual personalization baselines. [Direct/refinement report](SiT/continual/memory_refinement/artifacts_conservative/report.html), [5% protocol](SiT/continual/memory_refinement/artifacts_conservative/replay_p05_s125/protocol.json), [memory training protocol](SiT/continual/memory_budget/artifacts/protocol.json).

### 6. Robotics and composition

The MT10-trained FTN primitive library contains **36,883 primitives**. The newer blind start-to-goal planner solves **5/40 held-out MT50 tasks**, equal to the prior blind planner’s count, with different wins and losses. The historical replay ceiling is 19/40. This is evidence of some composition across tasks, but weak practical competence. “Blind” means no expert decomposition during planning; the planner still receives simulator state and a goal state, described as a first-success observation. It is not equivalent to visual robot control from a natural-language instruction, nor the official MT50 training protocol. [Results](ftn_robotics/MT50-eval/claude-v3/RESULTS.md), [method](ftn_robotics/MT50-eval/claude-v3/METHOD.md).

An existing symbolic-task audit reports .979 per-position / .918 exact-match held-out performance with labeled support and mask optimization, while direct data-free mask algebra fails. That supports reusable representations under adaptation, not general compositional program induction. I treat this as a secondary existing audit, not a rerun. [NAR audit](nar/codex_audit/HYGIENE_REPORT.md).

## Answers to the audit questions

### 1. Does the method scale?

**It scales to useful prototypes and large frozen backbones; indefinite learning and frontier-scale training remain unestablished.**

There is operational scaling: million-observation M2E runs, hundreds of thousands of integrated FTN observations per replicate, large-model adapters, small generative pretraining, and multi-domain diffusion. Dismissing the whole project as MNIST research would be wrong.

The unresolved axes are number and diversity of retained skills, stream length, support/routing cost, representation complexity, and resource growth. A fixed bank has finite capacity; combinatorially many masks do not imply that many independent learned skills. Shared atoms still interfere. The long LM results show forgetting despite repaired recall. Globo’s additional FTN coverage benefit does not increase monotonically with lifetime. A frozen visual reservoir also limits what the integrated agent can learn to represent; parameter isolation in its decoder does not itself solve representation learning.

The necessary next evidence is a capability-versus-compute/capacity curve across longer, more diverse streams, including saturation and recovery. Running more seeds on separate GPUs establishes replication throughput, not distributed scaling of one learner.

### 2. Does it beat SOTA baselines on the same problem domain?

**It beats selected local comparators on specific metrics. Broad SOTA superiority is not established.**

Globo has a meaningful same-environment comparison. However, its P2E is explicitly a local adaptation onto DreamerV3, not a numerical replication of the original TensorFlow agent. Dreamer and P2E have different model sizes; M2E has much larger replay capacity. The comparison concerns intrinsic exploration without survival rewards, not the full task-solving performance for which Dreamer is known. The local documentation discloses these differences. [Globo README](globo_dreamer/README.md), [Phase 2 configuration](globo_dreamer/PHASE2.md), [original P2E implementation](https://github.com/ramanans1/plan2explore).

DreamerV3 already demonstrates broad control-task competence; Dreamer 4 additionally demonstrates long-horizon Minecraft behavior learned through offline world-model training. Their different data/reward regimes prevent treating those published scores as a direct rebuttal or direct baseline win in Globo. They do show that beating a locally configured P2E exploration curve is insufficient to claim the industry frontier in generally capable agents. [DreamerV3 paper](https://www.nature.com/articles/s41586-025-08744-2), [Dreamer 4 project](https://danijar.com/project/dreamer4/).

For continual learning, compare against more than naive sequential tuning: strong replay/distillation, orthogonal adapters, shared-subspace adaptation, routed independent adapters, and matched-budget MoE. [DER](https://arxiv.org/abs/2004.07211), [O-LoRA](https://arxiv.org/abs/2310.14152), and the 2026 [Share preprint](https://arxiv.org/abs/2602.06043) are relevant examples, not a claim that any automatically wins this project’s protocols. Share is particularly relevant because it addresses shared low-rank continual adaptation across modalities. Published continual-learning successes also contradict the proposal’s blanket implication that existing neural methods cannot retain previous skills. [Continual-T0](https://arxiv.org/abs/2205.12393).

### 3. What are the compute and memory demands?

**Manageable on research accelerators, but substantial and incompletely benchmarked. Sparse activity is not the same as low total cost.**

| Component | Inspected evidence | Implication |
|---|---|---|
| Globo FTN decoder | 3,414,952 active vs 3,411,506 baseline parameters; **27,319,616 bank parameters** | Active budgets match closely; FTN stores about 8× the decoder parameters, before optimizer/target state. |
| Globo replay | 2M main + 500k frontier transitions; 100k density centroids | Bounded, but large. P2E replay is 100k transitions, with a different stored representation. Counts alone are not byte comparisons. |
| Reservoir replay tensors | Each state is 128×4×9, with both current and next state stored in FP16 | At full combined 2.5M capacity, those two tensors alone require approximately **46.1 GB decimal / 42.9 GiB**. This is a capacity calculation, not measured peak allocation. |
| Density lookup | Chunked distance calculations; centroid nearest-distance refresh compares centroid blocks against all centroids | Chunking bounds temporary memory, not total arithmetic. Full nearest-centroid refresh is quadratic in centroid count. |
| Small LM FTN | 218.37M trainable bank + 52.79M frozen base; 54.59M active; completed checkpoint 2.64GB | Storage and training state substantially exceed the nominal backbone label. |
| Recall/repair | Four-example routing; repair can use 32 fit + 32 validation training examples, 100 search steps × 3 restarts | Report adaptation work and retained data separately from the outer weight-update count. |

Sources: saved remote `phase4_conv_r1_s19/architecture.json`; [FTN decoder](globo_dreamer/m2e_ftn.py); [replay representation](globo_dreamer/legacy/agent.py); [scalable replay](globo_dreamer/scalable_replay.py); LM artifacts cited above.

The decoder genuinely indexes selected atoms for ordinary forwards, which is good. But mask-search contributions evaluate the full bank. Inference-time mask optimization therefore has a different cost from a single sparse forward. Similarly, a large frozen LLM still incurs its backbone forward and relevant backward-through-activation costs during routing even when most weights are not trainable.

I found no consolidated, matched report of total accelerator-hours, peak VRAM/RAM, inference latency, energy, and hyperparameter-search expenditure for the flagship methods. Observation-count gains cannot be converted to compute savings from the present charts. Resumes, rendering, different devices, and changing process allocations also make elapsed monitor timestamps a poor substitute for instrumented training time.

### 4. Are the environments toy or practical?

**The portfolio spans toys, serious simulations, and realistic data types; practical autonomous competence is still weakly evidenced.**

Synthetic tasks, MNIST variants, and permutation programs are controlled mechanism tests. Globo is a materially richer 3D visual environment with terrain, animals, resources, and hazards. It is a useful intermediate benchmark, but the evaluated behavior remains chiefly exploration in a game, with simplified controls and a fixed-world repeated comparison. Long lifetimes and realistic-looking videos do not establish robust manipulation, survival, tool use, or industrial utility.

MetaWorld supplies recognizable manipulation tasks, but the project’s 5/40 held-out result is low and simulation-state-conditioned. Text and image experiments use real data types, but two-corpus pretraining, project-document QA, and eleven small image domains are narrow task distributions. The strongest next practical demonstration would be a reliably acquired skill subsequently recalled and composed to solve a new objective, with success measured independently of discovery counters.

### 5. Is it domain general?

**FTN is plausibly a general adaptation mechanism; Nascent is not yet a demonstrated domain-general autonomous learner.**

The architecture has been adapted to classification, language, diffusion, and control. This is genuine architectural breadth. However, each application supplies different representations, losses, support-generation procedures, routing machinery, and replay policies. Those are separate engineered systems, not one agent autonomously acquiring all those competencies.

M2E’s general idea—using memory familiarity as recurrent feedback—is portable. Its practical behavior depends on what the representation regards as novel. A coordinate maze, a fixed visual reservoir, and semantic scientific exploration pose very different problems. Novel observations can be irrelevant, and important causal knowledge can require revisiting familiar observations. The current evidence does not solve that distinction.

### 6. Is the implementation modern by 2026 standards?

**Yes in several engineering choices; no evidence of a SOTA systems implementation.**

Positive evidence includes low-rank banks, mixed precision, JAX compilation in relevant paths, selected-atom forwards, Bayesian/warm routing, optimizer isolation, bounded replay, incremental checkpointing, and extensive diagnostic instrumentation. The correction that prevents inactive atoms moving through Adam/RMSprop state is substantive, not cosmetic. Direct input-conditioned routing also reduces the need to optimize masks from scratch.

These techniques make the project a contemporary research implementation. They do not establish a systems advantage. The saved large-model QA protocol uses eager attention; the small LM protocol records BF16 compute with FP32 parameters/state and no forced FlashAttention. These choices may be justified for compatibility, but they are not evidence of maximized accelerator efficiency. Efficient attention already has mature alternatives such as [FlashAttention-3](https://arxiv.org/abs/2407.08608). I did not find flagship evidence of fused custom bank kernels, demonstrated distributed expert scaling, or latency/throughput comparisons against optimized inference stacks. Absence from the audited evidence is not proof that no such code exists elsewhere.

### 7. What recognized methods and standards can be tested on its own environments?

There is no universal industry certification for continual learners or AGI. There are accepted comparators, public benchmarks, and evaluation practices that can make the claims much more decisive.

| Existing project setting | Comparators to run in that setting | What must be matched/measured |
|---|---|---|
| Globo reward-free exploration | Same control learner with RND, ICM, episodic reachability novelty, density-only novelty; validated P2E/Dreamer configurations; random and simple persistent-movement policies | Same sensors/actions/reset rules; memory bytes; tuning budget; observation and accelerator-hour curves; unseen worlds; achievement success rates after discovery. |
| FTN contribution inside M2E | Same total-capacity bank with fixed/random masks; learned router; retrieval-only router; no support replay; compute-matched larger dense decoder | Separate extra capacity, replay, optimizer protection, and mask optimization from FTN’s claimed reuse benefit. |
| Continual classification | DER-style replay/distillation, parameter isolation and support-conditioned retrieval, prompt/adapter approaches where applicable | Give every method the same support information and optimization budget; evaluate both with-support and strict no-support settings. |
| Continual LM and document QA | Replay LoRA, O-LoRA, shared-subspace methods, routed adapters, matched MoE; retrieval-augmented base for document QA | Held-out questions/documents, general-capability retention, answer correctness, retained bytes, repair latency, model capacity, and data access. |
| Diffusion domains | Sequential LoRA, replay LoRA, independently stored adapters plus routing, continual personalization/shared-subspace methods | Identical pretrained generator and memory budget; held-out identities/instances, multiple support draws, identity and diversity metrics, standard image-quality metrics where sample sizes permit. |
| Robotics | Same-data behavioral cloning and goal-conditioned policy/planner baselines, alongside standard RL reference protocols | Same demonstrations, state/goal privileges, task splits and interaction budgets; repeated success rates rather than one attempt per task. |

Useful implementations and protocol anchors include [URLB’s intrinsic-learning baselines](https://github.com/rll-research/url_benchmark), [episodic curiosity](https://github.com/google-research/episodic-curiosity), [TRACE for continual LLM evaluation](https://github.com/BeyonderXX/TRACE), [MetaWorld](https://github.com/Farama-Foundation/Metaworld), and [ContinualWorld](https://github.com/awarelab/continual_world). Crafter provides a complementary standardized achievement-oriented agent test; it does not replace Globo’s 3D setting. [Crafter](https://github.com/danijar/crafter).

Use predeclared endpoints, held-out environments, independent learner seeds, uncertainty intervals, and probability-of-improvement or robust aggregate measures. The statistical unit is a run/world, not a logged timestep. [Reliable RL evaluation methodology](https://arxiv.org/abs/2108.13264). Publish complete learning curves, including the later horizon where an advantage disappears.

### 8. Do the best experiments “cheat” in ways other methods do not?

**There are important information and accounting advantages, but I found no basis to allege fabricated results or deliberate deception.** Many caveats are already explicitly recorded by the project. The problem is promoting a conditional result into a stronger claim.

| Issue | Finding | Audit judgment |
|---|---|---|
| Labeled support at evaluation | Original FTN receives current-task input-output examples; manuscript comparators receive the same support. | Fair within that protocol. It is not label-free inference. “No task ID” does not mean no information identifying the task. |
| Contradictory-label benchmarks | Identical inputs can require different outputs across tasks. Support-conditioned FTN can disambiguate; an input-only replay model cannot. | A real identifiability advantage. Give baselines contextual evidence too before interpreting the gap as superior memory. |
| Head-start / “zero-shot” language | Mask logits can be optimized using support labels before model weights change. | Pre-weight-update transfer is valid; calling it no-adaptation or zero-supervision transfer would be misleading. |
| Replay-free LM | Weight training has zero replay hits, but retained examples support recall repair and mask selection. | Valid “no gradient replay”; invalid “no memory of old data” or cost-free recall. |
| Large-model QA split | October QA questions come from cached training data. | Retention/fit diagnostic, not held-out generalization. Existing artifacts explicitly disclose this. |
| Robotics “blind” planning | No expert decomposition, but explicit goal and current simulator states; earlier variants used stronger guidance. | Do not compare to pixels-only or instruction-only agents without matching privileges. |
| M2E maze sensors | The original continuous maze provides true coordinates. | Substantially easier perception/localization than visual navigation; useful as a mechanism test. |
| Globo coordinates and events | Inspected environment code keeps position out of returned policy observations; achievements are diagnostic. Frontier documentation uses visual embeddings. | I did not find evidence that the flagship Globo explorer is secretly given the coverage map or achievement labels. Do not import the maze-coordinate criticism into Globo without evidence. |
| Capacity matching | Globo matches active decoder parameters while FTN has about 8× total bank parameters. | Valid conditional-computation comparison, not equal total capacity/memory. The small-LM MoE comparison does better on active and trainable parameter matching. |
| Tuning/world selection | M2E uses tuned config 24; main long replicate set is one world. | Need held-out worlds and equal tuning budgets. This can inflate external validity without corrupting the recorded in-world result. |
| Warm starts and restarts | Some earlier render experiments are warm-started; Phase 4/5 documented fresh starts, then checkpoint resumes with new game instances. | Keep phases separate. Do not mix pretrained initialization into a from-scratch claim or equate resumed lives with an uninterrupted embodiment. |
| Achievement interpretation | Counts mix categories with discovered resource/biome identities; climbing and predator survival are operational proxies. | Discovery is not proof of learned intentional skill. Use repeated goal-directed tests. |
| Best-run and horizon selection | Preliminary headline is individual runs; proposal repeated curve stops at 341k; later data change FTN’s coverage conclusion. | Report historical and latest matched horizons together. A blanket 6× advantage is unsupported by the repeated results. |

The strict-support distinction matters especially for arbitrary label permutations: if the input distribution is unchanged, no model can identify an unseen permutation without additional evidence. Solving that setting with support is useful few-shot adaptation, not a demonstration that a context-free baseline has inherently inferior continual learning.

## Potential as the primary contribution leading to AGI

My sincere assessment is **plausible as an enabling contribution, currently an unconvincing candidate for the primary breakthrough**. This judgment is about the proposed mechanism, not merely the absence of an AGI demonstration.

The positive case is coherent. Persistent useful knowledge and self-directed experience acquisition are real bottlenecks. FTN offers a way to protect and recombine computation; M2E makes exploration responsive to accumulated familiarity. Their integration could improve the data an agent collects and the knowledge it retains. The results provide concrete reasons to investigate that loop, including backward transfer and supported recall outside classification.

The missing causal steps are larger than an engineering scale-up. Selecting and preserving computations does not specify how an agent acquires abstractions, chooses productive experiments, learns causal models, or builds compositional reasoning. Maximizing novelty does not specify what is worth understanding. The current integrated flagship has a fixed visual reservoir, independent decoder neurons, and no Dreamer world model in the M2E/FTN arms. It is not yet exercising the proposed rich cycle of learned representation, predictive understanding, abstract planning, and scientific discovery. The separate large-model applications inherit much of their competence from pretrained backbones; they cannot establish that FTN itself generates that competence.

The most informative observation is that **more exploration has not yet been shown to cause an accelerating accumulation of reusable, independently tested skills**. At the longer matched horizon, FTN’s average spatial advantage disappears and its category breadth is essentially the same as M2E’s. This does not disprove the research direction. It does weaken the specific claim that the current integrated system already demonstrates a self-amplifying intelligence process.

I would prioritize Nascent as a falsifiable research program in continual skill acquisition, with funding and claims tied to that milestone. I would not present “primordial AGI” or emergent scientific reasoning by a particular year as conclusions supported by these experiments. A decisive positive update would be sustained acquisition and recall of many useful skills under fixed resources, followed by composition on genuinely new goals/worlds, beating simpler memory-and-routing alternatives at matched compute.

## Highest-value next evidence

1. **Freeze the current claim set and publish the complete Globo comparison through 825k**, including per-run curves, world count, capability-category counts separately from discoveries, and the active/total-capacity distinction.
2. **Run held-out-world, compute- and memory-accounted ablations of FTN inside M2E.** Add matched total-capacity random/fixed/learned-routing banks. This distinguishes retention/reuse from extra storage and replay.
3. **Convert discoveries into competence tests.** Ask saved agents to repeat resource collection, safe traversal, and interaction outcomes from novel starting states. Then test whether earlier skills survive later learning and compose on unseen objectives.
4. **Extend continual learning beyond two text domains and training QA.** Use independent evaluation data, many domains/revisits, mixed streams, support-budget accounting, and strong adapter/replay baselines. Preserve both direct and repaired recall results.
5. **Measure the full cost and saturation curve.** Include bank/optimizer/replay bytes, training and routing latency, repair work, and tuning expenditure. Plot competence against lifetime data and compute, not just observations.

These are proposed experiments, not runs performed for this audit.
