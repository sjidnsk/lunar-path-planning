# 巡策: Topology-Aware Coverage Policy Network Design

## Summary

`巡策` is the short code name for `Topology-Aware Coverage Policy Network
Design`. It is a research-only network architecture line for improving
exploration coverage and generalization while keeping inference small enough for
the current `model-explorer` policy path. Its first objective is higher
exploration coverage and stronger generalization. Lower parameter count and
faster inference are hard constraints. Novelty is a research goal, but it must
remain compatible with the current project contracts.

The design upgrades the current candidate scorer from local candidate ranking to
coverage-memory-aware candidate graph ranking. It does not replace the planner,
change the action space, install default policy, connect a real executor,
publish checkpoints, or claim real-world performance.

Stage 18.7 is a prerequisite audit before using candidate-count arguments to
justify architecture or reward changes. It tests only candidate-set cardinality:
`dynamic_max_candidates_per_step` is swept over `6/12/24/36`, with proposal pool
limits `48/96/192/288`. Candidate generation mode, selection mode, validation
mode, coverage metric, canonical reward/guard profile, checkpoints, action
space, network, and default A* must remain unchanged. Each sweep must emit
candidate metric audit rows and must pass through Stage 18.6 guard refinement;
Stage 18.7 may report that higher counts expose more guard-clean candidates, but
it cannot authorize Stage 19 or training by itself.

## Platform Boundary

The Xunce research chain must use the Python-first cross-platform runner for
new offline evidence refreshes. `scripts/run_stage.py` is the canonical entry
for registered Stage 15-18 commands and the adjacent supported path-feedback /
guarded research orchestration stages. Bash/PowerShell files are wrappers only.
Windows support is scoped to the non-Drake offline profile; `pydrake` IRIS/GCS
backends remain Ubuntu/Linux optional diagnostics and must not block Windows
validation.

Full platform validation uses `scripts/run_platform_validation_matrix.py`.
Windows and Ubuntu non-Drake profiles cover platform tests, Xunce Stage 15-18,
path-feedback, model-explorer, path-planner non-Drake, and visual-workbench
backend/frontend checks. Ubuntu Drake is optional and only runs when `pydrake`
imports successfully.

Do not write `/home/kai/...` or machine-local `C:\Users\...` paths into tracked
configs. Large downloads and raw map products stay outside Git, with Windows
defaults under `D:\CodexDownloads\lunar-path-planning`.

## Current Fit

The existing policy path is already shaped as a candidate-set scorer:

- `ModelExplorerContract` carries `GoalCandidate` entries with stable fields and
  extensible `experimental` features.
- `extract_policy_observation` builds candidate features, global features,
  missing indicators, candidate cells, and action masks.
- `TorchPolicyScorer` produces masked logits for guarded ranking.
- Existing architectures are `mlp_v1`, `mlp_missing_v1`, and
  `candidate_attention_v1`.

This means the new architecture can remain a drop-in candidate-logit network:
it can add topology and memory features without changing default policy,
planner, executor, or action-space contracts.

## Research References

The design draws from these paper families:

- Set-structured attention: Set Transformer,
  https://arxiv.org/abs/1810.00825
- Attention for routing heuristics: Attention, Learn to Solve Routing Problems!,
  https://arxiv.org/abs/1803.08475
- Graph attention over structured relations: Graph Attention Networks,
  https://arxiv.org/abs/1710.10903
- Long-horizon sequence conditioning: Decision Transformer,
  https://arxiv.org/abs/2106.01345
- Efficient long-context sequence modeling: Mamba,
  https://arxiv.org/abs/2312.00752
- Modular learned exploration with classical planning boundaries: Active Neural
  SLAM, https://arxiv.org/abs/2004.05155
- Learned path-planning heuristics as diagnostic inspiration only: Neural A*,
  https://arxiv.org/abs/2009.07476

These references are not implementation requirements. They define the design
space: set ranking, graph/topology reasoning, long-horizon memory, and modular
learning around existing planners.

## Innovation Points

### Candidate Graph Reasoning

The current candidate scorer primarily evaluates candidate rows. The proposed
network builds a candidate graph:

- nodes are frontier or waypoint candidates;
- edges encode BFS distance, frontier-cluster membership, ROI relation,
  bottleneck relation, coverage overlap, and revisit relation;
- graph-level state encodes coverage memory, remaining budget, recent waypoint
  behavior, and replay context.

The model asks not only which candidate is locally attractive, but what role
the candidate plays in the global exploration structure.

### Topology-Biased Attention

The design extends ordinary candidate self-attention with exploration-specific
edge bias:

- candidates in the same frontier cluster can share local context;
- candidates that cover overlapping cells can suppress redundant choices;
- bottleneck candidates can receive explicit structural emphasis;
- high-cost / low-new-coverage relations can be penalized;
- ROI-group and split/family context can influence candidate contrast.

This is different from generic attention because the attention weights are
conditioned on exploration topology and budget structure.

### Coverage Memory Token

The network receives a compact memory token rather than a dense map encoder. The
token summarizes:

- current covered-rate and uncovered-rate;
- remaining budget;
- recent waypoint direction and path-cost trend;
- repeated-path tendency;
- ROI-group coverage state;
- fallback, risk, and path-feedback counters;
- optional replay family context.

This gives the network a small long-horizon signal without introducing a large
map-scale Transformer or CNN.

### Ranking-Only Integration

The network remains a candidate ranking model:

- input: existing candidate/global features plus optional graph/memory features;
- output: masked candidate logits and value estimate;
- consumer: guarded blended ranking and shadow/replay runners.

It does not take over route generation. It does not call path-planner. It does
not issue executor commands.

## Architecture Sketch

```text
ModelExplorerContract
  -> PolicyObservation v1.1 candidate/global features
  -> TopologyObservation extension
       candidate_edge_features
       candidate_graph_mask
       coverage_memory_token
       roi/family context
  -> Candidate encoder
  -> Topology-biased candidate attention / message passing
  -> Memory-conditioned fusion
  -> masked logits + value
```

The first implementation candidate should be small:

- 1 or 2 topology attention/message-passing layers;
- 16 or 32 maximum candidates by default;
- hidden dimension no larger than existing baseline sweeps unless justified;
- no dense image/map encoder;
- no large sequence model in v1;
- optional Mamba/sequence module deferred until memory-token evidence shows a
  clear bottleneck.

## Observation Contract

The initial extension should be additive and backwards compatible.

Candidate-level additions may include:

- `frontier_cluster_id`
- `roi_group_id`
- `new_coverage_cell_count`
- `coverage_overlap_count`
- `bfs_distance_from_current`
- `path_bottleneck_score`
- `revisit_path_cell_count`
- `budget_fraction_cost`
- `fallback_risk`

Edge-level additions may include:

- `same_frontier_cluster`
- `same_roi_group`
- `bfs_distance_between_candidates`
- `coverage_overlap_ratio`
- `shared_bottleneck`
- `mutual_redundancy_score`

Memory-token additions may include:

- `coverage_rate`
- `remaining_budget_fraction`
- `recent_path_cost_trend`
- `recent_new_coverage_trend`
- `revisit_rate`
- `fallback_rate`
- `roi_group_completion_ratio`

All added fields should remain optional in the source contract. Missing fields
must use explicit missing indicators rather than silent zero-only semantics.

## Evaluation Contract

The research line must compare against existing architectures before any
promotion:

- `mlp_v1`
- `mlp_missing_v1`
- `candidate_attention_v1`
- `topology_aware_coverage_graph_v1`

Primary metrics:

- required-scenario pass rate;
- aggregate and minimum achieved coverage;
- real-map multi-ROI family pass rate;
- policy better/worse than baseline counts;
- controlled regression count;
- guard fallback rate;
- path-budget efficiency;
- deterministic source-match stability.

Efficiency constraints:

- parameter count should not exceed `candidate_attention_v1` by more than 2x
  unless the design review explicitly approves it;
- inference latency should not exceed `candidate_attention_v1` by more than
  1.5x on the same candidate count;
- peak candidate count and default hidden size must be reported in artifacts.

## Required Artifacts

The research line should start with evidence before training:

- `network-literature-and-bottleneck-audit-summary.json`
- `topology-aware-observation-contract-audit.json`
- `topology-aware-architecture-inventory.json`
- `architecture-contrast-evaluation-summary.json`
- `architecture-contrast-results.jsonl`
- `topology-aware-coverage-policy-report.md`

If later training is added, it must use a separate guarded training stage and
must not publish a default-policy checkpoint by default.

## Non-Goals

This design does not:

- replace default policy;
- directly replace default policy / 不直接替换 default policy;
- publish a checkpoint;
- connect a real executor;
- start online canary traffic;
- run PPO update in the design stage;
- modify action space;
- modify default A*;
- claim Ackermann-feasible trajectory;
- treat IRIS/GCS/path-planner diagnostics as release proof;
- claim real-world performance;
- block the already passing Global 99 release-governance path.

## Auditable Development Chain

This is the authoritative 巡策 stage chain. 每一个阶段都必须先设计详细计划，再执行实现。 The plan for each stage must be
written to `docs/superpowers/plans/YYYY-MM-DD-xunce-<stage>.md` before the
stage runner/config/test/docs are implemented. Each stage must have an
independent output root with summary, manifest, audit, report, and rejection
artifacts. Each stage should be committed and pushed separately.

0 冻结巡策设计
   - Freeze this design, README, architecture report, and Global 99 cross-link.
   - Current implementation target: `scripts/run_xunce_design_freeze_audit.py`.
   - Output root: `outputs/path_feedback_batch_xunce_design_freeze_v1/`.
   - Passing next gate: `current_head_evidence_refresh`.
1 Current-HEAD Evidence Refresh
   - Refresh current evidence for Global 99, real-map replay, and controlled
     default-policy candidate installation preflight.
   - If provenance is dirty or source-match fails, repair evidence before
     network research.
   - Current implementation target:
     `scripts/run_xunce_current_head_evidence_refresh.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1/`.
   - Passing next gate: `network_literature_bottleneck_review`.
2 文献与项目瓶颈审计
   - Map literature to actual project bottlenecks.
   - Decide whether the bottleneck is coverage, generalization, fallback,
     latency, parameter count, or not currently network-related.
   - Current implementation target:
     `scripts/run_xunce_network_literature_bottleneck_review.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1/`.
   - Passing next gate: `topology_observation_contract`.
   - Passing does not approve training; it only authorizes additive observation
     contract design.
3 Topology Observation Contract
   - Design additive topology/edge/memory fields.
   - Preserve `policy-observation/v1.1`, candidate mask, old checkpoint loading,
     and scorer compatibility.
   - Current implementation target:
     `scripts/run_xunce_topology_observation_contract.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_topology_observation_contract_v1/`.
   - Passing next gate: `topology_feature_extraction_audit`.
   - This gate writes a contract only; field extraction starts in Stage 4.
4 Topology Feature Extraction Audit
   - Verify stable generation of frontier cluster, ROI group, BFS distance,
     coverage overlap, revisit, budget, fallback risk, and missing indicators.
   - Current implementation target:
     `scripts/run_xunce_topology_feature_extraction_audit.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/`.
   - Passing next gate: `topology_aware_coverage_graph_proto`.
   - No training.
5 小型巡策原型
   - Implement `topology_aware_coverage_graph_proto_v1`.
   - Verify candidate graph plus memory token can output masked logits/value.
   - Current implementation target:
     `scripts/run_xunce_topology_graph_proto.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_topology_graph_proto_v1/`.
   - Passing next gate: `xunce_proto_mechanism_validation`.
   - No checkpoint publication.
6 原型机制验证
   - Prove topology/memory signals can alter ranking without breaking masks,
     guarded ranking, or fallback boundaries.
   - Current implementation target:
     `scripts/run_xunce_proto_mechanism_validation.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_proto_mechanism_validation_v1/`.
   - Passing next gate: `architecture_contrast_evaluation`.
7 架构对比评测
   - Compare `mlp_v1`, `mlp_missing_v1`, `candidate_attention_v1`, and the
     巡策 prototype.
   - Track coverage, minimum scenario coverage, policy better/worse,
     controlled regression, fallback, path-budget efficiency, params, latency.
   - Current implementation target:
     `scripts/run_xunce_architecture_contrast_evaluation.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1/`.
   - Passing next gate: `full_xunce_network_v1_design`.
   - This gate is a deterministic contrast audit, not training or performance
     proof.
8 完整巡策网络 v1
   - Build candidate graph encoder, topology-biased attention/message passing,
     coverage memory token, ROI/family/budget fusion, masked logits, value head,
     and metadata compatibility.
   - Current implementation target:
     `scripts/run_xunce_full_network_v1.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_full_network_v1/`.
   - Passing next gate: `full_network_static_contract_validation`.
   - This implements a research-only forward module, not production architecture
     registration or training.
9 完整网络静态合同验证
   - Verify shape, mask, metadata, missing-field handling, and old-observation
     compatibility.
   - Current implementation target:
     `scripts/run_xunce_full_network_static_contract_validation.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1/`.
   - Passing next gate: `full_network_ablation_experiments`.
   - This gate checks static compatibility only, not performance.
10 完整网络消融实验
   - Isolate candidate graph, topology bias, memory token, ROI/family context,
     and budget fusion contributions.
   - Current implementation target:
     `scripts/run_xunce_full_network_ablation_experiments.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1/`.
   - Passing next gate: `full_network_stress_evaluation`.
   - This gate verifies module contribution only, not performance.
11 完整网络压力评测
   - Test candidate count, missing fields, no-NaN behavior, latency, parameters,
     fallback rate, and deterministic replay stability.
   - Current implementation target:
     `scripts/run_xunce_full_network_stress_evaluation.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_full_network_stress_evaluation_v1/`.
   - Passing next gate: `guarded_training_candidate_preflight`.
   - This gate validates stability only, not training readiness by itself.
12 Guarded Training Candidate Preflight
   - Permit controlled training only if the full network passes contrast,
     ablation, stress, and efficiency gates.
   - Current implementation target:
     `scripts/run_xunce_guarded_training_candidate_preflight.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_guarded_training_candidate_preflight_v1/`.
   - Passing next gate: `controlled_training_candidate`.
   - This gate authorizes the next stage only; it does not run PPO.
13 受控训练候选
   - Current implementation target:
     `scripts/run_xunce_controlled_training_candidate.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_controlled_training_candidate_v1/`.
   - Run bounded deterministic supervised surrogate training as a research
     candidate only, after Stage 12 authorization.
   - It may write `xunce-controlled-training-candidate.pt` as a research-only
     checkpoint under the output root.
   - It must keep `publishes_checkpoint=false`, `runs_new_ppo_update=false`,
     `ppo_update_executed=false`, and must not replace default policy.
   - Passing next gate: `post_training_offline_evaluation`.
14 训练后离线评测
   - Current implementation target:
     `scripts/run_xunce_post_training_offline_evaluation.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_post_training_offline_evaluation_v1/`.
   - Load the Stage 13 research checkpoint read-only and compare it with a
     deterministic fresh `xunce_full_network_v1`.
   - Audit target loss, target probability, target logit, latency, parameter
     count, checkpoint schema, and boundary fields.
   - It must not run new training or PPO update, and must not publish the
     checkpoint.
   - Passing next gate: `shadow_replay_validation`.
15 Shadow / Replay 验证
   - Current implementation target:
     `scripts/run_xunce_shadow_replay_validation.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_shadow_replay_validation_v1/`.
   - Reload the Stage 13 research checkpoint read-only and replay the Stage 14
     deterministic topology-contract cases.
   - Verify source-match, determinism, finite outputs, and closed release /
     executor / online-canary boundaries.
   - Passing next gate: `sandbox_candidate_preflight`.
16 Sandbox Candidate Preflight
   - Current implementation target:
     `scripts/run_xunce_sandbox_candidate_preflight.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/`.
   - Package and load as sandbox candidate with checkpoint hash verification,
     kill-switch, rollback, telemetry, read-only default-policy boundary, and
     executor isolation.
   - This is not default-policy installation.
   - Passing next gate: `xunce_release_governance_gate`.
17 发布治理门禁
   - Current implementation target:
     `scripts/run_xunce_release_governance_gate.py`.
   - Output root:
     `outputs/path_feedback_batch_xunce_release_governance_gate_v1/`.
   - Audit evidence lineage, release scope, boundary fields, and final
     governance verdict from the sandbox candidate evidence.
   - Passing next gate: `xunce_research_track_complete`.
   - This completes the research evidence chain only; it still does not approve
     default-policy replacement, real-world release, executor connection, or
     online canary traffic.

18 高保真真实地图对比延伸
   - This is a post-research-chain extension, not a mandatory release stage.
   - ROI expansion target:
     `scripts/run_xunce_high_fidelity_real_map_roi_expansion.py`.
   - ROI expansion output root:
     `outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1/`.
   - Expand quasi-real LOLA evidence from 12 slices / 4 ROI groups to 24 slices
     / 8 ROI groups while auditing context IDs, contract/sidecar paths,
     path-feedback evidence, and open-grid fallback.
   - Use the existing LOLA LDEM/LDEC quasi-real data by default; do not download
     additional map products unless the 24/8 expansion remains low-spread or a
     later illumination/shadow-specific question requires it.
   - Passing next gate: `xunce_high_fidelity_real_map_policy_comparison`.
   - Policy comparison target:
     `scripts/run_xunce_high_fidelity_real_map_comparison.py`.
   - Policy comparison output root:
     `outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1/`.
   - Compare the Xunce sandbox candidate and incumbent experimental policy in
     read-only mode on the expanded ROI evidence by loading both checkpoints and
     running the same observation/candidate batch through real model inference.
   - The comparison must write
     `xunce-high-fidelity-model-inference-audit.json` and
     `xunce-high-fidelity-model-inference-results.jsonl`, including logits,
     masked logits, probabilities, selected action/rank, value, latency,
     checkpoint load state, and actual parameter counts.
   - Path-feedback candidate rows are observation/evidence inputs only; they
     must not be used as a proxy for Xunce model selection.
   - If evidence is valid but advantage is not established, next gate:
     `xunce_research_iteration_required`.
   - If advantage is established without worse/regression/fallback/efficiency
     blockers, next gate:
     `xunce_default_policy_candidate_authorization_preflight`.
   - This still does not publish checkpoints, replace default policy, train PPO,
     modify network/action space/default A*, connect a real executor, start
     online canary traffic, or claim real-world performance.
   - Stage 18C target:
     `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`.
   - Stage 18C output root:
     `outputs/path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_v1/`.
   - Run a 24-scenario x 10-step offline shadow rollout over the Stage 18A
     high-fidelity scenarios with the same Xunce and incumbent checkpoints used
     by Stage 18B.
   - Stage 18B is the one-step true-inference comparison. Stage 18C is the
     multi-step exploration coverage comparison: coverage return, cumulative
     coverage delta, coverage curve AUC, new/revisited cells, ROI-family
     minimum coverage, path/risk efficiency, mask/fallback violations, action
     entropy, selected probability/rank, latency, and parameter count.
   - Stage 18C must keep `proxy_selection_used=false`. Path-feedback evidence
     can define candidates, path cost, risk, and coverage effects, but it must
     not choose actions for either model.
   - Coverage advantage is valid only when coverage return and coverage AUC
     improve without mask, safety, fallback, path-cost, or risk-efficiency
     regression. Otherwise the route is `xunce_research_iteration_required` or
     `refine_coverage_reward_and_cost_guard`, not default-policy replacement
     and not additional network complexity by default.
   - Stage 18D target:
     `scripts/run_xunce_coverage_discriminability_audit.py`.
   - Stage 18D output root:
     `outputs/path_feedback_batch_xunce_coverage_discriminability_audit_v1/`.
   - Stage 18D diagnoses why Stage 18C may show many model disagreements but
     no coverage separation. It checks candidate coverage spread, static
     candidate reuse, Pareto coverage/cost opportunities, ROI-group spread,
     oracle coverage return, oracle regret, useful disagreement opportunities,
     and whether endpoint footprint coverage is too coarse.
   - Stage 18C-v2 is enabled through dynamic coverage rollout options:
     `candidate_refresh_mode=dynamic_from_coverage_memory`,
     `coverage_metric_mode=path_line_plus_endpoint`,
     `include_oracle_baselines=true`, and
     `include_roi_weighted_coverage=true`.
   - If oracle baselines cannot beat incumbent, the route is ROI/map or
     candidate-task complexity expansion. If oracle baselines can beat incumbent
     but Xunce cannot, the route is training objective, adapter, coverage
     reward, and candidate materialization iteration. Neither case authorizes
     checkpoint publication, default-policy replacement, executor connection,
     online canary traffic, or network complexity increases by default.
   - Stage 18E target:
     `scripts/run_xunce_candidate_level_coverage_opportunity_materialization.py`.
   - Stage 18E output root:
     `outputs/path_feedback_batch_xunce_candidate_level_coverage_opportunity_materialization_v1/`.
   - Stage 18E materializes endpoint, path-line, expected, ROI-weighted,
     revisit, cost/risk-efficiency, and opportunity-rank fields for every
     candidate, then writes an enriched Stage 18A-compatible ROI expansion root.
     The source label is
     `geometric_counterfactual_from_stage18a_candidate/v1`; it is offline
     counterfactual evidence, not real executor evidence.
   - Stage 18F target:
     `scripts/run_xunce_oracle_separability_benchmark.py`.
   - Stage 18F output root:
     `outputs/path_feedback_batch_xunce_oracle_separability_benchmark_v1/`.
   - Stage 18F verifies whether greedy and cost-aware coverage oracles can beat
     incumbent on Stage 18E materialized candidates without mask, fallback, or
     cost-aware efficiency regressions. After gate simplification, oracle
     separability is diagnostic only: `oracle_separable=false` no longer blocks
     Stage 18C-v2 model comparison. Stage 18F passes when evidence authenticity
     and candidate validity gates pass, then routes to
     `run_stage18c_v2_model_comparison`; research advice is recorded in
     `diagnostic_reason_codes` and `diagnostic_recommended_change`. Oracle
     remains a diagnostic baseline and must not become the default policy.
   - Stage 18F.1 target:
     `scripts/run_xunce_cost_efficient_coverage_opportunity_refinement.py`.
   - Stage 18F.1 output root:
     `outputs/path_feedback_batch_xunce_cost_efficient_coverage_opportunity_refinement_v1/`.
   - Stage 18F.1 refines Stage 18E candidates into a cost-efficient coverage
     root by adding cost-, risk-, and budget-adjusted coverage fields,
     Pareto/dominance status, and `safe_efficient_opportunity`. The cost-aware
     oracle in Stage 18F reports this field but does not use it as a hard pool
     restriction; it selects from valid candidates only: reachable,
     path-feedback validated, non-proposal, and no open-grid fallback. Missing
     safe-efficient opportunity is diagnostic, not a blocker for model
     comparison. This remains offline counterfactual evidence and does not
     authorize checkpoint publication, default-policy replacement, executor
     connection, or online canary.
   - Stage 18G targets:
     `scripts/run_xunce_true_incumbent_selection_binding.py`,
     `scripts/run_xunce_safe_efficient_opportunity_root_cause_audit.py`, and
     `scripts/run_xunce_safe_efficient_candidate_repair.py`.
   - Stage 18G removes `fallback_action_index_0` from safe-efficient decisions
     by binding true incumbent checkpoint inference back to Stage 18E
     candidates, audits why safe-efficient opportunities are absent, and writes
     a repaired candidate root. Only path-feedback-validated candidates can
     count as safe-efficient opportunities; interpolation proposals without
     validation stay diagnostic-only. If safe-efficient opportunities remain
     absent after true binding and repair, the next route is ROI/map or
     candidate-generation complexity, not actor/critic training.
   - Stage 18H.0 target:
     `scripts/run_xunce_risk_coverage_cost_quantization_audit.py`.
   - Stage 18H.0 output root:
     `outputs/path_feedback_batch_xunce_risk_coverage_cost_quantization_audit_v1/`.
   - Stage 18H.0 separates coverage, risk, and cost into atomic candidate
     vectors before any guard, Pareto, reward, or critic target is considered.
     Coverage carries provenance fields for source, cell-set kind, and dedupe
     scope; risk uses the existing candidate risk scalar as a path-risk proxy in
     v1 and records that source explicitly; cost stays lightweight with path
     cost, budget ratio, and planning latency only. This stage is a diagnostic
     and label-standardization gate, not actor/critic training.
   - Stage 18H.0 no longer requires safe-efficient candidates, zero risk
     regression, or zero cost regression before Stage 18F / Stage 18C-v2 can
     continue. It hard-fails only on missing true incumbent binding, missing
     path-feedback validation, metric coupling, normalization failure, or zero
     valid candidates. Risk/cost regressions and safe-efficient absence are
     diagnostics.
   - Stage 18I target:
     `scripts/run_xunce_risk_constrained_frontier_nbv_candidate_generation.py`.
   - Stage 18I output root:
     `outputs/path_feedback_batch_xunce_risk_constrained_frontier_nbv_candidate_generation_v1/`.
   - Stage 18I introduces an offline risk-constrained frontier-guided NBV
     candidate generator. Frontier logic proposes where to explore, NBV-style
     estimates measure endpoint/path-line/ROI coverage opportunity, risk/cost
     guards decide which path-feedback-validated proposals are admissible, and
     Pareto compression produces a small stable action set. The output root is
     unbound; it must pass Stage 18B true checkpoint inference and Stage 18G.0
     true incumbent binding before Stage 18H.0 quantization. Stage 18I is not
     actor/critic training, checkpoint publication, default-policy replacement,
     executor connection, online canary traffic, or a new map download.
   - Stage 18I.2 target:
     `scripts/run_xunce_risk_aware_frontier_nbv_candidate_repair.py`.
   - Stage 18I.2 output root:
     `outputs/path_feedback_batch_xunce_risk_aware_frontier_nbv_candidate_repair_v1/`.
   - Stage 18I.2 repairs the Stage 18I candidate-generation bottleneck by using
     Stage 18G.0 true incumbent binding as the baseline. It matches the
     incumbent by candidate cell instead of treating `action_index=0` as a
     pseudo-incumbent, then emits `incumbent_neighborhood`,
     `frontier_boundary`, and `roi_undercovered_boundary` candidates. Coverage
     remains geometric path-line plus endpoint provenance; cost/risk/reachable
     labels must come from path-feedback evidence. Proposal-only rows cannot
     count as safe-efficient candidates. After gate simplification, missing
     frontier candidates, safe-efficient count 0, low spread, and cost/risk
     regression are diagnostics; Stage 18I.2 still passes and routes to
     `rerun_true_model_inference_and_binding` when authenticity and candidate
     validity gates pass.
   - Stage 18I.3 target:
     `scripts/run_xunce_true_frontier_nbv_candidate_source_replacement.py`.
   - Stage 18I.3 output root:
     `outputs/path_feedback_batch_xunce_true_frontier_nbv_candidate_source_replacement_v1/`.
   - Stage 18I.3 is the first true candidate-source replacement step. It
     creates frontier/NBV proposals from ROI geometry, start cell, and true
     incumbent binding instead of only relabeling old candidate rows. A proposal
     remains `proposal_only=true` until the validation adapter can attach
     path-feedback-backed `reachable`, `path_cost`, `risk`, and
     `open_grid_fallback_used=false`; unvalidated proposals are retained only in
     audit artifacts and never enter the formal action set. Stage 18I.4 changes
     the default validator from exact-cell evidence matching to an in-process
     adapter over `evaluate_candidate_paths()`: proposal cells are injected into
     an in-memory `ModelExplorerContract` copy, `reachable=true` is used only as
     a planner-attempt seed, and the formal candidate fields come back from the
     candidate-level planner/path-feedback evaluation. Coverage provenance stays
     geometric and is not relabelled as planner-validated. The path-feedback CLI
     remains only a small integration smoke path, not the bulk validation route.
   - Stage 18C-v2 now treats oracle policies as offline rollout baselines:
     oracle steps can execute without checkpoint inference and are labelled
     `policy_inference_kind=oracle_offline_policy`, while Xunce and incumbent
     still require true checkpoint inference. The legacy
     `dynamic_from_coverage_memory` mode maps to `dynamic_validated_only`; dynamic
     candidates must carry validated path-feedback fields, otherwise the runner
     keeps the static validated candidate instead of moving cells with stale
     path cost/risk.
   - Stage 18I.1 target:
     `scripts/run_xunce_stage18i_evidence_closure_audit.py`.
   - Stage 18I.1 output root:
     `outputs/path_feedback_batch_xunce_stage18i_evidence_closure_v1/`.
   - Stage 18I.1 closes the after-Stage18I evidence chain by reading Stage 18I
     candidate generation, Stage 18B true checkpoint inference, Stage 18G.0
     true incumbent binding, Stage 18H.0 quantization, Stage 18F oracle
     separability, and Stage 18C-v2 rollout summaries. In this closure, Stage
     18F must consume the true-bound / quantized Stage 18H.0 after-Stage18I
     root rather than the unbound Stage 18I root.
   - `safe_efficient_candidate` and `safe_efficient_opportunity` are compatible
     aliases and report fields, not model-comparison prerequisites. Stage 18I.1
     checks only artifact completeness, true model inference, no proxy
     selection, true incumbent binding, and candidate validity. When these pass,
     it routes to `review_xunce_incumbent_comparison_metrics`, even if oracle is
     not separable or Xunce has not improved. This still does not authorize PPO,
     checkpoint publication, default-policy replacement, executor connection,
     online canary traffic, or map download.

   - Simplified hard-gate system:
     `evidence_authenticity_gate_passed` blocks fake inference, proxy
     selection, checkpoint load failure, action-0 fallback, and candidate-cell
     mismatch. `candidate_validity_gate_passed` blocks unvalidated proposals,
     unreachable candidates, open-grid fallback, missing cost/risk/coverage
     provenance, and zero valid candidates. Safe-efficient counts, frontier
     family counts, oracle separability, cost-efficiency deltas, and Xunce
     advantage are diagnostics and comparison metrics only.

   - Consolidated Stage 18 mainline:
     `xunce-stage18-research-evidence-pipeline` is the recommended read-only
     Stage 18 entry point. It resolves the active Stage 18 evidence roots,
     checks root lineage, and writes a single summary/report without duplicating
     the business logic in the underlying runners.
   - Stage 18.1 maps to Stage 18A scenario/ROI evidence. Stage 18.2 maps to
     Stage 18I.3/I.4 candidate-source replacement and proposal validation.
     Stage 18.3 maps to Stage 18B and Stage 18G.0 true inference/binding.
     Stage 18.4 maps to Stage 18H.0, Stage 18F, and Stage 18C-v2
     quantization/oracle/rollout comparison. Stage 18.5 maps to closure,
     attribution, and next-stage routing.
   - Stage 18.4D dynamic rollout now uses
     `candidate_refresh_mode=dynamic_frontier_nbv_in_process` with
     `dynamic_candidate_validation_mode=in_process_path_planner_astar_batch`
     as the mainline validation path. It loads the sidecar/grid once per step
     and calls `path_planner.search.AStarPlanner.plan()` in-process for the
     proposal batch. The evidence kind is `in_process_astar_screening`: it can
     create formal offline rollout candidates, but it is not full
     `PathPlannerRouteAdapter` evidence. `PathPlannerRouteAdapter` is reserved
     for explicit smoke or sampled audit. Sampled audit is explicitly enabled
     with `dynamic_adapter_audit_enabled=true`; it compares a bounded route
     sample and does not participate in policy action selection.
   - Stage 18.4 reports closed-loop dynamic rollout performance
     (`dynamic generator + policy`) separately from same-candidate-set policy
     selection evidence (`same state + same mask + same candidates`). Candidate
     set hash divergence is expected after trajectories split and must not be
     treated as an error.
   - Dynamic candidate exhaustion is a clean terminal condition, not a model
     inference failure. When dynamic generation executes but returns no formal
     valid candidates, the episode records
     `terminal_reason=candidate_generation_exhausted` and does not count the
     step as a mask violation, path-planning failure, or
     `true_model_inference_not_executed`.
   - Coverage reports raw cells and saturation-aware rates separately. Raw
     cell counts and `comparison-pairs.jsonl` deltas remain the primary model
     comparison facts; `coverage_rate_capped` and
     `coverage_saturation_exceeded` are reporting diagnostics for long rollouts
     whose legacy denominator can be exceeded.
  - Stage 18.4E updates the dynamic generator to
    `candidate_generation_algorithm_source=map_aware_coverage_frontier_nbv/v1`
    while keeping the public refresh source
    `dynamic_frontier_nbv_in_process/v1` for compatibility. The generator now
    treats frontier as a coverage frontier over valid ROI/passable cells,
    proposes undercovered component centroid/boundary candidates, keeps
    low-cost bridge and conservative local backups, clips coverage estimates
    to valid cells, and selects validated candidates with a Pareto-diverse
    rule instead of pure coverage-first sorting. A* validates reachability,
    path cost, and path length; `risk` remains a documented sidecar/path-cost
    proxy, not a physical executor risk integral.
  - Stage 18.5 is implemented as `xunce-stage18-5-evidence-attribution-review`
    with runner `scripts/run_xunce_stage18_5_evidence_attribution_review.py`.
    It consumes the Stage 18.4E coverage comparison summary, aggregate, pairs,
    episodes, and paired decision audit, then writes attribution, guard,
    routing, report, and manifest artifacts under
    `outputs/path_feedback_batch_xunce_stage18_5_evidence_attribution_review_v1/`.
    Its guard requires coverage gains to remain inside path-cost and risk
    budgets: positive raw coverage alone is not an advantage when path cost,
    risk proxy, risk-cost-weighted exposure, coverage per 100m, or coverage gain
    per path cost regress. Attribution classes are non-exclusive:
    `candidate_generation`, `policy_preference`, `path_cost`, `risk_proxy`, and
    `candidate_exhaustion`.
  - The current Stage 18.4E dynamic root is evidence-readable but not release or
    training ready. Stage 18.5 routes it to
    `refine_coverage_reward_and_cost_guard` because coverage gain is paired with
    cost/risk regression and same-candidate-set policy advantage is not
    established. `xunce-stage18-research-evidence-pipeline` can optionally
    consume this output through `stage18_5_attribution_root` or
    `--stage18-5-attribution-root`; without it, an unestablished Xunce advantage
    routes to `review_xunce_incumbent_comparison_metrics`.
  - Stage 18.6 is implemented as
    `xunce-stage18-6-coverage-reward-cost-risk-guard-refinement` with runner
    `scripts/run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py`.
    It consumes the Stage 18.5 attribution root, the Stage 18.4E coverage
    comparison root, and `configs/xunce_canonical_reward_guard_profile_v2.json`
    read-only. It does not change the canonical profile; it replays the v2
    budget guard with path cost delta <= 20m, risk delta <= 0.5,
    risk-cost-weighted delta <= 25, and non-regressing coverage per 100m.
    Stage 18.6 also validates profile id/version/hash lineage across Stage
    18.5, Stage 18.4E pairs, and optional candidate metric audit rows.
  - Stage 18.6 can only claim candidate-level guarded reselection when Stage
    18.4E emits `xunce-exploration-coverage-candidate-metric-audit.jsonl`
    through `emit_candidate_metric_audit=true` or `--emit-candidate-metric-audit`.
    Without that file it must set
    `full_candidate_metric_replay_available=false`,
    `counterfactual_reselection_claimed=false`, and route to
    `rerun_stage18_4e_with_candidate_metric_audit`. Pipeline consumption of
    Stage 18.6 is allowlist-only and cannot route to legacy PPO training.
  - Stage 18.8 fixes the candidate-level risk guard semantics without changing
    the canonical v2 profile or reward formula. Candidate-level guard-clean
    replay compares each candidate against the incumbent selected candidate in
    the same paired decision row: `risk_delta = candidate_risk_proxy -
    incumbent_selected_risk_proxy`, with the v2 budget applied to that delta.
    Absolute candidate `risk` / `risk_proxy` is recorded as sidecar/path-cost
    proxy provenance only; it is not a physical executor risk and is not gated
    by `risk <= 0.5`. Extra candidate metric keys are diagnostic because
    Xunce/incumbent/oracle trajectories can emit additional candidate sets; only
    missing paired decision keys block full candidate replay.
  - Stage 18D, 18E, 18F.1, 18G.1, and 18G.2 remain available as legacy
    diagnostics or optional experiments; they are no longer the mainline
    preconditions for comparing Xunce and incumbent.
   - The consolidated summary must separate `evidence_status`,
     `candidate_validity_status`, `comparison_verdict`, `overall_conclusion`,
     `release_readiness`, and `training_readiness`. A valid evidence chain can
     still conclude `evidence_valid_but_xunce_advantage_not_established`; this
     does not authorize PPO, checkpoint publication, default-policy replacement,
     executor connection, or online canary traffic.

## Acceptance Criteria

The design is ready for implementation planning when:

- the project documents clearly mark this as a research line, not a release
  gate;
- all non-goals are explicit;
- the first implementation target is the observation/bottleneck audit, not PPO
  training;
- evaluation includes both coverage/generalization and parameter/latency
  constraints;
- the route preserves current policy scoring interfaces and guarded ranking.

## Stage 18.9 Risk/Reward Contract Update

Stage 18.9 changes the readiness authority from candidate-level risk deltas to a
whole-trajectory risk boundary and reward audit. The intended contract is:

- hard risk filters paths before reward: failed planning, unreachable route,
  mask violation, open-grid fallback, out-of-bounds/no-go traversal, platform
  capability violation, low autonomous terrain confidence, or unrecoverable
  path segment;
- soft risk remains path telemetry: `path_risk_peak`, `path_risk_exposure`,
  `high_risk_distance_m`, and `recovery_margin_min`;
- reward ranks only allowed paths and uses `path_cost` as the dominant cost
  term;
- if `path_cost` already includes terrain/risk proxy cost, the v3
  `soft_risk_component` must stay tiny or audit-only;
- Stage 18.6 and Stage 18.7 are diagnostic views and cannot bypass Stage 18.9
  for Stage 19 readiness;
- new v3 PPO inputs require Stage 18.9 readiness lineage and cannot rely on
  legacy v2 reward artifacts alone.

## Stage 18.9 Strict V3 Evidence Run

Strict v3 evidence is the clean route after the risk/reward contract update. The
6/12/24/36 candidate-count sweeps must be regenerated with
`configs/xunce_canonical_reward_guard_profile_v3.json` from Stage 18.4E through
Stage 18.9, so every artifact carries the same
`xunce-coverage-cost-risk-boundary-v3` `profile_hash`.

`scripts/xunce_stage18_guard_thresholds.py` is the compatibility boundary. For
v2 it preserves candidate-level risk-delta guards. For v3 it maps the profile to
trajectory thresholds, sets `risk_delta_hard_gate_enabled=false`, and marks
candidate-level risk-delta clean rates as diagnostic only. Stage 18.9 remains
the readiness authority.

`scripts/run_xunce_stage18_9_strict_v3_evidence_rollup.py` aggregates the four
strict v3 Stage 18.9 roots plus the strict Stage 18.7 diagnostic root. It routes
missing or non-v3 lineage to `rerun_stage18_9_strict_v3_required_inputs`, hard
risk violations to `repair_path_risk_boundary_filtering`, coverage/cost
failures to `refine_coverage_cost_reward_weights`, abnormal soft-risk exposure
to `calibrate_soft_risk_exposure_weight`, and only trajectory-clean evidence to
`prepare_stage19_evaluator_critic_preflight`. The rollup always keeps
`stage19_authorized=false` and never starts PPO, publishes checkpoints, replaces
the default policy, connects a real executor, or starts canary traffic.
### Stage 18.11 Path Cost Weight Calibration

Stage 18.11 calibrates path-cost reward weight under the strict v3 evidence
contract. The primary mission metric remains final whole-task coverage above
99%; path cost and soft risk are constraints and secondary objectives. A lower
path cost result is not sufficient when final coverage remains below 99%.

The stage reads existing strict v3 Stage 18.4E candidate metric audits and
performs one-step observed-candidate-set replay for stable v3 path-cost profile
variants. This replay is counterfactual ranking only: it does not modify the
fixed Xunce checkpoint, does not claim trajectory outcome, and does not
authorize Stage 19.

The high-fidelity coverage runner may optionally include
`canonical_reward_rerank_oracle`. This policy is diagnostic only. It uses the
selected canonical v3 reward profile to pick the highest-reward candidate among
the current valid candidates and writes reward components plus profile lineage.
It must not replace Xunce or incumbent rows, must not enter paired decision
readiness, and must keep PPO, checkpoint publication, default policy
replacement, executor connection, and canary traffic disabled.

If Stage 18.11 cannot show observed or diagnostic rollout final coverage at or
above 99%, the next route is
`stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage`.

### Stage 19 Evaluator / Critic Preflight

Stage 19 consumes the passed Stage 18.11 path-cost calibration evidence and
prepares a human-review-only evaluator/critic handoff. It is not a training
stage. The current practical target is the diagnostic reward-rerank oracle with
`candidate_count=36` and `path_cost_weight=0.1`; the `path_cost_weight=0.0`
rollout remains only a coverage upper-bound reference because it is more costly.

The preflight uses capped final coverage for the 99% gate. Raw coverage may
exceed 1.0 after denominator saturation and is retained only as diagnostic
telemetry. The Stage 19 runner also compares the fixed Xunce checkpoint against
the oracle target and keeps `xunce_checkpoint_advantage_established=false` when
the checkpoint has not learned the oracle behavior.

Stage 19 writes `xunce-stage19-evaluator-critic-preflight-summary.json`,
`xunce-stage19-diagnostic-rollout-evaluator.json`,
`xunce-stage19-practical-target-selection.json`,
`xunce-stage19-preference-pair-audit.jsonl`,
`xunce-stage19-critic-target-readiness.json`,
`xunce-stage19-next-stage-routing.json`, report, and manifest. A passing Stage
19 route is `stage20_reward_rerank_oracle_preference_dataset_preparation`, but
`stage20_authorized=false`: Stage 20 is preference evidence preparation and
review, not PPO training, checkpoint publication, default-policy replacement,
executor connection, or canary launch.
