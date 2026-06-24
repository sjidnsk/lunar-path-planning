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

### Stage 20 Reward-Rerank Oracle Imitation Dataset

Stage 20 converts the Stage 19 preference-pair audit into strict teacher-label
samples for Xunce. A row is trainable only when `baseline_policy=xunce`,
`same_candidate_set=true`, `hard_risk_clean_pair=true`, and the oracle selected
a different action from Xunce within the same `candidate_set_hash`.
Cross-trajectory comparisons, incumbent-only rows, candidate-set mismatches, and
hard-risk-unclean pairs are excluded and retained only in the exclusion report.

The Stage 20 runner writes
`xunce-stage20-oracle-imitation-summary.json`,
`xunce-stage20-oracle-imitation-teacher-samples.jsonl`,
`xunce-stage20-oracle-imitation-exclusion-report.jsonl`,
`xunce-stage20-oracle-imitation-dataset-stats.json`,
`xunce-stage20-oracle-imitation-dry-run-summary.json`,
`xunce-stage20-next-stage-routing.json`, report, and manifest. Its dry-run uses
the existing Xunce network/training path for teacher-imitation loss only. It
does not run PPO and does not save a publishable checkpoint.

If the strict same-candidate teacher sample count is below 24, the dry-run is
skipped. If it is between 24 and 199, the dry-run may execute, but the route
remains `collect_more_reward_rerank_same_candidate_preference_evidence`.
Only at 200 or more trainable samples with a passing dry-run may Stage 20 route
to `stage20_1_supervised_oracle_imitation_checkpoint_preflight`, and even then
`stage20_authorized=false`.

### Stage 20.1 Same-Candidate Oracle Imitation Evidence

Stage 20.1 fixes the sample-count blocker exposed by Stage 20. Independent
oracle and Xunce rollouts quickly diverge, so many Stage 19 preference rows do
not share the same `candidate_set_hash`. Stage 20.1 therefore collects labels on
Xunce on-policy states: Xunce still executes its own checkpoint action, while
the reward-rerank oracle is evaluated on the exact same current cell, covered
cells hash, and candidate set.

The high-fidelity coverage comparison runner supports
`--emit-on-policy-oracle-teacher-labels`,
`--on-policy-oracle-teacher-baseline-policy xunce`, and
`--on-policy-oracle-teacher-profile <profile>`. It writes
`xunce-exploration-coverage-on-policy-oracle-teacher-labels.jsonl`; these rows
are audit labels and do not change Xunce trajectory execution.

The Stage 20.1 runner writes
`xunce-stage20-1-same-candidate-oracle-imitation-summary.json`,
`xunce-stage20-1-on-policy-teacher-labels.jsonl`,
`xunce-stage20-1-teacher-label-exclusion-report.jsonl`,
`xunce-stage20-1-dataset-stats.json`, routing, report, and manifest. Stage 20
can merge these labels through
`--stage20-1-same-candidate-oracle-imitation-root`. Stage 20.1 is still evidence
collection only: no PPO, no checkpoint publication, no default-policy
replacement, no executor connection, and no canary launch.

### Stage 21.0 Pure PPO Readiness Audit

Stage 21 starts a separate pure PPO mainline. Oracle-imitation evidence from
Stage 20/20.1 remains useful as diagnostic signal, but it is not the gate for
pure PPO. The new contract is:

```text
Xunce on-policy rollout -> PPO trainable batch -> coverage-first reward ->
tiny PPO update -> post-update trajectory evaluation -> multi-seed PPO pilot
```

Stage 21.0 audits the current codebase before any PPO collection or update. It
checks that the Xunce full network exposes masked logits, probabilities, and a
value head; that the high-fidelity runner can produce state, candidate sets,
action masks, coverage/path/risk artifacts; that generic PPO loss and rollout
transition schemas exist; and that the older PPO smoke path validates old log
probability and value consistency. It also marks the missing pieces for later
stages: stochastic Xunce on-policy collector, Xunce-specific PPO batch adapter,
coverage-first PPO reward profile, post-update trajectory evaluation, and
multi-seed pilot.

The Stage 21.0 runner is
`scripts/run_xunce_stage21_0_pure_ppo_readiness_audit.py`. It writes summary,
capability audit, next-stage routing, report, and manifest under
`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\...`.
Passing Stage 21.0 routes to
`implement_stage21_1_xunce_on_policy_ppo_rollout_collector`. It does not run
PPO, publish a checkpoint, replace default policy, connect executor, or start
canary traffic.

### Stage 21.1 Xunce On-Policy PPO Rollout Collector

Stage 21.1 implements the first trainable-data boundary for the pure PPO
mainline. It does not change the high-fidelity evaluator's argmax behavior.
Instead, a separate collector runner samples Xunce actions from
`Categorical(logits=masked_logits)` and writes a PPO transition contract with
`observation`, serialized `xunce_batch`, `action_index`, `old_log_prob`,
`old_value`, reward, `next_observation`, `done`, and audit `info`.

The sampling mask is stricter than the legacy candidate validity mask:
`sampling_mask = action_mask & hard_risk_clean_mask`. This preserves the Stage
18.9 risk-boundary principle: hard-risk paths are not allowed into trainable
PPO samples and cannot be offset by reward. Full rows remain auditable, but only
mask-valid, hard-risk-clean, finite reward/log-prob/value transitions are
trainable.

Stage 21.1 uses canonical v3 as an interim collector reward only. The
coverage-first reward contract for 99% final coverage is intentionally deferred
to Stage 21.2. Stage 21.1 still does not run PPO, publish checkpoints, replace
the default policy, connect a real executor, or start canary traffic.

### Stage 21.2 Coverage-First PPO Reward Contract

Stage 21.2 adds a Stage21-specific reward contract instead of extending
canonical v3. The profile schema
`xunce-stage21-coverage-first-ppo-reward-profile/v1` is stored in
`configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json`, and the
computation helper is `model_explorer.policy.coverage_first_reward`.

The fixed component set is:
`step_coverage_gain_component`, `coverage_progress_component`,
`final_coverage_bonus_component`, `success_99pct_bonus_component`,
`path_cost_component`, `soft_risk_component`, `failure_component`, and
`hard_risk_component`. Terminal final-coverage and 99% success bonuses are only
active when `done=true`; 40-step rollouts below the 99% target must route to
horizon or mission-budget scaling rather than being judged by low path cost.

Hard risk remains a boundary, not a tradeable reward term. Hard-risk rows are
not trainable; if audited, positive coverage components are suppressed or the
total reward is clamped non-positive. When path cost already includes a risk
proxy, soft risk is capped to a tiny audit-weighted penalty to avoid double
counting. Stage 21.2 validates the contract and writes reward evaluation
artifacts only; it does not run PPO or authorize release.

### Stage 21.3 PPO Batch Validation

Stage 21.3 is the pre-update PPO batch gate. It consumes Stage 21.1 on-policy
transitions and Stage 21.2 reward evaluations, joins them one-to-one by
`transition_id`, and writes a trainable batch plus return/advantage and lineage
audits. The validator rejects blank or duplicate ids, missing or extra reward
rows, transition/reward `scenario_id`, `step_index`, or `done` mismatches,
non-trainable reward rows, `hard_risk_rejected` reward rows, negative actions,
mask-invalid actions, hard-risk mask violations, non-finite old logprob/value,
reward, return, or advantage, stale old-logprob recomputation, missing
non-terminal next observations, and malformed episode boundaries.

Episode validation is explicit: each scenario must have unique, monotonic,
contiguous step indices and exactly one terminal transition, with the terminal
transition last. Discounted returns and raw advantages are computed per
scenario. Advantage normalization uses train-split statistics only, and every
batch row includes `stage21_3_split` so Stage 21.4 can train only on train rows.

The current smoke artifact under
`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_3_ppo_batch_validation_v1`
passes these gates and routes to `implement_stage21_4_tiny_ppo_update_smoke`.
Stage 21.3 still does not run PPO, publish checkpoints, replace the default
policy, connect an executor, or start canary traffic.

### Stage 21.4 Tiny PPO Update Smoke

Stage 21.4 validates the pure-PPO update mechanics for `XunceFullNetworkV1`.
It is intentionally separate from the generic masked-candidate PPO helper,
because Xunce requires candidate, edge, memory, context, missing-indicator, and
action-mask tensors rather than the generic candidate/global feature contract.
The update recomputes action log probabilities with the Stage 21.1
`sampling_mask` and `sampling_temperature`, then applies the standard clipped
PPO objective plus value and entropy terms.

Only Stage 21.3 train-split rows may be consumed. The output checkpoint is
experimental-only and must include complete Xunce model dimensions in metadata
so the existing `_load_xunce_checkpoint` path can reload it. Stage 21.4 may
write `runs_new_ppo_update=true` for this local smoke test, but publication,
default-policy replacement, executor connection, and canary traffic remain
forbidden. Any coverage improvement claim is deferred to Stage 21.5
post-update trajectory evaluation.

### Stage 21.5 Post-Update Offline Trajectory Evaluation

Stage 21.5 compares the Stage 21.4 source checkpoint against the Stage 21.4
experimental-only checkpoint in a small offline high-fidelity trajectory smoke.
It does not run another PPO update and does not authorize checkpoint release,
default-policy replacement, executor connection, or canary traffic.

The runner is
`scripts/run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py`.
It runs the same high-fidelity configuration twice, with only the Xunce
checkpoint and output/work roots changed. The default smoke scope is 2
scenarios, 4 rollout steps, 36 candidates, and a proposal pool of 288.

Stage 21.5 must compare pre/post Xunce episodes directly. It must not use
summary `xunce_*_delta_vs_incumbent` fields, because those describe Xunce versus
incumbent within one run rather than Xunce before versus after the PPO update.
Instead it reads both `xunce-exploration-coverage-episodes.jsonl` files, filters
`policy == "xunce"`, verifies matching `scenario_id` sets, and computes deltas
for final coverage, capped final coverage, coverage AUC, capped coverage AUC,
new covered cells, path cost, coverage per 100m, soft-risk exposure, hard-risk
violations, model-inference failures, mask violations, unreachable selections,
path-planning failures, open-grid fallbacks, and candidate exhaustion.
It writes `xunce-stage21-5-scenario-trajectory-delta.jsonl` and treats any
per-scenario coverage/AUC regression as blocking, even when aggregate means do
not regress.

Routing to `implement_stage21_6_multi_seed_ppo_pilot` is allowed only when both
final coverage and AUC families do not regress, post-update hard-risk and
execution-boundary counts remain zero, and Stage 21.4 checkpoint metadata remains
experimental-only. Because Stage 21.4 currently trains on a tiny smoke batch,
Stage 21.5 keeps `sample_count_too_low_for_performance_claim=true`; passing it
is a mechanics and regression signal, not a release-quality performance claim.

### Stage 21.6 Multi-Seed PPO Pilot

Stage 21.6 runs the first conservative multi-seed PPO pilot for the pure-PPO
mainline. The runner is
`scripts/run_xunce_stage21_6_multi_seed_ppo_pilot.py`, with config
`configs/xunce_stage21_6_multi_seed_ppo_pilot_v1.json`. By default it uses
three seeds, `2101`, `2102`, and `2103`, and keeps the current smoke scope:
2 scenarios, 4 rollout steps, 36 candidates, and 288 proposal-pool limit.

Each seed reruns the whole Stage21.1 -> Stage21.2 -> Stage21.3 -> Stage21.4 ->
Stage21.5 chain with its own `sampling_seed`. Stage 21.6 must not reuse one
batch as multi-seed evidence. The lineage audit records per-seed Stage21.1
roots, Stage21.3 batch fingerprints, and transition-id fingerprints; duplicates
route to `repair_stage21_6_seed_lineage_or_batch_reuse`.

The aggregate gate includes mean/std/variance/min/max for final coverage delta
and coverage-AUC delta, plus the worst seed. It also gates PPO numerical
stability: KL, entropy, and gradient norms must remain finite and within budget.
Mean final coverage delta and mean coverage-AUC delta must be strictly positive;
holding flat is not enough to route to Stage22. Hard risk, model-inference
failures, mask violations, unreachable selected candidates, path-planning
failures, and open-grid fallbacks must remain zero.

Stage 21.6 may set `runs_new_ppo_update=true` because it performs local offline
PPO updates for the pilot. It still must keep `publishes_checkpoint=false`,
`replaces_default_policy=false`, `connects_real_executor=false`,
`starts_online_canary=false`, and `canary_traffic_fraction=0.0`. Because the
default pilot remains small, `sample_count_too_low_for_performance_claim=true`
prevents treating it as release-quality performance evidence.

### Stage 21.7 Reward / Collector / Advantage / Horizon Repair

Stage 21.7 is the repair stage after a failed Stage 21.6 multi-seed PPO pilot.
Its runner is
`scripts/run_xunce_stage21_7_reward_collector_advantage_horizon_repair.py`,
with config
`configs/xunce_stage21_7_reward_collector_advantage_horizon_repair_v1.json`.

The stage answers why Stage 21.6 can execute three offline PPO updates while
final coverage and coverage AUC remain flat. It audits five causes: reward
signal quality, advantage signal quality, trainable transition count, PPO
update strength / observable policy shift, and holdout horizon. It derives
policy shift by joining Stage 21.5 pre/post model-inference rows on scenario,
step, and candidate-set hash.

Current Stage 21.6 evidence is diagnostic-only: 24 trainable transitions are
below the 200-transition performance-claim threshold, the holdout covers only
4 steps, and pre/post policy probabilities move too little to change selected
actions. Stage 21.7 therefore writes a repaired Stage21.6 smoke config with
8 scenarios, 10 steps, 3 seeds, 2 PPO epochs, and learning rate `2e-5`, then may
execute one repaired pilot smoke. The repaired pilot remains experimental
evidence only: no checkpoint publication, default-policy replacement, executor
connection, online canary, network change, action-space change, or default A*
change is authorized.

### Stage 21.8 PPO Update Strength Calibration

Stage 21.8 follows the Stage21.7 repaired pilot when all repaired seeds fail
with `grad_unstable`. Its runner is
`scripts/run_xunce_stage21_8_ppo_update_strength_calibration.py`, with config
`configs/xunce_stage21_8_ppo_update_strength_calibration_v1.json`.

The stage separates two concepts that were easy to conflate: Stage21.4
`max_grad_norm` clips gradients during the local update, while Stage21.6
`max_grad_norm` is also used as a stability gate against the recorded
`pre_clip_grad_norm`. Stage21.8 therefore writes per-combo Stage21.4 and
Stage21.6 configs, runs a bounded calibration sweep, and reports whether each
combo is numerically stable, has observable policy shift, and does not regress
coverage/AUC. Runtime-blocked combos are treated as explicit blockers and are
not retried by default; retry requires `retry_runtime_blocked_combinations=true`
or a new `combo_id` / `sweep_work_root`.

Passing Stage21.8 only recommends a repaired Stage21.6 config when the source
sweep has Stage21.6 `status=passed`, seed stage status, batch fingerprint,
transition fingerprint, post-clip grad norm, stable gradient/KL/entropy, and
non-regressing coverage/AUC. It does not authorize formal PPO training,
checkpoint publication, default-policy replacement, executor connection, online
canary traffic, network changes, action-space changes, or default A* changes.

### Stage 21.9 Gradient Normalization / Loss Scaling Repair

Stage 21.9 follows Stage21.8 when even the low-learning-rate completed combo
still has raw `pre_clip_grad_norm` above the Stage21.6 stability gate. The key
interpretation is that lowering learning rate reduces parameter step size, but
does not materially reduce the raw gradient produced by the loss. Stage21.9
therefore repairs the loss and advantage scale rather than continuing to tune
learning rate blindly.

The stage first audits Stage21.3 train split advantage normalization and then
reads Stage21.4 loss/gradient artifacts to identify whether policy, value,
entropy, or total loss dominates the gradient. The default repaired Stage21.4
config sets `advantage_clip_abs=5.0`,
`normalize_minibatch_advantages=true`, `loss_scale=0.25`, and
`value_loss_coefficient=0.1`; it keeps the coverage-first reward target,
collector, network, action space, and default A* unchanged.

The repaired smoke must be judged from the Stage21.8 sweep result row, not from
Stage21.4 `status=passed` alone. It must directly check pre-clip gradient, the
post-clip norm, KL, entropy, parameter delta, observable policy shift,
coverage/AUC non-regression, and all release/executor/canary boundaries.
Passing Stage21.9 only routes to another Stage21.6 multi-seed pilot with the
repaired config. It still does not authorize formal PPO training, checkpoint
publication, default-policy replacement, executor connection, online canary,
network changes, action-space changes, or default A* changes.

### Stage 21.10 Stage21.9 Repaired Multi-Seed PPO Pilot

Stage 21.10 runs the next repaired Stage21.6 pilot after Stage21.9 has proven
that the loss/advantage scale repair brings raw gradients back under the
Stage21.6 stability gate. The runner is
`scripts/run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot.py`, with
config `configs/xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot_v1.json`.
The default Windows config sets `stage21_6_output_subdir=s6` for the nested
Stage21.6 execution root so high-fidelity dynamic-validation artifact paths stay
under the legacy 260-character path limit. The Stage21.10 summary records the
actual nested root in `stage21_6_root`.

The wrapper materializes a new Stage21.4 config instead of reusing the raw
Stage21.9 config verbatim. It fixes `learning_rate=2e-6`, `epochs=1`,
`clip_ratio=0.2`, Stage21.4 training `max_grad_norm=1.0`,
`advantage_clip_abs=5.0`, `normalize_minibatch_advantages=true`,
`loss_scale=0.25`, and `value_loss_coefficient=0.1`. Stage21.4 `max_grad_norm`
is the post-clip update limit; Stage21.6/21.10
`stage21_6_pre_clip_grad_norm_gate=25.0` is the raw pre-clip gradient audit
gate.

Stage21.10 must not claim improvement from raw Stage21.6 means alone. It also
reads each seed's Stage21.5 capped final-coverage and capped coverage-AUC
deltas. Routing to `scale_stage21_ppo_pilot_scenarios_and_horizon` requires
stable gradients, trusted Stage21.6 lineage, zero hard-risk / execution-boundary
counts, strictly positive raw mean final coverage and AUC, strictly positive
capped mean final coverage and AUC, and no worst-seed regression. Otherwise the
route remains repair-focused, typically
`repair_stage21_reward_signal_or_advantage_separation`.

This stage may execute local offline PPO updates through Stage21.6, but it is
still a repaired pilot only. It does not authorize formal PPO training,
checkpoint publication, default-policy replacement, executor connection, online
canary traffic, reward-target changes, collector changes, network changes,
action-space changes, or default A* changes.

Current Stage21.10 evidence completed the repaired three-seed pilot with
`trainable_transition_count_total=240`, trusted lineage, zero hard-risk /
execution-boundary counts, and stable gradients
(`pre_clip_grad_norm_max=2.334965467453003`). However raw and capped final
coverage / coverage-AUC deltas all remained `0.0`, so the active next route is
`repair_stage21_reward_signal_or_advantage_separation` rather than a larger PPO
pilot.

### Stage 21.11 Coverage-Constrained Path-Cost Objective Audit

Stage21.11 is a read-only audit for the Stage21.10 blocker. Its goal is to
identify whether the no-uplift result comes from the reward objective, return /
advantage credit assignment, PPO probability shift, or pre/post evaluation
binding. The audit consumes Stage21.10 and the nested Stage21.6 seed artifacts;
it does not run another PPO update.

The objective is no longer described as pure coverage-first in isolation. The
mission target remains final coverage above 99%, but the next reward direction
must be coverage-constrained path-cost optimization: below 99% coverage, actions
should still receive strong positive signal for expanding coverage; among
actions with comparable coverage progress, lower path cost and better
coverage-per-cost should receive better rank. A low-cost trajectory with poor
coverage is not a success.

Pre/post probability shift must use strong state binding whenever available:
`scenario_id + step_index + candidate_set_hash + covered_cells_hash +
current_cell`. A weaker join can be reported as diagnostic only and cannot be
used to claim policy behavior changed. Stage21.11 outputs reward separation,
advantage separation, action-probability shift, objective recommendation,
recommended Stage21.6 config, routing, report, and manifest artifacts while
keeping all publish/default-policy/executor/canary fields false.

### Stage 21.12 Coverage-Constrained Reward Profile Repair

Stage21.12 implements the reward-profile repair recommended by Stage21.11. It
adds `configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json`
without overwriting the v1 profile. The v2 profile keeps the 99% final coverage
target as the first success criterion, keeps hard-risk rejection outside the
positive reward tradeoff, and adds a bounded `coverage_per_cost_component` so
that path cost matters most when coverage progress is close.

The v2 `coverage_per_cost_component` is computed from
`coverage_gain / max(path_cost, floor)` and uses a log-bounded component cap.
This fixes the previous behavior where the coverage-per-cost term could be
effectively capped by the candidate's own coverage gain and therefore fail to
prefer lower-cost candidates with nearly equal coverage progress. Soft risk
remains tiny / audit-weighted when path cost already includes risk proxy.

The Stage21.12 runner
`scripts/run_xunce_stage21_12_coverage_constrained_reward_profile_repair.py`
replays existing Stage21.10 transitions with both v1 and v2 profiles, validates
v1 compatibility with Stage21.2 reward rows, checks advantage correlation on the
train split, and writes recommended Stage21.2/Stage21.6 configs. It does not run
PPO. Current evidence passed with v2 reward best-action alignment improving from
`0.20833333333333334` to `0.29583333333333334`, v2 alignment improvement
`0.0875`, coverage-priority violation rate `0.020833333333333332`, and the
route is `run_stage21_13_coverage_constrained_multi_seed_ppo_smoke`.

### Stage 21.13 Coverage-Constrained Multi-Seed PPO Smoke

Stage21.13 executes the Stage21.12 recommended Stage21.6 config with
`configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json`. The
runner is `scripts/run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke.py`,
with config `configs/xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke_v1.json`.
It is intentionally a wrapper around Stage21.6 rather than a second PPO
implementation.

The wrapper requires Stage21.12 to be passed and to route to Stage21.13. It then
checks that the recommended Stage21.6 config still uses seeds `2101/2102/2103`,
8 scenarios, 10 rollout steps, 36 candidates, proposal pool 288, and the
coverage-constrained v2 reward profile. The nested Stage21.6 run uses a short
`s6` output directory on Windows to avoid long dynamic-validation artifact paths.

Routing remains conservative. Stage21.13 advances only when raw final coverage
delta, raw coverage AUC delta, capped final coverage delta, and capped coverage
AUC delta are all strictly positive with no worst-seed regression, clean
lineage, stable KL/entropy/gradient norms, observable parameter delta, and zero
hard-risk / execution-boundary counts. Otherwise it routes back to policy-signal
or return/advantage repair. It may execute local offline PPO updates through
Stage21.6, but it never publishes checkpoints, replaces the default policy,
connects an executor, starts canary traffic, or changes network/action space/A*.

The current Stage21.13 evidence completed three seeds with 240 trainable
transitions. Lineage and execution boundaries passed, `pre_clip_grad_norm_max`
was `3.6654789447784424`, post-clip grad stayed near `1.0`, KL was finite, and
parameter delta was observable. Raw and capped final coverage / coverage AUC
deltas all remained `0.0`, so the route is
`repair_stage21_return_advantage_credit_assignment`.

### Stage 21.14 Multi-Epoch PPO Update Depth Calibration

Stage21.14 tests whether the Stage21.13 failure is caused by PPO update depth
being too shallow for the discrete candidate-action policy. It keeps the
Stage21.12 coverage-constrained reward v2 objective and the Stage21.9
loss-scaling repair, then reuses Stage21.6 with deeper epoch counts such as 2,
4, and 8 epochs at `learning_rate=2e-6`.

The runner is
`scripts/run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration.py`,
with config
`configs/xunce_stage21_14_multi_epoch_ppo_update_depth_calibration_v1.json`.
It is an orchestration wrapper, not a second PPO implementation. For every
sweep combo it writes a dedicated Stage21.4 config and points Stage21.6
`stage21_4_base_config` to that generated file. Sweep outputs stay on D drive
under the Stage21 pure PPO output tree.

The fixed pilot scope remains three seeds, eight scenarios, ten rollout steps,
36 candidates, proposal pool 288, coverage-constrained reward v2,
`loss_scale=0.25`, `value_loss_coefficient=0.1`, `advantage_clip_abs=5.0`, and
`normalize_minibatch_advantages=true`. Stage21.14 does not change reward
targets, network, action space, candidate generation, or default A*.
The default required core sweep is limited to `e2/e4/e8` at
`learning_rate=2e-6`; optional `lr=5e-6` combinations require an explicit config
change and are not launched automatically after the core sweep is complete.

Action-rank auditing uses the strong state key
`scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash`
when those fields are present. Missing strong-key fields are diagnostic only and
must not be treated as proof of action change. However, true raw and capped
coverage/AUC improvement with no worst-seed regression may still route to
`scale_stage21_ppo_pilot_scenarios_and_horizon`, because trajectory improvement
is the primary evidence.

Routing remains conservative: unstable gradients/KL/entropy route back to
Stage21.9 repair; stable but tiny probability/rank movement routes to
`calibrate_stage21_policy_update_signal_strength`; probability/rank movement
without trajectory improvement routes to
`repair_stage21_return_advantage_credit_assignment`; and only raw plus capped
coverage/AUC improvement advances to larger pilot scaling. This stage still
does not publish checkpoints, replace the default policy, connect an executor,
or start canary traffic.

### Stage 21.15 Policy Update Signal Strength Calibration

Stage21.15 addresses the Stage21.14 evidence gap rather than changing the model.
Stage21.14 completed stable deeper PPO updates, but old pre/post inference rows
could not be strongly joined because `current_cell` and `covered_cells_hash`
were missing. Therefore a zero probability delta in Stage21.14 was not valid
evidence that the policy failed to move.

The high-fidelity inference row contract now requires top-level
`scenario_id`, `step_index`, `current_cell`, `current_cell_before`,
`covered_cells_hash`, `candidate_set_hash`, `selected_action_index`,
`selected_rank`, `selected_probability`, `action_probs`, `logits`,
`masked_logits`, `value`, `finite_outputs`, and `latency_ms`. Stage21.15 only
claims action-signal evidence when pre/post rows match on
`scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash`.
Weak joins remain diagnostic only.

The Stage21.15 runner is
`scripts/run_xunce_stage21_15_policy_update_signal_strength_calibration.py`
with config
`configs/xunce_stage21_15_policy_update_signal_strength_calibration_v1.json`.
It reuses Stage21.6 for bounded offline sweeps and then reports probability
delta, best coverage-per-cost probability delta, selected action probability
delta, selected-rank changes, argmax changes, selected-action changes, and raw
plus capped trajectory coverage/AUC deltas. It does not change reward targets,
network, action space, candidate generation, or default A*, and it still cannot
publish checkpoints, replace the default policy, connect an executor, or start
canary traffic.

### Stage 21.16 Policy Signal Margin And Credit Attribution

Stage21.16 is the next diagnostic layer after Stage21.15. Stage21.15 proved the
strong state binding contract was repaired, but its best observed policy movement
was still too small to change rank, argmax, selected action, coverage, or AUC.
Stage21.16 therefore attributes that failure across four axes:

- PPO update strength: learning rate, epochs, clip ratio, loss scale,
  probability delta, KL, entropy, and parameter delta.
- Reward and advantage separation: whether high coverage-per-cost actions get
  higher reward and positive normalized advantage.
- Discrete action margin: the logit/probability/rank gap between the current top
  action and the best coverage-per-cost candidate.
- Loss-gradient attribution: whether policy, value, or entropy loss dominates
  the total gradient norm.

The runner is
`scripts/run_xunce_stage21_16_policy_signal_margin_credit_attribution.py` and the
config is `configs/xunce_stage21_16_policy_signal_margin_credit_attribution_v1.json`.
It may run one bounded offline Stage21.6 combo, but it does not publish
checkpoints, replace the default policy, connect an executor, start canary
traffic, or modify reward targets, network, action space, candidate generation,
or default A*.

### Stage 21.17 Policy Signal Amplification Value Balance

Stage21.17 follows Stage21.16 when probability movement is still below the
action-changing threshold and value loss dominates the component-gradient audit.
The runner is
`scripts/run_xunce_stage21_17_policy_signal_amplification_value_balance.py` and
the config is
`configs/xunce_stage21_17_policy_signal_amplification_value_balance_v1.json`.

Stage21.4 now accepts `policy_loss_coefficient` with a default of `1.0`, keeping
older configs compatible. Stage21.17 uses that knob only for bounded offline
calibration: it first lowers `value_loss_coefficient`, then tests whether
raising `policy_loss_coefficient` strengthens policy gradients enough to move
action probabilities. The audit records policy/value/entropy gradient ratios,
KL, entropy, parameter delta, strong-join probability delta, rank/argmax/action
changes, and raw/capped coverage/AUC deltas. It can recommend a new Stage21.6
config, but it does not authorize checkpoint publication, default-policy
replacement, executor connection, canary traffic, reward-target changes,
network changes, action-space changes, candidate-generation changes, or default
A* changes.

### Stage 22.0 Theta-Aware Sensor Action Space Contract

Stage22.0 changes the action contract definition before any new PPO work. The
Stage21 PPO batch still represents an action as an index into point-only
`candidate_cells=(x,y)`. That contract cannot express that the same point may
produce different coverage when the sensor points in different directions.

The Stage22 contract defines an action as a candidate viewpoint
`(x,y,theta_deg)`. The default audit model uses 8 theta bins, 45 degree spacing,
90 degree FOV, and the convention `0` degrees = positive x and `90` degrees =
positive y. Path planning still targets `(x,y)` through the existing A* path
planner; theta only changes observation coverage after arrival. Occlusion is
recorded as audit-only in Stage22.0.

The runner is
`scripts/run_xunce_stage22_0_theta_aware_sensor_action_space_contract.py` with
config `configs/xunce_stage22_0_theta_aware_sensor_action_space_contract_v1.json`.
It reads Stage21.18/Stage21.6/Stage21.1/Stage21.3 artifacts, expands each
candidate point into 8 viewpoints offline, and reports whether theta materially
changes coverage or the best action ranking. If theta is material and Stage21
artifacts remain point-only, the next route is
`implement_stage22_1_theta_aware_candidate_viewpoint_generation`; old point-only
Stage21 PPO evidence is not theta-aware readiness.

The current Stage22.0 audit reviewed 720 Stage21 transitions. It found
`theta_material_to_coverage=true`, `theta_changes_action_preference=true`,
`point_only_action_space_detected=true`, and
`stage21_ppo_batch_theta_incompatible=true`, so the active route is
`implement_stage22_1_theta_aware_candidate_viewpoint_generation`.

### Stage 22.1 Theta-Aware Candidate Viewpoint Generation

Stage22.1 is the first implementation smoke for the theta-aware action
contract. It expands each base candidate point `(x,y)` into viewpoint actions
`(x,y,theta_deg)` while keeping `candidate["cell"]` as the 2D path-planning
target. Default A* and the network architecture are not changed in this stage.

The helper `scripts/xunce_theta_viewpoint_candidates.py` writes the viewpoint
metadata used by high-fidelity comparison and PPO batch contracts:
`candidate_theta_deg`, `candidate_viewpoint`, `base_candidate_index`,
`viewpoint_index`, `base_candidate_set_hash`, sensor model fields,
`theta_coverage_hash`, and `theta_coverage_gain_per_path_cost`. The
viewpoint-level `candidate_set_hash` includes theta and sensor fields, so a
point-only hash is no longer accepted as theta-aware evidence.

Stage21.1 can now carry viewpoint-level `candidate_cells` and selected
viewpoint metadata in transition `info`. Stage21.3 can be configured with
`require_theta_aware_viewpoint_contract=true`; under that contract it rejects
point-only batches before they reach PPO.

The Stage22.1 runner is
`scripts/run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation.py`
with config
`configs/xunce_stage22_1_theta_aware_candidate_viewpoint_generation_v1.json`.
The current run audited 720 transitions, expanded them to 177672 viewpoint
candidates, found zero theta-hash or mask-length mismatches, and routed to
`run_stage22_2_theta_aware_coverage_reward_contract`. It remains a contract and
smoke stage only: no PPO, checkpoint publication, default-policy replacement,
executor connection, canary, network-architecture change, or default A* change
is authorized.

### Stage22.2 theta-aware coverage reward contract

Stage22.2 connects the viewpoint action contract to the reward and PPO batch
gates. Stage21.2 now supports `require_theta_aware_reward_contract=true`; with
that flag, coverage gain is computed from
`theta_new_visible_cell_count / theta_coverage_denominator_cells`, the reward
row records `coverage_source=theta_aware_sensor_footprint/v1`, and point-only
coverage fallback is rejected.

Stage21.3 also supports `require_theta_aware_reward_contract=true`. It rejects
batches that lack theta reward provenance, use point-only fallback, or bind the
reward viewpoint to a different selected action than the transition. This means
a theta-aware action batch cannot enter PPO if the reward was still computed
from old point-only coverage.

The Stage22.2 runner is
`scripts/run_xunce_stage22_2_theta_aware_coverage_reward_contract.py` with
config `configs/xunce_stage22_2_theta_aware_coverage_reward_contract_v1.json`.
The current run replayed 5000 sampled Stage22.1 viewpoint rows, found 625
same-point groups where theta changed coverage, and all 625 groups had
discriminating reward values. It passed with route
`run_stage22_3_theta_aware_ppo_collector_smoke`.

Stage22.2 is still a contract smoke stage. It does not run PPO, publish
checkpoints, replace the default policy, connect executor, start canary, modify
the network, introduce continuous theta policy, or change default A*.

### Stage22.3 theta-aware PPO collector smoke

Stage22.3 validates that the real PPO data path can carry the theta-aware action
contract from source, not only through offline replay. The runner is
`scripts/run_xunce_stage22_3_theta_aware_ppo_collector_smoke.py` with config
`configs/xunce_stage22_3_theta_aware_ppo_collector_smoke_v1.json`. It creates
three internal roots, `s21_1`, `s21_2`, and `s21_3`, then runs Stage21.1,
Stage21.2, and Stage21.3 with the theta viewpoint and theta reward gates
enabled.

The Stage22.3 transition audit requires Stage21.1 trainable transitions to
carry `candidate_viewpoints`, `candidate_theta_deg`,
`theta_new_visible_cell_counts`, `theta_coverage_hashes`,
`theta_coverage_gain_per_path_costs`, `selected_viewpoint`, and
`selected_theta_deg`. The selected action index must bind to the selected
viewpoint and its `(x,y)` target. Stage21.2 must emit
`coverage_source=theta_aware_sensor_footprint/v1` with no point-only fallback,
and Stage21.3 must preserve theta reward provenance while rejecting point-only
or mismatched theta batches.

Passing Stage22.3 routes only to `run_stage22_4_theta_aware_ppo_update_smoke`.
It does not execute a PPO update, publish checkpoints, replace default policy,
connect executor, start canary, introduce continuous theta policy, modify the
network, or change default A*.

### Stage22.4 theta-aware PPO update smoke

Stage22.4 validates the first offline PPO update path for theta-aware viewpoint
actions. The runner is
`scripts/run_xunce_stage22_4_theta_aware_ppo_update_smoke.py` with config
`configs/xunce_stage22_4_theta_aware_ppo_update_smoke_v1.json`. It wraps the
existing Stage21.4 tiny PPO update and points it at the Stage22.3 `s21_3` batch.

Before running Stage21.4, Stage22.4 audits the trainable batch contract:
`action_index` must identify the selected `(x,y,theta)` viewpoint,
`candidate_viewpoints[action_index]` must match the selected reward viewpoint,
and `xunce_batch.action_mask`, `candidate_features`, and `context_features`
must all use the viewpoint-level action dimension. Point-only reward fallback or
shape mismatch routes to `repair_stage22_4_theta_batch_update_contract`.

After Stage21.4, the smoke checks finite loss rows, finite gradient audit,
readable policy/value/entropy component gradients, experimental-only checkpoint
reload, and all release/default-policy/executor/canary boundary fields. Passing
Stage22.4 only routes to
`run_stage22_5_theta_aware_post_update_trajectory_eval_smoke`; it does not run
trajectory evaluation, publish checkpoints, replace default policy, connect an
executor, start canary traffic, introduce continuous theta, change the reward
target, modify the network, or change default A*.

The current real Stage22.4 run passed with 8 theta-aware trainable rows, no
viewpoint/action/mask contract blocker, and a reloadable experimental
checkpoint. Its gradient audit also shows the pre-clip gradient is dominated by
value loss, so Stage22.4 is chain-readiness evidence only; trajectory impact is
left to Stage22.5.

### Stage22.5 theta-aware post-update trajectory eval smoke

Stage22.5 is the first trajectory-level smoke for the theta-aware PPO branch.
It wraps Stage21.5 rather than reimplementing trajectory evaluation. The wrapper
points Stage21.5 at the Stage22.4 `s21_4` root, keeps the Stage22.4 source
checkpoint as the pre baseline, evaluates the Stage22.4 experimental-only
checkpoint as post, and forces the high-fidelity config to use the theta-aware
viewpoint contract.

The audit reads pre/post `xunce-exploration-coverage-model-inference.jsonl`
and only uses the strong key
`scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash`
to compare inference rows. It reports `selected_viewpoint_changed_count`,
`selected_theta_changed_count`, `selected_action_changed_count`,
`mean_abs_probability_delta`, final coverage delta, coverage AUC delta,
path-cost delta, hard-risk/mask/path-planning/open-grid regressions, and whether
the sample count is too small for a performance claim.

Passing Stage22.5 only means the small theta-aware pre/post evaluation moved in
the right direction without a worst-scenario or safety regression. It routes to
`run_stage22_6_theta_aware_multi_seed_ppo_pilot`; it does not publish a
checkpoint, replace the default policy, connect executor, start canary traffic,
introduce continuous theta, or claim production performance.

### Stage23.0 endpoint obstacle-aware theta sensor coverage contract

Stage23.0 tightens the theta-aware sensor model without changing PPO or path
planning. Stage22 endpoint coverage uses the viewpoint `(x,y,theta)` and a
range/FOV footprint, but it does not make obstacles block line of sight.
Stage23.0 keeps the simplified endpoint observation assumption: the rover first
plans to `(x,y)` using the existing A* path planner, then looks along
`theta_deg`; theta does not imply continuous observation along the path.

The helper `scripts/xunce_obstacle_aware_theta_sensor_coverage.py` first calls
the existing theta FOV footprint and then applies a 2D grid line-of-sight test
from the sensor cell to each target cell. A target obstacle cell is not counted
as visible coverage, and a non-obstacle cell is removed when the line segment
from sensor to target crosses an intermediate obstacle cell. With no obstacle
cells, the helper must return the same visible set and hash as the Stage22
theta FOV helper.

The runner
`scripts/run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract.py`
reads high-fidelity candidate/viewpoint audit rows and compares unobstructed
theta coverage against obstacle-aware theta coverage. Explicit obstacle
sources are required: `obstacle_cells`, `blocked_cells`, `no_go_cells`, or
obstacle/blocked rectangles. Summary statistics such as `blocked_count=0`,
`passable_ratio=1.0`, or an absent sidecar are not valid line-of-sight obstacle
sources. If required sources are missing, Stage23.0 routes to
`rerun_stage23_0_required_obstacle_sources` rather than false-passing the
occlusion audit.

If obstacle occlusion materially changes viewpoint coverage while Stage22
reward still uses unobstructed theta coverage, the next route is
`implement_stage23_1_obstacle_aware_theta_reward_contract`. If occlusion is not
material on the audited roots, the route is
`document_obstacle_occlusion_audit_only`. Stage23.0 is read-only: it does not
run PPO, publish checkpoints, replace default policy, connect executor, start
canary traffic, introduce continuous theta, model slope/height/3D occlusion,
modify candidate generation, modify the network, or change default A*.

### Stage 23.0A Obstacle Source Materialization

Stage23.0A closes the input gap exposed by Stage23.0. High-fidelity candidate
rows should not carry full obstacle cell lists; instead the run writes a single
`xunce-exploration-coverage-obstacle-sources.json` artifact and candidate rows
reference it through `obstacle_source_id`, `obstacle_source_hash`, and
`obstacle_source_kind`. This keeps LOS replay reproducible without duplicating
large maps in every candidate row.

Physical `obstacle_cells` / `obstacle_rectangles` are preferred. If no physical
source exists, `blocked_cells`, `blocked_rectangles`, or sidecar
`passable_mask == false` may be exported as `blocked_as_obstacle_proxy`. The
runner must not derive obstacles from summary-only values such as
`blocked_count` or `passable_ratio`. `no_go_cells` are not LOS blockers unless
`no_go_blocks_los=true` is explicitly configured.

After materializing sources, Stage23.0A reruns Stage23.0. Physical obstacle
materiality routes to `implement_stage23_1_obstacle_aware_theta_reward_contract`;
blocked-only proxy sources route to `review_blocked_as_obstacle_proxy_semantics`;
missing sources route to `repair_stage23_map_obstacle_source_export`. Stage23.0A
does not train, publish checkpoints, replace policy, connect executor, start
canary traffic, introduce continuous theta, model 3D occlusion, or change
default A*.

### Stage 23.0B Map Obstacle Source Export

Stage23.0B repairs the upstream sidecar source needed by Stage23.0A. The
quasi-real bridge now exports `blocked_cells` from `passable_mask == false` and
sets `blocked_source_kind=passable_mask_false`. High-fidelity obstacle source
selection reads explicit sidecar `obstacle_cells` first, then sidecar
`blocked_cells`, and only falls back to deriving blocked cells from the
`passable_mask` grid when no explicit source is present.

The semantic boundary is strict. `physical_obstacle_cells` can only come from
explicit physical obstacle fields; `blocked_cells` are exported as
`blocked_as_obstacle_proxy`; `no_go_cells` remain non-LOS blockers unless
`no_go_blocks_los=true`. Risk, slope, or summary fields such as `blocked_count`
and `passable_ratio` cannot be promoted to physical obstacles.

The runner `scripts/run_xunce_stage23_0b_map_obstacle_source_export.py` wraps a
Stage23.0A rerun, writes a sidecar-source audit, and routes physical obstacle
materiality to `implement_stage23_1_obstacle_aware_theta_reward_contract`,
blocked-only proxy evidence to `review_blocked_as_obstacle_proxy_semantics`, and
missing sources to `continue_stage23_map_obstacle_source_export_repair`. It is
audit-only: no PPO, checkpoint publishing, default-policy replacement,
executor connection, canary traffic, continuous theta, 3D occlusion, network
change, or default A* change is authorized.

### Stage23.1 slope-derived obstacle source for endpoint theta LOS

Stage23.1 adopts the simpler terrain occlusion contract selected for the current
20m quasi-real DEMs. Instead of interpolating a DEM ray height for every target
cell, the bridge computes physical `slope_deg` from DEM height differences and
`resolution_m`. Cells above `max_traversable_slope_deg` are materialized as
`slope_blocked_cells`.

`slope_blocked_cells` have source kind `slope_blocked_as_obstacle_proxy`. They
mean “too steep to traverse and treated as a grid LOS blocker in this simplified
sensor model”; they are not physical obstacle labels such as rocks, walls, or
cracks. Stage23.2B aligns the default hard gate to Scout Mini platform capability:
`max_traversable_slope_deg=30.0`. The old `20.0` degree threshold is retained
only as a sensitivity/audit baseline, so Stage23 records 20/30 degree comparison
instead of treating 20 degrees as the default.

High-fidelity obstacle source priority is fixed as physical obstacle >
slope-blocked proxy > blocked/passable-mask proxy > optional no-go proxy. The
Stage23.1 runner reruns Stage23.0A/23.0 with slope derivation enabled. Material
slope occlusion routes to
`implement_stage23_2_slope_obstacle_aware_theta_reward_contract`; non-material
occlusion routes to `document_slope_obstacle_occlusion_audit_only`. It remains
audit-only: no PPO, checkpoint publishing, policy replacement, executor, canary,
continuous theta, DEM ray interpolation, network change, or default A* change is
authorized.

### Stage23.2A high-resolution terrain data ingestion

Stage23.2A replaces the default Stage23 sample-map source with higher-resolution
terrain inputs while keeping the Stage23 slope-blocked proxy semantics. The
primary source is the USGS Moon LRO South Pole DEM + Slope Map at 4m/pixel,
recorded in
`model-explorer/data/manifests/lunar_south_pole_usgs_lro_dem_slope_4m.json`.
LROC NAC DTM 2-5m products are represented separately in
`model-explorer/data/manifests/lunar_lroc_nac_dtm_roi_2m_5m.json` as
ROI-specific enhancement candidates; they must not replace the default map until
a selected product is proven to overlap the current ROI.

The runner
`scripts/run_xunce_stage23_2a_high_resolution_terrain_data_prepare.py` downloads
or copies raw GeoTIFF products to `D:\CodexDownloads`, records runtime sha256
hashes, reads bounded ROI windows through the GeoTIFF adapter, and emits a
high-fidelity compatible ROI expansion root. If a slope map is available,
`terrain_layers.slope_deg` comes from that product; otherwise slope is derived
from DEM height differences and physical resolution. Cells above
`max_traversable_slope_deg` become `slope_blocked_cells` with source kind
`slope_blocked_as_obstacle_proxy`. This is an obstacle proxy for traversal and
endpoint LOS, not physical rock/crack/wall truth.

Stage23.2A may rerun a bounded Stage23.1 smoke against the new high-resolution
root. Passing this stage only means the high-resolution DEM/slope source can
feed Stage23 LOS/reward work. It does not run PPO, publish checkpoints, replace
default policy, connect executor, start canary traffic, introduce continuous
theta, model 3D visibility, modify the network, or change default A*.

### Stage23.2B platform geometry and sensor contract alignment

Stage23.2B centralizes the SCOUT MINI + PiPER platform parameters in
`configs/platforms/agilex_scout_mini_piper_v1.json`. The contract records Scout
Mini geometry, four-wheel independent drive, differential skid-steer control,
ground clearance, maximum speed, minimum turning radius, `platform_max_climb_deg
=30.0`, and Livox Mid360 `360 x 59` degree FOV / 40m 10% reflectivity range.
Orbbec Dabai camera FOV and range remain `calibration_required=true` because the
source page does not provide those values.

Stage23.1 and Stage23.2A must propagate `platform_contract_id`,
`platform_contract_hash`, `platform_max_climb_deg`, and
`max_traversable_slope_deg` into sidecars, high-fidelity slices, obstacle source
audits, and summaries. Stage23.2B reruns bounded 30 degree platform-aligned and
20 degree sensitivity smokes and routes material 30 degree slope LOS occlusion to
`implement_stage23_2_slope_obstacle_aware_theta_reward_contract`. It remains
audit-only: no PPO, checkpoint publication, default-policy replacement, executor
connection, canary, continuous theta policy, 3D LOS, network change, or default
A* change is authorized.

### Stage 21.19 Policy Update Signal Source Repair

Stage21.19 checks whether the Stage21.18 updates actually reach the
candidate-action layer. It uses
`scripts/run_xunce_stage21_19_policy_update_signal_source_repair.py` and
`configs/xunce_stage21_19_policy_update_signal_source_repair_v1.json`.

The audit validates action/log-prob binding from `old_sampling_logits`, the
three masks (`sampling_mask`, `action_mask`, hard-risk clean mask),
candidate-set hashes, Stage21.4 ratio/clip/advantage behavior, policy/value
gradient components, and checkpoint parameter deltas by module. Pre/post action
movement uses only `policy=xunce` inference rows and the strong key
`scenario_id + step_index + current_cell + covered_cells_hash +
candidate_set_hash`; incumbent rows, weak joins, duplicate xunce keys, or vector
length mismatches cannot be used as evidence.

The current Stage21.19 evidence shows the policy update path is not broken:
old-log-prob recomputation passes, policy-head and candidate-encoder parameters
move, and strict xunce-only strong binding is available. However, only a few
selected actions change and final coverage / coverage AUC do not improve. The
current next route is `repair_stage21_return_advantage_credit_assignment`, not
another policy-head update-path repair.

### Stage 21.18 Iterative PPO Learning Curve Audit

Stage21.18 checks whether policy improvement needs several stable small PPO
updates instead of a single larger update. It uses
`scripts/run_xunce_stage21_18_iterative_ppo_learning_curve_audit.py` and
`configs/xunce_stage21_18_iterative_ppo_learning_curve_audit_v1.json`.

The key contract is checkpoint lineage. Stage21.18 must maintain a per-seed
checkpoint chain: each seed's round `N` experimental-only checkpoint is the
only valid source checkpoint for that same seed's round `N+1`. The collector
config used by Stage21.1 and the PPO update config used by Stage21.4 must point
to the same source checkpoint, and the audit verifies sha256, reload status,
and `experimental_only=true`.

The audit tracks learning-curve trends for probability movement, best
coverage-per-cost probability movement, rank/argmax/action changes, KL,
entropy, gradient norms, parameter deltas, final coverage, and coverage AUC.
If it writes a recommended Stage21.6 config, the selected checkpoint is only a
representative checkpoint because Stage21.6 cannot express the full per-seed
chain as one config input; the artifact also records the final per-seed
checkpoint map. It does not authorize checkpoint publication, default-policy
replacement, executor connection, canary traffic, reward-target changes,
network changes, action-space changes, candidate-generation changes, or default
A* changes.
