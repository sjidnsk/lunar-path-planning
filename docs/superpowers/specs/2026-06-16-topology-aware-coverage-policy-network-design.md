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
   - Run bounded training as a research candidate only.
   - Do not publish default policy.
14 训练后离线评测
   - Re-run Global 99, multi-map, real-map replay, source-match, guard fallback,
     parameter, and latency audits.
15 Shadow / Replay 验证
   - Validate the trained candidate in offline shadow/replay only.
16 Sandbox Candidate Preflight
   - Package and load as sandbox candidate with kill-switch, rollback,
     telemetry, read-only default-policy boundary, and executor isolation.
17 发布治理门禁
   - Only after all prior gates pass may it enter release governance.
   - This still does not directly replace default policy.

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
