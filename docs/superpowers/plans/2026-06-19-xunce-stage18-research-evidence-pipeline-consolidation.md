# Xunce Stage 18 Research Evidence Pipeline Consolidation

## Summary

Stage 18 is consolidated into a read-only research evidence pipeline. The new
entry point is `xunce-stage18-research-evidence-pipeline`, backed by
`scripts/run_xunce_stage18_research_evidence_pipeline.py` and
`scripts/xunce_stage18_pipeline.py`.

The pipeline does not delete or replace legacy runners. It resolves active roots,
checks root lineage, summarizes existing artifacts, and separates evidence
validity from model advantage, training readiness, and release readiness.

## Mainline Mapping

- Stage 18.1: Stage 18A scenario and quasi-real ROI evidence.
- Stage 18.2: Stage 18I.3/I.4 candidate generation and planner/path-feedback
  validation.
- Stage 18.3: Stage 18B true checkpoint inference plus Stage 18G.0 true
  incumbent binding.
- Stage 18.4: Stage 18H.0 quantization, Stage 18F oracle benchmark, and
  Stage 18C-v2 rollout comparison.
- Stage 18.5: closure, attribution, and next-stage routing.

## Legacy Handling

Legacy diagnostic runners remain available for historical reproduction:
Stage 18D, Stage 18E, Stage 18F.1, Stage 18G.1, and Stage 18G.2. Earlier proxy
comparison and strict safe-efficient hard-gate experiments are not deleted, but
they are not the recommended mainline.

## Artifacts

The consolidated output root is:

```text
outputs/path_feedback_batch_xunce_stage18_research_evidence_pipeline_v1/
```

It writes:

```text
xunce-stage18-pipeline-summary.json
xunce-stage18-resolved-roots.json
xunce-stage18-module-results.jsonl
xunce-stage18-pipeline-report.md
```

## Boundaries

This is an offline research evidence closure. It does not start PPO, train an
actor/critic, publish checkpoints, replace the default policy, connect an
executor, start online canary traffic, modify the action space, or change the
default A* behavior.

The summary must keep `release_readiness=not_authorized` and
`training_readiness=not_authorized` unless a future stage explicitly changes the
governance model.
