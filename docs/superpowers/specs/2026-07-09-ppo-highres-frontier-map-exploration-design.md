# PPO High-Resolution Frontier Map Exploration v1

## Status

Design approved for documentation. This document defines the first formal version of a new PPO exploration-mapping task. It does not implement code, publish checkpoints, replace any default policy, connect an executor, start canary traffic, or change the current Stage26 mainline.

## Purpose

The task is to train a PPO policy for lunar surface exploration when the robot starts with only a low-resolution global prior map. During rollout, onboard sensing incrementally builds a high-resolution observed map. The policy chooses safe frontier observation poses so that the high-resolution observed map reaches more than 99 percent coverage.

The design separates three concerns:

- Low-resolution prior maps provide global guidance.
- High-resolution observed maps provide trusted local and frontier state.
- Planner and safety checks remain responsible for execution feasibility.

## Success Metric

The only task success metric is:

```text
highres_observed_coverage_rate >= 0.99
```

For v1, coverage and confidence are equivalent:

```text
unobserved cell -> confidence = 0
observed cell   -> confidence = 1
```

Future versions may replace binary confidence with continuous confidence from repeated, multi-angle, or distance-weighted observations. That is out of scope for v1.

## Assumptions

- The initial map available at deployment is a low-resolution global prior, not a high-resolution truth map.
- The robot can build high-resolution observed map data only for areas already sensed during the current rollout.
- Vehicle footprint is smaller than the low-resolution prior cell size, so safe target cells must be selected in the high-resolution observed map.
- Unknown high-resolution cells cannot be used as landing or stopping targets.
- Target cells are observed-safe frontier cells. Unknown cells are exploration targets only through sensor observation, not physical target locations.

## Map Scale Profiles

The first implementation uses three fixed scale profiles. The kilometer profile represents the large lunar polar use case, but its high-resolution map is not passed to the policy as a dense tensor.

| Profile | Purpose | ROI | High-resolution map | Low-resolution global map | Local high-resolution crop | Frontier cap |
| --- | --- | --- | --- | --- | --- | --- |
| Smoke v1 | Architecture smoke tests, tiny PPO overfit, unit tests | 64m x 64m | 128 x 128 @ 0.5m/cell | 32 x 32 @ 2m/cell | 64 x 64, covering 32m x 32m | 512 |
| Standard v1 | Main v1 training and ablations | 128m x 128m | 256 x 256 @ 0.5m/cell | 32 x 32 @ 4m/cell | 96 x 96, covering 48m x 48m | 1024 |
| Kilometer v1 | Large lunar polar scenes and long-horizon coverage stress tests | 1024m x 1024m | 2048 x 2048 @ 0.5m/cell, maintained by the environment only | 128 x 128 @ 8m/cell | 192 x 192, covering 96m x 96m | 4096 |

The recommended development order is:

```text
Smoke v1 -> Standard v1 -> Kilometer v1
```

The following metadata must be stored in checkpoints, rollout manifests, and evaluation reports:

```text
scale_profile
roi_size_m
world_origin_m
highres_resolution_m
highres_shape
lowres_resolution_m
lowres_shape
tile_ratio
local_crop_shape
local_crop_size_m
frontier_top_m
coordinate_convention
theta_convention
sensor_model_id
sensor_range_m
sensor_fov_deg
sensor_range_cells
coverage_update_mode
```

For Kilometer v1, the environment may maintain a 2048 x 2048 high-resolution grid internally for mapping, frontier extraction, coverage accounting, and planner validation. The policy observation must remain hierarchical and sparse:

```text
global low-resolution map
+ global high-resolution coverage summary
+ local high-resolution crop
+ sparse global frontier candidates
+ pose and budget features
```

The full kilometer-scale high-resolution map must not be stored in every PPO transition as a dense policy tensor.

## Non-Goals

- Do not train PPO to output low-level velocity or steering commands.
- Do not use complete future high-resolution ground truth as a policy input.
- Do not feed the full kilometer-scale high-resolution map directly into the policy network as a dense tensor.
- Do not bypass planner or safety checks.
- Do not claim mathematical global optimality.
- Do not replace default A* or Hybrid A* semantics.
- Do not model continuous confidence, repeated observations, or multi-angle confidence in v1.

## MDP Definition

Each PPO step represents selecting and attempting one next observation pose:

```text
state_t:
  current low-resolution prior state
  current high-resolution observed map state
  current robot pose
  current global frontier action set

action_t:
  selected frontier target cell
  selected continuous target theta

transition:
  planner validates and executes the target pose
  sensor updates high-resolution observed coverage
  reward and done are computed from the updated state
```

An episode starts from an initial observed area around the robot and ends on success, budget exhaustion, stagnation, or severe safety violation.

## Sensor And Coverage Update Model

The v1 sensor model is a forward field-of-view endpoint observation model:

```text
sensor_model_id = endpoint-forward-fov-90-range-20m-los/v1
sensor_range_m = 20.0
sensor_fov_deg = 90.0
sensor_direction = target_theta
observation_origin = target_pose
coverage_update_mode = endpoint_observation_only
```

At each PPO step, the policy selects a target frontier cell and a continuous `target_theta`. The planner attempts to reach the target pose. If the pose is valid and executed, the environment observes from that endpoint pose only:

```text
visible_cell =
  inside_map_bounds
  AND distance(endpoint_cell_center, cell_center) <= sensor_range_m
  AND angular_distance(bearing(endpoint, cell), target_theta) <= sensor_fov_deg / 2
  AND line_of_sight_not_blocked
```

Line-of-sight blockers include hard obstacle cells and slope-blocked cells when those sources are available. The v1 hard slope threshold stays aligned with the existing platform contract:

```text
max_traversable_slope_deg = 30.0
```

All currently unknown high-resolution cells satisfying this predicate become observed. For v1:

```text
observed cell -> coverage = 1, confidence = 1
unknown cell  -> coverage = 0, confidence = 0
```

The sensor range is defined in meters. For implementation on a grid:

```text
sensor_range_cells = round(sensor_range_m / highres_resolution_m)
```

Because all v1 scale profiles use `highres_resolution_m = 0.5`, the fixed v1 value is:

```text
sensor_range_cells = 40
```

The continuous policy angle is the source of truth. If a helper expects degrees, convert from radians:

```text
target_theta_deg = degrees(target_theta_rad) mod 360
```

Legacy project stages used `theta_bin_count = 8`, `theta_step_deg = 45`, and smaller `sensor_range_cells` values for smoke-scale audits. Those values are historical compatibility settings and are not the v1 action-space contract for this design.

Coverage gain is measured after endpoint observation:

```text
coverage_gain_cells = count(cells that changed from unknown to observed)
coverage_gain_rate = coverage_gain_cells / total_highres_coverage_denominator_cells
```

Along-path continuous sensing, repeated-observation confidence accumulation, sensor noise, and multi-angle confidence are out of scope for v1.

## Observation Schema

Each policy observation contains:

```text
observation = {
  global_lowres_prior_state,
  global_highres_coverage_summary,
  local_highres_observed_crop,
  frontier_cells,
  frontier_features,
  frontier_mask,
  pose_features
}
```

### global_lowres_prior_state

This tensor covers the whole task region at low resolution. It is derived from deployment-available prior data.

Recommended v1 channels:

```text
height_prior
value_prior
obstacle_prior
traversability_prior
current_position_marker_lowres
heading_sin_marker_lowres
heading_cos_marker_lowres
```

This input provides global geography, value, and coarse traversability context.

### global_highres_coverage_summary

This tensor is a low-resolution summary of the current high-resolution observed map. It must be computed only from cells observed so far.

Recommended v1 channels per summary tile:

```text
covered_ratio
unknown_ratio
frontier_count
observed_safe_frontier_count
mean_uncovered_value_prior
reachable_frontier_count
```

This input prevents purely local behavior such as repeatedly exploring east while west remains uncovered.

For Kilometer v1, this summary is the primary way the policy sees global high-resolution coverage progress. The full 2048 x 2048 high-resolution map remains an environment-side state, not a dense policy input.

### local_highres_observed_crop

This tensor is a high-resolution crop around the current robot pose. It contains only currently observed local information plus unknown markers.

Recommended v1 channels:

```text
observed_height
coverage_mask
obstacle
traversability
frontier_mask
current_position_marker
heading_sin_marker
heading_cos_marker
```

This input supports local safety, local terrain reasoning, and near-field frontier geometry.

### frontier_cells

The global high-resolution frontier action set for the current step. Each item is a high-resolution cell coordinate:

```text
frontier_cells[i] = (x_i, y_i)
```

These cells are action candidates, not unknown cells to be mapped directly.

### frontier_features

Each frontier cell has a feature row used by the policy to score that action.

Recommended v1 fields:

```text
x_norm
y_norm
distance_from_robot_norm
bearing_sin
bearing_cos
potential_coverage_gain_norm
visible_unknown_count_norm
value_gain_norm
traversability
clearance_norm
reachable_prefilter_cost_norm
region_coverage_ratio
region_unknown_ratio
same_connected_component
```

These features must be derived from observed high-resolution state, low-resolution prior, and sensor geometry only. They must not use future high-resolution truth.

### frontier_mask

Padding and validity mask for batched training:

```text
frontier_mask[i] = true  if frontier_features[i] is a valid action
frontier_mask[i] = false if row i is padding or invalid
```

### pose_features

Robot pose and episode progress scalars:

```text
x_norm
y_norm
sin(theta)
cos(theta)
current_coverage_rate
remaining_step_budget_norm
remaining_path_budget_norm
no_gain_steps_norm
```

These features help the policy and value head distinguish early exploration from late coverage completion.

## Frontier Action Set

The action cell set is generated by the environment at every step:

```text
frontier_cells = extract_global_observed_safe_frontier_cells(state_t)
```

A frontier cell is valid only if:

```text
observed == true
obstacle == false
traversability >= threshold
clearance >= vehicle_radius + safety_margin
near_unknown == true
potential_gain > 0
reachable_prefilter == true
```

`near_unknown` means the cell is an observed-safe landing cell near currently unobserved high-resolution cells, typically within sensor range or near an observed-unobserved boundary.

The extractor should prefer high recall. If the action set is too large, top-M pruning must preserve spatial diversity instead of only selecting the nearest or highest immediate-gain frontier cells.

For Kilometer v1, top-M pruning must be region-aware. It should preserve candidates across directions, connected components, and low-resolution coverage-summary tiles so that distant unexplored regions are not permanently removed from the policy action set.

Frontier potential gain must use the fixed v1 sensor model above. Candidate gain estimates should evaluate the unknown cells visible from the candidate endpoint under a forward 90 degree FOV with 20m range and line-of-sight filtering. Since `target_theta` is continuous, the estimator may use an analytic best-facing direction toward nearby unknown cells or a small internal sampling heuristic, but the stored PPO action remains continuous theta.

## Action Space

The v1 action space is B-sparse:

```text
action = {
  target_frontier_index,
  target_cell,
  target_theta
}
```

The policy distribution factorizes as:

```text
P(action | state)
= P(target_frontier_index | state, frontier_cells)
  * P(target_theta | state, selected_frontier_cell)
```

The PPO log probability is:

```text
log_prob_total = log_prob_frontier_index + log_prob_theta
```

### target_frontier_index

`target_frontier_index` is sampled from a masked categorical distribution over the current `frontier_cells`.

### target_theta

`target_theta` is sampled from a continuous Von Mises distribution conditioned on the state and selected frontier cell.

The theta head should output:

```text
theta_mu_sin
theta_mu_cos
theta_kappa_raw
```

These are converted to:

```text
theta_mu_rad = atan2(theta_mu_sin, theta_mu_cos)
theta_kappa  = softplus(theta_kappa_raw) + epsilon
```

Rollout and update must preserve separate theta and frontier log probabilities for auditability.

## Execution Flow

```text
1. Build observation from current map state and pose.
2. Extract global observed-safe frontier cells.
3. Build frontier_features and frontier_mask.
4. Policy scores frontier actions.
5. Sample target_frontier_index.
6. Policy samples target_theta using the selected frontier cell.
7. Planner validates current_pose -> (target_cell, target_theta).
8. If valid, execute and update high-resolution observed coverage.
9. Compute reward and done.
10. Store the full transition contract for PPO update.
```

Planner failure should be classified by cause where possible:

```text
cell_unreachable
theta_unreachable
path_collision
safety_violation
planner_timeout
```

## Reward

The task success metric is coverage above 99 percent. Reward is training support, not a separate success definition.

The v1 reward is temporarily:

```text
reward =
  coverage_gain_shaping
+ success_bonus_if_coverage_rate_gt_0_99
- invalid_action_penalty
- safety_violation_penalty
```

Where:

```text
coverage_gain_shaping:
  positive reward for newly observed high-resolution cells

success_bonus_if_coverage_rate_gt_0_99:
  terminal success bonus when highres_observed_coverage_rate >= 0.99

invalid_action_penalty:
  penalty for invalid target, planner failure, or action-mask violation

safety_violation_penalty:
  stronger penalty for collision, hard obstacle entry, unsafe slope, or clearance violation
```

Reward weights remain open for later tuning. The success metric does not change.

## Episode Termination

An episode ends under any of these conditions:

```text
success_done:
  highres_observed_coverage_rate >= 0.99

budget_done:
  step_count >= max_steps
  or path_length_used >= path_budget

stagnation_done:
  no_gain_steps >= N

safety_done:
  severe safety violation occurred
```

Planner failure may be treated as invalid action and continue unless it indicates an unrecoverable safety condition.

## PPO Transition Contract

Every trainable transition must store enough information to recompute PPO log probabilities without re-extracting frontier actions.

Required fields:

```text
observation tensors
frontier_cells
frontier_features
frontier_mask
selected_frontier_index
selected_cell_xy
selected_theta
old_log_prob_frontier
old_log_prob_theta
old_log_prob_total
old_value
reward
done
advantage
return
frontier_extractor_version
observation_schema_version
action_space_version
reward_version
planner_validation_summary
```

Important rule:

```text
PPO update must use the saved frontier list, features, and mask from rollout.
It must not re-run frontier extraction and silently replace the action space.
```

## Network Shape

A v1 network should include:

```text
global encoder:
  encodes global_lowres_prior_state and global_highres_coverage_summary

local encoder:
  encodes local_highres_observed_crop

pose encoder:
  encodes pose_features

frontier encoder:
  encodes each frontier_features row

frontier scoring head:
  produces masked logits over frontier cells

theta head:
  produces Von Mises parameters for the selected frontier cell

value head:
  estimates state value from global, local, pose, and pooled frontier context
```

The first implementation may share encoders between policy and value heads, but checkpoint metadata must record the architecture and observation schema.

## Leakage And Consistency Rules

The following are hard failures:

- Policy input uses complete future high-resolution truth.
- Frontier generation filters on unknown high-resolution obstacle or height truth.
- Training and deployment use different channel order, normalization, or map resolution semantics.
- Update-time action set differs from rollout-time action set.
- Unknown cells are selected as physical landing targets.
- Planner or safety checker is bypassed.

Allowed information:

- Current observed high-resolution map.
- Low-resolution prior map.
- Current coverage mask.
- Sensor geometry.
- Prior value map.
- Derived summaries from already observed state.

## Evaluation

Primary evaluation:

```text
final highres_observed_coverage_rate
success_rate where coverage >= 0.99
```

Diagnostic evaluation:

```text
coverage curve over steps
path length used
invalid action count
planner failure count by cause
safety violation count
stagnation termination count
frontier action entropy
theta distribution diagnostics
frontier extractor recall diagnostics
```

Evaluation must verify that all inference observations use deployment-available information only.

## Open Decisions For Planning

The following values are intentionally not fixed in this design:

```text
frontier spatial diversity strategy
coverage gain scaling
invalid and safety penalty weights
max_steps and path_budget
stagnation N
```

The three scale profiles above are fixed for v1. The remaining values should be selected in the implementation plan and validated through small smoke tests before larger PPO experiments.
