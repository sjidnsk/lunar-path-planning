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

For v1, the rate is measured against a reachable and observable high-resolution free-cell denominator:

```text
highres_observed_coverage_rate =
  count(observed_highres_cells AND coverable_mask)
  / count(coverable_mask)

coverage_denominator_source =
  reachable_observable_free_highres_cells/v1
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

The first implementation uses three scale profiles. The kilometer profile represents the large lunar polar use case, but its high-resolution map is not passed to the policy as a dense tensor.

| Profile | Purpose | ROI | High-resolution map | Low-resolution global map | Local high-resolution crop | Frontier cap |
| --- | --- | --- | --- | --- | --- | --- |
| Smoke v1 | Architecture smoke tests, tiny PPO overfit, unit tests | 64m x 64m | 128 x 128 @ 0.5m/cell | 32 x 32 @ 2m/cell | 64 x 64, covering 32m x 32m | 512 |
| Standard v1 | Main v1 training and ablations | 128m x 128m | 256 x 256 @ 0.5m/cell | 32 x 32 @ 4m/cell | 96 x 96, covering 48m x 48m | 1024 |
| Kilometer v1 | Large lunar polar scenes and long-horizon coverage stress tests | 1024m x 1024m | 2048 x 2048 @ 0.5m/cell, maintained by the environment only | 128 x 128 @ 8m/cell | 192 x 192, covering 96m x 96m | 2048 default, 4096 max after overflow audit |

Scale-profile metadata is derived directly from this table:

```text
roi_size_m:
  ROI physical size

highres_shape:
  high-resolution map cell shape

lowres_resolution_m:
  low-resolution global map meters per cell

lowres_shape:
  low-resolution global map cell shape

tile_ratio:
  lowres_resolution_m / highres_resolution_m

local_crop_shape:
  local high-resolution crop cell shape

local_crop_size_m:
  local high-resolution crop physical size
```

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
frontier_top_m_max
top_m_selection_policy
coordinate_convention
theta_convention
platform_safety_constants_source
vehicle_radius_m
safety_margin_m
traversability_threshold
max_traversable_slope_deg
coverage_progress_feature_source
theta_kappa_parameterization
baseline_cost_score_policy
sensor_model_id
sensor_range_m
sensor_fov_deg
sensor_range_cells
coverage_update_mode
los_model
los_implementation_source
ray_angle_step_deg
observation_overlap_policy
coverage_denominator_source
reachable_prefilter_source
planner_validation_source
planner_search_state
execution_observation_source
path_observation_step_m
path_budget_mode
budget_done_mode
coverable_mask_algorithm_id
coverable_mask_hash
coverable_mask_exact
coverable_mask_precompute_scope
frontier_segment_candidate_policy
normal_confidence_threshold
max_candidates_per_segment
standoff_distance_m
potential_gain_source
candidate_priority_source
reward_scaling_source
stagnation_policy
efficiency_pressure_source
max_steps_by_scale
primary_eval_metric
baseline_comparison_contract
baseline_algorithms_v1
baseline_candidate_set_contract
baseline_information_contract
scenario_split_policy
eval_episode_count_policy
main_result_table_metrics
confidence_reporting
ppo_eval_policy_mode
evaluation_seed_policy
tie_break_policy
progress_reporting_policy
implementation_file_structure
legacy_code_isolation_policy
integration_adapter_policy
implementation_roadmap_version
stage1_acceptance_policy
stage2_acceptance_policy
stage3_acceptance_policy
stage4_acceptance_policy
stage5_acceptance_policy
stage6_acceptance_policy
stage7_acceptance_policy
stage8_acceptance_policy
rollout_transition_storage
rollout_collection_mode
num_envs
rollout_steps_per_env
rollout_batch_size
advantage_method
policy_loss_type
frontier_distribution_type
theta_distribution_type
value_loss_type
entropy_regularization
invalid_action_training_policy
ppo_update_schedule
checkpoint_policy
network_architecture_version
network_memory_mode
network_diagram_path
```

For Kilometer v1, the environment may maintain a 2048 x 2048 high-resolution grid internally for mapping, frontier extraction, coverage accounting, and planner validation. The policy observation must remain hierarchical and sparse:

```text
global low-resolution map
+ global high-resolution coverage summary
+ local high-resolution crop
+ sparse global frontier candidates
+ pose and episode progress features
```

The full kilometer-scale high-resolution map must not be stored in every PPO transition as a dense policy tensor.

## Coordinate And Angle Convention

The v1 coordinate and heading convention is fixed so that observation generation, frontier cells, planner validation, network features, and report artifacts use the same frame:

```text
coordinate_convention =
  world_xy_grid_col_row_cell_center/v1

theta_convention =
  radians_world_x_ccw_normalized_minus_pi_to_pi/v1

world_origin_m:
  lower-left ROI corner unless explicitly overridden by config

world_x:
  increases to the right / east direction of the ROI

world_y:
  increases upward / north direction of the ROI

grid_cell:
  (cell_x, cell_y)
  where cell_x = column index
  and cell_y = row index

cell_center_world:
  x_m = world_origin_x_m + (cell_x + 0.5) * highres_resolution_m
  y_m = world_origin_y_m + (cell_y + 0.5) * highres_resolution_m

theta_rad:
  0 points along +world_x
  positive rotation is counter-clockwise toward +world_y
  normalized to [-pi, pi)
```

All stored `frontier_cells`, `selected_cell_xy`, planned path cells, and map-index features must use `(cell_x, cell_y)` order. If any internal library uses `(row, col)`, the adapter must convert at the boundary and record the conversion in the audit artifact.

## Platform Safety Constants

V1 does not silently invent vehicle geometry. Safety constants must come from the stage config or platform config and must be written into checkpoints, rollout manifests, and evaluation reports:

```text
platform_safety_constants_source =
  required_stage_or_platform_config/v1

required constants:
  vehicle_radius_m
  safety_margin_m
  traversability_threshold

fixed v1 constant:
  max_traversable_slope_deg = 30.0

derived constant:
  min_clearance_m = vehicle_radius_m + safety_margin_m
```

Any environment reset must fail fast if `vehicle_radius_m`, `safety_margin_m`, or `traversability_threshold` is missing. All clearance checks use `clearance >= min_clearance_m`.

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
  planner validates target cell reachability and endpoint theta
  sensor updates high-resolution observed coverage
  reward and done are computed from the updated state
```

An episode starts from an initial observed area around the robot and ends on success, fixed-budget failure, stagnation, no-candidate termination, or severe safety violation.

## Sensor And Coverage Update Model

The v1 sensor model observes along the executed path and then observes once more at the endpoint using the PPO-selected theta:

```text
sensor_model_id = path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1
execution_observation_source = path_tangent_samples_plus_endpoint_theta/v1
sensor_range_m = 20.0
sensor_fov_deg = 90.0
path_observation_step_m = 1.0

sensor_direction =
  path_tangent_heading for intermediate path samples
  target_theta for endpoint observation

observation_origin =
  sampled poses along executed path
  plus endpoint pose

coverage_update_mode =
  path_continuous_observation_plus_endpoint_theta_observation

los_model = two_dimensional_grid_line_of_sight/v1
los_implementation_source = grid_ray_casting_fov/v1
ray_angle_step_deg = 1.0
observation_overlap_policy = update_newly_observed_only/v1
```

At each PPO step, the policy selects a target frontier cell and a continuous `target_theta`. The planner attempts to reach the target cell. If the path is valid and executed, the environment observes from regularly sampled path poses using the local path tangent heading, then observes from the endpoint using `target_theta`.

Each observation sample uses the same 20m / 90 degree FOV / 2D LOS geometry:

```text
visible_cell =
  inside_map_bounds
  AND distance(observation_origin, cell_center) <= sensor_range_m
  AND angular_distance(bearing(observation_origin, cell), sensor_direction) <= sensor_fov_deg / 2
  AND line_of_sight_not_blocked
```

For an intermediate sample, `sensor_direction` is:

```text
path_tangent_heading =
  atan2(next_y - current_y, next_x - current_x)
```

At corners, the heading follows the currently executed path segment. At the final endpoint, the robot may rotate in place and `sensor_direction = target_theta`.

Line-of-sight blockers include hard obstacle cells and slope-blocked cells when those sources are available. The v1 hard slope threshold stays aligned with the existing platform contract:

```text
max_traversable_slope_deg = 30.0
```

The v1 LOS model is a 2D grid line-of-sight model, not a 3D DEM visibility model. The implementation should use 2D grid ray casting over the FOV instead of checking a separate LOS segment for every cell in the sector.

The ray-casting rule is:

Recommended implementation semantics:

```text
ray_angle_set =
  angles from sensor_direction - sensor_fov_deg / 2
  to sensor_direction + sensor_fov_deg / 2
  in ray_angle_step_deg increments

ray traversal =
  deterministic Bresenham-style or DDA integer grid traversal
  from observation_origin
  until sensor_range_cells
  or map boundary
  or los_blocker_cell

los_blocker_cell =
  hard_obstacle_cell
  OR slope_blocked_cell

slope_blocked_cell =
  slope_deg > max_traversable_slope_deg

visible_cells_for_sample =
  unique cells visited by any ray before the ray stops,
  including the first blocker cell if encountered
```

With `sensor_range_m = 20m`, `highres_resolution_m = 0.5m`, and `ray_angle_step_deg = 1.0`, a single sample uses about 91 rays and at most 40 cell visits per ray before blocker early-stopping. This is roughly 3,640 cell visits per observation sample before deduplication, and it avoids repeated per-cell LOS checks over the whole FOV sector.

The observation origin cell itself does not block its own observation. If a ray reaches a blocker, that blocker cell may be observed as obstacle or slope-blocked evidence, but cells behind it remain unknown for that sample.

Rays may visit the same cell more than once because neighboring rays overlap. The implementation must deduplicate visible cells within a sample before map update and coverage-gain accounting.

Height is still part of the observed high-resolution map, but v1 does not perform continuous 3D height-profile visibility interpolation. Terrain height may be used to derive slope-blocked cells; full 3D DEM LOS is reserved for a later version.

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

Coverage gain is measured after the full action observation batch:

```text
action_visible_cells =
  union(visible_cells_for_sample over all path samples and endpoint sample)

newly_observed_cells =
  cells in action_visible_cells
  where observed_mask changed from 0 to 1 during this action

coverage_gain_cells =
  count(newly_observed_cells AND cells in coverable_mask)

coverage_gain_rate = coverage_gain_cells / total_highres_coverage_denominator_cells
```

Overlapping observations are expected and are handled by set union plus unknown-to-observed accounting:

```text
already observed free cell:
  may be traversed by rays
  not written again
  not counted again in coverage_gain_cells

already observed blocker cell:
  may be traversed by rays
  still stops the ray
  not counted again in coverage_gain_cells

unknown visible cell:
  written once into observed map
  counted in coverage_gain_cells only if it is in coverable_mask and not blocked
```

Repeated-observation confidence accumulation, sensor noise, and multi-angle confidence are out of scope for v1.

## Observed Map Update Rules

The environment may hold high-resolution scenario truth internally. The policy sees only the observed high-resolution map and prior-derived low-resolution maps. Truth values may enter the observed map only through a successful sensor observation.

For each currently unknown cell in `action_visible_cells`:

```text
observed_mask[cell] = 1
confidence[cell] = 1
observed_height[cell] = truth_height[cell]
observed_obstacle[cell] = truth_obstacle[cell]
observed_slope_deg[cell] = truth_or_derived_slope_deg[cell]
observed_slope_blocked[cell] = observed_slope_deg[cell] > max_traversable_slope_deg
observed_traversability[cell] =
  0 if observed_obstacle[cell] or observed_slope_blocked[cell]
  otherwise truth_or_derived_traversability[cell]
```

For cells that fail LOS because they are behind a blocker:

```text
observed_mask[cell] remains unchanged
confidence[cell] remains unchanged
height / obstacle / slope / traversability remain unknown
```

For a visible hard obstacle or slope-blocked cell:

```text
observed_mask[cell] = 1
confidence[cell] = 1
obstacle or slope_blocked evidence is written
cell is excluded from coverable_mask
cell is not counted in coverage_gain_cells
cell is never a safe frontier target
```

Coverage and observation are related but not identical:

```text
observed:
  any visible cell whose evidence is written into the observed map

coverage_gain:
  newly observed cells that are also in coverable_mask
```

The `value` signal is not written from hidden high-resolution truth. In v1 it remains a deployment-available prior:

```text
value_prior:
  comes from low-resolution prior data
  may be sampled or aggregated for frontier_features
  may be resampled into a local crop if the network uses a local value channel
  must not be replaced by hidden high-resolution future value truth
```

`local_frontier_channel` is not a persistent sensed channel. It is recomputed after each map update from the current observed map, safety rules, reachability prefilter, sensor model, and frontier extraction rules. It is a local crop map channel and is separate from the sparse action-set `candidate_valid_mask`.

## Coverage Denominator And Coverable Mask

The 99 percent success denominator is not the full ROI. It is the set of high-resolution cells that the robot can theoretically cover from the start pose under the v1 safety, sensor, and LOS contracts.

The v1 denominator is:

```text
coverage_denominator_source = reachable_observable_free_highres_cells/v1
coverable_mask_precompute_scope = episode_initialization
```

The semantic construction is:

```text
safe_free_mask =
  inside_roi
  AND not hard_obstacle
  AND not slope_blocked
  AND clearance >= min_clearance_m

reachable_safe_mask =
  start-cell connected component within safe_free_mask

coverable_mask =
  free high-resolution cells
  AND exists a cell in reachable_safe_mask
      within sensor_range_m
      with two_dimensional_grid_line_of_sight/v1 visible
```

This excludes free-looking cells that are enclosed by obstacle or slope-blocked regions and cannot be reached or seen from any reachable safe observation pose.

For denominator construction, the 90 degree FOV is not a limiting factor because `target_theta` is continuous. If a cell is within 20m and has clear LOS from some reachable safe pose, the robot can theoretically face that cell. The 90 degree FOV still applies to each executed endpoint observation, reward step, and frontier potential-gain estimate.

Blocked cells are treated as follows:

```text
hard_obstacle_cell:
  may be observed as map evidence
  excluded from coverable_mask
  never a safe frontier target

slope_blocked_cell:
  may be observed as map evidence
  excluded from coverable_mask
  never a safe frontier target

free_cell unreachable and not observable from reachable_safe_mask:
  excluded from coverable_mask
```

The coverable mask is environment-side accounting, not a policy input. In simulation, it may be computed from high-resolution scenario truth at episode initialization and cached by map, start pose, scale profile, sensor model, LOS model, and safety thresholds. It must not be passed to the policy as a dense channel, and frontier generation must not use it to reveal unknown high-resolution truth.

For implementation efficiency:

```text
Smoke v1 and Standard v1:
  compute an exact LOS-refined coverable_mask during episode initialization

Kilometer v1:
  compute once during episode initialization or load from cache
  use a distance-transform or dilation prefilter for within-20m candidates
  then apply LOS refinement to produce the official coverable_mask
```

Any approximation must be explicit in metadata:

```text
coverable_mask_exact = true | false
coverable_mask_algorithm_id
coverable_mask_hash
```

Official success-rate comparisons should use `coverable_mask_exact = true`. Approximate masks are allowed only for development smoke tests or clearly labeled diagnostics.

## Reachability Prefilter

The v1 reachability prefilter source is:

```text
reachable_prefilter_source =
  observed_safe_connected_component_prefilter/v1
```

It is a fast action-set prefilter, not the final path planner. Its job is to remove candidate landing cells that are clearly disconnected from the robot in the currently observed safe map.

The observed-safe cells are:

```text
observed_safe_cell =
  observed == true
  AND obstacle == false
  AND slope_blocked == false
  AND traversability >= traversability_threshold
  AND clearance >= min_clearance_m
```

At each decision step, compute the observed-safe connected component from the current robot cell:

```text
reachable_component =
  BFS(current_robot_cell, observed_safe_cell graph)

reachable_prefilter == true if:
  candidate_cell in reachable_component
```

The graph should use the same grid connectivity convention as the local planner, such as 8-connected cells for holonomic grid prefiltering. The implementation must record the chosen connectivity in metadata if it affects results.

The boundary between prefiltering and planner validation is:

```text
reachable_prefilter == false:
  candidate does not enter the PPO action set

reachable_prefilter == true:
  candidate may enter the PPO action set
  but execution still requires planner validation

planner validation failure:
  apply invalid_action_penalty
  done = false
  unless the failure is a severe safety violation
```

V1 does not use a hard path budget:

```text
path_budget_m =
  none for v1

path_budget_mode =
  disabled for v1

budget_done =
  disabled for v1

budget_done_mode =
  disabled for v1

remaining_path_budget_norm =
  omitted for v1
```

`detour_factor` and `remaining_path_budget` are not hard filters in v1. Path length remains a diagnostic metric, and may be used later for efficiency analysis, but it is not part of the v1 success/failure condition.

If `reachable_prefilter_cost_norm` is retained as a candidate feature, it is only a soft feature for scoring or pruning among already reachable candidates. It may be computed from observed-map BFS distance or Euclidean distance and normalized within the current candidate set. It must not use hidden truth, `coverable_mask`, `path_budget`, or `detour_factor` as a hard gate.

## Observation Schema

Each policy observation contains:

```text
observation = {
  global_lowres_prior_state,
  global_highres_coverage_summary,
  local_highres_observed_crop,
  frontier_cells,
  frontier_features,
  candidate_valid_mask,
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

The summary grid has the same shape as the low-resolution global map in the selected scale profile. Each summary tile aggregates the high-resolution cells whose centers fall inside that low-resolution tile.

V1 channels per summary tile:

```text
observed_ratio
unknown_ratio
observed_free_ratio
observed_blocked_ratio
frontier_count
observed_safe_frontier_count
reachable_frontier_count
mean_uncovered_value_prior
```

The aggregation rules are:

```text
tile_cell_count =
  count(high-resolution cells in this low-resolution tile)

observed_ratio =
  count(observed_mask == 1 in tile) / tile_cell_count

unknown_ratio =
  count(observed_mask == 0 in tile) / tile_cell_count

observed_free_ratio =
  count(
    observed_mask == 1
    AND observed_obstacle == 0
    AND observed_slope_blocked == 0
    AND observed_traversability >= traversability_threshold
    in tile
  ) / tile_cell_count

observed_blocked_ratio =
  count(
    observed_mask == 1
    AND (observed_obstacle == 1 OR observed_slope_blocked == 1)
    in tile
  ) / tile_cell_count

frontier_count =
  count(frontier cells in tile) / tile_cell_count

observed_safe_frontier_count =
  count(observed-safe frontier cells in tile) / tile_cell_count

reachable_frontier_count =
  count(frontier cells passing reachability prefilter in tile) / tile_cell_count

mean_uncovered_value_prior =
  mean(value_prior for unobserved high-resolution cells in tile)
  or 0 when the tile has no unobserved cells
```

These summary channels must not use the dense `coverable_mask` as a per-tile denominator or spatial mask, because that would expose hidden high-resolution scenario structure to the policy. They may use the current observed map, current frontier set, reachability prefilter result, sensor geometry, and deployment-available low-resolution value prior.

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
local_frontier_channel
current_position_marker
heading_sin_marker
heading_cos_marker
```

This input supports local safety, local terrain reasoning, and near-field frontier geometry.

### frontier_cells

The global high-resolution frontier action set for the current step. Each item is a high-resolution cell coordinate:

```text
frontier_cells[i] = (cell_x_i, cell_y_i)
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
frontier_segment_id_norm
segment_length_norm
normal_sin
normal_cos
normal_confidence
candidate_generation_mode
recommended_theta_sin
recommended_theta_cos
traversability
clearance_norm
reachable_prefilter_cost_norm
region_coverage_ratio
region_unknown_ratio
same_connected_component
```

These features must be derived from observed high-resolution state, low-resolution prior, and sensor geometry only. They must not use future high-resolution truth.

`candidate_generation_mode` should be encoded as a stable categorical feature, such as `0` for `regular_normal_standoff` and `1` for `irregular_local_gain_sampling`, or as an equivalent one-hot representation recorded in `observation_schema_version`.

### candidate_valid_mask

Padding and validity mask for batched training:

```text
candidate_valid_mask[i] = true  if frontier_features[i] is a valid action
candidate_valid_mask[i] = false if row i is padding or invalid
```

The environment must handle empty action sets before calling the policy:

```text
valid_candidate_count =
  count(candidate_valid_mask == true)

if valid_candidate_count == 0:
  do not run the policy network
  terminate the episode as no_candidate_done
  record the empty-action-set reason in diagnostics
```

This rule prevents undefined masked categorical sampling, masked entropy/log-probability recomputation, and masked mean/max pooling when every candidate row is padding or invalid. Because no policy action is sampled, this terminal environment outcome must not be stored as a trainable PPO action transition with fake action log probabilities.

`stagnation_done` is reserved for the consecutive zero-coverage-gain counter and must not be used as the empty-candidate terminal reason.

### pose_features

Robot pose and episode progress scalars:

```text
x_norm
y_norm
sin(theta)
cos(theta)
observed_roi_ratio
remaining_step_budget_norm
```

These features help the policy and value head distinguish early exploration from late coverage completion.

`remaining_path_budget_norm` is intentionally omitted in v1 because hard path budget termination is disabled.

The policy input must not include the true `highres_observed_coverage_rate`, because that scalar uses the environment-side `coverable_mask` denominator. V1 uses an observed-only progress feature instead:

```text
coverage_progress_feature_source =
  observed_roi_ratio_no_coverable_denominator/v1

observed_roi_ratio =
  count(observed_mask == 1 inside ROI)
  / count(high-resolution cells inside ROI)
```

`observed_roi_ratio` is a policy feature only. It is not the success metric, not the reward denominator, and not a replacement for `highres_observed_coverage_rate` in reward, done, or evaluation.

## Frontier Action Set

The action cell set is generated by the environment at every step:

```text
frontier_cells = extract_global_observed_safe_frontier_cells(state_t)
```

A frontier cell is valid only if:

```text
observed == true
obstacle == false
slope_blocked == false
traversability >= traversability_threshold
clearance >= min_clearance_m
near_unknown == true
potential_gain > 0
reachable_prefilter == true
```

`near_unknown` means the cell is an observed-safe landing cell near currently unobserved high-resolution cells, typically within sensor range or near an observed-unobserved boundary.

Cells that fail `reachable_prefilter` are excluded before PPO sees the action set. Cells that pass it are only quick-screen reachable; they must still pass planner validation before execution.

The extractor should prefer high recall before hard filtering. If the action set is too large, top-M pruning only controls the maximum number of candidates exposed to the network; it is not a replacement for PPO policy selection.

The v1 top-M policy is:

```text
top_m_selection_policy =
  score_first_top_m/v1
```

The selection rule is:

```text
candidate_count_before_top_m = N

if N <= frontier_top_m:
  keep all valid candidates
  pad frontier_features to frontier_top_m
  set candidate_valid_mask false for padding rows

if N > frontier_top_m:
  sort candidates by candidate_priority descending
  keep the first frontier_top_m candidates
  set candidate_valid_mask true for kept rows
  prune the rest
```

V1 does not use mandatory region quota, direction quota, FOV-overlap suppression, or extra duplicate-suppression rules in top-M. Repetition is primarily controlled by the segment candidate generator, including `max_candidates_per_segment`.

The scale-specific top-M defaults are:

```text
Smoke v1:
  frontier_top_m = 512

Standard v1:
  frontier_top_m = 1024

Kilometer v1:
  frontier_top_m = 2048
  frontier_top_m_max = 4096 after overflow audit
```

The following diagnostics must be recorded:

```text
candidate_count_before_top_m
candidate_count_after_top_m
candidate_overflow_count
kept_min_priority
pruned_max_priority
selected_candidate_original_rank
```

If high-priority candidates are frequently pruned, increase `frontier_top_m` rather than introducing hard regional quotas.

### Frontier Segment Candidate Policy

The v1 frontier candidate generator uses a binary segment policy:

```text
frontier_segment_candidate_policy =
  binary_regular_normal_or_irregular_gain_sampling/v1
```

The extractor first groups observed-safe frontier cells into frontier segments. Each segment is classified as either regular or irregular:

```text
regular segment:
  normal_confidence >= normal_confidence_threshold

irregular segment:
  normal_confidence < normal_confidence_threshold
```

For v1, `normal_confidence` should be simple and auditable. It is primarily based on unknown-side consistency: a segment is regular when its local boundary has a stable observed side and a stable unknown side. Optional supporting diagnostics may include segment chord ratio, curvature variance, and normal variance, but v1 does not introduce a separate mild-irregular mode. The default threshold is:

```text
normal_confidence_threshold = 0.6
```

Regular segment candidates:

```text
mode = regular_normal_standoff
anchor_count = 1 to 3 based on segment length
look_at_point = segment midpoint or arc-length anchors
recommended_theta = outward normal from observed side to unknown side
candidate_position = look_at_point - normal_vector * standoff_distance_m
standoff_distance_m = 5.0
```

The candidate position must be corrected onto the observed-safe side. If the ideal standoff cell is not observed-safe, does not have enough clearance, or fails reachability prefilter, the generator searches nearby observed-safe cells within a small radius. If no valid cell is found, that anchor produces no candidate.

Irregular segment candidates:

```text
mode = irregular_local_gain_sampling
sample observed-safe cells near the segment
estimate one or more recommended theta values from local unknown distribution
score samples with 20m / 90 degree FOV / 2D LOS visible unknown gain
keep the best 1 to 3 valid candidates for that segment
```

Irregular segments are not discarded merely because their geometry is noisy. They are filtered by the same safety, reachability, and potential-gain checks as regular candidates. This avoids two failure modes:

```text
do not delete useful irregular frontiers only because the boundary is not smooth
do not force a single unreliable normal direction onto an irregular frontier
```

The number of candidates per segment is capped:

```text
max_candidates_per_segment = 3

segment_length_m <= 10:
  target anchor_count = 1

10 < segment_length_m <= 30:
  target anchor_count = 2

segment_length_m > 30:
  target anchor_count = 3
```

All generated candidates must satisfy:

```text
observed == true
obstacle == false
slope_blocked == false
traversability >= traversability_threshold
clearance >= min_clearance_m
reachable_prefilter == true
potential_gain > 0
```

`recommended_theta` is a feature and potential-gain estimate direction, not a hard action. PPO still samples a continuous `target_theta`, and the theta log probability is computed from that continuous policy distribution.

### Potential Gain And Value Gain

The v1 gain source is:

```text
potential_gain_source =
  endpoint_fov_los_visible_unknown_gain/v1
```

For each generated candidate, estimate gain from the candidate endpoint and its `recommended_theta`:

```text
candidate_pose_for_gain =
  (candidate_x, candidate_y, recommended_theta)
```

The candidate's visible unknown set is:

```text
visible_unknown_cells =
  unique cells swept by grid_ray_casting_fov/v1
  from candidate endpoint
  inside 20m range
  inside 90 degree FOV centered at recommended_theta
  stopping at current observed blockers
  AND observed_mask == 0
```

Line-of-sight blockers for this estimate may use only current observed evidence:

```text
current_observed_blocker =
  observed_mask == 1
  AND (observed_obstacle == 1 OR observed_slope_blocked == 1)
```

The estimator must not use hidden high-resolution truth to decide whether an unknown cell is free, blocked, high-value, or coverable. It also must not use the dense `coverable_mask` to filter `visible_unknown_cells`.

Candidate potential-gain estimation should use the same ray-casting FOV implementation as execution observation, but with current observed blockers only. This keeps action ranking computationally aligned with execution while avoiding hidden-truth leakage.

The geometric potential gain is:

```text
potential_gain_cells =
  count(visible_unknown_cells)
```

The low-resolution value gain is:

```text
value_gain =
  sum(value_prior_lowres[tile(cell)] for cell in visible_unknown_cells)
```

Candidate retention uses only the geometric gain plus safety and reachability:

```text
retain candidate if:
  potential_gain_cells > 0
  AND observed-safe
  AND reachable_prefilter == true
```

The v1 normalized feature values are computed within the current step's candidate set:

```text
epsilon = 1.0e-6

visible_unknown_count_norm =
  potential_gain_cells / max(1, max_candidate_potential_gain_cells)

value_gain_norm =
  value_gain / max(epsilon, max_candidate_value_gain)

potential_coverage_gain_norm =
  potential_gain_cells / max(1, visible_footprint_cell_count)
```

`visible_footprint_cell_count` is the number of cells visible from the candidate endpoint under the same 20m / 90 degree FOV / 2D LOS estimate before filtering by `observed_mask == 0`.

Top-M pruning uses a score-first priority:

```text
candidate_priority_source =
  score_first_gain_value_cost_priority/v1

candidate_priority =
  0.5 * potential_coverage_gain_norm
+ 0.3 * value_gain_norm
- 0.1 * distance_from_robot_norm
- 0.1 * reachable_prefilter_cost_norm
```

This priority score is not reward and is not the success metric. It is only an action-set pruning heuristic.

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
theta_mu_sin_raw
theta_mu_cos_raw
theta_kappa_raw
```

These are converted to:

```text
theta_kappa_parameterization =
  clamped_softplus_kappa/v1

theta_mu_norm =
  sqrt(theta_mu_sin_raw^2 + theta_mu_cos_raw^2) + epsilon

theta_mu_rad =
  atan2(theta_mu_sin_raw / theta_mu_norm, theta_mu_cos_raw / theta_mu_norm)

theta_kappa =
  clamp(
    softplus(theta_kappa_raw) + 1e-3,
    min = 1e-3,
    max = 20.0
  )
```

Rollout and update must preserve separate theta and frontier log probabilities for auditability.

## Planner Validation And Execution

The v1 planner validation source is:

```text
planner_validation_source =
  observed_safe_grid_astar_with_endpoint_theta_check/v1

planner_search_state =
  (x, y)
```

The PPO policy outputs an exploration goal pose, not a final trajectory:

```text
exploration_goal_pose =
  target_cell_x
  target_cell_y
  target_theta
```

The actual path must be produced by the planner before execution:

```text
planned_path =
  current_cell -> target_cell
```

For v1, the planner should use the current project's stable 2D A*/grid-planner semantics over the currently observed safe map. It should not require Hybrid A* as the default execution planner. Existing Hybrid A* pose-path machinery may remain an optional diagnostic or later validation source, but v1 does not replace the default 2D A* route and does not claim Ackermann-feasible execution.

Because the robot can rotate in place, `target_theta` is not part of the path search state. It does not make the target position unreachable. It only affects the final endpoint observation, and it must pass a lightweight endpoint check:

```text
target_theta_check =
  finite numeric value
  normalized to [-pi, pi)
  endpoint cell has enough clearance for in-place rotation
```

Planner validation succeeds only if:

```text
selected frontier index is valid under candidate_valid_mask
target_cell is observed_safe_cell
target_cell passes reachable_prefilter
2D A*/grid planner finds a path through observed_safe_cell
planned_path does not cross obstacle, slope_blocked, unknown, or insufficient-clearance cells
target_theta_check passes
```

If validation fails before movement, the environment does not execute the path and does not update coverage for that action. It applies `invalid_action_penalty`, sets `done = false`, and returns control to PPO for the next decision unless the failure exposes a severe safety violation.

Execution after successful validation is:

```text
1. Follow planned_path.
2. Sample observation poses every path_observation_step_m along the path.
3. Use path_tangent_heading for each intermediate FOV observation.
4. At the endpoint, rotate in place to target_theta.
5. Perform one endpoint FOV observation using target_theta.
6. Update the high-resolution observed map with the union of all visible cells.
```

The action-level coverage gain is:

```text
newly_observed_cell_count =
  count(cells where observed_mask changed from 0 to 1 after all path and endpoint observations)

coverage_gain_cells =
  count(cells where observed_mask changed from 0 to 1 AND coverable_mask)

coverage_gain_per_meter =
  coverage_gain_cells / max(path_length_m, epsilon)
```

`newly_observed_cell_count` records all newly written observed-map evidence. `coverage_gain_cells` is the reward and success-progress quantity and always counts only newly observed cells inside `coverable_mask`. `path_length_m`, `coverage_gain_cells`, and `coverage_gain_per_meter` are diagnostic fields. They do not create a hard path budget in v1.

## Execution Flow

```text
1. Build observation from current map state and pose.
2. Extract global observed-safe frontier cells and apply reachability prefilter.
3. Build frontier_features and candidate_valid_mask.
4. Policy scores frontier actions.
5. Sample target_frontier_index.
6. Policy samples target_theta using the selected frontier cell.
7. Planner validates current_pose -> target_cell using 2D observed-safe A*/grid search.
8. Endpoint theta check validates target_theta for in-place rotation and observation.
9. If valid, execute path observations plus endpoint theta observation.
10. Compute reward and done.
11. Store the full transition contract for PPO update.
```

Planner failure should be classified by cause where possible:

```text
mask_invalid
cell_unsafe
cell_unreachable
theta_invalid
path_collision
safety_violation
planner_timeout
execution_safety_stop
```

The prefilter/planner contract is intentionally asymmetric:

```text
prefilter false:
  do not expose the candidate to PPO

prefilter true:
  expose the candidate if other candidate rules pass
  still run planner validation before execution
```

## Reward

The task success metric is coverage above 99 percent. Reward is training support, not a separate success definition.

The v1 reward is:

```text
reward =
  coverage_gain_reward
+ success_bonus_if_coverage_rate_gt_0_99
- invalid_action_penalty
- safety_violation_penalty
```

Where:

```text
coverage_gain_cells =
  count(newly_observed_cells AND coverable_mask)

coverable_cell_count =
  count(coverable_mask)

normalized_coverage_gain =
  coverage_gain_cells / coverable_cell_count

coverage_gain_reward =
  w_coverage * normalized_coverage_gain

w_coverage =
  100.0

success_bonus_if_coverage_rate_gt_0_99:
  100.0 when highres_observed_coverage_rate >= 0.99
  otherwise 0

invalid_action_penalty:
  2.0 for planner validation failure, theta check failure, or target-cell validation failure

safety_violation_penalty:
  20.0 for collision, hard obstacle entry, unsafe slope, clearance violation, or severe safety checker failure
```

```text
reward_scaling_source =
  normalized_coverable_coverage_gain/v1

reward_constants_v1 =
  w_coverage: 100.0
  success_bonus: 100.0
  invalid_action_penalty: 2.0
  safety_violation_penalty: 20.0
```

The success metric remains coverage-based, but success is evaluated under a fixed step budget. V1 intentionally does not add path-cost, repeat-coverage, value-weighted coverage, distance, or turning penalties to the reward.

## Episode Termination

V1 uses fixed decision-step budgets to create exploration efficiency pressure without adding path-cost or repeat-coverage terms to the reward:

```text
efficiency_pressure_source =
  fixed_step_budget_terminal_success/v1

max_steps_by_scale =
  Smoke v1: 64
  Standard v1: 128
  Kilometer v1: 512
```

Each step is one frontier decision followed by planner validation, path execution if valid, path observations, endpoint theta observation, and map update. Path length remains diagnostic; the hard efficiency budget is the number of high-level frontier decisions.

The episode budget check order is:

```text
episode_budget_check_order =
  execute_action_then_check_success_then_budget/v1

1. Policy selects an action.
2. Environment validates and executes the action if valid.
3. Observed map and coverage are updated.
4. step_count += 1.
5. If coverage_rate >= 0.99, emit success_done.
6. Else if step_count >= max_steps, emit failure_done.
```

An episode ends under any of these conditions:

```text
success_done:
  highres_observed_coverage_rate >= 0.99 after action execution
  AND step_count <= max_steps

failure_done:
  highres_observed_coverage_rate < 0.99 after action execution
  AND step_count >= max_steps

stagnation_done:
  consecutive_no_gain_steps >= scale_N

no_candidate_done:
  valid_candidate_count == 0 before policy sampling

safety_done:
  severe safety violation occurred
```

Planner failure may be treated as invalid action and continue unless it indicates an unrecoverable safety condition.

There is no `path_length_used >= path_budget` termination in v1. Path length is logged for diagnostics only.

V1 does not add a separate failure penalty:

```text
failure_penalty_policy =
  no_extra_failure_penalty/v1

failure_done:
  reward does not include separate failure_penalty
  terminal = true
  bootstrap = 0

failure pressure comes from:
  no success_bonus
  finite step budget
  no future coverage_gain_reward after termination
```

Stagnation termination is based on consecutive zero-coverage-gain actions:

```text
stagnation_policy =
  consecutive_zero_coverage_gain/v1

stagnation_no_gain_steps_by_scale =
  Smoke v1: 8
  Standard v1: 16
  Kilometer v1: 32

no_gain_step =
  coverage_gain_cells == 0

stagnation_done =
  consecutive_no_gain_steps >= scale_N
```

If `coverage_gain_cells > 0`, the consecutive no-gain counter resets. V1 does not use a minimum positive gain threshold.

## PPO Transition Contract

Every trainable transition must store enough information to recompute PPO log probabilities without re-extracting frontier actions.

The v1 storage mode is:

```text
rollout_transition_storage =
  rollout_time_snapshot_storage/v1
```

The PPO data lifecycle is:

```text
1. Collect T rollout steps with the current policy.
2. For every step, save the observation tensors, frontier action-set snapshot,
   selected action, old log probabilities, old value, reward, and done.
3. After rollout collection, compute advantage and return.
4. Run PPO update epochs using only the saved rollout snapshots.
5. Clear the rollout buffer after the update.
6. Persist checkpoints, metrics, manifests, and small audit samples only.
```

The rollout buffer is not a permanent history memory for the policy. It exists to make PPO updates mathematically consistent with the action set that was available when each action was sampled.

Required fields:

```text
observation tensors
frontier_cells
frontier_features
candidate_valid_mask
top_m_selection_summary
selected_frontier_index
selected_cell_xy
selected_theta
planned_path_cells
path_length_m
path_observation_step_m
coverage_gain_cells
coverage_gain_per_meter
observation_sample_count
ray_count_per_sample
ray_cell_visit_count
newly_observed_cell_count
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
execution_observation_summary
```

Important rule:

```text
PPO update must use the saved frontier list, features, and candidate_valid_mask from rollout.
It must not re-run frontier extraction and silently replace the action space.
```

## PPO Training And Update Rules

The v1 PPO trainer uses on-policy vectorized rollout collection:

```text
rollout_collection_mode =
  on_policy_vectorized_env/v1

num_envs = 8
rollout_steps_per_env = 128
rollout_batch_size = 1024
after_update_clear_rollout_buffer = true
```

Each PPO update collects 128 steps from each of 8 parallel environments, then updates on the resulting 1024 trainable transitions. After the update, the rollout buffer is cleared. Old rollout batches must not be reused across later policy versions.

The transition storage contract is:

```text
rollout_transition_storage =
  rollout_time_snapshot_storage/v1

must_store_candidate_snapshot = true
must_not_rerun_frontier_extraction_during_update = true
store_full_highres_truth_in_transition = false
```

A stored transition must include the observation snapshot, action-set snapshot, selected action, old log probabilities, old value, reward, done state, diagnostics, and version metadata. In structured form:

```text
transition = {
  observation_snapshot,
  action_set_snapshot,
  selected_action,
  old_policy_outputs,
  reward_and_done,
  execution_diagnostics,
  version_metadata
}
```

Empty candidate sets are terminal but non-trainable:

```text
empty_candidate_handling =
  terminal_non_trainable/v1

if valid_candidate_count == 0:
  policy_forward = skipped
  trainable_transition = false
  done = true
  done_reason = no_candidate_done
  diagnostics_only = true
```

Invalid sampled actions are trainable penalized transitions:

```text
invalid_sampled_action_handling =
  trainable_penalized_transition/v1

if policy sampled action and validation failed:
  trainable_transition = true
  reward includes invalid_action_penalty
  done = false
  done_reason = none
  severe safety violation may set done = true
```

Advantage and return are computed with GAE:

```text
advantage_method =
  gae_lambda/v1

gamma = 0.995
gae_lambda = 0.95
normalize_advantage = true
```

The recurrence is:

```text
delta_t =
  reward_t + gamma * value_{t+1} * not_done_t - value_t

advantage_t =
  delta_t + gamma * gae_lambda * not_done_t * advantage_{t+1}

return_t =
  advantage_t + value_t
```

Fixed step budget failure is a terminal task outcome, not an artificial truncation:

```text
terminal_done_reasons = {
  success_done,
  failure_done,
  no_candidate_done,
  safety_done
}

truncated_done_reasons = {}

bootstrap_rule:
  terminal_done -> next_value = 0
  non_done -> next_value = value(next_observation)
```

The PPO policy loss uses the joint action log probability:

```text
policy_loss_type =
  clipped_surrogate_joint_action/v1

old_log_prob_total =
  old_log_prob_frontier + old_log_prob_theta

new_log_prob_total =
  new_log_prob_frontier + new_log_prob_theta

ppo_ratio =
  exp(new_log_prob_total - old_log_prob_total)

clip_eps = 0.2

policy_loss =
  -mean(
    min(
      ppo_ratio * normalized_advantage,
      clip(ppo_ratio, 1 - clip_eps, 1 + clip_eps) * normalized_advantage
    )
  )

candidate_snapshot_consistency_required = true
```

The frontier index distribution is a strictly masked categorical distribution:

```text
frontier_distribution_type =
  masked_categorical/v1

invalid_logit_value = -1e9

masked_frontier_logits =
  frontier_logits.masked_fill(~candidate_valid_mask, invalid_logit_value)

frontier_dist =
  Categorical(logits = masked_frontier_logits)

new_log_prob_frontier =
  frontier_dist.log_prob(selected_frontier_index)

frontier_entropy =
  frontier_dist.entropy()
```

Frontier mask requirements:

```text
candidate_valid_mask.any() must be true
selected index must be valid
padding candidates never affect softmax/logprob/entropy/pooling
```

The continuous theta distribution is:

```text
theta_distribution_type =
  von_mises_continuous/v1

theta_mu_parameterization =
  normalized_sin_cos_to_atan2/v1

theta_kappa =
  clamp(
    softplus(theta_kappa_raw) + 1e-3,
    min = 1e-3,
    max = 20.0
  )

selected_theta_params =
  gather candidate theta params by selected_frontier_index

new_log_prob_theta =
  VonMises(selected_theta_mu, selected_theta_kappa)
    .log_prob(saved_selected_theta)

saved_selected_theta_normalization =
  [-pi, pi)

theta_entropy_enabled_initially = false
theta_exploration_initialization = low_kappa_init
```

The value loss is clipped:

```text
value_target =
  return

value_loss_type =
  clipped_value_loss/v1

value_clip_eps = 0.2
value_loss_coef = 0.5

value_pred_clipped =
  old_value + clamp(new_value - old_value, -0.2, 0.2)

value_loss =
  0.5 * mean(
    max(
      (new_value - return)^2,
      (value_pred_clipped - return)^2
    )
  )

value_mask_rule:
  candidate_valid_mask must be used for candidate pooling
  padding candidates must not affect V(s)
```

Entropy regularization is frontier-only in v1:

```text
entropy_regularization =
  frontier_only_entropy/v1

frontier_entropy_coef = 0.01
theta_entropy_enabled_initially = false
entropy_schedule = constant/v1

entropy_bonus =
  frontier_entropy

total_loss =
  policy_loss
  + value_loss_coef * value_loss
  - frontier_entropy_coef * frontier_entropy
```

Required entropy diagnostics:

```text
frontier_entropy_mean
frontier_entropy_min
frontier_entropy_by_valid_candidate_count
theta_kappa_mean
theta_kappa_max
```

Invalid action training policy:

```text
invalid_action_training_policy =
  validation_failure_trainable/v1

mask_violation:
  trainable_transition = false
  reason = implementation_error
  action = dropped
  raise_critical_diagnostic = true

planner_validation_failed:
  trainable_transition = true
  reward += invalid_action_penalty
  done = false
  invalid_action_flag = true

theta_check_failed:
  trainable_transition = true
  reward += invalid_action_penalty
  done = false
  theta_invalid_flag = true

severe_safety_violation:
  trainable_transition = true
  reward += safety_violation_penalty
  done = true
  done_reason = safety_done
```

The update schedule is:

```text
ppo_update_schedule =
  fixed_epoch_minibatch/v1

ppo_epochs = 4
minibatch_size = 256
shuffle_minibatches_each_epoch = true

optimizer = AdamW
learning_rate = 3e-4
adam_eps = 1e-5
weight_decay = 1e-4

max_grad_norm = 0.5
```

KL monitoring is required:

```text
kl_monitoring:
  approx_kl_source = old_log_prob_total_minus_new_log_prob_total
  target_kl = 0.03
  early_stop_update_epoch_if_kl_exceeds = true
```

Checkpoint and evaluation policy:

```text
checkpoint_policy =
  latest_periodic_best/v1

latest:
  save_every_update = true
  keep_count = 1

periodic:
  save_every_n_updates = 50
  keep_count = 5

best:
  primary = best_success_rate_under_fixed_step_budget
  secondary = best_mean_final_coverage

eval:
  eval_every_n_updates = 10
  eval_episodes = scale-specific validation_eval_episodes
  eval_policy_mode = deterministic_argmax_frontier_mean_theta/v1
```

Deterministic evaluation uses the argmax valid frontier action and the selected frontier's `theta_mu_rad`. Training rollout remains stochastic.

Checkpoint files must include:

```text
model_state_dict
optimizer_state_dict
update_step
observation_schema_version
action_space_version
network_architecture_version
reward_version
normalization_stats
top_m_config
scale_profile
git_commit
training_config
eval_metrics
```

## Network Shape

The v1 network architecture is:

```text
network_architecture_version =
  cross_attention_frontier_policy/v1

network_memory_mode =
  stateless_observation_only/v1
```

The network does not use RNN, LSTM, recurrent hidden state, previous action input, visited-path history, or candidate failure history. History is represented only through the current observed map, coverage summary, frontier set, and pose/progress features.

The architecture diagram is stored at:

```text
network_diagram_path =
  docs/superpowers/diagrams/ppo-cross-attention-frontier-policy.drawio
```

The v1 token budget is fixed so kilometer-scale maps remain computationally bounded:

```text
frontier_top_m:
  follows scale profile defaults

global token count:
  Kg <= 1024

local token count:
  Kl <= 1024

context token count:
  K = Kg + Kl + 1

token dimension:
  D = 128 by default
  D = 256 maximum for larger ablations

cross_attention_layers:
  2

attention_heads:
  4 by default
  8 maximum when D = 256
```

The CNN encoders must use stride, patching, or adaptive pooling to satisfy `Kg` and `Kl`. The policy must not create one token per high-resolution cell in kilometer-scale scenes.

The forward structure is:

```text
global encoder:
  CNN(global_lowres_prior_state + global_highres_coverage_summary)
  + 2D positional encoding
  -> global_map_tokens [B, Kg, D]

local encoder:
  CNN(local_highres_observed_crop)
  + 2D positional encoding
  -> local_map_tokens [B, Kl, D]

pose encoder:
  MLP(pose_features)
  -> pose_token [B, 1, D]

frontier encoder:
  shared MLP(frontier_features)
  + frontier positional encoding from x_norm, y_norm, bearing_sin, bearing_cos
  -> frontier_tokens [B, M, D]

context_tokens:
  concat(global_map_tokens, local_map_tokens, pose_token)
  -> [B, K, D]

cross-attention:
  Q = frontier_tokens
  K,V = context_tokens
  CrossAttentionBlock x 2
  -> refined_frontier_tokens [B, M, D]

output input:
  concat(original_frontier_tokens, refined_frontier_tokens, raw frontier_features)
  -> shared_output_mlp
  -> action_hidden [B, M, H]

frontier logit head:
  Linear(H, 1)
  -> frontier_logits [B, M]

theta parameter head:
  Linear(H, 3)
  -> theta_mu_sin_raw [B, M]
  -> theta_mu_cos_raw [B, M]
  -> theta_kappa_raw [B, M]

value head:
  masked_mean(refined_frontier_tokens)
  + masked_max(refined_frontier_tokens)
  + pooled context_tokens
  -> independent value MLP
  -> value [B]
```

The cross-attention block must use residual connections and normalization:

```text
x = frontier_tokens
x = LayerNorm(x + CrossAttention(Q=x, K=context_tokens, V=context_tokens))
x = LayerNorm(x + FeedForward(x))
```

The policy distribution is:

```text
frontier_logits[candidate_valid_mask == false] = -inf
selected_frontier_index ~ Categorical(masked frontier_logits)

theta_mu_norm =
  sqrt(theta_mu_sin_raw^2 + theta_mu_cos_raw^2) + epsilon

theta_mu_sin =
  theta_mu_sin_raw / theta_mu_norm

theta_mu_cos =
  theta_mu_cos_raw / theta_mu_norm

theta_mu_rad =
  atan2(theta_mu_sin, theta_mu_cos)

theta_kappa =
  clamp(
    softplus(theta_kappa_raw) + 1e-3,
    min = 1e-3,
    max = 20.0
  )

selected_theta_distribution =
  VonMises(
    theta_mu_rad[selected_frontier_index],
    theta_kappa[selected_frontier_index]
  )
```

All candidate-dependent operations must respect `candidate_valid_mask`, including masked softmax, entropy, log-probability recomputation, masked mean pooling, and masked max pooling. Padding candidates must not affect the action distribution or the value estimate.

The value head may share map, pose, frontier, and cross-attention encoders with the policy, but its final MLP layers must be separate from the action output MLP and heads.

Required stability settings:

```text
frontier/map positional encoding enabled
cross-attention residual + LayerNorm enabled
raw frontier_features preserved into output MLP
shared output MLP before action heads
theta_mu_sin/cos normalized to unit direction
theta_kappa uses clamped_softplus_kappa/v1
strict candidate_valid_mask handling
separate final value MLP
conservative initialization for frontier_logit_head
conservative low initial theta_kappa
```

The v1 network must not include:

```text
RNN / LSTM
previous-action history input
candidate self-attention over all M candidates
multiple alternative network branches
candidate-neighborhood token gather
```

Checkpoint metadata must record the architecture version, memory mode, token dimensions, attention layer count, head count, hidden dimensions, initialization settings, and observation schema.

## Leakage And Consistency Rules

The following are hard failures:

- Policy input uses complete future high-resolution truth.
- Policy input includes the dense `coverable_mask` or any equivalent hidden-truth spatial denominator channel.
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

Environment-only accounting may use high-resolution scenario truth for reward, done, and evaluation denominators, but those masks must not become spatial policy inputs or candidate-filtering shortcuts.

## Evaluation

Primary evaluation:

```text
primary_eval_metric =
  success_rate_under_fixed_step_budget/v1

success_threshold =
  0.99

success_rate_under_fixed_step_budget =
  count(episodes where coverage_rate >= 0.99 before or at max_steps)
  / total_eval_episodes

coverage_denominator_source
coverable_mask_exact
```

This is the primary metric because unlimited exploration would make final coverage alone too weak for comparing learned and rule-based exploration policies.

The v1 baseline set is:

```text
baseline_algorithms_v1 =
  random_valid_frontier
  nearest_frontier
  max_potential_gain_frontier
  gain_over_cost_frontier
  ppo_policy

non_learning_baselines =
  random_valid_frontier
  nearest_frontier
  max_potential_gain_frontier
  gain_over_cost_frontier

learning_method =
  ppo_policy
```

Baseline behavior definitions:

```text
baseline_cost_score_policy =
  gain_over_one_plus_cost/v1

random_valid_frontier:
  uniformly select one valid candidate

nearest_frontier:
  select min(distance_from_robot_norm)

max_potential_gain_frontier:
  select max(potential_coverage_gain_norm)

gain_over_cost_frontier:
  score =
    potential_coverage_gain_norm
    / (1.0 + reachable_prefilter_cost_norm)
  select max(score)

ppo_policy:
  use the trained PPO policy under deterministic evaluation mode
```

All methods use the same candidate set:

```text
baseline_candidate_set_contract =
  shared_observed_safe_reachable_frontier_candidates/v1

all_methods_use_same_candidate_set = true
all_methods_use_same_planner_validation = true
all_methods_use_same_sensor_model = true
all_methods_use_same_step_budget = true
```

The shared candidate set includes the same frontier extraction, regular/severe irregular segment handling, observed-safe filtering, reachability prefilter, top-M cap when applicable, candidate features, and planner validation. Algorithms differ only in how they choose among valid candidates.

Baseline information access is restricted to deployment-available information:

```text
baseline_information_contract =
  deployment_available_information_only/v1

forbidden_for_baselines:
  hidden_highres_truth
  dense_coverable_mask_as_spatial_input
  future_observation_result
  unknown_cell_true_obstacle_or_height
  unknown_cell_true_traversability

allowed_for_baselines:
  current_observed_highres_map
  lowres_prior_map
  global_coverage_summary
  frontier_cells
  frontier_features
  candidate_valid_mask
  current_pose
  sensor_model
  planner_validation_result
```

Baseline comparison must use the same environment and budget:

```text
baseline_comparison_contract =
  same_env_same_budget/v1

all algorithms must use:
  same map set
  same start pose set
  same sensor model
  same coverable_mask denominator
  same planner validation
  same max_steps
  same success threshold = 0.99
```

Scenario splitting is fixed-seed and disjoint:

```text
scenario_split_policy =
  fixed_seed_disjoint_split/v1

splits:
  train_scenarios
  validation_scenarios
  test_scenarios
  unseen_test_scenarios

split_ratio_default:
  train = 0.70
  validation = 0.15
  test = 0.15

rules:
  validation and test do not update PPO
  test is not used for checkpoint selection
  unseen_test is reported separately as generalization
  scenario_seed, start_pose_seed, terrain_seed must be logged
```

Evaluation episode counts are scale-specific:

```text
eval_episode_count_policy =
  scale_specific_eval_counts/v1

validation_eval_episodes:
  Smoke v1: 8
  Standard v1: 16
  Kilometer v1: 8

final_test_episodes:
  Smoke v1: 16
  Standard v1: 64
  Kilometer v1: 32

unseen_test_episodes:
  Smoke v1: 16
  Standard v1: 64
  Kilometer v1: 32
```

Main result tables must report:

```text
main_result_table_metrics =
  method
  scale_profile
  success_rate_under_fixed_step_budget
  mean_final_coverage
  steps_to_99_success_only
  path_length_to_99_success_only
  coverage_auc_over_steps
  coverage_per_meter
  invalid_action_count_mean
  planner_failure_count_mean
  safety_violation_count

confidence_reporting =
  bootstrap_95ci_by_episode/v1
```

PPO evaluation is deterministic:

```text
ppo_eval_policy_mode =
  deterministic_argmax_frontier_mean_theta/v1

frontier_eval_action =
  argmax(masked_frontier_logits)

theta_eval_action =
  theta_mu_rad[selected_frontier_index]

evaluation_seed_policy =
  fixed_eval_seed_set/v1

tie_break_policy =
  stable_candidate_order/v1

argmax_tie_break =
  lowest_candidate_index
```

Training rollout remains stochastic. Evaluation uses fixed scenario seeds, start-pose seeds, terrain seeds, and deterministic tie-breaking so that repeated evaluations of the same checkpoint are comparable.

Additional diagnostic metrics include:

```text
observation_sample_count
ray_cell_visit_count
newly_observed_cells_per_action
invalid_action_count_by_reason
planner_failure_count_by_cause
safety_violation_count_by_cause
stagnation_termination_count
frontier_action_entropy
theta_distribution_diagnostics
frontier_extractor_recall_diagnostics
top_m_overflow_diagnostics
selected_candidate_original_rank
```

Evaluation must verify that all inference observations use deployment-available information only.

## Progress Reporting

Progress reporting is a monitoring and experiment-management feature. It is not a reward term, not a policy input, and not part of the PPO action distribution.

The v1 progress reporting policy is:

```text
progress_reporting_policy =
  console_progress_and_metrics_jsonl/v1
```

The implementation should show console progress bars and write the same state into append-only metrics JSONL records.

Episode progress:

```text
episode_progress:
  coverage_progress =
    coverage_rate / 0.99

  step_budget_progress =
    step_count / max_steps

  stagnation_progress =
    consecutive_no_gain_steps / stagnation_N
```

Training progress:

```text
training_progress:
  update_id / total_updates
  rollout_steps_collected / rollout_batch_size
  ppo_epoch / ppo_epochs
  minibatch_id / minibatch_count
  latest_eval_success_rate_under_fixed_step_budget
  best_eval_success_rate_under_fixed_step_budget
```

Evaluation and baseline progress:

```text
evaluation_progress:
  method_id / method_count
  episode_id / eval_episode_count
  scale_profile
  current_success_count
  current_mean_final_coverage
```

Metrics JSONL records should include:

```text
metrics_jsonl_fields:
  progress_type
  timestamp
  scale_profile
  method
  update_id
  episode_id
  step_count
  max_steps
  coverage_rate
  coverage_progress
  step_budget_progress
  stagnation_progress
  rollout_steps_collected
  rollout_batch_size
  ppo_epoch
  minibatch_id
  minibatch_count
  success_rate_so_far
  mean_final_coverage_so_far
  best_eval_success_rate_under_fixed_step_budget
```

Progress records must use deployment-available and already-computed diagnostics only. They must not expose hidden high-resolution truth, dense `coverable_mask` spatial structure, or future observation results to the policy.

## Implementation File Structure And Isolation

The v1 implementation must be separated from previous Stage26 / xunce runners and legacy experiment scripts. Existing code remains usable as dependencies, but the new PPO exploration system must have its own package, configs, tests, runners, and output root.

```text
implementation_file_structure =
  isolated_core_package_with_thin_stage_runners/v1

legacy_code_isolation_policy =
  do_not_modify_legacy_runners_or_default_policy/v1

integration_adapter_policy =
  reuse_existing_capabilities_only_through_adapters/v1
```

Recommended repository layout:

```text
lunar-path-planning/
  src/
    lunar_exploration_ppo/
      __init__.py

      env/
        __init__.py
        map_state.py
        sensor_model.py
        coverage.py
        frontier.py
        reachability.py
        action_execution.py
        env.py

      policy/
        __init__.py
        observation.py
        network.py
        distributions.py
        action_sampling.py

      ppo/
        __init__.py
        rollout_buffer.py
        advantage.py
        losses.py
        trainer.py
        checkpoint.py

      eval/
        __init__.py
        baselines.py
        evaluator.py
        metrics.py
        reports.py

      integrations/
        __init__.py
        path_planner_adapter.py
        artifact_io_adapter.py
        config_loader.py

      configs/
        __init__.py
        schema.py

      utils/
        __init__.py
        geometry.py
        masks.py
        rng.py
        artifact_io.py

  scripts/
    run_ppo_stage1_smoke_env.py
    run_ppo_stage2_observation_candidates.py
    run_ppo_stage3_network_forward.py
    run_ppo_stage4_update_smoke.py
    run_ppo_stage5_baselines.py
    run_ppo_stage6_standard_train_eval.py
    run_ppo_stage7_kilometer_stress.py
    run_ppo_stage8_final_package.py

  configs/
    ppo_highres_frontier_smoke_v1.json
    ppo_highres_frontier_standard_v1.json
    ppo_highres_frontier_kilometer_v1.json

  tests/
    ppo_highres_frontier/
      test_stage1_smoke_env.py
      test_stage2_observation_candidates.py
      test_stage3_network_forward.py
      test_stage4_ppo_update.py
      test_stage5_baselines.py
      test_stage6_standard_config.py
      test_stage7_kilometer_config.py
      test_stage8_report_package.py
```

Boundary rules:

```text
new PPO core package:
  src/lunar_exploration_ppo/

thin stage runners:
  scripts/run_ppo_stage*.py

new config namespace:
  configs/ppo_highres_frontier_*.json

new test namespace:
  tests/ppo_highres_frontier/

default output root:
  D:/xunce/out/ppo_frontier/
```

The `scripts/run_ppo_stage*.py` files should only load config, call the package API, and write artifacts. Environment semantics, candidate generation, network forward, PPO update, baseline evaluation, and report generation belong in the package, not in the runner scripts.

The existing `path-planner/src/path_planner/` package remains an external planning dependency. The new PPO package must not move planner code into itself and must not modify the default planner policy. Planner calls should go through `lunar_exploration_ppo.integrations.path_planner_adapter`.

Legacy xunce / Stage26 runners, configs, and tests are not part of this new PPO v1 implementation. They may be referenced for artifact IO conventions or planner behavior, but the new package must not depend on legacy runner internals.

## Implementation Roadmap

The v1 implementation roadmap is:

```text
implementation_roadmap_version =
  eight_stage_smoke_to_final_package_v1

stages:
  1. Smoke v1 environment closed loop
  2. Observation + frontier candidate generator
  3. PPO network forward + action sampling
  4. Rollout buffer + PPO update
  5. Baseline evaluator
  6. Standard v1 training/eval
  7. Kilometer v1 stress test
  8. Final report package
```

### Stage 1: Smoke v1 environment closed loop

Goal: run the minimum environment loop end to end on Smoke v1. This stage validates environment semantics, not PPO performance.

Scope:

```text
map reset
initial observed area
observation build
frontier candidate generation
candidate_valid_mask
action input
planner validation
path execution or invalid action handling
path + endpoint observation update
coverage_gain_cells
reward calculation
done reason
progress reporting
metrics_jsonl
```

Out of scope:

```text
PPO training
network optimization
baseline comparison
Standard / Kilometer scale
performance claim
```

Acceptance:

```text
stage1_acceptance_policy =
  smoke_env_closed_loop_acceptance/v1

1. reset produces the Smoke v1 map dimensions:
   ROI = 64m x 64m
   highres = 128 x 128 @ 0.5m/cell
   lowres = 32 x 32 @ 2m/cell
   local crop = 64 x 64
   frontier_top_m = 512
   max_steps = 64

2. initial observed area is created around the robot start pose.

3. build observation completes without error.

4. frontier candidate generation completes without error.

5. candidate_valid_mask shape = [512].

6. if no valid candidate exists:
   no_candidate_done is emitted
   policy forward is not called
   no trainable PPO action transition with fake logprob is stored

7. if at least one valid candidate exists:
   a rule/test action can select a valid candidate.

8. planner validation returns valid or invalid without crashing.

9. valid action executes path observation plus endpoint theta observation.

10. observed map updates after valid observation.

11. coverage_gain_cells >= 0.

12. reward matches the v1 formula:
    coverage_gain_reward + success_bonus - invalid_action_penalty - safety_violation_penalty.

13. done_reason belongs to the allowed set:
    success_done, failure_done, stagnation_done, no_candidate_done, safety_done, none.

14. progress metrics JSONL is written.

15. one complete Smoke episode can run from reset to done.

16. ten consecutive Smoke episodes run without NaN / inf in observation, reward, coverage, done, progress, or trace fields.
```

Recommended artifacts:

```text
smoke_env_closed_loop_report.md
smoke_episode_trace.jsonl
smoke_progress_metrics.jsonl
smoke_config.json
```

### Stage 2: Observation + frontier candidate generator

Goal: stabilize the full policy observation schema and sparse frontier action set before network training.

Scope:

```text
global_lowres_prior_state
global_highres_coverage_summary
local_highres_observed_crop
frontier_cells
frontier_features
candidate_valid_mask
top-M padding
top-M pruning
top-M overflow diagnostics
candidate priority
regular / severe irregular frontier handling
potential gain estimate
value gain estimate
```

Out of scope:

```text
PPO update
network training
baseline comparison
Kilometer stress test
```

Acceptance:

```text
stage2_acceptance_policy =
  observation_candidate_generator_acceptance/v1

1. global_lowres_prior_state shape matches the Smoke v1 schema.

2. global_lowres_prior_state channel order is fixed and documented.

3. global_highres_coverage_summary shape matches the Smoke v1 schema.

4. global_highres_coverage_summary is aggregated only from the current observed map, current frontier set, reachability prefilter result, sensor geometry, and deployment-available low-resolution prior. It does not use dense coverable_mask spatial structure.

5. local_highres_observed_crop shape matches the Smoke v1 schema.

6. local_highres_observed_crop channel order is fixed and documented.

7. frontier_cells shape is valid and every coordinate lies inside the high-resolution map bounds.

8. frontier_features shape = [M, F].

9. frontier_features field order is fixed and documented.

10. candidate_valid_mask shape = [M].

11. padding rows have candidate_valid_mask = false.

12. every valid row has finite feature values.

13. if N <= M:
    all valid candidates are kept
    remaining rows are padding.

14. if N > M:
    score-first top-M pruning is applied.

15. top-M pruning does not use hidden high-resolution truth or dense coverable_mask spatial leakage.

16. potential_gain uses the current observed map, current observed blockers, sensor model, and ray-casting estimate only. It does not use future truth.

17. empty candidate sets are represented cleanly:
    frontier_cells padded
    frontier_features padded
    candidate_valid_mask all false
    no NaN / inf feature values

18. sample observation batch can be saved and reloaded with unchanged shape, dtype, channel order, feature order, and candidate_valid_mask.
```

Recommended artifacts:

```text
observation_schema_report.md
candidate_generation_audit.json
top_m_audit.json
sample_observation_batch.pt or sample_observation_batch.npz
sample_candidate_table.csv
```

### Stage 3: PPO network forward + action sampling

Goal: verify that the network consumes batched observations and returns legal actions, log probabilities, and state values. This stage does not update network weights.

Scope:

```text
global encoder
local encoder
pose encoder
frontier encoder
cross-attention blocks
shared output MLP
frontier logit head
theta parameter head
value head
candidate_valid_mask handling
masked categorical sampling
Von Mises theta sampling
selected_frontier_index
selected_theta
old_log_prob_frontier
old_log_prob_theta
old_log_prob_total
value
deterministic eval action
```

Out of scope:

```text
PPO optimizer update
rollout advantage calculation
baseline comparison
full training
```

Acceptance:

```text
stage3_acceptance_policy =
  network_forward_action_sampling_acceptance/v1

1. forward accepts batched Smoke observations.

2. frontier_logits shape = [B, M].

3. theta_mu_sin_raw, theta_mu_cos_raw, and theta_kappa_raw each have shape = [B, M].

4. value shape = [B].

5. candidates with candidate_valid_mask = false are never sampled.

6. all-false candidate_valid_mask is rejected before network forward.

7. selected_frontier_index satisfies candidate_valid_mask[selected_frontier_index] = true.

8. selected_theta is normalized to [-pi, pi).

9. theta_kappa is finite and clipped to [1e-3, 20.0].

10. old_log_prob_total = old_log_prob_frontier + old_log_prob_theta.

11. deterministic eval uses argmax masked frontier logits plus theta_mu of the selected candidate.

12. value pooling ignores padding candidates.

13. forward outputs contain no NaN / inf.

14. same saved observation and same weights recompute the same deterministic action.

15. same saved observation and same weights recompute the same log probabilities.

16. batch samples may have different valid candidate counts.
```

Recommended artifacts:

```text
network_forward_audit.md
network_shape_audit.json
sample_policy_outputs.pt or sample_policy_outputs.npz
logprob_recompute_audit.json
mask_handling_audit.json
```

### Stage 4: Rollout buffer + PPO update

Goal: run one complete PPO update chain on Smoke v1 and verify that the rollout snapshot, GAE, losses, optimizer step, checkpoint, and progress logs are correct.

Scope:

```text
vectorized rollout collection
rollout_time_snapshot_storage
observation snapshot
candidate snapshot
selected action
old logprobs
old value
reward / done
GAE advantage
return
joint logprob recomputation
PPO ratio
clipped policy loss
clipped value loss
frontier entropy bonus
gradient clipping
optimizer step
KL monitoring
latest checkpoint save
progress reporting
metrics JSONL
```

Out of scope:

```text
baseline comparison
Standard scale training
Kilometer stress test
performance claim
```

Acceptance:

```text
stage4_acceptance_policy =
  rollout_ppo_update_smoke_acceptance/v1

1. The runner collects rollout_batch_size = 1024 trainable transitions.

2. If an episode terminates early, terminal transitions are handled correctly and do not contaminate the PPO update.

3. Each trainable transition stores the rollout-time observation snapshot and candidate snapshot.

4. PPO update uses the saved candidate snapshot, not re-extracted frontier candidates.

5. old_log_prob_frontier, old_log_prob_theta, and old_log_prob_total exist and have correct shape.

6. new_log_prob_total can be recomputed on the same saved snapshot.

7. old_log_prob_total and new_log_prob_total have matching shape.

8. advantages and returns are finite.

9. normalized advantages have mean approximately 0 and standard deviation approximately 1.

10. PPO ratio = exp(new_log_prob_total - old_log_prob_total) is finite.

11. clipped policy loss is finite.

12. value loss is finite.

13. entropy bonus is finite.

14. grad_norm is finite and clipped by max_grad_norm.

15. optimizer step changes at least one trainable parameter.

16. approx_kl is recorded.

17. early stop triggers when approx_kl > target_kl.

18. rollout buffer is cleared after update.

19. latest checkpoint can be saved.

20. latest checkpoint can be loaded.

21. loaded checkpoint reproduces the same deterministic action on the same observation.

22. training progress JSONL is written.

23. Three consecutive Smoke PPO updates run without NaN, inf, or crash.
```

Recommended artifacts:

```text
stage4_smoke_update_report.md
rollout_snapshot_audit.json
gae_audit.json
ppo_loss_audit.json
logprob_recompute_audit.json
optimizer_step_audit.json
checkpoint_load_audit.json
training_progress.jsonl
checkpoint_latest.pt
```

### Stage 5: Baseline evaluator

Goal: run all v1 baselines and PPO evaluation under the same environment, candidate set, seed set, planner validation, sensor model, and step budget.

Scope:

```text
baseline_algorithms_v1
shared candidate set
same planner validation
same sensor model
same max_steps
same success threshold = 0.99
fixed eval seeds
deterministic PPO eval
stable tie-break
main result table
bootstrap 95% CI
baseline progress reporting
```

Out of scope:

```text
PPO training improvement
new reward tuning
Kilometer stress test
paper final claim
```

Acceptance:

```text
stage5_acceptance_policy =
  baseline_evaluator_acceptance/v1

1. random_valid_frontier can run a complete Smoke evaluation episode.

2. nearest_frontier can run a complete Smoke evaluation episode.

3. max_potential_gain_frontier can run a complete Smoke evaluation episode.

4. gain_over_cost_frontier can run a complete Smoke evaluation episode.

5. All baselines and PPO evaluation use the same environment, map seed, start pose seed, sensor model, frontier candidate generator, reachability prefilter, planner validation, and max_steps.

6. Baselines do not access hidden highres truth except through final metric computation.

7. Baselines do not access dense coverable_mask as a spatial decision input.

8. Every selected baseline action comes from the reachable observed-safe candidate set.

9. Empty candidate set behavior is consistent across methods and is recorded as no_candidate_done according to the environment contract.

10. deterministic baselines are reproducible under the same seed.

11. random_valid_frontier is reproducible under a fixed random seed.

12. PPO eval uses argmax frontier plus theta_mu.

13. Tie-break is deterministic.

14. All methods write the same metrics schema.

15. success_rate_under_fixed_step_budget is computed.

16. path_length_to_99_success_only is computed.

17. coverage_per_meter is computed.

18. invalid_action_count_mean and planner_failure_count_mean are computed.

19. bootstrap 95% CI is computed.

20. baseline_comparison_table.csv is written.

21. baseline_coverage_curves.csv is written.

22. baseline_eval_report.md is written.
```

Recommended artifacts:

```text
baseline_eval_report.md
baseline_comparison_table.csv
baseline_metrics.json
baseline_coverage_curves.csv
baseline_episode_traces.jsonl
baseline_fairness_audit.json
bootstrap_ci_audit.json
```

### Stage 6: Standard v1 training/eval

Goal: run the minimum complete Standard v1 training and evaluation workflow. This stage is a system-closure milestone, not a new algorithm-design stage.

Minimum configuration:

```text
stage6_acceptance_policy =
  standard_v1_minimal_training_eval_acceptance/v1

scale_profile:
  Standard v1

train_seed_count:
  1

train_seed:
  20260716

train_updates:
  100

eval_every_updates:
  10

max_steps:
  128

validation_eval_episodes:
  16

final_test_episodes:
  64

unseen_test_episodes:
  64

checkpoint_policy:
  inherit latest_periodic_best/v1
  save latest every update
  save periodic every 50 updates
  save best by validation success_rate_under_fixed_step_budget
  use validation mean_final_coverage only as a tie-break

eval_policy:
  deterministic PPO

baseline_set:
  random_valid_frontier
  nearest_frontier
  max_potential_gain_frontier
  gain_over_cost_frontier
```

Stage 6A is frozen as `single_seed_system_closure/v1`: it completes the
training, recovery, checkpoint, evaluation, fairness, audit, and reproduction
loop for seed `20260716`, but is not a cross-seed performance conclusion.
An additional 3--5-seed extension is optional: 只有用户明确要求才追加，且它不阻塞
Stage 6 Gate、Stage 7 或 Stage 8。

Acceptance:

```text
1. Standard training completes 100 PPO updates without NaN, inf, or crash.

2. latest checkpoint can resume training.

3. best checkpoint is selected by validation success_rate_under_fixed_step_budget, with validation mean_final_coverage used only as a deterministic tie-break.

4. validation evaluation uses deterministic PPO mode.

5. final test evaluation does not update PPO.

6. final test evaluation does not select or change the checkpoint.

7. baseline comparison uses same_env_same_budget.

8. PPO is compared against all four v1 non-learning baselines.

9. success_rate_under_fixed_step_budget, path_length_to_99_success_only, coverage_per_meter, invalid_action_count_mean, and planner_failure_count_mean are reported.

10. coverage curves are generated.

11. checkpoint manifest is generated.

12. Implementation acceptance does not require PPO to beat gain_over_cost_frontier; that remains a later paper-level performance target.
```

Recommended artifacts:

```text
standard_training_report.md
standard_training_metrics.jsonl
standard_eval_report.md
standard_baseline_comparison.csv
standard_coverage_curves.csv
standard_checkpoint_manifest.json
checkpoint_latest.pt
checkpoint_best_success_rate.pt
```

### Stage 7: Kilometer v1 stress test

Goal: verify that the Standard v1 policy and the sparse hierarchical observation/action pipeline can run on kilometer-level scenes without changing reward, network, planner, or action semantics.

Minimum configuration:

```text
stage7_acceptance_policy =
  kilometer_v1_eval_only_stress_acceptance/v1

scale_profile:
  Kilometer v1

mode:
  eval_only

policy_source:
  Stage 6A seed 20260716 frozen validation-best checkpoint

max_steps:
  512

primary_baseline:
  gain_over_cost_frontier

full_baseline_set:
  optional only if runtime is acceptable
```

Acceptance:

```text
1. Kilometer episode can reset and run to done.

2. policy observation remains hierarchical and sparse.

3. full highres map is not fed as a dense policy tensor.

4. PPO directly uses the Stage 6A frozen validation-best checkpoint without
   Kilometer-specific fine-tuning.

5. candidate generation completes within the implementation runtime threshold.

6. top-M overflow diagnostics are recorded.

7. PPO forward fits the available CPU/GPU memory budget.

8. runtime and memory profiles are recorded.

9. PPO is compared against gain_over_cost_frontier under same_env_same_budget.

10. coverage_rate and success_rate_under_fixed_step_budget are reported.

11. no hidden truth leakage enters policy or baseline decision inputs.

12. no NaN, inf, or crash occurs in observation, reward, policy outputs, or metrics.
```

Recommended artifacts:

```text
kilometer_stress_report.md
kilometer_runtime_profile.json
kilometer_memory_profile.json
kilometer_candidate_overflow_audit.json
kilometer_eval_metrics.jsonl
kilometer_progress_metrics.jsonl
kilometer_coverage_curves.csv
kilometer_baseline_summary.csv
```

### Stage 8: Final report package

Goal: package the design, configs, checkpoints, metrics, comparisons, and audits into one reproducible experiment report. This stage does not add new training or algorithm changes.

Scope:

```text
stage8_acceptance_policy =
  final_report_package_acceptance/v1

design document
experiment configs
checkpoint manifest
training curves
baseline comparison table
coverage curves
failure audit
truth-leakage audit
reproduction commands
final report markdown
```

Acceptance:

```text
1. artifact_manifest.json references all required files.

2. final_report.md states the task success criterion: coverage_rate >= 0.99 under the fixed step budget.

3. final_report.md includes Standard v1 training/eval results.

4. final_report.md includes Kilometer v1 stress-test results.

5. final_report.md compares PPO with the four v1 non-learning baselines.

6. success_rate_under_fixed_step_budget, path_length_to_99_success_only, coverage_per_meter, invalid_action_count_mean, and planner_failure_count_mean are included.

7. coverage curves and baseline comparison tables are included.

8. checkpoint_manifest.json identifies latest and best checkpoints.

9. reproduction_commands.md contains the commands needed to rerun Stage 6 and Stage 7.

10. leakage_audit.json confirms that hidden truth is used only for environment updates and final metrics, not policy or baseline decisions.

11. report claims do not exceed the evidence; Kilometer v1 is reported as a stress test, not final deployment validation.

12. Open decisions are limited to implementation-specific runtime thresholds, hardware scheduling, and optional future ablations.
```

Recommended artifacts:

```text
final_report.md
artifact_manifest.json
reproduction_commands.md
config_manifest.json
checkpoint_manifest.json
final_metrics_summary.json
baseline_comparison_table.csv
coverage_curves.csv
failure_audit.json
leakage_audit.json
```

## Open Decisions For Planning

The v1 default constants for reward scaling, invalid and safety penalties, fixed step budgets, stagnation termination, baseline comparison, and implementation roadmap are fixed in this design. The remaining choices are implementation logistics only and must not replace the v1 defaults without creating a new version.

```text
remaining implementation-plan choices:
  implementation-specific runtime thresholds
  hardware scheduling
  optional future ablation ranges around v1 defaults
```

The three scale profiles, fixed step budgets, reward constants, and stagnation thresholds are fixed for v1. Hard path-length budget is disabled for v1.
