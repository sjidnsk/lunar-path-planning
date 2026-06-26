import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_xunce_full_network_outputs_continuous_theta_distribution() -> None:
    from scripts.xunce_full_network_common import XunceFullNetworkV1

    model = XunceFullNetworkV1(
        candidate_feature_count=4,
        edge_feature_count=2,
        memory_feature_count=3,
        context_feature_count=5,
        hidden_dim=8,
        message_passing_layers=1,
    )
    output = model(
        candidate_features=torch.randn(1, 3, 4),
        edge_features=torch.zeros(0, 2),
        edge_index=torch.zeros(0, 2, dtype=torch.long),
        memory_features=torch.randn(1, 3),
        context_features=torch.randn(1, 3, 5),
        action_mask=torch.tensor([[True, False, True]]),
    )

    assert output.theta_mu_rad.shape == (1, 3)
    assert output.theta_kappa.shape == (1, 3)
    assert torch.isfinite(output.theta_mu_rad).all()
    assert torch.isfinite(output.theta_kappa).all()
    assert torch.all(output.theta_kappa > 0)


def test_legacy_checkpoint_missing_only_theta_heads_is_compatible() -> None:
    from scripts.xunce_full_network_common import (
        XunceFullNetworkV1,
        load_xunce_full_network_state_dict_compatible,
    )

    model = XunceFullNetworkV1(
        candidate_feature_count=4,
        edge_feature_count=2,
        memory_feature_count=3,
        context_feature_count=5,
        hidden_dim=8,
        message_passing_layers=1,
    )
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not (key.startswith("theta_mu_head.") or key.startswith("theta_kappa_head."))
    }
    target = XunceFullNetworkV1(
        candidate_feature_count=4,
        edge_feature_count=2,
        memory_feature_count=3,
        context_feature_count=5,
        hidden_dim=8,
        message_passing_layers=1,
    )

    loaded, missing, disallowed, unexpected = load_xunce_full_network_state_dict_compatible(target, legacy_state)

    assert loaded is True
    assert missing
    assert disallowed == []
    assert unexpected == []

    bad_state = dict(legacy_state)
    bad_state.pop("policy_head.0.weight")
    bad_target = XunceFullNetworkV1(
        candidate_feature_count=4,
        edge_feature_count=2,
        memory_feature_count=3,
        context_feature_count=5,
        hidden_dim=8,
        message_passing_layers=1,
    )
    loaded, _missing, disallowed, _unexpected = load_xunce_full_network_state_dict_compatible(bad_target, bad_state)
    assert loaded is False
    assert "policy_head.0.weight" in disallowed


def test_continuous_theta_logprob_is_point_plus_theta() -> None:
    from scripts.xunce_continuous_theta_action import (
        CONTINUOUS_THETA_ACTION_SPACE,
        continuous_theta_log_prob,
    )

    point_logits = torch.tensor([0.2, 1.0, -0.5], dtype=torch.float32)
    theta_mu = torch.tensor([0.0, math.pi / 2.0, math.pi], dtype=torch.float32)
    theta_kappa = torch.tensor([2.0, 4.0, 3.0], dtype=torch.float32)
    theta = torch.tensor(math.pi / 2.0 + 0.1, dtype=torch.float32)

    detail = continuous_theta_log_prob(
        point_logits=point_logits,
        theta_mu_rad=theta_mu,
        theta_kappa=theta_kappa,
        action_index=1,
        theta_rad=theta,
    )

    point_distribution = torch.distributions.Categorical(logits=point_logits)
    theta_distribution = torch.distributions.VonMises(theta_mu[1], theta_kappa[1])
    expected = point_distribution.log_prob(torch.tensor(1)) + theta_distribution.log_prob(theta)

    assert detail["action_space_type"] == CONTINUOUS_THETA_ACTION_SPACE
    assert abs(float(detail["old_log_prob"]) - float(expected)) < 1.0e-6
    assert abs(detail["old_log_prob"] - (detail["old_point_log_prob"] + detail["old_theta_log_prob"])) < 1.0e-6


def test_stage21_3_continuous_theta_contract_rejects_missing_fields() -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import _continuous_theta_action_contract_missing_count
    from scripts.xunce_continuous_theta_action import continuous_theta_log_prob

    point_logits = torch.tensor([0.1, 1.2, -0.3], dtype=torch.float32)
    theta_mu_rad = torch.tensor([0.0, 0.2, -1.0], dtype=torch.float32)
    theta_kappa = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    log_prob = continuous_theta_log_prob(
        point_logits=point_logits,
        theta_mu_rad=theta_mu_rad,
        theta_kappa=theta_kappa,
        action_index=1,
        theta_rad=0.25,
    )

    valid = {
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "action_index": 1,
        "selected_base_candidate_index": 1,
        "selected_theta_rad": log_prob["selected_theta_rad"],
        "selected_theta_deg": log_prob["selected_theta_deg"],
        "old_point_log_prob": log_prob["old_point_log_prob"],
        "old_theta_log_prob": log_prob["old_theta_log_prob"],
        "old_log_prob": log_prob["old_log_prob"],
        "base_candidate_set_hash": "base-hash",
        "action_sample_hash": "sample-hash",
        "info": {
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "selected_base_candidate_index": 1,
            "selected_theta_rad": log_prob["selected_theta_rad"],
            "selected_theta_deg": log_prob["selected_theta_deg"],
            "base_candidate_set_hash": "base-hash",
            "action_sample_hash": "sample-hash",
            "old_point_log_prob": log_prob["old_point_log_prob"],
            "old_theta_log_prob": log_prob["old_theta_log_prob"],
            "old_sampling_logits": [float(value) for value in point_logits],
            "theta_mu_rad": [float(value) for value in theta_mu_rad],
            "theta_kappa": [float(value) for value in theta_kappa],
        },
    }
    missing = dict(valid)
    missing["info"] = dict(valid["info"])
    missing.pop("selected_theta_rad")
    missing["info"].pop("selected_theta_rad")

    assert _continuous_theta_action_contract_missing_count([valid]) == 0
    assert _continuous_theta_action_contract_missing_count([missing]) == 1


def test_stage21_1_continuous_theta_sampling_records_split_logprob() -> None:
    from scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector import _sample_xunce_action
    from scripts.xunce_full_network_common import XunceFullNetworkOutput

    class Model:
        def __call__(self, **_kwargs):
            logits = torch.tensor([[0.1, 1.2, -0.4]], dtype=torch.float32)
            return XunceFullNetworkOutput(
                logits=logits,
                masked_logits=logits,
                action_probs=torch.softmax(logits, dim=-1),
                value=torch.tensor([0.3], dtype=torch.float32),
                theta_mu_sin=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
                theta_mu_cos=torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32),
                theta_kappa_raw=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                theta_mu_rad=torch.tensor([[0.0, math.pi / 2.0, math.pi]], dtype=torch.float32),
                theta_kappa=torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32),
            )

    batch = {"action_mask": torch.tensor([[True, True, True]])}
    detail = _sample_xunce_action(
        Model(),
        batch,
        sampling_mask=(False, True, False),
        temperature=1.0,
        continuous_theta_action_space_enabled=True,
    )

    assert detail["action_space_type"] == "hybrid_discrete_xy_continuous_theta/v1"
    assert detail["action_index"] == 1
    assert detail["selected_base_candidate_index"] == 1
    assert math.isfinite(detail["selected_theta_rad"])
    assert math.isfinite(detail["old_point_log_prob"])
    assert math.isfinite(detail["old_theta_log_prob"])
    assert abs(detail["old_log_prob"] - (detail["old_point_log_prob"] + detail["old_theta_log_prob"])) < 1.0e-6


def test_stage25_runner_generates_continuous_theta_substage_configs(tmp_path, monkeypatch) -> None:
    import scripts.run_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation as runner

    stage24_5a_root = tmp_path / "stage24_5a"
    stage24_5a_root.mkdir()
    _write_json(stage24_5a_root / "xunce-stage24-5a-summary.json", {"status": "passed"})
    _write_json(stage24_5a_root / "xunce-stage24-5a-rerun-stage24-5-summary.json", {"stage23_2b_high_res_root": "D:/fake/high-res"})
    base_paths = {}
    for name in ("s21_1", "s21_2", "s21_3", "s21_4", "s21_5", "hf"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {})
        base_paths[name] = path
    canonical = tmp_path / "canonical.json"
    reward = tmp_path / "reward.json"
    _write_json(canonical, {})
    _write_json(reward, {})
    config_path = tmp_path / "stage25.json"
    _write_json(
        config_path,
        {
            "stage24_5a_root": str(stage24_5a_root),
            "stage21_1_base_config": str(base_paths["s21_1"]),
            "stage21_2_base_config": str(base_paths["s21_2"]),
            "stage21_3_base_config": str(base_paths["s21_3"]),
            "stage21_4_base_config": str(base_paths["s21_4"]),
            "stage21_5_base_config": str(base_paths["s21_5"]),
            "high_fidelity_config": str(base_paths["hf"]),
            "canonical_reward_profile": str(canonical),
            "coverage_first_reward_profile": str(reward),
            "xunce_candidate_checkpoint": "checkpoint.pt",
            "min_mean_abs_probability_delta_for_signal": 1.0e-6,
        },
    )

    def fake_s21_1(*, config_path, output_root, repo_root):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert cfg["action_space_type"] == "hybrid_discrete_xy_continuous_theta/v1"
        output_root.mkdir(parents=True, exist_ok=True)
        transition = {
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "selected_theta_rad": 0.25,
            "old_point_log_prob": -0.5,
            "old_theta_log_prob": -0.2,
            "base_candidate_set_hash": "base",
            "action_sample_hash": "sample",
            "info": {"candidate_viewpoints": None},
        }
        (output_root / runner.stage21_1.TRANSITIONS_FILE).write_text(json.dumps(transition) + "\n", encoding="utf-8")
        sampling = {"selected_theta_rad": 0.25, "theta_mu_rad": [0.25, 0.5], "theta_kappa": [1.0, 2.0]}
        (output_root / runner.stage21_1.SAMPLING_AUDIT_FILE).write_text(json.dumps(sampling) + "\n", encoding="utf-8")
        return {"status": "passed", "trainable_transition_count": 1}

    def fake_s21_2(*, config_path, output_root, repo_root):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert cfg["slope_obstacle_aware_theta_reward_enabled"] is True
        assert cfg["require_hybrid_astar_path_cost_contract"] is True
        output_root.mkdir(parents=True, exist_ok=True)
        return {"status": "passed", "reward_evaluation_row_count": 1}

    def fake_s21_3(*, config_path, output_root, repo_root):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert cfg["require_continuous_theta_action_contract"] is True
        output_root.mkdir(parents=True, exist_ok=True)
        batch_row = {
            "stage21_3_split": "train",
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        }
        (output_root / runner.stage21_3.BATCH_FILE).write_text(json.dumps(batch_row) + "\n", encoding="utf-8")
        return {
            "status": "passed",
            "trainable_transition_count": 1,
            "continuous_theta_action_contract_missing_count": 0,
            "slope_obstacle_reward_contract_missing_count": 0,
            "hybrid_astar_path_cost_contract_missing_count": 0,
        }

    def fake_s21_4(*, config_path, output_root, repo_root):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert cfg["continuous_theta_head_init_seed"] == 2101
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / runner.stage21_4.LOSS_AUDIT_FILE).write_text(json.dumps({"loss": 1.0}) + "\n", encoding="utf-8")
        return {
            "status": "passed",
            "train_transition_count": 1,
            "checkpoint_reload_passed": True,
            "final_post_update_approx_kl": 0.0,
            "parameter_delta_l2": 1.0e-4,
        }

    def fake_s21_5(*, config_path, output_root, repo_root):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert cfg["continuous_theta_action_space_enabled"] is True
        pre = output_root / "pre"
        post = output_root / "post"
        pre.mkdir(parents=True, exist_ok=True)
        post.mkdir(parents=True, exist_ok=True)
        base = {
            "scenario_id": "s",
            "step_index": 0,
            "current_cell": [0, 0],
            "covered_cells_hash": "covered",
            "candidate_set_hash": "base",
            "selected_action_index": 0,
            "selected_theta_rad": 0.25,
            "selected_theta_deg": 14.323944878,
            "theta_mu_rad": [0.25],
            "theta_kappa": [2.0],
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "policy": "xunce",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "action_probs": [1.0],
            "candidate_viewpoint": [1, 1, 14.323944878],
            "candidate_theta_deg": 14.323944878,
            "hybrid_astar_path_cost": 1.0,
            "hybrid_astar_pose_path_hash": "pose-hash",
            "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
            "legacy_grid_astar_path_cost": 0.8,
            "hybrid_vs_grid_path_cost_delta": 0.2,
            "point_grid_path_cost_fallback_used": False,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
            "max_traversable_slope_deg": 30.0,
            "slope_obstacle_source_hash": "slope-hash",
            "platform_contract_hash": "platform-hash",
        }
        (pre / runner.MODEL_INFERENCE_FILE).write_text(json.dumps(base) + "\n", encoding="utf-8")
        (post / runner.MODEL_INFERENCE_FILE).write_text(json.dumps(base) + "\n", encoding="utf-8")
        summary = {"status": "passed", "pre_evaluation_root": str(pre), "post_evaluation_root": str(post)}
        _write_json(output_root / runner.stage21_5.SUMMARY_FILE, summary)
        return summary

    monkeypatch.setattr(runner.stage21_1, "run_xunce_stage21_1_on_policy_ppo_rollout_collector", fake_s21_1)
    monkeypatch.setattr(runner.stage21_2, "run_xunce_stage21_2_coverage_first_ppo_reward_contract", fake_s21_2)
    monkeypatch.setattr(runner.stage21_3, "run_xunce_stage21_3_ppo_batch_validation", fake_s21_3)
    monkeypatch.setattr(runner.stage21_4, "run_xunce_stage21_4_tiny_ppo_update_smoke", fake_s21_4)
    monkeypatch.setattr(runner.stage21_5, "run_xunce_stage21_5_post_update_offline_trajectory_evaluation", fake_s21_5)

    summary = runner.run_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "calibrate_stage25_continuous_theta_policy_signal_strength"
    generated_hf = json.loads(Path(summary["generated_configs"]["high_fidelity_config"]).read_text(encoding="utf-8"))
    assert generated_hf["continuous_theta_action_space_enabled"] is True
    assert generated_hf["theta_aware_candidate_viewpoints_enabled"] is False
    assert generated_hf["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert summary["publishes_checkpoint"] is False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
