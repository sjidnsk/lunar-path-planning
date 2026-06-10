import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


class _FakeCuda:
    def __init__(self, *, available: bool, name: str = "Fake CUDA Device") -> None:
        self._available = available
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int = 0) -> str:
        return self._name


class _FakeTorch:
    def __init__(self, *, available: bool, name: str = "Fake CUDA Device") -> None:
        self.cuda = _FakeCuda(available=available, name=name)


class PolicyTrainingCudaDeviceSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def test_device_contract_defaults_to_cpu(self) -> None:
        from model_explorer.policy.device import resolve_training_device

        resolved = resolve_training_device(None, torch_module=_FakeTorch(available=True))

        self.assertEqual(resolved.requested_device, "cpu")
        self.assertEqual(resolved.resolved_device, "cpu")
        self.assertFalse(resolved.fallback_to_cpu)
        self.assertEqual(resolved.reason_codes, ())

    def test_auto_uses_cuda_when_available(self) -> None:
        from model_explorer.policy.device import resolve_training_device

        resolved = resolve_training_device("auto", torch_module=_FakeTorch(available=True, name="RTX Test"))

        self.assertEqual(resolved.requested_device, "auto")
        self.assertEqual(resolved.resolved_device, "cuda")
        self.assertTrue(resolved.cuda_available)
        self.assertEqual(resolved.cuda_device_name, "RTX Test")
        self.assertFalse(resolved.fallback_to_cpu)

    def test_cuda_requested_but_unavailable_is_a_hard_failure(self) -> None:
        from model_explorer.policy.device import resolve_training_device

        resolved = resolve_training_device("cuda", torch_module=_FakeTorch(available=False))

        self.assertEqual(resolved.requested_device, "cuda")
        self.assertEqual(resolved.resolved_device, "cuda")
        self.assertIn("cuda_requested_but_unavailable", resolved.reason_codes)

    def test_limited_update_batch_tensors_are_created_on_requested_device(self) -> None:
        import torch
        from scripts.run_limited_ppo_update_smoke import _ppo_batch

        observation = SimpleNamespace(
            candidate_feature_names=("path_cost", "risk"),
            candidate_features=((1.0, 0.1), (0.4, 0.05)),
            global_features=(0.0,),
            action_mask=(True, True),
            candidate_missing_indicator_names=(),
            candidate_missing_indicators=((), ()),
        )
        transition = SimpleNamespace(
            observation=observation,
            action_index=1,
            log_prob=-0.5,
            value=0.25,
            reward=1.0,
            done=True,
        )
        config = {"training": {"discount_factor": 0.99}}

        batch = _ppo_batch((transition,), config=config, torch=torch, device="cpu")

        tensor_fields = (
            "candidate_features",
            "global_features",
            "action_mask",
            "candidate_missing_indicators",
            "actions",
            "old_log_probs",
            "returns",
            "advantages",
        )
        self.assertTrue(all(batch[field].device.type == "cpu" for field in tensor_fields))

    def test_checkpoint_state_dict_is_serialized_on_cpu(self) -> None:
        import torch
        from scripts.run_limited_ppo_update_smoke import _cpu_state_dict

        state = {"weight": torch.ones((2, 2), device="cpu")}

        cpu_state = _cpu_state_dict(state)

        self.assertEqual(cpu_state["weight"].device.type, "cpu")

    def test_readiness_accepts_passed_cuda_device_support_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _policy_training_cuda_device_support_readiness,
        )

        readiness = _policy_training_cuda_device_support_readiness(
            {
                "schema_version": "policy-training-cuda-device-support-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "requested_device": "auto",
                "resolved_device": "cuda",
                "cuda_available": True,
                "cuda_device_name": "RTX Test",
                "fallback_to_cpu": False,
                "optimizer_train_transition_count": 37,
                "source_fallback_trainable_count": 0,
                "old_log_prob_max_abs_error": 0.0,
                "old_value_max_abs_error": 0.0,
                "loss_non_finite_count": 0,
                "non_finite_gradient_count": 0,
                "non_finite_reward_count": 0,
                "non_finite_return_count": 0,
                "non_finite_advantage_count": 0,
                "parameter_l2_delta": 0.001,
                "approx_kl": 0.01,
                "max_grad_norm_after_clip": 0.5,
                "checkpoint_cpu_loadable": True,
                "experimental_checkpoint": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            }
        )

        self.assertTrue(readiness["present"])
        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])
        self.assertEqual(readiness["resolved_device"], "cuda")


if __name__ == "__main__":
    unittest.main()
