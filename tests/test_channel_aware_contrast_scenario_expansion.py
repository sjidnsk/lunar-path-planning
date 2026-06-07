import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ChannelAwareContrastScenarioExpansionTests(unittest.TestCase):
    def test_stress_generator_includes_channel_contrast_scenarios(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        dev_root = repo_root / "dev-platform-constraints"
        root = Path(tempfile.mkdtemp(prefix="channel-aware-contrast-scenarios-"))
        scenario_config = root / "npz_validation_scenarios.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(dev_root / "scripts" / "generate_npz_validation_maps.py"),
                "--scenario-set",
                "stress",
                "--output-dir",
                str(root / "maps"),
                "--scenario-config",
                str(scenario_config),
            ],
            cwd=dev_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        scenarios = json.loads(scenario_config.read_text(encoding="utf-8"))["scenarios"]
        by_id = {item["scenario_id"]: item for item in scenarios}
        expected_ids = {
            "npz_low_centerline_bad_channel",
            "npz_blocked_nearby_clearance_detour",
            "npz_high_cost_exposure_rock_detour",
        }
        self.assertTrue(expected_ids.issubset(by_id))

        centerline = by_id["npz_low_centerline_bad_channel"]
        self.assertEqual(centerline["scenario_group"], "channel_contrast")
        self.assertEqual(centerline["contrast_focus"], "low_centerline_cost_bad_channel_quality")
        with np.load(Path(centerline["map_source"]["path"]), allow_pickle=False) as grid:
            rx0, rx1, ry0, ry1 = centerline["risk_region"]
            risk_region = grid["confidence"][ry0:ry1, rx0:rx1]
            upper_y0 = max(0, ry0 - 2)
            upper_detour = grid["confidence"][upper_y0:ry0, rx0:rx1]
            lower_y1 = min(centerline["height"], ry1 + 2)
            lower_detour = grid["confidence"][ry1:lower_y1, rx0:rx1]
            self.assertLess(float(np.max(risk_region)), 0.35)
            if upper_detour.size:
                self.assertGreater(float(np.mean(upper_detour)), float(np.mean(risk_region)))
            if lower_detour.size:
                self.assertGreater(float(np.mean(lower_detour)), float(np.mean(risk_region)))

        clearance = by_id["npz_blocked_nearby_clearance_detour"]
        self.assertEqual(clearance["scenario_group"], "channel_contrast")
        self.assertEqual(clearance["contrast_focus"], "blocked_nearby_clearance")
        with np.load(Path(clearance["map_source"]["path"]), allow_pickle=False) as grid:
            blocked_count = int(np.count_nonzero(grid["obstacle"] >= 0.5))
            self.assertGreaterEqual(blocked_count, 20)
            self.assertLess(float(np.min(grid["illumination"])), 0.30)

        rock_detour = by_id["npz_high_cost_exposure_rock_detour"]
        self.assertEqual(rock_detour["scenario_group"], "channel_contrast")
        self.assertEqual(rock_detour["contrast_focus"], "high_cost_exposure_rock_field_detour")
        with np.load(Path(rock_detour["map_source"]["path"]), allow_pickle=False) as grid:
            self.assertGreaterEqual(int(np.count_nonzero(grid["obstacle"] >= 0.5)), 24)
            self.assertTrue(np.any(grid["value"] > 0.0))


if __name__ == "__main__":
    unittest.main()
