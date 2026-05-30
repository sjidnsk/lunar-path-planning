import shutil
import subprocess
import unittest
from pathlib import Path


class BootstrapUbuntuCondaTests(unittest.TestCase):
    def test_parent_environment_declares_shared_runtime(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "environment.yml").read_text(encoding="utf-8")

        self.assertIn("name: lunar-explorer", content)
        self.assertIn("python=3.12", content)
        self.assertIn("numpy>=1.26,<2.3", content)
        self.assertIn("matplotlib>=3.8", content)
        self.assertIn("pytest>=8", content)
        self.assertIn("pip", content)

    def test_ubuntu_bootstrap_dry_run_prints_expected_steps(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is required for bootstrap script validation")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "bootstrap_ubuntu_conda.sh"

        syntax = subprocess.run(
            [bash, "-n", str(script)],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stdout + syntax.stderr)

        result = subprocess.run(
            [
                bash,
                str(script),
                "--dry-run",
                "--run-validation",
                "--install-editable",
                "--with-training",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("DRY RUN", output)
        self.assertIn("git submodule update --init --recursive path-planner model-explorer dev-platform-constraints", output)
        self.assertNotIn("a_gcs_ws-2.0.1", output)
        self.assertIn("conda env create/update", output)
        self.assertIn("environment.yml", output)
        self.assertIn("python=3.12", output)
        self.assertIn("assert sys.version_info[:2] == (3, 12)", output)
        self.assertIn("path_planner import ok", output)
        self.assertIn("model_explorer import ok", output)
        self.assertIn("dev_platform_constraints import ok", output)
        self.assertIn("python -m pytest", output)
        self.assertIn("python -m unittest discover -s tests -v", output)
        self.assertIn("python -m model_explorer verify", output)
        self.assertIn("python -m pip install -e path-planner -e dev-platform-constraints -e model-explorer[training]", output)


if __name__ == "__main__":
    unittest.main()
