import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GitProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(self.repo_root / "scripts"))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="git-provenance-"))

    def _git(self, path: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    def _init_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._git(path, "init", "-q")
        self._git(path, "config", "user.email", "codex@example.invalid")
        self._git(path, "config", "user.name", "Codex")

    def _commit_file(self, path: Path, name: str, content: str) -> None:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._git(path, "add", name)
        self._git(path, "commit", "-q", "-m", f"add {name}")

    def test_snapshot_treats_ignored_untracked_files_as_clean_but_records_them(self) -> None:
        from git_provenance import git_snapshot

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        (repo / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
        self._git(repo, "add", ".gitignore")
        self._git(repo, "commit", "-q", "-m", "ignore local scratch")
        (repo / "ignored.tmp").write_text("local scratch\n", encoding="utf-8")

        snapshot = git_snapshot(repo, submodules=())

        self.assertFalse(snapshot["parent"]["dirty"])
        self.assertEqual(snapshot["parent"]["untracked_count"], 0)
        self.assertEqual(snapshot["parent"]["ignored_untracked_count"], 1)
        self.assertFalse(snapshot["dirty"])

    def test_snapshot_marks_tracked_and_untracked_parent_changes_dirty(self) -> None:
        from git_provenance import git_snapshot

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        self._commit_file(repo, "tracked.txt", "clean\n")

        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
        (repo / "new.txt").write_text("new\n", encoding="utf-8")
        snapshot = git_snapshot(repo, submodules=())

        self.assertTrue(snapshot["parent"]["dirty"])
        self.assertEqual(snapshot["parent"]["tracked_modified_count"], 1)
        self.assertEqual(snapshot["parent"]["untracked_count"], 1)
        self.assertTrue(snapshot["dirty"])

    def test_snapshot_marks_dirty_submodule_like_repo_dirty_without_parent_noise(self) -> None:
        from git_provenance import git_snapshot

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        (repo / ".gitignore").write_text("child/\n", encoding="utf-8")
        self._git(repo, "add", ".gitignore")
        self._git(repo, "commit", "-q", "-m", "ignore child checkout")
        child = repo / "child"
        self._init_repo(child)
        self._commit_file(child, "tracked.txt", "clean\n")
        (child / "tracked.txt").write_text("modified\n", encoding="utf-8")

        snapshot = git_snapshot(repo, submodules=("child",))

        self.assertFalse(snapshot["parent"]["dirty"])
        self.assertTrue(snapshot["submodules"]["child"]["dirty"])
        self.assertEqual(snapshot["submodules"]["child"]["tracked_modified_count"], 1)
        self.assertTrue(snapshot["dirty"])

    def test_snapshot_match_blocks_dirty_declared_source_but_keeps_legacy_sha_compatibility(self) -> None:
        from git_provenance import git_snapshot, git_snapshots_match

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        self._commit_file(repo, "tracked.txt", "clean\n")
        clean_snapshot = git_snapshot(repo, submodules=())
        legacy_source = {"parent": {"path": ".", "sha": clean_snapshot["parent"]["sha"]}, "submodules": {}}

        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
        dirty_snapshot = git_snapshot(repo, submodules=())

        self.assertFalse(git_snapshots_match(dirty_snapshot, dirty_snapshot, submodules=()))
        self.assertTrue(git_snapshots_match(legacy_source, dirty_snapshot, submodules=()))

    def test_snapshot_match_can_allow_identical_dirty_fingerprint_when_explicit(self) -> None:
        from git_provenance import git_snapshot, git_snapshots_match

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        self._commit_file(repo, "tracked.txt", "clean\n")
        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        (repo / "scratch.txt").write_text("scratch\n", encoding="utf-8")
        dirty_source = git_snapshot(repo, submodules=())
        dirty_current = git_snapshot(repo, submodules=())

        self.assertFalse(git_snapshots_match(dirty_source, dirty_current, submodules=()))
        self.assertTrue(
            git_snapshots_match(
                dirty_source,
                dirty_current,
                submodules=(),
                allow_dirty_match=True,
            )
        )

        (repo / "scratch.txt").write_text("changed scratch\n", encoding="utf-8")
        changed_current = git_snapshot(repo, submodules=())

        self.assertFalse(
            git_snapshots_match(
                dirty_source,
                changed_current,
                submodules=(),
                allow_dirty_match=True,
            )
        )

    def test_inspect_source_git_provenance_fails_missing_current_with_specific_reasons(self) -> None:
        from git_provenance import git_snapshot, inspect_source_git_provenance

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        self._commit_file(repo, "tracked.txt", "clean\n")
        current = git_snapshot(repo, submodules=())
        reason_codes: list[str] = []

        source_matches = inspect_source_git_provenance(
            {"git_provenance": {"current_matches_sources": True}},
            label="fixture_summary",
            current_git=current,
            require_current_git_match=True,
            reason_codes=reason_codes,
        )

        self.assertFalse(source_matches)
        self.assertIn("current_git_provenance_missing", reason_codes)
        self.assertIn("fixture_summary_current_git_provenance_missing", reason_codes)
        self.assertNotIn("current_git_provenance_mismatch", reason_codes)

    def test_inspect_source_git_provenance_distinguishes_dirty_submodule_and_legacy_sha_only(self) -> None:
        from git_provenance import git_snapshot, inspect_source_git_provenance

        repo = self.temp_dir / "repo"
        self._init_repo(repo)
        (repo / ".gitignore").write_text("child/\n", encoding="utf-8")
        self._git(repo, "add", ".gitignore")
        self._git(repo, "commit", "-q", "-m", "ignore child checkout")
        child = repo / "child"
        self._init_repo(child)
        self._commit_file(child, "tracked.txt", "clean\n")
        clean = git_snapshot(repo, submodules=("child",))
        legacy = {
            "parent": {"path": ".", "sha": clean["parent"]["sha"]},
            "submodules": {"child": {"path": "child", "sha": clean["submodules"]["child"]["sha"]}},
        }

        (child / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        dirty_current = git_snapshot(repo, submodules=("child",))
        dirty_source_reasons: list[str] = []
        legacy_reasons: list[str] = []

        dirty_source_matches = inspect_source_git_provenance(
            {"git_provenance": {"current": dirty_current}},
            label="dirty_summary",
            current_git=dirty_current,
            require_current_git_match=True,
            reason_codes=dirty_source_reasons,
            submodules=("child",),
        )
        legacy_matches = inspect_source_git_provenance(
            {"git_provenance": {"current": legacy}},
            label="legacy_summary",
            current_git=dirty_current,
            require_current_git_match=True,
            reason_codes=legacy_reasons,
            submodules=("child",),
        )

        self.assertFalse(dirty_source_matches)
        self.assertIn("current_git_provenance_mismatch", dirty_source_reasons)
        self.assertIn("dirty_summary_current_git_provenance_mismatch", dirty_source_reasons)
        self.assertTrue(legacy_matches)
        self.assertEqual(legacy_reasons, [])


if __name__ == "__main__":
    unittest.main()
