"""Tests for data models."""

import unittest
from pathlib import Path

from aind_disrnn_result_access.models import ArtifactInfo, RunInfo


class TestRunInfo(unittest.TestCase):
    """Tests for RunInfo dataclass."""

    def test_create_with_required_fields(self):
        """Test creating RunInfo with only required fields."""
        run = RunInfo(id="abc123", name="my-run", state="finished")
        self.assertEqual(run.id, "abc123")
        self.assertEqual(run.name, "my-run")
        self.assertEqual(run.state, "finished")
        self.assertEqual(run.tags, [])
        self.assertEqual(run.config, {})
        self.assertEqual(run.summary, {})
        self.assertEqual(run.created_at, "")
        self.assertEqual(run.url, "")
        self.assertEqual(run.project, "")
        self.assertEqual(run.entity, "")

    def test_create_with_all_fields(self):
        """Test creating RunInfo with all fields populated."""
        run = RunInfo(
            id="abc123",
            name="my-run",
            state="finished",
            tags=["disrnn", "mouse"],
            config={"model": {"type": "disrnn"}},
            summary={"likelihood": 0.95},
            created_at="2026-01-01T00:00:00",
            url="https://wandb.ai/run/abc123",
            project="test",
            entity="AIND-disRNN",
        )
        self.assertEqual(run.tags, ["disrnn", "mouse"])
        self.assertEqual(run.config, {"model": {"type": "disrnn"}})
        self.assertEqual(run.summary, {"likelihood": 0.95})
        self.assertEqual(run.created_at, "2026-01-01T00:00:00")
        self.assertEqual(run.url, "https://wandb.ai/run/abc123")
        self.assertEqual(run.project, "test")
        self.assertEqual(run.entity, "AIND-disRNN")

    def test_default_mutable_fields_are_independent(self):
        """Test that default list/dict fields are independent instances."""
        run1 = RunInfo(id="a", name="a", state="finished")
        run2 = RunInfo(id="b", name="b", state="finished")
        run1.tags.append("tag1")
        run1.config["key"] = "val"
        self.assertEqual(run2.tags, [])
        self.assertEqual(run2.config, {})


class TestArtifactInfo(unittest.TestCase):
    """Tests for ArtifactInfo dataclass."""

    def test_create_with_required_fields(self):
        """Test creating ArtifactInfo with only required fields."""
        artifact = ArtifactInfo(
            name="disrnn-output-abc123",
            type="training-output",
            version="v0",
            download_path=Path("/tmp/artifacts"),
            run_id="abc123",
        )
        self.assertEqual(artifact.name, "disrnn-output-abc123")
        self.assertEqual(artifact.type, "training-output")
        self.assertEqual(artifact.version, "v0")
        self.assertEqual(artifact.download_path, Path("/tmp/artifacts"))
        self.assertEqual(artifact.run_id, "abc123")
        self.assertEqual(artifact.files, [])

    def test_create_with_files(self):
        """Test creating ArtifactInfo with file list."""
        artifact = ArtifactInfo(
            name="disrnn-output-abc123",
            type="training-output",
            version="v0",
            download_path=Path("/tmp/artifacts"),
            run_id="abc123",
            files=["params.json", "output_summary.csv"],
        )
        self.assertEqual(
            artifact.files, ["params.json", "output_summary.csv"]
        )

    def test_default_files_are_independent(self):
        """Test that default files list is independent across instances."""
        a1 = ArtifactInfo(
            name="a",
            type="t",
            version="v0",
            download_path=Path("."),
            run_id="r1",
        )
        a2 = ArtifactInfo(
            name="b",
            type="t",
            version="v0",
            download_path=Path("."),
            run_id="r2",
        )
        a1.files.append("file.txt")
        self.assertEqual(a2.files, [])


if __name__ == "__main__":
    unittest.main()
