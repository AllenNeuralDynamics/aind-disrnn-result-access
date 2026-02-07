"""Tests for WandbClient."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_disrnn_result_access.wandb_client import WandbClient


def _make_mock_run(
    run_id="abc123",
    name="my-run",
    state="finished",
    tags=None,
    config=None,
    summary=None,
    created_at="2026-01-01T00:00:00",
    url="https://wandb.ai/run/abc123",
    project="test",
    entity="AIND-disRNN",
):
    """Create a mock W&B run object.

    Parameters
    ----------
    run_id : str
        Run identifier.
    name : str
        Run display name.
    state : str
        Run state.
    tags : list, optional
        Run tags.
    config : dict, optional
        Run config.
    summary : dict, optional
        Run summary metrics.
    created_at : str
        Creation timestamp.
    url : str
        Run URL.
    project : str
        Project name.
    entity : str
        Entity name.

    Returns
    -------
    MagicMock
        Mock run object.
    """
    run = MagicMock()
    run.id = run_id
    run.name = name
    run.state = state
    run.tags = tags or ["disrnn"]
    run.config = config or {"model": {"type": "disrnn"}}
    run.summary = summary or {"likelihood": 0.95}
    run.created_at = created_at
    run.url = url
    run.project = project
    run.entity = entity
    return run


def _make_mock_artifact(
    name="disrnn-output-abc123",
    artifact_type="training-output",
    version="v0",
    file_names=None,
):
    """Create a mock W&B artifact object.

    Parameters
    ----------
    name : str
        Artifact name.
    artifact_type : str
        Artifact type.
    version : str
        Artifact version.
    file_names : list, optional
        List of file names in the artifact.

    Returns
    -------
    MagicMock
        Mock artifact object.
    """
    artifact = MagicMock()
    artifact.name = name
    artifact.type = artifact_type
    artifact.version = version
    files = []
    for fname in file_names or ["params.json", "output_summary.csv"]:
        f = MagicMock()
        f.name = fname
        files.append(f)
    artifact.files.return_value = files
    return artifact


class TestInitApi(unittest.TestCase):
    """Tests for _init_api and WANDB_API_KEY validation."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    @patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
    def test_init_api_with_key(self, mock_api_cls):
        """Test _init_api succeeds when WANDB_API_KEY is set."""
        api = WandbClient._init_api()
        mock_api_cls.assert_called_once()
        self.assertEqual(api, mock_api_cls.return_value)

    @patch.dict(os.environ, {}, clear=True)
    def test_init_api_raises_without_key(self):
        """Test _init_api raises EnvironmentError without key."""
        with self.assertRaises(EnvironmentError) as ctx:
            WandbClient._init_api()
        msg = str(ctx.exception)
        self.assertIn("WANDB_API_KEY", msg)
        self.assertIn("export WANDB_API_KEY", msg)
        self.assertIn("wandb login", msg)
        self.assertIn("Code Ocean", msg)
        self.assertIn("https://wandb.ai/authorize", msg)


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestWandbClientInit(unittest.TestCase):
    """Tests for WandbClient initialization."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_default_init(self, mock_api_cls):
        """Test client initializes with default entity."""
        client = WandbClient()
        self.assertEqual(client.entity, "AIND-disRNN")
        self.assertIsNone(client.project)
        mock_api_cls.assert_called_once()

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_custom_init(self, mock_api_cls):
        """Test client initializes with custom entity and project."""
        client = WandbClient(entity="my-team", project="my-project")
        self.assertEqual(client.entity, "my-team")
        self.assertEqual(client.project, "my-project")


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestResolveProject(unittest.TestCase):
    """Tests for project resolution logic."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_argument_overrides_default(self, mock_api_cls):
        """Test that explicit project argument takes precedence."""
        client = WandbClient(project="default-proj")
        result = client._resolve_project("override-proj")
        self.assertEqual(result, "override-proj")

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_falls_back_to_default(self, mock_api_cls):
        """Test that None argument falls back to default project."""
        client = WandbClient(project="default-proj")
        result = client._resolve_project(None)
        self.assertEqual(result, "default-proj")

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_raises_when_no_project(self, mock_api_cls):
        """Test that ValueError is raised when no project is set."""
        client = WandbClient()
        with self.assertRaises(ValueError):
            client._resolve_project(None)


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestRunPath(unittest.TestCase):
    """Tests for run path construction."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_run_path(self, mock_api_cls):
        """Test run path is entity/project/run_id."""
        client = WandbClient(entity="AIND-disRNN")
        path = client._run_path("abc123", "test")
        self.assertEqual(path, "AIND-disRNN/test/abc123")


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestToRunInfo(unittest.TestCase):
    """Tests for converting W&B Run to RunInfo."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_converts_run(self, mock_api_cls):
        """Test that a W&B run is correctly converted to RunInfo."""
        client = WandbClient()
        mock_run = _make_mock_run()
        info = client._to_run_info(mock_run)
        self.assertEqual(info.id, "abc123")
        self.assertEqual(info.name, "my-run")
        self.assertEqual(info.state, "finished")
        self.assertEqual(info.tags, ["disrnn"])
        self.assertEqual(info.config, {"model": {"type": "disrnn"}})
        self.assertEqual(info.summary, {"likelihood": 0.95})
        self.assertEqual(info.project, "test")
        self.assertEqual(info.entity, "AIND-disRNN")

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_handles_none_fields(self, mock_api_cls):
        """Test conversion handles None tags/config/summary gracefully."""
        client = WandbClient()
        mock_run = _make_mock_run(tags=None, config=None, summary=None)
        # Set to None explicitly (overriding default in helper)
        mock_run.tags = None
        mock_run.config = None
        mock_run.summary = None
        info = client._to_run_info(mock_run)
        self.assertEqual(info.tags, [])
        self.assertEqual(info.config, {})
        self.assertEqual(info.summary, {})


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestGetProjects(unittest.TestCase):
    """Tests for get_projects."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_returns_project_names(self, mock_api_cls):
        """Test that project names are returned as a list of strings."""
        mock_api = mock_api_cls.return_value
        p1 = MagicMock()
        p1.name = "test"
        p2 = MagicMock()
        p2.name = "han_mice_disrnn"
        p3 = MagicMock()
        p3.name = "han_cpu_gpu_test"
        mock_api.projects.return_value = [p1, p2, p3]

        client = WandbClient()
        projects = client.get_projects()
        self.assertEqual(
            projects, ["test", "han_mice_disrnn", "han_cpu_gpu_test"]
        )
        mock_api.projects.assert_called_once_with(entity="AIND-disRNN")

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_returns_empty_list(self, mock_api_cls):
        """Test that empty project list is handled."""
        mock_api = mock_api_cls.return_value
        mock_api.projects.return_value = []
        client = WandbClient()
        projects = client.get_projects()
        self.assertEqual(projects, [])


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestGetRuns(unittest.TestCase):
    """Tests for get_runs."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_returns_run_infos(self, mock_api_cls):
        """Test that runs are returned as RunInfo list."""
        mock_api = mock_api_cls.return_value
        mock_api.runs.return_value = [
            _make_mock_run(run_id="r1", name="run-1"),
            _make_mock_run(run_id="r2", name="run-2"),
        ]
        client = WandbClient(project="test")
        runs = client.get_runs()
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].id, "r1")
        self.assertEqual(runs[1].id, "r2")
        mock_api.runs.assert_called_once_with(
            "AIND-disRNN/test",
            filters={},
            order="-created_at",
            per_page=50,
        )

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_passes_filters_and_params(self, mock_api_cls):
        """Test that filters, order, and per_page are forwarded."""
        mock_api = mock_api_cls.return_value
        mock_api.runs.return_value = []
        client = WandbClient(project="test")
        client.get_runs(
            filters={"state": "finished"},
            order="created_at",
            per_page=10,
        )
        mock_api.runs.assert_called_once_with(
            "AIND-disRNN/test",
            filters={"state": "finished"},
            order="created_at",
            per_page=10,
        )

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_project_override(self, mock_api_cls):
        """Test that project argument overrides default."""
        mock_api = mock_api_cls.return_value
        mock_api.runs.return_value = []
        client = WandbClient(project="default")
        client.get_runs(project="han_mice_disrnn")
        mock_api.runs.assert_called_once_with(
            "AIND-disRNN/han_mice_disrnn",
            filters={},
            order="-created_at",
            per_page=50,
        )

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_raises_without_project(self, mock_api_cls):
        """Test that ValueError is raised when no project is set."""
        client = WandbClient()
        with self.assertRaises(ValueError):
            client.get_runs()


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestGetRun(unittest.TestCase):
    """Tests for get_run."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_returns_run_info(self, mock_api_cls):
        """Test that a single run is returned as RunInfo."""
        mock_api = mock_api_cls.return_value
        mock_api.run.return_value = _make_mock_run()
        client = WandbClient(project="test")
        info = client.get_run("abc123")
        self.assertEqual(info.id, "abc123")
        mock_api.run.assert_called_once_with("AIND-disRNN/test/abc123")

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_project_override(self, mock_api_cls):
        """Test that project argument overrides default."""
        mock_api = mock_api_cls.return_value
        mock_api.run.return_value = _make_mock_run()
        client = WandbClient(project="default")
        client.get_run("abc123", project="han_cpu_gpu_test")
        mock_api.run.assert_called_once_with(
            "AIND-disRNN/han_cpu_gpu_test/abc123"
        )


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestDownloadArtifact(unittest.TestCase):
    """Tests for download_artifact."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_downloads_matching_artifacts(self, mock_api_cls):
        """Test that artifacts of the correct type are downloaded."""
        mock_api = mock_api_cls.return_value
        mock_run = _make_mock_run()
        mock_artifact = _make_mock_artifact()
        mock_run.logged_artifacts.return_value = [mock_artifact]
        mock_api.run.return_value = mock_run

        client = WandbClient(project="test")
        results = client.download_artifact("abc123")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "disrnn-output-abc123")
        self.assertEqual(results[0].type, "training-output")
        self.assertEqual(results[0].run_id, "abc123")
        self.assertEqual(
            results[0].files, ["params.json", "output_summary.csv"]
        )
        expected_path = Path(
            "/root/capsule/results/artifacts/abc123/disrnn-output-abc123"
        )
        self.assertEqual(results[0].download_path, expected_path)
        mock_artifact.download.assert_called_once_with(root=str(expected_path))

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_filters_by_artifact_type(self, mock_api_cls):
        """Test that only artifacts of the specified type are returned."""
        mock_api = mock_api_cls.return_value
        mock_run = _make_mock_run()
        matching = _make_mock_artifact(artifact_type="training-output")
        non_matching = _make_mock_artifact(
            name="other-artifact", artifact_type="dataset"
        )
        mock_run.logged_artifacts.return_value = [matching, non_matching]
        mock_api.run.return_value = mock_run

        client = WandbClient(project="test")
        results = client.download_artifact("abc123")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "disrnn-output-abc123")
        non_matching.download.assert_not_called()

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_custom_output_dir(self, mock_api_cls):
        """Test custom output_dir is used as base with run_id appended."""
        mock_api = mock_api_cls.return_value
        mock_run = _make_mock_run()
        mock_artifact = _make_mock_artifact()
        mock_run.logged_artifacts.return_value = [mock_artifact]
        mock_api.run.return_value = mock_run

        client = WandbClient(project="test")
        results = client.download_artifact(
            "abc123", output_dir="/tmp/my-output"
        )

        expected_path = Path("/tmp/my-output/abc123/disrnn-output-abc123")
        self.assertEqual(results[0].download_path, expected_path)
        mock_artifact.download.assert_called_once_with(root=str(expected_path))

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_no_matching_artifacts(self, mock_api_cls):
        """Test that empty list is returned when no artifacts match."""
        mock_api = mock_api_cls.return_value
        mock_run = _make_mock_run()
        mock_run.logged_artifacts.return_value = []
        mock_api.run.return_value = mock_run

        client = WandbClient(project="test")
        results = client.download_artifact("abc123")
        self.assertEqual(results, [])

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_download_specific_files(self, mock_api_cls):
        """Test downloading only specific files from artifact."""
        mock_api = mock_api_cls.return_value
        mock_run = _make_mock_run()
        mock_artifact = _make_mock_artifact()

        # Mock get_entry for specific file
        mock_entry = MagicMock()
        mock_artifact.get_entry.return_value = mock_entry

        mock_run.logged_artifacts.return_value = [mock_artifact]
        mock_api.run.return_value = mock_run

        client = WandbClient(project="test")
        results = client.download_artifact("abc123", files=["params.json"])

        self.assertEqual(len(results), 1)
        # Only the requested file should be in the list
        self.assertEqual(results[0].files, ["params.json"])
        # get_entry should be called for the specific file
        mock_artifact.get_entry.assert_called_once_with("params.json")
        # download() should NOT be called (we use entry.download instead)
        mock_artifact.download.assert_not_called()
        # entry.download should be called
        mock_entry.download.assert_called_once()

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_download_specific_files_not_found(self, mock_api_cls):
        """Test downloading files that don't exist is handled gracefully."""
        mock_api = mock_api_cls.return_value
        mock_run = _make_mock_run()
        mock_artifact = _make_mock_artifact()

        # Simulate file not found
        mock_artifact.get_entry.side_effect = KeyError("File not found")

        mock_run.logged_artifacts.return_value = [mock_artifact]
        mock_api.run.return_value = mock_run

        client = WandbClient(project="test")
        results = client.download_artifact(
            "abc123", files=["nonexistent.json"]
        )

        self.assertEqual(len(results), 1)
        # No files should be downloaded
        self.assertEqual(results[0].files, [])


@patch.dict(os.environ, {"WANDB_API_KEY": "test-key"})
class TestDownloadArtifacts(unittest.TestCase):
    """Tests for download_artifacts (batch)."""

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_downloads_for_multiple_runs(self, mock_api_cls):
        """Test batch download returns dict keyed by run_id."""
        mock_api = mock_api_cls.return_value

        mock_run1 = _make_mock_run(run_id="r1")
        art1 = _make_mock_artifact(name="disrnn-output-r1")
        mock_run1.logged_artifacts.return_value = [art1]

        mock_run2 = _make_mock_run(run_id="r2")
        art2 = _make_mock_artifact(name="disrnn-output-r2")
        mock_run2.logged_artifacts.return_value = [art2]

        mock_api.run.side_effect = [mock_run1, mock_run2]

        client = WandbClient(project="test")
        results = client.download_artifacts(["r1", "r2"])

        self.assertIn("r1", results)
        self.assertIn("r2", results)
        self.assertEqual(len(results["r1"]), 1)
        self.assertEqual(len(results["r2"]), 1)
        self.assertEqual(results["r1"][0].run_id, "r1")
        self.assertEqual(results["r2"][0].run_id, "r2")

    @patch("aind_disrnn_result_access.wandb_client.wandb.Api")
    def test_empty_run_ids(self, mock_api_cls):
        """Test batch download with empty run list."""
        client = WandbClient(project="test")
        results = client.download_artifacts([])
        self.assertEqual(results, {})


if __name__ == "__main__":
    unittest.main()
