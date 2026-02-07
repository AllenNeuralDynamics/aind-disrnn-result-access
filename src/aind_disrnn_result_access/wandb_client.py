"""Client for accessing disRNN W&B run data and artifacts."""

import os
from pathlib import Path
from typing import Optional

import wandb

from aind_disrnn_result_access.models import ArtifactInfo, RunInfo


class WandbClient:
    """Client for accessing disRNN W&B run data and artifacts.

    Parameters
    ----------
    entity : str
        W&B entity (team or user). Defaults to "AIND-disRNN".
    project : str, optional
        Default W&B project name. Can be overridden per method call.
    """

    DEFAULT_ENTITY = "AIND-disRNN"

    def __init__(
        self,
        entity: str = DEFAULT_ENTITY,
        project: Optional[str] = None,
    ):
        """Initialize the client.

        Parameters
        ----------
        entity : str
            W&B entity (team or user).
        project : str, optional
            Default project name.
        """
        self.entity = entity
        self.project = project
        self._api = self._init_api()

    @staticmethod
    def _init_api() -> wandb.Api:
        """Initialize the W&B API, checking for authentication.

        Returns
        -------
        wandb.Api
            Authenticated W&B API client.

        Raises
        ------
        EnvironmentError
            If WANDB_API_KEY is not set in the environment.
        """
        if not os.environ.get("WANDB_API_KEY"):
            raise EnvironmentError(
                "WANDB_API_KEY not found in environment.\n\n"
                "To fix this, do one of the following:\n"
                "  1. Set it in your shell:\n"
                "       export WANDB_API_KEY=<your-key>\n\n"
                "  2. Add it to a .env file in the project root:\n"
                "       WANDB_API_KEY=<your-key>\n\n"
                "  3. Run 'wandb login' to authenticate "
                "interactively.\n\n"
                "  4. If you are in Code Ocean, attach your "
                "Weights & Biases\n"
                "     API key as a secret to the capsule and set "
                "the environment\n"
                "     variable name to WANDB_API_KEY.\n\n"
                "You can find your API key at: "
                "https://wandb.ai/authorize"
            )
        return wandb.Api()

    def _resolve_project(self, project: Optional[str]) -> str:
        """Resolve the project name from argument or default.

        Parameters
        ----------
        project : str, optional
            Project name override.

        Returns
        -------
        str
            Resolved project name.

        Raises
        ------
        ValueError
            If no project is specified and no default is set.
        """
        resolved = project or self.project
        if not resolved:
            raise ValueError(
                "No project specified. Pass project= or set a "
                "default in the constructor."
            )
        return resolved

    def _run_path(self, run_id: str, project: str) -> str:
        """Build the W&B run path string.

        Parameters
        ----------
        run_id : str
            The run ID.
        project : str
            The project name.

        Returns
        -------
        str
            Path in the form 'entity/project/run_id'.
        """
        return f"{self.entity}/{project}/{run_id}"

    def _to_run_info(self, run: wandb.apis.public.Run) -> RunInfo:
        """Convert a W&B Run object to a RunInfo dataclass.

        Parameters
        ----------
        run : wandb.apis.public.Run
            The W&B run object.

        Returns
        -------
        RunInfo
            Structured run metadata.
        """
        return RunInfo(
            id=run.id,
            name=run.name,
            state=run.state,
            tags=list(run.tags or []),
            config=dict(run.config or {}),
            summary=dict(run.summary or {}),
            created_at=run.created_at,
            url=run.url,
            project=run.project,
            entity=run.entity,
        )

    def get_projects(self) -> list[str]:
        """List all project names under the entity.

        Returns
        -------
        list[str]
            Project names.
        """
        projects = self._api.projects(entity=self.entity)
        return [p.name for p in projects]

    def get_runs(
        self,
        project: Optional[str] = None,
        filters: Optional[dict] = None,
        order: str = "-created_at",
        per_page: int = 50,
    ) -> list[RunInfo]:
        """List runs with optional filters.

        Parameters
        ----------
        project : str, optional
            Project name. Falls back to self.project.
        filters : dict, optional
            W&B run filters (MongoDB-style query).
        order : str
            Sort order. Default is most recent first.
        per_page : int
            Number of runs per page.

        Returns
        -------
        list[RunInfo]
            List of run metadata.
        """
        proj = self._resolve_project(project)
        path = f"{self.entity}/{proj}"
        runs = self._api.runs(
            path,
            filters=filters or {},
            order=order,
            per_page=per_page,
        )
        return [self._to_run_info(r) for r in runs]

    def get_run(
        self,
        run_id: str,
        project: Optional[str] = None,
    ) -> RunInfo:
        """Get detailed metadata for a single run.

        Parameters
        ----------
        run_id : str
            The W&B run ID.
        project : str, optional
            Project name. Falls back to self.project.

        Returns
        -------
        RunInfo
            Structured run metadata.
        """
        proj = self._resolve_project(project)
        run = self._api.run(self._run_path(run_id, proj))
        return self._to_run_info(run)

    def download_artifact(
        self,
        run_id: str,
        project: Optional[str] = None,
        output_dir: Optional[str] = None,
        artifact_type: str = "training-output",
        files: Optional[list[str]] = None,
    ) -> list[ArtifactInfo]:
        """Download training output artifacts for a given run.

        Parameters
        ----------
        run_id : str
            The W&B run ID.
        project : str, optional
            Project name. Falls back to self.project.
        output_dir : str, optional
            Base directory for downloads. The run_id will be automatically
            appended to create the final path. Defaults to
            '/root/capsule/data', resulting in
            '/root/capsule/data/<run_id>'.
        artifact_type : str
            Artifact type filter. Default is 'training-output'.
        files : list[str], optional
            List of specific files to download from each artifact.
            If None (default), downloads all files. Uses W&B's path_prefix
            parameter to filter files by name pattern.

        Returns
        -------
        list[ArtifactInfo]
            List of downloaded artifact metadata.
        """
        proj = self._resolve_project(project)
        run = self._api.run(self._run_path(run_id, proj))
        base_dir = Path(output_dir or "/root/capsule/data")
        base_dir = base_dir / run_id

        results = []
        for artifact in run.logged_artifacts():
            if artifact.type != artifact_type:
                continue
            download_path = base_dir / artifact.name

            # Download all files or specific files
            if files is None:
                artifact.download(root=str(download_path))
                downloaded_files = [f.name for f in artifact.files()]
            else:
                # Download each specified file individually
                downloaded_files = []
                for file_pattern in files:
                    # Use get_entry to download specific files
                    try:
                        entry = artifact.get_entry(file_pattern)
                        entry.download(root=str(download_path))
                        downloaded_files.append(file_pattern)
                    except KeyError:
                        # File not found in artifact, skip
                        pass

            results.append(
                ArtifactInfo(
                    name=artifact.name,
                    type=artifact.type,
                    version=artifact.version,
                    download_path=download_path,
                    run_id=run_id,
                    files=downloaded_files,
                )
            )
        return results

    def download_artifacts(
        self,
        run_ids: list[str],
        project: Optional[str] = None,
        output_dir: Optional[str] = None,
        artifact_type: str = "training-output",
        files: Optional[list[str]] = None,
    ) -> dict[str, list[ArtifactInfo]]:
        """Batch download artifacts for multiple runs.

        Parameters
        ----------
        run_ids : list[str]
            List of W&B run IDs.
        project : str, optional
            Project name. Falls back to self.project.
        output_dir : str, optional
            Base directory for downloads. Each run gets a subdirectory.
            Defaults to '/root/capsule/data'.
        artifact_type : str
            Artifact type filter. Default is 'training-output'.
        files : list[str], optional
            List of specific files to download from each artifact.
            If None (default), downloads all files.

        Returns
        -------
        dict[str, list[ArtifactInfo]]
            Mapping from run ID to list of downloaded artifact metadata.
        """
        results = {}
        for run_id in run_ids:
            results[run_id] = self.download_artifact(
                run_id=run_id,
                project=project,
                output_dir=output_dir,
                artifact_type=artifact_type,
                files=files,
            )
        return results
