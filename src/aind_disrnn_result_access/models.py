"""Data models for W&B run metadata and artifacts."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunInfo:
    """Metadata for a single W&B run."""

    id: str
    name: str
    state: str
    tags: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    created_at: str = ""
    url: str = ""
    project: str = ""
    entity: str = ""


@dataclass
class ArtifactInfo:
    """Metadata for a downloaded W&B artifact."""

    name: str
    type: str
    version: str
    download_path: Path
    run_id: str
    files: list[str] = field(default_factory=list)
