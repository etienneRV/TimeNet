"""A reusable base for connectors that read files from a pinned HuggingFace Hub dataset revision.

Subclasses set ``HF_REPO``, ``SOURCE_REVISION``, and ``SOURCE_FILES``, ship a ``dataset.yaml`` card
beside the connector, and implement ``convert()``. ``download()`` returns a mapping from each part
in ``SOURCE_FILES`` to its local path. ``convert()`` receives that mapping.
:mod:`timenet_connectors.sources.huggingface_hub` describes how the files download. For the Hub's
parquet conversion, pin the commit ``refs/convert/parquet`` points to.
"""

from abc import ABC
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class BaseHuggingFaceConnector(BaseConnector[Mapping[str, Path]], ABC):
    """Base class for connectors that read files from a pinned Hub dataset revision.

    Subclasses set :attr:`HF_REPO`, :attr:`SOURCE_REVISION`, and :attr:`SOURCE_FILES`, and implement
    :meth:`convert`, which receives the local path of each part.
    """

    HF_REPO: ClassVar[str]
    SOURCE_REVISION: ClassVar[str]
    """An immutable commit: a mutable branch can silently change rebuilt datasets."""
    SOURCE_FILES: ClassVar[Mapping[str, str]]
    """The repo-relative filename of each part, in download-order. A filename can end
    in a glob like ``data/*.parquet``; the part's path is then the directory of the matching files."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Check the declaration when the class is defined
        revision = getattr(cls, "SOURCE_REVISION", None)
        if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
            raise TimeFValidationError(
                f"{cls.__name__}.SOURCE_REVISION must be a 40-character commit, got {revision!r}"
            )
        if not getattr(cls, "SOURCE_FILES", None):
            raise TimeFValidationError(f"{cls.__name__}.SOURCE_FILES must declare at least one file")

    async def download_async(self, cache_dir: Path) -> dict[str, Path]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION` concurrently.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            The local path of each part, in :attr:`SOURCE_FILES` order: a file, or the directory of a pattern part.

        Raises:
            ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
            TimeNetDownloadError: If a declared file is absent from the revision or cannot be downloaded.
        """  # ruff: ignore[docstring-extraneous-exception] - also raised by hub_snapshot
        root = await hub_snapshot(self.HF_REPO, self.SOURCE_REVISION, cache_dir, tuple(self.SOURCE_FILES.values()))
        files = {}
        for part, name in self.SOURCE_FILES.items():
            path = root / name if "*" not in name else root / Path(name).parent
            if not path.exists():
                raise TimeNetDownloadError(f"{self.HF_REPO}@{self.SOURCE_REVISION} has no {name}")
            files[part] = path
        return files
