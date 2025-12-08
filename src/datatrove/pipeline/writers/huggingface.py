import os
import random
import tempfile
import time
from typing import Any, Callable, Literal

from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    preupload_lfs_files,
)
from huggingface_hub.utils import HfHubHTTPError

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.logging import logger


MAX_RETRIES = 12
BASE_DELAY = 0.1


class HuggingFaceDatasetWriter(ParquetWriter):
    default_output_filename: str = "data/${rank}.parquet"
    name = "🤗 HuggingFace"

    def __init__(
        self,
        dataset: str,
        private: bool = True,
        local_working_dir: DataFolderLike | None = None,
        output_filename: str = None,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "snappy",
        adapter: Callable = None,
        cleanup: bool = True,
        expand_metadata: bool = True,
        max_file_size: int = round(4.5 * 2**30),  # 4.5GB, leave some room for the last batch
        schema: Any = None,
        revision: str | None = None,
    ):
        """
        This class is intended to upload VERY LARGE datasets. Consider using `push_to_hub` or just using a
        `hf://datasets/...` output path if your dataset is small enough.

        Args:
            dataset: A namespace (user or an organization) and a repo name separated by a `/`.
            private: whether to set the repo to private if it has to be created
            local_working_dir: where to save files before they are uploaded
            output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
            compression: if any compression scheme should be used. By default, "infer" - will be guessed from the filename
            adapter: a custom function to "adapt" the Document format to the desired output format
            cleanup: delete the created files from local storage after upload
            expand_metadata: save each metadata entry in a different column instead of as a dictionary
            max_file_size: will create a new file when this size is exceeded (in bytes). -1 for no limit.
                Filenames will have a number prepended (000_..., 001_..., etc)
            revision: The git revision to commit from. Defaults to the head of the `"main"` branch
        """
        self.dataset = dataset
        self.private = private
        self.local_working_dir = get_datafolder(
            local_working_dir if local_working_dir else tempfile.TemporaryDirectory()
        )
        self.cleanup = cleanup
        if not self.local_working_dir.is_local():
            raise ValueError("local_working_dir must be a local path")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
            logger.warning(
                "You should now use xet for uploads.\nSee https://hf.co/docs/huggingface_hub/en/guides/download#faster-downloads\nexport HF_HUB_ENABLE_HF_TRANSFER=0"
            )
        super().__init__(
            output_folder=local_working_dir,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            expand_metadata=expand_metadata,
            max_file_size=max_file_size,
            schema=schema,
        )
        self.operations = []
        self._repo_init = False
        self.revision = revision

    def upload_files(self, *filenames):
        if not self._repo_init:
            create_repo(self.dataset, private=self.private, repo_type="dataset", exist_ok=True)
            self._repo_init = True
        additions = [
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=self.local_working_dir.resolve_paths(filename))
            for filename in filenames
        ]
        logger.info(f"Uploading {','.join(filenames)} to the hub...")
        preupload_lfs_files(self.dataset, repo_type="dataset", additions=additions, revision=self.revision)
        logger.info(f"Upload of {','.join(filenames)} to the hub complete!")
        if self.cleanup:
            for filename in filenames:
                self.local_working_dir.rm(filename)
        self.operations.extend(additions)

    def close(self, rank: int = 0):
        filelist = list(self.output_mg.get_open_files().keys())
        super().close()
        if filelist:
            logger.info(f"Starting upload of {len(filelist)} files to {self.dataset}")
            self.upload_files(*filelist)
        retries = 0
        while True:
            try:
                create_commit(
                    self.dataset,
                    repo_type="dataset",
                    operations=self.operations,
                    commit_message=f"DataTrove upload ({len(self.operations)} files)",
                    revision=self.revision,
                )
                break
            except HfHubHTTPError as e:
                if (
                    "A commit has happened since" in e.server_message
                    or "maximum queue size reached" in e.server_message
                    or "maximum time in concurrency queue reached" in e.server_message
                ):
                    if retries >= MAX_RETRIES:
                        logger.error(f"Failed to create commit after {MAX_RETRIES=}. Giving up.")
                        raise e
                    logger.info("Commit creation race condition issue. Waiting...")
                    time.sleep(BASE_DELAY * 2**retries + random.uniform(0, 2))
                    retries += 1
                else:
                    logger.error(f"Failed to create commit: {e.server_message}")
                    raise e

    def _on_file_switch(self, original_name, old_filename, new_filename):
        """
            Called when we are switching file from "old_filename" to "new_filename" (original_name is the filename
            without 000_, 001_, etc)
        Args:
            original_name: name without file counter
            old_filename: old full filename
            new_filename: new full filename

        """
        super()._on_file_switch(original_name, old_filename, new_filename)
        self.upload_files(old_filename)


class StreamingHuggingFaceDatasetWriter(HuggingFaceDatasetWriter):
    """HuggingFaceDatasetWriter variant that pushes commits after each upload."""

    def __init__(
        self,
        *args: Any,
        commit_every_n_uploads: int = 1,
        commit_message_prefix: str = "DataTrove upload",
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the writer with incremental commit configuration.

        Args:
            commit_every_n_uploads: Number of uploaded files to batch per commit.
            commit_message_prefix:  Prefix applied to every incremental commit.
            batch_size:             Size of the batch to write to the Parquet file. Defaults to the upstream writer value when not provided.
        """
        if batch_size is not None and batch_size < 1:
            msg = "batch_size must be >= 1 when provided."
            raise ValueError(msg)
        super().__init__(*args, **kwargs)
        if batch_size is not None:
            self.batch_size = batch_size
        if commit_every_n_uploads < 1:
            msg = "commit_every_n_uploads must be at least 1."
            raise ValueError(msg)
        self.commit_every_n_uploads = commit_every_n_uploads
        self.commit_message_prefix = commit_message_prefix
        self._uploads_since_commit = 0
        logger.info(
            "Initialized streaming writer for %s (commit_every=%s, max_file_size=%s, batch_size=%s)",
            self.dataset,
            self.commit_every_n_uploads,
            self.max_file_size,
            self.batch_size,
        )

    def _commit_pending_operations(self, reason: str) -> None:
        """Push the staged operations to the Hub with retries."""
        if not self.operations:
            logger.debug("No pending operations to commit (%s).", reason)
            return
        operation_count = len(self.operations)
        logger.info(
            "Creating commit with %s pending operations for %s (reason=%s)",
            operation_count,
            self.dataset,
            reason,
        )
        retries = 0
        while True:
            try:
                create_commit(
                    self.dataset,
                    repo_type="dataset",
                    operations=self.operations,
                    commit_message=f"{self.commit_message_prefix} ({reason})",
                    revision=self.revision,
                )
                self.operations.clear()
                self._uploads_since_commit = 0
                logger.info(
                    "Commit finished for %s (reason=%s, committed_operations=%s)",
                    self.dataset,
                    reason,
                    operation_count,
                )
                break
            except HfHubHTTPError as exc:
                race_condition = False
                if hasattr(exc, "server_message"):
                    race_condition = "A commit has happened since" in exc.server_message
                elif hasattr(exc, "response") and exc.response is not None:
                    race_condition = "A commit has happened since" in exc.response.text
                if race_condition and retries < MAX_RETRIES:
                    logger.warning(
                        "Commit race detected for %s (attempt %s/%s). Retrying...",
                        self.dataset,
                        retries + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(BASE_DELAY * 2**retries + random.uniform(0, 2))
                    retries += 1
                    continue
                logger.exception("Commit failed for %s (reason=%s)", self.dataset, reason)
                raise

    def _on_file_switch(self, original_name: str, old_filename: str, new_filename: str) -> None:
        """Log file rotations to make sure uploads happen mid-run."""
        logger.info(
            "File switch triggered for %s (original=%s, old=%s, new=%s)",
            self.dataset,
            original_name,
            old_filename,
            new_filename,
        )
        super()._on_file_switch(original_name, old_filename, new_filename)

    def upload_files(self, *filenames: str) -> None:  # type: ignore[override]
        """Upload files and immediately create a commit if threshold reached."""
        if not filenames:
            logger.debug("upload_files called with no filenames for %s", self.dataset)
            return
        logger.info(
            "Uploading %s file(s) to %s: %s",
            len(filenames),
            self.dataset,
            ", ".join(filenames),
        )
        super().upload_files(*filenames)
        self._uploads_since_commit += len(filenames)
        if self._uploads_since_commit >= self.commit_every_n_uploads:
            self._commit_pending_operations(reason=f"{self._uploads_since_commit} files")

    def close(self, rank: int = 0) -> None:  # type: ignore[override]
        """Flush remaining data, upload leftovers, and ensure the final commit exists."""
        logger.info("Closing streaming writer for %s", self.dataset)
        ParquetWriter.close(self)
        filelist = list(self.output_mg.get_open_files().keys())
        if filelist:
            logger.info("Starting upload of %s files to %s", len(filelist), self.dataset)
            self.upload_files(*filelist)
        self._commit_pending_operations(reason="final chunk")
