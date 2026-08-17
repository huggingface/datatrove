import os
import random
import tempfile
import time
from typing import Any, Callable, Literal

from huggingface_hub import (
    CommitOperationAdd,
    batch_bucket_files,
    create_bucket,
    create_commit,
    create_repo,
    list_bucket_tree,
    preupload_lfs_files,
)
from huggingface_hub.utils import HfHubHTTPError

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.logging import logger


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
        save_media_bytes=False,
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
            save_media_bytes=save_media_bytes,
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
                if self.is_retryable_hf_hub_error(e):
                    if retries >= self.HF_MAX_RETRIES:
                        logger.error(f"Failed to create commit after {self.HF_MAX_RETRIES=}. Giving up.")
                        raise e
                    logger.info("Commit creation race condition issue. Waiting...")
                    time.sleep(self.HF_BASE_DELAY * 2**retries + random.uniform(0, 2))
                    retries += 1
                else:
                    logger.error(f"Failed to create commit: {e.server_message or str(e)}")
                    raise e
        self.operations = []

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


class HuggingFaceBucketWriter(ParquetWriter):
    """High-level writer for Hugging Face storage buckets.

    Buckets are mutable, S3-like object storage backed by Xet. Unlike the
    Git-based ``HuggingFaceDatasetWriter``, there are no commits or revisions:
    files are uploaded directly via ``batch_bucket_files``. This writer stages
    output locally and then pushes completed files to the bucket on rotation
    (``max_file_size``) and on ``close()``.

    Use this writer for very large raw / intermediate datasets. For published
    datasets, use ``HuggingFaceDatasetWriter`` instead.
    """

    default_output_filename: str = "data/${rank}.parquet"
    name = "🪣 HuggingFace Bucket"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        private: bool = True,
        overwrite: bool = False,
        local_working_dir: DataFolderLike | None = None,
        output_filename: str = None,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "snappy",
        adapter: Callable = None,
        cleanup: bool = True,
        expand_metadata: bool = True,
        max_file_size: int = round(4.5 * 2**30),  # 4.5GB, leave some room for the last batch
        schema: Any = None,
        save_media_bytes: bool = False,
    ):
        """
        Args:
            bucket: bucket id, e.g. ``"myorg/my-bucket"`` or ``"my-bucket"``.
            prefix: optional path prefix inside the bucket (no leading or trailing ``/``).
            private: whether to create the bucket as private if it does not exist yet.
            overwrite: if ``True``, delete all existing files under ``prefix`` before the
                first upload. Only deletes once per writer instance (safe for multi-file
                pipelines). Default is ``False`` (append).
            local_working_dir: where files are staged before upload. Must be local.
                A ``TemporaryDirectory`` is used when not provided.
            output_filename: filename template, may contain ``${rank}`` and metadata tags.
            compression: parquet compression codec.
            adapter: optional adapter function ``(self, document) -> dict``.
            cleanup: delete each local staging file after successful upload.
            expand_metadata: store each metadata key in its own column.
            max_file_size: rotate to a new file when this size (bytes) is exceeded; -1 disables.
            schema: optional pyarrow schema.
            save_media_bytes: include raw media bytes in the parquet output.
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.private = private
        self.overwrite = overwrite
        # Hold the TemporaryDirectory so it survives until ``self`` is garbage-collected.
        self._local_working_tmpdir = tempfile.TemporaryDirectory() if local_working_dir is None else None
        self.local_working_dir = get_datafolder(
            local_working_dir if local_working_dir else self._local_working_tmpdir.name
        )
        self.cleanup = cleanup
        if not self.local_working_dir.is_local():
            raise ValueError("local_working_dir must be a local path")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
            logger.warning(
                "You should now use xet for uploads.\nSee https://hf.co/docs/huggingface_hub/en/guides/download#faster-downloads\nexport HF_HUB_ENABLE_HF_TRANSFER=0"
            )
        super().__init__(
            output_folder=self.local_working_dir,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            expand_metadata=expand_metadata,
            max_file_size=max_file_size,
            schema=schema,
            save_media_bytes=save_media_bytes,
        )
        self._bucket_init = False
        self._overwrite_done = False

    def _remote_path(self, filename: str) -> str:
        """Compute the in-bucket path for a staging filename."""
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def _delete_existing_files(self) -> None:
        """Delete all existing files under ``self.prefix`` in the bucket (overwrite mode)."""
        existing = [
            entry.path
            for entry in list_bucket_tree(self.bucket, prefix=self.prefix or None, recursive=True)
            if getattr(entry, "type", None) == "file"
        ]
        if existing:
            logger.info(f"Overwrite mode: deleting {len(existing)} existing files at {self.prefix!r}")
            batch_bucket_files(self.bucket, delete=existing)

    def upload_files(self, *filenames: str) -> None:
        """Upload one or more staged files to the bucket via Xet, then optionally clean them up."""
        if not self._bucket_init:
            create_bucket(self.bucket, private=self.private, exist_ok=True)
            self._bucket_init = True
        if self.overwrite and not self._overwrite_done:
            self._delete_existing_files()
            self._overwrite_done = True
        add = [(self.local_working_dir.resolve_paths(filename), self._remote_path(filename)) for filename in filenames]
        logger.info(f"Uploading {','.join(filenames)} to bucket {self.bucket}...")
        batch_bucket_files(self.bucket, add=add)
        logger.info(f"Upload of {','.join(filenames)} to bucket {self.bucket} complete!")
        if self.cleanup:
            for filename in filenames:
                self.local_working_dir.rm(filename)

    def close(self, rank: int = 0) -> None:
        filelist = list(self.output_mg.get_open_files().keys())
        super().close()
        if filelist:
            logger.info(f"Starting upload of {len(filelist)} files to bucket {self.bucket}")
            self.upload_files(*filelist)

    def _on_file_switch(self, original_name, old_filename, new_filename):
        """When ``max_file_size`` triggers a rotation, upload the completed file immediately."""
        super()._on_file_switch(original_name, old_filename, new_filename)
        self.upload_files(old_filename)
