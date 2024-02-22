import os
import random
import tempfile
import time
from typing import Callable

from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    get_repo_discussions,
    preupload_lfs_files,
)
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.writers import ParquetWriter


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
        compression: str | None = None,
        adapter: Callable = None,
        cleanup: bool = True,
        expand_metadata: bool = True,
    ):
        """

        Args:
            dataset: A namespace (user or an organization) and a repo name separated by a `/`.
            private:
            local_working_dir:
            output_filename:
            compression:
            adapter:
        """
        self.dataset = dataset
        self.private = private
        self.local_working_dir = get_datafolder(
            local_working_dir if local_working_dir else tempfile.TemporaryDirectory()
        )
        self.cleanup = cleanup
        if not isinstance(self.local_working_dir.fs, LocalFileSystem):
            raise ValueError("local_working_dir must be a local path")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
            logger.warning(
                'HF_HUB_ENABLE_HF_TRANSFER is not set to "1". Install hf_transfer and set the env '
                "variable for faster uploads:\npip install hf-transfer\nexport HF_HUB_ENABLE_HF_TRANSFER=1"
            )
        super().__init__(
            output_folder=local_working_dir,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            expand_metadata=expand_metadata,
        )

    def close(self, rank: int = 0):
        repo_id = create_repo(self.dataset, private=self.private, repo_type="dataset", exist_ok=True).repo_id
        filelist = list(self._writers.keys())
        super().close()
        logger.info(f"Starting upload of {len(filelist)} files to {repo_id}")
        operations = []  # List of all `CommitOperationAdd` objects that will be generated
        for file in filelist:
            addition = CommitOperationAdd(
                path_in_repo=file, path_or_fileobj=self.local_working_dir.resolve_paths(file)
            )
            preupload_lfs_files(repo_id, repo_type="dataset", additions=[addition])
            if self.cleanup:
                self.local_working_dir.rm(file)
            operations.append(addition)
        retries = 0
        while True:
            try:
                create_commit(
                    repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"DataTrove upload ({len(filelist)} files)",
                )
                break
            except HfHubHTTPError as e:
                if "A commit has happened since" in e.server_message:
                    if retries >= MAX_RETRIES:
                        logger.error(f"Failed to create commit after {MAX_RETRIES=}. Giving up.")
                        raise e
                    logger.info("Commit creation race condition issue. Waiting...")
                    time.sleep(BASE_DELAY * 2**retries + random.uniform(0, 2))
                    retries += 1

    def _get_open_datatrove_prs(self):
        return filter(
            lambda x: x.title.startswith("DataTrove upload"),
            get_repo_discussions(
                repo_id=self.dataset,
                repo_type="dataset",
                discussion_type="pull_request",
                discussion_status="open",
            ),
        )
