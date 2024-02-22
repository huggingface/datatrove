import os
import tempfile
import time
from typing import Callable

from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    get_repo_discussions,
    merge_pull_request,
    preupload_lfs_files,
)
from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.writers import ParquetWriter


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
            output_folder=local_working_dir, output_filename=output_filename, compression=compression, adapter=adapter
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
        create_commit(
            repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"DataTrove upload ({len(filelist)} files)",
            create_pr=True,
        )

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

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        super(ParquetWriter, self).run(data, rank, world_size)
        if rank == 0:
            # wait for all the PRs to have been submitted
            while True:
                nr_open_datatrove_prs = len(list(self._get_open_datatrove_prs()))
                if nr_open_datatrove_prs == world_size:
                    break
                logger.info(f"Found {nr_open_datatrove_prs}/{world_size} open datatrove PRs, waiting...")
                time.sleep(30)
            # merge all the PRs
            for discussion in self._get_open_datatrove_prs():
                merge_pull_request(
                    repo_id=self.dataset,
                    repo_type="dataset",
                    discussion_num=discussion.num,
                    comment="Auto merged by DataTrove",
                )
