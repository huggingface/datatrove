import os
import random
import tempfile
import time
from typing import Callable

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
        compression: str | None = None,
        adapter: Callable = None,
        cleanup: bool = True,
        expand_metadata: bool = True,
        max_file_size: int = round(4.5 * 2**30),  # 4.5GB, leave some room for the last batch
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
        """
        self.dataset = dataset
        self.private = private
        self.local_working_dir = get_datafolder(
            local_working_dir if local_working_dir else tempfile.TemporaryDirectory()
        )
        self.cleanup = cleanup
        if not self.local_working_dir.is_local():
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
            max_file_size=max_file_size,
        )
        self.operations = []
        self._repo_init = False

    def upload_files(self, *filenames):
        if not self._repo_init:
            create_repo(self.dataset, private=self.private, repo_type="dataset", exist_ok=True)
            self._repo_init = True
        additions = [
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=self.local_working_dir.resolve_paths(filename))
            for filename in filenames
        ]
        logger.info(f"Uploading {','.join(filenames)} to the hub...")
        preupload_lfs_files(self.dataset, repo_type="dataset", additions=additions)
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
