import json
from typing import IO, Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter
from swiss_ai.utils.language_list import LANGUAGE_CODES
from datetime import datetime

current_year = datetime.now().year


class SwissAIJsonlWriter(DiskWriter):
    """Write data to datafolder (local or remote) in JSONL format

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        compression: if any compression scheme should be used. By default, "infer" - will be guessed from the filename
        adapter: a custom function to "adapt" the Document format to the desired output format
    """

    default_output_filename: str = "${rank}.jsonl"
    name = "🐿 Jsonl"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "gzip",
        adapter: Callable = None,
    ):
        super().__init__(output_folder, output_filename=output_filename, compression=compression, adapter=adapter)

    @staticmethod
    def check_document(document: dict):
        """
        Add checks to metadata.
        Needs to have document -> metadata -> required, optional
        required:
        - language: in ISO 639-1 format (en, de, it, ...)
        - year (2014, 1992)
        - token_count (using fixed tokenizer)
        optional: can be anything

        Args:
            document:

        Returns: True if document is valid, False otherwise

        """

        if document.get('text', None) is None:
            return False

        metadata = document.get('metadata', None)
        if metadata is None or type(metadata) is not dict:
            return False

        required_metadata = metadata.get('required', None)

        if required_metadata is None or type(required_metadata) is not dict:
            return False

        required_check = SwissAIJsonlWriter._check_required_metadata(required_metadata)
        if not required_check:
            return False

        optional_metadata = metadata.get('optional')
        if optional_metadata is None or type(optional_metadata) is not dict:
            return False

        return True

    @staticmethod
    def _check_required_metadata(required_metadata: dict):
        language = required_metadata.get('language')
        if language is None or language not in LANGUAGE_CODES:
            return False

        year = required_metadata.get('year')
        if year is None or type(year) is not int or year > current_year:
            return False

        tok_count = required_metadata.get('token_count')
        if tok_count is None or type(tok_count) is not int:
            return False

        return True

    def _write(self, document: dict, file_handler: IO, _filename: str):
        passed_check = SwissAIJsonlWriter.check_document(document)
        if not passed_check:
            #TODO handle this better and give more descriptive feedback
            raise ValueError('Document is not valid')

        file_handler.write(json.dumps(document, ensure_ascii=False) + "\n")
