import codecs
from email.message import Message
from typing import TYPE_CHECKING, Callable, Literal

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


if TYPE_CHECKING:
    from warcio.recordloader import ArcWarcRecord


class WarcReader(BaseDiskReader):
    """Read data from WARC files.
        Will read each record as a separate document.

        Valid UTF-8 payloads are preserved. For other payloads without a Unicode BOM,
        supported HTTP charset declarations are tried before heuristic encoding detection.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        compression: the compression to use (default: "infer")
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "🕷 Warc"
    _requires_dependencies = ["warcio", ("cchardet", "faust-cchardet"), ("magic", "python-magic")]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        self.compression = compression
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )

    def read_file(self, filepath: str):
        from warcio.archiveiterator import ArchiveIterator

        with self.data_folder.open(filepath, "rb", compression=self.compression) as f:
            for ri, record in enumerate(ArchiveIterator(f)):
                with self.track_time():
                    extracted_data = process_record(record)
                    if not extracted_data:
                        continue
                    document = self.get_document_from_dict(extracted_data, filepath, ri)
                    if not document:
                        continue
                yield document


def process_record(record: "ArcWarcRecord") -> dict | None:
    """Process a WARC record to extract the html and metadata (id, url, date)."""
    import magic

    # record type
    if record.rec_type != "response" and record.rec_type != "conversion":  # wet files have "conversion" type
        return

    # content type filtering
    mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)
    if mime_type is not None and (
        mime_type != "text/html"
        and mime_type != "application/xhtml+xml"
        and (record.rec_type != "conversion" or mime_type != "text/plain")
    ):
        return

    content_bytes = record.content_stream().read()
    if mime_type is None:
        # fallback for older crawls without payload types
        mime_type = magic.from_buffer(content_bytes, mime=True)
        if (
            mime_type != "text/html"
            and mime_type != "application/xhtml+xml"
            and (record.rec_type != "conversion" or mime_type != "text/plain")
        ):
            return

    content_type = record.http_headers.get_header("Content-Type") if record.http_headers else None
    html = _decode_content(content_bytes, content_type, mime_type)
    if html is None:
        return

    id_ = record.rec_headers["WARC-Record-ID"]
    url = record.rec_headers.get("WARC-Target-URI", None)
    date = record.rec_headers.get("WARC-Date", None)
    # handle older formats
    if not url:
        url = dict(record.rec_headers.headers)["uri"]
    if not date:
        date = dict(record.rec_headers.headers)["archive-date"]

    return {"text": html, "id": id_, "url": url, "date": date}


def _decode_content(content: bytes, content_type: str | None, mime_type: str) -> str | None:
    """Decode a payload, retaining the UTF-8 fast path and detector fallback."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Keep BOM-marked payloads on the existing detector path; a conflicting HTTP
    # charset must not reinterpret them as a single-byte encoding.
    unicode_boms = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)
    if content_type and not content.startswith(unicode_boms):
        message = Message()
        message["content-type"] = content_type
        encoding = message.get_content_charset()
        if encoding:
            try:
                # HTML uses Windows-1252 for the legacy Latin-1 and ASCII labels.
                if mime_type == "text/html" and codecs.lookup(encoding).name in {"iso8859-1", "ascii"}:
                    encoding = "windows-1252"
                return content.decode(encoding)
            except (UnicodeError, LookupError, ValueError):
                pass

    import cchardet

    encoding = cchardet.detect(content)["encoding"]
    if not encoding or encoding == "UTF-8":
        return None
    try:
        return content.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return None
