from typing import TYPE_CHECKING, Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseReader


if TYPE_CHECKING:
    from warcio.recordloader import ArcWarcRecord


class WarcReader(BaseReader):
    name = "🕷 Warc"
    _requires_dependencies = ["warcio", ("cchardet", "faust-chardet"), ("magic", "python-magic")]

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        self.compression = compression
        super().__init__(data_folder, limit, progress, adapter, text_key, id_key, default_metadata)

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
    import cchardet
    import magic

    # record type
    if record.rec_type != "response" and record.rec_type != "conversion":  # wet files have "conversion" type
        return

    # content type filtering
    mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)
    if mime_type is not None and (
        mime_type != "text/html" and (record.rec_type != "conversion" or mime_type != "text/plain")
    ):
        return

    content_bytes = record.content_stream().read()
    if mime_type is None:
        # fallback for older crawls without payload types
        mime_type = magic.from_buffer(content_bytes, mime=True)
        if mime_type != "text/html" and (record.rec_type != "conversion" or mime_type != "text/plain"):
            return

    # Decode the response bytes
    charset = "UTF-8"
    try:
        html = content_bytes.decode(charset)
    except UnicodeDecodeError:
        encoding_det = cchardet.detect(content_bytes)["encoding"]
        if not encoding_det or encoding_det == charset:
            return
        charset = encoding_det

        try:
            html = content_bytes.decode(charset)
        except (UnicodeDecodeError, LookupError):
            return

    id = record.rec_headers["WARC-Record-ID"]
    url = record.rec_headers.get("WARC-Target-URI", None)
    date = record.rec_headers.get("WARC-Date", None)
    # handle older formats
    if not url:
        url = dict(record.rec_headers.headers)["uri"]
    if not date:
        date = dict(record.rec_headers.headers)["archive-date"]

    return {"text": html, "id": id, "url": url, "date": date}
