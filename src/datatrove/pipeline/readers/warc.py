from typing import Callable, Literal

import cchardet
import magic
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class WarcReader(BaseReader):
    name = "🕷️ Warc"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        compression: Literal["guess", "gzip", "zst"] | None = "guess",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
    ):
        self.compression = compression
        super().__init__(data_folder, limit, progress, adapter, content_key, id_key)

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(compression=self.compression, binary=True) as f:
            for ri, record in enumerate(ArchiveIterator(f)):
                with self.track_time():
                    extracted_data = process_record(record)
                    if not extracted_data:
                        continue
                    document = self.get_document_from_dict(extracted_data, datafile, ri)
                    if not document:
                        continue
                yield document


def process_record(record: ArcWarcRecord) -> dict | None:
    # record type
    if record.rec_type != "response":
        return

    # content type filtering
    mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)
    if mime_type is not None and mime_type != "text/html":
        return

    content_bytes = record.content_stream().read()
    if mime_type is None:
        # fallback for older crawls without payload types
        mime_type = magic.from_buffer(content_bytes, mime=True)
        if mime_type != "text/html":
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

    data_id = record.rec_headers["WARC-Record-ID"]
    url = record.rec_headers.get("WARC-Target-URI", None)
    date = record.rec_headers.get("WARC-Date", None)
    # handle older formats
    if not url:
        url = dict(record.rec_headers.headers)["uri"]
    if not date:
        date = dict(record.rec_headers.headers)["archive-date"]

    return {"content": html, "data_id": data_id, "url": url, "date": date}
