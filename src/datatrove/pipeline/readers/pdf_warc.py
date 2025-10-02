from typing import TYPE_CHECKING, Callable, Literal
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.warc import WarcIndexReprocess
from datatrove.data import Media, MediaType


if TYPE_CHECKING:
    from warcio.recordloader import ArcWarcRecord


class PDFWarcReader(WarcIndexReprocess):
    """Read PDF files from WARC archives.

    Extends WarcIndexReprocess to filter only PDF documents.
    Records PDF-specific metadata including truncation status.

    Args:
        pdf_mime_types: MIME types to consider as PDFs
        **kwargs: Arguments passed to WarcIndexReprocess
    """

    name = "📄 PDF Warc"

    def __init__(
        self,
        data_folder: DataFolderLike,
        pdf_mime_types: list[str] = None,
        **kwargs
    ):
        if pdf_mime_types is None:
            pdf_mime_types = ["application/pdf"]
        self.pdf_mime_types = set(pdf_mime_types)
        super().__init__(data_folder, **kwargs)

    def read_file(self, filepath: str):
        """Only yield PDF documents from WARC file."""
        from warcio.archiveiterator import ArchiveIterator

        with self.data_folder.open(filepath, "rb") as f:
            archive_iterator = ArchiveIterator(f)
            for ri, record in enumerate(archive_iterator):
                offset = archive_iterator.offset
                with self.track_time():
                    name = f.path[len("commoncrawl/"):] if f.path.startswith("commoncrawl/") else f.path
                    extracted_data = process_pdf_record(record, offset, name, self.pdf_mime_types)
                    if not extracted_data:
                        continue
                    document = self.get_document_from_dict(extracted_data, filepath, ri)
                    if not document:
                        continue
                yield document


def process_pdf_record(record: "ArcWarcRecord", offset: int, name: str, pdf_mime_types: set) -> dict | None:
    """Process a WARC record to extract PDFs and metadata."""
    import magic

    # record type
    if record.rec_type not in ["response", "conversion", "resource"]:
        return None

    # Get content
    content_bytes = record.content_stream().read()
    if not content_bytes:
        return None

    # MIME type detection
    original_mime_type = record.content_type.split(";")[0] if record.content_type else None
    detected_mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)

    if detected_mime_type is None:
        detected_mime_type = magic.from_buffer(content_bytes, mime=True)

    # Filter for PDFs only
    is_pdf = (
        (detected_mime_type and detected_mime_type in pdf_mime_types) or
        (original_mime_type and original_mime_type in pdf_mime_types) or
        # URL heuristic for early crawls
        record.rec_headers.get("WARC-Target-URI", "").lower().endswith(".pdf")
    )

    if not is_pdf:
        return None

    # Truncation detection
    truncated = None
    content_truncated_field = record.rec_headers.get("WARC-Truncated")

    if content_truncated_field:
        truncated = "warc_field"
    elif len(content_bytes) == 1024 * 1024:  # 1MB truncation for older dumps
        truncated = "length"

    # Extract metadata
    id_ = record.rec_headers.get("WARC-Record-ID", f"unknown_{offset}")
    url = record.rec_headers.get("WARC-Target-URI", None)
    date = record.rec_headers.get("WARC-Date", None)

    # Handle older formats
    if not url:
        headers_dict = dict(record.rec_headers.headers)
        url = headers_dict.get("uri", None)
    if not date:
        headers_dict = dict(record.rec_headers.headers)
        date = headers_dict.get("archive-date", None)

    return {
        "text": "",  # Empty until text extraction
        "id": id_,
        "media": [
            Media(
                id=id_,
                type=MediaType.DOCUMENT,
                media_bytes=content_bytes,
                url=url,
            )
        ],
        "url": url,
        "date": date,
        "fetch_status": getattr(record.http_headers, 'statusline', None),
        "content_mime_type": original_mime_type,
        "content_mime_detected": detected_mime_type,
        "is_truncated": truncated is not None,
        "truncation_reason": truncated,
        "warc_record_offset": offset,
        "warc_filename": name,
        "content_digest": record.rec_headers.get("WARC-Payload-Digest", None),
        "content_length": len(content_bytes)
    }