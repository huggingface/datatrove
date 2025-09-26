from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from warcio.archiveiterator import ArchiveIterator

class WarcReader(PipelineStep):
    type = "Media Reader"
    name = "🌐 - Warc Reader"
    def __init__(self, data_folder: DataFolderLike):
        self.data_folder = get_datafolder(data_folder)
        self.current_fp = None
        self.current_file = None
        self.format = None
        super().__init__()

    def update_record(self, record: Document, warc_record):
        id_ = warc_record.rec_headers["WARC-Record-ID"]
        url = warc_record.rec_headers.get("WARC-Target-URI", None)
        date = warc_record.rec_headers.get("WARC-Date", None)
        # handle older formats
        if url is None:
            url = dict(warc_record.rec_headers.headers)["uri"]
        if date is None:
            date = dict(warc_record.rec_headers.headers)["archive-date"]

        
        content_bytes = warc_record.content_stream().read()
        record.media.append(Media(
            id=id_,
            type=MediaType.DOCUMENT,
            media_bytes=content_bytes,
            url=url,
            metadata=dict(warc_record.rec_headers.headers) | {"date": date},
        ))

        self.stat_update("media_fetched", value=1, unit="documents")
        self.stat_update("media_fetched_bytes", value=len(content_bytes), unit="bytes")
        self.update_media_stats(record.media[-1])
        return record
        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            return

        for record in data:
            with self.track_time():
                warc_file = record.metadata["warc_filename"]
                warc_record_offset = record.metadata["warc_record_offset"]

                if self.current_fp is None or self.current_file != warc_file:
                    if self.current_fp:
                        self.format = None
                        self.current_fp.close()

                    self.current_fp = self.data_folder.open(warc_file)
                    self.current_file = warc_file

                # seek to the record offset
                self.current_fp.seek(warc_record_offset)

                # read the record
                ait = ArchiveIterator(self.current_fp)
                ait.known_format = self.format
                warc_record = next(ait)
                self.format = warc_record.format
                record = self.update_record(record, warc_record)
                ait.close()
            yield record
