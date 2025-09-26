import PyPDF2
from datatrove.data import DocumentsPipeline, MediaType
from datatrove.pipeline.base import PipelineStep
import io


class PdfPageCounter(PipelineStep):
    type = "⚙️ - REFINE"
    name = "🧮 PDF Page Counter"
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for record in data:
            with self.track_time():
                for media in record.media:
                    if media.type == MediaType.DOCUMENT and media.media_bytes is not None:
                        try:
                            reader = PyPDF2.PdfReader(io.BytesIO(media.media_bytes))
                            num_pages = len(reader.pages)
                            self.stat_update("pages", value=num_pages, unit="pages")
                            media.metadata["num_pages"] = num_pages
                        except Exception as e:
                            pass
            yield record


