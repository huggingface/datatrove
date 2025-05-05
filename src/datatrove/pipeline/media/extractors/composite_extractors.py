from io import BytesIO
from docling.document_converter import DocumentConverter, PdfFormatOption, DocumentStream
from datatrove.pipeline.media.extractors.media_extractor import BaseMediaExtractor
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from pypdfium2 import PdfiumError


class DoclingExtractor(BaseMediaExtractor):
    def __init__(self):
        super().__init__()
        pipeline_options = PdfPipelineOptions(
            do_table_structure=False,
            do_ocr=False,
        )
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        try:
            result = self.doc_converter.convert(DocumentStream(name="pog.pdf", stream=BytesIO(media_bytes)))
            return result.document.export_to_markdown(), {}
        except:
            pass



