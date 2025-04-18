from datatrove.data import Document, Media
from datatrove.pipeline.media.extractors.media_extractor import BaseMediaExtractor
import io
import contextlib
from loguru import logger
import numpy as np
from docling.datamodel.settings import settings
import warnings
import logging
import pdfplumber
from pypdf.generic import BooleanObject, NullObject, FloatObject
from datatrove.pipeline.base import PipelineStep

def keep_only_valid_metadata(metadata: dict) -> dict:
    return {k:v for k,v in metadata.items() if v}

class StreamValue:
    def __init__(self, handler: logging.StreamHandler):
        self._handler = handler

    def value(self):
        self._handler.flush()
        value = self._handler.stream.getvalue()
        self._handler.stream.truncate(0)
        self._handler.stream.seek(0)
        return value

class LoggerStream():
    def __init__(self, loggers: logging.Logger | list[logging.Logger]):
        if isinstance(loggers, logging.Logger):
            loggers = [loggers]
        self._loggers = loggers
        self._handler = None

    @property
    def handler(self):
        if self._handler is None:
            self._handler = logging.StreamHandler(io.StringIO())
            for logger in self._loggers:
                # Remove all existing handlers
                for h in logger.handlers:
                    logger.removeHandler(h)

                logger.addHandler(self._handler)
                # Don't propagate to root logger 
                logger.propagate = False
                # Set level to DEBUG
                logger.setLevel(logging.INFO)
        return self._handler

    def __enter__(self):
        self.handler.flush()
        return StreamValue(self.handler)

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class PyMuPDFExtractor(BaseMediaExtractor):
    def __init__(self, as_markdown: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.as_markdown = as_markdown

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        try:
            if media_bytes is None or len(media_bytes) == 0:
                logger.warning(f"Media has no bytes")
                return "", {}
            metadata = {}
            import pymupdf
            parsed_file = pymupdf.open(None, io.BytesIO(media_bytes))
            if self.as_markdown:
                import pymupdf4llm
                chunks = pymupdf4llm.to_markdown(parsed_file, page_chunks=True)
                page_texts = [chunk["text"] for chunk in chunks]
            else:
                page_texts = [page.get_text() for page in parsed_file]

            length_cumsum = np.cumsum([len(page_text) for page_text in page_texts]).tolist()
            metadata["num_pages"] = len(page_texts)
            metadata["page_offsets"] = length_cumsum
            metadata.update(_clean_metadata_value(parsed_file.metadata or {}))
            return "".join(page_texts), metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

# Helper function to recursively clean metadata values and keys
def _clean_metadata_value(value):
    if isinstance(value, bytes):
        # Decode bytes to string, replacing errors
        cleaned_value = value.decode('utf-8', errors='replace')
        # Clean potential surrogates from the decoded string
        return cleaned_value.encode('utf-8', errors='replace').decode('utf-8', 'replace')
    elif isinstance(value, BooleanObject):
        # Convert pypdf BooleanObject to Python bool
        return bool(value)
    elif isinstance(value, NullObject):
        # Convert pypdf NullObject to Python None (which serializes to JSON null)
        return None
    elif isinstance(value, FloatObject):
        # Convert pypdf FloatObject to Python float
        return float(value)
    elif isinstance(value, list):
        # Recursively clean list elements
        return [_clean_metadata_value(item) for item in value]
    elif isinstance(value, tuple):
        # Recursively clean tuple elements
        return tuple(_clean_metadata_value(item) for item in value)
    elif isinstance(value, dict):
        # Recursively clean dictionary keys and values
        # Ensure keys are strings and values are cleaned
        return {str(k.lstrip("/")): _clean_metadata_value(v) for k, v in value.items()}
    elif isinstance(value, str):
        # Clean potential surrogates from existing strings
        return value.encode('utf-8', errors='replace').decode('utf-8', 'replace')
    # Add checks for other potential non-serializable types if needed
    else:
        # Return other types as is (int, float, None, etc.)
        # Note: We assume other base types like int, float are JSON serializable.
        # If pypdf returns other custom objects, more checks might be needed here.
        return value

class PyPdfExtractor(BaseMediaExtractor):
    """
    Extracts text from a PDF file using the PyPDF library.
    """
    name = "PyPDF Extractor"

    def __init__(self, remove_bytes: bool = False):
        super().__init__(remove_bytes=remove_bytes)
        self.logger_stream = LoggerStream(logging.getLogger("pypdf"))

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        import pypdf
        if media_bytes is None or len(media_bytes) == 0:
            logger.warning(f"Media has no bytes")
            return "", {}

        metadata = {}
        try:
            with warnings.catch_warnings(record=True) as caught_warnings, self.logger_stream as log_output:
                warnings.simplefilter("always")
                parsed_file = pypdf.PdfReader(io.BytesIO(media_bytes), strict=False)
                logger_content = log_output.value()
                if caught_warnings or logger_content:
                    caught_waning_content = [str(w.message) for w in caught_warnings] if caught_warnings else []
                    logger_content = [logger_content] if logger_content else []
                    metadata["extraction_warnings"] = caught_waning_content + logger_content

            page_texts = [page.extract_text(extraction_mode="plain") for page in parsed_file.pages]
            # Clean potential surrogates from page texts *before* joining using built-in methods
            cleaned_page_texts = [text.encode('utf-8', errors='replace').decode('utf-8', 'replace') for text in page_texts]
            joined_text = "".join(cleaned_page_texts)

            length_cumsum = np.cumsum([len(page_text) for page_text in cleaned_page_texts]).tolist()

            metadata["page_offsets"] = length_cumsum
            metadata["num_pages"] = len(parsed_file.pages)
            # Process metadata: use the recursive helper function
            processed_metadata = _clean_metadata_value(parsed_file.metadata)
            metadata.update(processed_metadata)

            # Return the cleaned text
            return joined_text, metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

class PdfPlumberExtractor(BaseMediaExtractor):
    def __init__(self, use_simple_extraction: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_simple_extraction = use_simple_extraction
        self.logger_stream = LoggerStream([logging.getLogger("pdfminer"), logging.getLogger("pdfplumber")])

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        from pdfminer.pdftypes import PDFStream
        try:
            metadata = {}
            with self.logger_stream as log_output:
                parsed_file = pdfplumber.open(io.BytesIO(media_bytes))
                logger_content = log_output.value()
                if logger_content:
                    metadata["extraction_warnings"] = [logger_content]

            if self.use_simple_extraction:
                page_texts = [page.extract_text_simple() for page in parsed_file.pages]
            else:
                page_texts = [page.extract_text(format="text") for page in parsed_file.pages]
            length_cumsum = np.cumsum([len(page_text) for page_text in page_texts]).tolist()
            metadata.update(parsed_file.metadata)
            metadata["num_pages"] = len(page_texts)
            metadata["page_offsets"] = length_cumsum
            # If any metadata is of Type PDFStream remove it
            metadata = {k:v for k,v in metadata.items() if not isinstance(v, PDFStream)}

            return "".join(page_texts), metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

# class Pypdfium2Extractor(BaseMediaExtractor):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def extract(self, media_bytes: bytes) -> tuple[str, list[int], dict]:
#         import pypdfium2
#         parsed_file = pypdfium2.PdfDocument(io.BytesIO(media_bytes))
#         page_texts = [page.get_textpage().get_text_bounded() for page in parsed_file]
#         length_cumsum = np.cumsum([len(page_text) for page_text in page_texts])
#         return "".join(page_texts), list(length_cumsum), {}


class ExtractousExtractor(BaseMediaExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            from extractous import Extractor, PdfParserConfig, PdfOcrStrategy
            self._extractor = Extractor().set_pdf_config(PdfParserConfig().set_ocr_strategy(PdfOcrStrategy.NO_OCR))
        return self._extractor

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        # Convert bytes to bytearray
        if len(media_bytes) == 0:
            logger.warning(f"Media has no bytes")
            return "", {}

        media_bytes = bytearray(media_bytes)
        try:
            joined_pages, metadata = self.extractor.extract_bytes_to_string(media_bytes)
            pages_cumsum = np.cumsum([int(chars) for chars in metadata["pdf:charsPerPage"]]).tolist()

            metadata = keep_only_valid_metadata({
                "format": metadata.get("pdf:format", [None])[0],
                "creator": metadata.get("dc:creator", [None])[0],
                "has_marked_content": metadata.get("pdf:hasMarkedContent", [None])[0],
                "version": metadata.get("pdf:version", [None])[0],
                "creator_tool": metadata.get("pdf:docinfo:creator_tool", [None])[0],
                "producer": metadata.get("pdf:producer", [None])[0],
                "keywords": metadata.get("pdf:docinfo:keywords", [None])[0],
                "creation_date": metadata.get("dcterms:created", [None])[0],
                "title": metadata.get("dc:title", [None])[0],
                "num_pages": len(pages_cumsum),
                "page_offsets": pages_cumsum,
            })

            misc_metadata = keep_only_valid_metadata({
                "contains_damanged_font": metadata.get("pdf:containsDamangedFont", [None])[0],
                "has_xfa": metadata.get("pdf:hasXFA", [None])[0],
                "has_xmp": metadata.get("pdf:hasXMP", [None])[0],
                "annotation_types": metadata.get("pdf:annotationTypes", [None])[0],
                "annotation_subtypes": metadata.get("pdf:annotationSubtypes", [None])[0],
            })
            metadata["misc"] = misc_metadata

            metadata.update({k:v for k,v in metadata.items() if k.startswith("pdf:docinfo:custom")})
            return joined_pages, metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

class PyPdfium2Extractor(BaseMediaExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        import pypdfium2
        try:
            parsed_file = pypdfium2.PdfDocument(io.BytesIO(media_bytes))
            page_texts = [page.get_textpage().get_text_bounded() for page in parsed_file]
            length_cumsum = np.cumsum([len(page_text) for page_text in page_texts])
            parsed_file.close()
            return "".join(page_texts), {}
        except Exception as e:
            return "", {"extraction_error": str(e)}

# class PyPdfium2Extractor(PipelineStep):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def extract(self, media_bytes: bytes) -> tuple[str, dict]:
#         import pypdfium2
#         try:
#             parsed_file = pypdfium2.PdfDocument(io.BytesIO(media_bytes))
#             page_texts = [page.get_textpage().get_text_bounded() for page in parsed_file]
#             length_cumsum = np.cumsum([len(page_text) for page_text in page_texts])
#             parsed_file.close()
#             return "".join(page_texts), {}
#         except Exception as e:
#             return "", {"extraction_error": str(e)}

#     def run(self, data, rank: int = 0, world_size: int = 1):
#         for record in data:
#             for media in record.media:
#                 media.text, media.metadata = self.extract(media.media_bytes)
#                 yield record

class Docling(PipelineStep):
    def __init__(self, *args, **kwargs):
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption, DocumentStream
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        # accelerator_options = AcceleratorOptions(
        #     num_threads=1, device=AcceleratorDevice.CPU
        # )
        pipeline_options = PdfPipelineOptions(
            do_table_structure=False,
            do_ocr=False,
            document_timeout=100
            # accelerator_options=accelerator_options,
        )
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        super().__init__(*args, **kwargs)

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        from docling.datamodel.settings import settings
        from docling_core.types.io import DocumentStream
        try:
            doc_stream = DocumentStream(name="tmp.pdf", stream=io.BytesIO(media_bytes))
            converted = self.doc_converter.convert(doc_stream,raises_on_error=False)
            pages = len(converted.document.pages)
            return converted.document.export_to_markdown(), {"num_pages": pages}
        except Exception as e:
            print(e)
            return "", {"extraction_error": str(e)}

    def run(self, data, rank: int = 0, world_size: int = 1):
        total_pages = 0
        for record in data:
            for media in record.media:
                with self.track_time():
                    media.text, media.metadata = self.extract(media.media_bytes)
                    total_pages += media.metadata.get("num_pages", 0)
            yield record
            if self.stats.time_stats.total > 0:
                print(f"Documents per second: {1/self.stats.time_stats.mean}")
                print(f"Pages per second: {total_pages/self.stats.time_stats.total}")

