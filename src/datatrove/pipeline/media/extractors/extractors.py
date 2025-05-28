from datatrove.data import Document, Media
from datatrove.pipeline.media.extractors.media_extractor import BaseMediaExtractor
import io
import contextlib
from loguru import logger
import numpy as np
import warnings
import logging
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
    name = "PyMuPDF Extractor"
    def __init__(self, as_markdown: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.as_markdown = as_markdown
        self.logger_stream = LoggerStream(logging.getLogger("pymupdf"))

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        if not media_bytes:
            return "", {}
        try:
            metadata = {}
            import pymupdf
            pymupdf.set_messages(pylogging=True)
            with self.logger_stream as log_output:
                parsed_file = pymupdf.open(None, io.BytesIO(media_bytes))

                if self.as_markdown:
                    import pymupdf4llm
                    chunks = pymupdf4llm.to_markdown(parsed_file, page_chunks=True)
                    page_texts = [chunk["text"] for chunk in chunks]
                else:
                    page_texts = [page.get_text() for page in parsed_file]

                # Ensure it's valid utf-8
                page_texts = [text.encode('utf-8', errors='ignore').decode('utf-8', 'ignore') for text in page_texts]

                # logger_content = log_output.value()
                # if logger_content:
                #     metadata["extraction_warnings"] = [logger_content]

            length_cumsum = np.cumsum([len(page_text) for page_text in page_texts]).tolist()
            metadata["num_pages"] = len(page_texts)
            metadata["page_offsets"] = length_cumsum
            # metadata.update(_clean_metadata_value(parsed_file.metadata or {}))
            return "".join(page_texts), metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

# Helper function to recursively clean metadata values and keys
def _clean_metadata_value(value):
    from pdfminer.pdftypes import PDFStream, PDFObject, PDFObjRef
    from pypdf.generic import BooleanObject, NullObject, FloatObject
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
    elif isinstance(value, PDFStream) or isinstance(value, PDFObjRef) or isinstance(value, PDFObject):
        return None
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

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        import pypdf
        if not media_bytes:
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

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        import pdfplumber
        if not media_bytes:
            return "", {}

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
            metadata = _clean_metadata_value(metadata)
            metadata["num_pages"] = len(page_texts)
            metadata["page_offsets"] = length_cumsum

            # Clean metadata

            return "".join(page_texts), metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}


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

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        # Convert bytes to bytearray
        if not media_bytes:
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

            metadata.update({k:v for k,v in metadata.items() if k.startswith("pdf:docinfo:custom")})
            return joined_pages, metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

class PyPdfium2Extractor(BaseMediaExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        if not media_bytes:
            return "", {}

        import pypdfium2
        try:
            parsed_file = pypdfium2.PdfDocument(io.BytesIO(media_bytes))
            page_texts = [page.get_textpage().get_text_bounded() for page in parsed_file]
            length_cumsum = np.cumsum([len(page_text) for page_text in page_texts]).tolist()
            metadata = {
                "num_pages": len(page_texts),
                "page_offsets": length_cumsum,
            }
            metadata.update(parsed_file.get_metadata_dict(skip_empty=True))
            parsed_file.close()
            return "".join(page_texts), metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

class PDFMinerExtractor(BaseMediaExtractor):
    def __init__(self, *args, **kwargs):
        # https://pdfminersix.readthedocs.io/en/latest/faq.html#why-are-there-cid-x-values-in-the-textual-output
        # Careful about CID-X values in the text output
        self.logger_stream = LoggerStream(logging.getLogger("pdfminer"))
        super().__init__(*args, **kwargs, timeout=60*5)

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        if not media_bytes:
            return "", {}

        import pdfminer.high_level
        try:
            with self.logger_stream as log_output:
                text = pdfminer.high_level.extract_text(io.BytesIO(media_bytes))
                page_texts = text.split("\f")
                # Add \n to the end of each page text but not to the last one
                page_texts = [page_text + "\n" for page_text in page_texts[:-1]] + [page_texts[-1]]
                length_cumsum = np.cumsum([len(page_text) for page_text in page_texts]).tolist()
                metadata = {
                    "num_pages": len(page_texts),
                    "page_offsets": length_cumsum,
                }
                logger_content = log_output.value()
                if logger_content:
                    metadata["extraction_warnings"] = [logger_content]

                return "".join(page_texts), metadata
        except Exception as e:
            return "", {"extraction_error": str(e)}

class MinerUExtractor(BaseMediaExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(timeout=10000000)
        # Note: MinerU/magic_pdf specific logging might need configuration
        # self.logger_stream = LoggerStream(logging.getLogger("...")) # Adjust if needed

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        if not media_bytes:
            return "", {}

        import os
        import tempfile
        # Import necessary magic_pdf components locally within the method
        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        except ImportError as e:
            return "", {"extraction_error": f"MinerU dependencies not installed: {e}"}

        md_data = ""
        metadata = {}
        # Use temporary directories for image and markdown output, ensuring cleanup
        try:
            with tempfile.TemporaryDirectory() as image_output_dir, \
                 tempfile.TemporaryDirectory() as md_output_dir:

                image_writer = FileBasedDataWriter(image_output_dir)
                md_writer = FileBasedDataWriter(md_output_dir)

                # Create dataset instance from bytes
                ds = PymuDocDataset(media_bytes)

                # Inference: decide whether to run OCR mode based on dataset classification
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

                # Generate markdown content to a temporary file and read it back
                image_dir_basename = os.path.basename(image_output_dir)

                md = pipe_result.get_markdown(image_dir_basename=image_dir_basename)
                import pdb; pdb.set_trace()


            return md_data, metadata
        except Exception as e:
            # Catch potential errors during MinerU processing
            return "", {"extraction_error": f"MinerU processing failed: {e}"}


class DoclingExtractor(BaseMediaExtractor):
    def __init__(self, timeout: int = 10*60, use_table_structure: bool = False, table_acc: bool = False):
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        from docling.backend.pymupdf_backend import PyMuPdfDocumentBackend
        from docling.datamodel.settings import settings
        from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode
        docling_timeout = timeout - 2 if timeout > 2 else None
        pipeline_options = PdfPipelineOptions(
            do_table_structure=use_table_structure,
            do_ocr=False,
            document_timeout=docling_timeout,
            table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST if not table_acc else TableFormerMode.ACCURATE),
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=1)
        )
        self.logger_stream = LoggerStream(logging.getLogger("docling"))
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyMuPdfDocumentBackend)
            }
        )
        super().__init__(timeout=timeout)

    def extract(self, media_bytes: bytes) -> tuple[str, dict]:
        if media_bytes is None or len(media_bytes) == 0:
            return "", {}

        from docling.datamodel.settings import settings
        from docling_core.types.io import DocumentStream
        from docling.datamodel.base_models import ConversionStatus
        with self.logger_stream as log_output:
            doc_stream = DocumentStream(name="file.pdf", stream=io.BytesIO(media_bytes))
            converted = self.doc_converter.convert(doc_stream,raises_on_error=False)
            logger_content = log_output.value()
        page_break_placeholder = "<--- page break --->"
        full_text = converted.document.export_to_markdown(page_break_placeholder=page_break_placeholder, image_placeholder="", escape_underscores=False)
        page_list = full_text.split(page_break_placeholder)
        # Ensure all page texts are valid utf-8
        page_list = [text.encode('utf-8', errors='ignore').decode('utf-8', 'ignore') for text in page_list]
        # Remove empty strings
        metadata = {
            "num_pages": len(page_list),
            "page_offsets": np.cumsum([len(t) for t in page_list]).tolist(),
            # "docling_doc_json": converted.document.model_dump(mode="json"),
        }
        if converted.status not in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
            errors = ", ".join([e.model_dump(mode="json") for e in converted.errors])
            metadata["extraction_error"] = f"Conversion failed with status {converted.status} and errors {errors} and logger content: {logger_content}"
        return "".join(page_list), metadata


class OOMTestExtractor(BaseMediaExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self, media_bytes: bytes | None) -> tuple[str, dict]:
        if not media_bytes:
            return "", {}
        
        # ALlocate huge amount of memory
        huge_list = bytearray(10*1024*1024*1024)
        return "".join([str(i) for i in range(1000000)]), {}
