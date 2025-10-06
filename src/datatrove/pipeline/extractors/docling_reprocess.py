from abc import abstractmethod
from multiprocessing import Pipe, Process
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import StatHints
from datatrove.pipeline.extractors.base import ExtractorSandbox
from docling_ibm_models.reading_fixer.reading_fix import ReadingOrderFixer
from docling_ibm_models.text_fix.text_fix import TextFix
from docling_ibm_models.para_fix.para_fix import ParagraphFixer
from docling_ibm_models.listitem_normalizer.list_marker_normalizer import ListItemMarkerProcessor
from docling_core.types.doc.document import DoclingDocument


class DoclingReprocessor(PipelineStep):
    """Reprocesses Docling documents"""

    type = "🛢 - REPROCESS Docling"

    @abstractmethod
    def __init__(self, timeout: float = 1):
        """

        Args:
            timeout: the timeout for extraction, per document, in seconds
        """
        super().__init__()
        self.timeout = timeout
        self.text_fix = TextFix()
        self.reading_order_fixer = ReadingOrderFixer()
        self.paragraph_fixer = ParagraphFixer()
        self.list_item_marker_processor = ListItemMarkerProcessor()

    def extract(self, docling_document: dict) -> str:
        if not docling_document:
            return ""

        docling_document_parsed = DoclingDocument.model_validate(docling_document)

        docling_document_parsed = self.text_fix.process_document(docling_document_parsed)
        docling_document_parsed = self.reading_order_fixer.process_document(docling_document_parsed)
        docling_document_parsed = self.paragraph_fixer.process_document(docling_document_parsed)
        docling_document_parsed = self.list_item_marker_processor.process_document(docling_document_parsed)

        return docling_document_parsed.to_dict()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document in data and calls `timeout_extract` on it.

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        with ExtractorSandbox(timeout=self.timeout, wamup_text={}) as extractor:
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    try:
                        doc.text = extractor.process_document(doc.text, self.extract)
                        self.stat_update("extracted")
                    except TimeoutError:
                        self.stat_update("timeout")
                        logger.warning("⏰ Timeout while cleaning record text. Skipping record.")
                        continue
                    except EOFError:
                        # Process died unexpectedly
                        self.stat_update("broken_process")
                        logger.warning("Process died unexpectedly, will create new process for next document")
                        continue
                    except Exception as e:
                        self.stat_update("clean_error")
                        if not self._warned_error:
                            logger.warning(
                                f'❌ Error "{e}" while cleaning record text. Skipping record. '
                                f"This message will only appear once."
                            )
                            self._warned_error = True
                        continue

                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)

