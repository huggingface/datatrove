from typing import Any, Optional, Union, override
from datatrove.pipeline.extractors.base import BaseExtractor
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
import textwrap
from docling_core.transforms.serializer.common import create_ser_result
import numpy as np
from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
from pathlib import Path
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
)
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    TextItem,
    PictureItem,
)
class ContentPictureSerializer(MarkdownPictureSerializer):
    def __init__(self, filter_non_text_items_ratio: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_non_text_items_ratio = filter_non_text_items_ratio
    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        text_parts: list[str] = []

        # reusing the existing result:
        parent_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **kwargs,
        )
        text_parts.append(parent_res.text)

        for it, level in doc.iterate_items(root=item, traverse_pictures=True):
            if isinstance(it, TextItem) and it.get_ref() not in item.captions:
                text_parts.append(it.text)

        text_res = (separator or "\n").join(text_parts)
        # If half of the text is non-text, return an empty string
        # Count alphabetic characters vs total length
        alpha_count = sum(c.isalpha() for c in text_res)
        if self.filter_non_text_items_ratio and alpha_count < len(text_res) * self.filter_non_text_items_ratio:
            text_res = ""
        return create_ser_result(text=text_res, span_source=item)


from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DocItem,
    DoclingDocument,
    FloatingItem,
    Formatting,
    FormItem,
    FormulaItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    NodeItem,
    OrderedList,
    PictureItem,
    PictureTabularChartData,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)
from pydantic import AnyUrl, BaseModel, PositiveInt

class TextSerializer(BaseModel, BaseTextSerializer):
    """Markdown-specific text item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []
        escape_html = True
        escape_underscores = True
        if isinstance(item, TitleItem):
            text_part = f"{item.text}"
        elif isinstance(item, SectionHeaderItem):
            text_part = f"{item.text}"
        elif isinstance(item, CodeItem):
            text_part = (
                f"`{item.text}`" if is_inline_scope else f"```\n{item.text}\n```"
            )
            escape_html = False
            escape_underscores = False
        elif isinstance(item, FormulaItem):
            if item.text:
                text_part = f"${item.text}$" if is_inline_scope else f"$${item.text}$$"
            else:
                text_part = ""
            escape_html = False
            escape_underscores = False
        elif params.wrap_width:
            text_part = textwrap.fill(item.text, width=params.wrap_width)
        else:
            text_part = item.text

        if text_part:
            text_res = create_ser_result(text=text_part, span_source=item)
            res_parts.append(text_res)

        if isinstance(item, FloatingItem):
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        text = (" " if is_inline_scope else "\n\n").join([r.text for r in res_parts])
        text = doc_serializer.post_process(
            text=text,
            escape_html=escape_html,
            escape_underscores=escape_underscores,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )
        return create_ser_result(text=text, span_source=res_parts)

class TextDocSerializer(MarkdownDocSerializer):
    """Markdown-specific document serializer."""

    text_serializer: BaseTextSerializer = TextSerializer()

    @override
    def serialize_bold(self, text: str, **kwargs: Any):
        """Apply Markdown-specific bold serialization."""
        return f"{text}"

    @override
    def serialize_italic(self, text: str, **kwargs: Any):
        """Apply Markdown-specific italic serialization."""
        return f"{text}"

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any):
        """Apply Markdown-specific strikethrough serialization."""
        return f"{text}"

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ):
        """Apply Markdown-specific hyperlink serialization."""
        return f"{text}"



class BaseDoclingSerializer(PipelineStep):
    def __init__(self, use_picture: bool = False, use_markdown: bool = True, filter_non_text_items_ratio: float = 0.5):
        self.use_picture = use_picture
        self.use_markdown = use_markdown
        self.filter_non_text_items_ratio = filter_non_text_items_ratio

    def run(self, pipeline: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # load the docling document
        page_break_placeholder = "<--- page break --->"
        params = MarkdownParams(
            page_break_placeholder=page_break_placeholder,
            image_placeholder="",
            escape_underscores=False,
        )
        for record in pipeline:
            docling_document = DoclingDocument.model_validate(record.media[0].metadata["pdf_metadata"]['docling_doc_dict'])
            if not self.use_markdown:
                serializer = TextDocSerializer(doc=docling_document, picture_serializer=ContentPictureSerializer(filter_non_text_items_ratio=self.filter_non_text_items_ratio) if self.use_picture else MarkdownPictureSerializer(), params=params)
            else:
                serializer = MarkdownDocSerializer(doc=docling_document, picture_serializer=ContentPictureSerializer(filter_non_text_items_ratio=self.filter_non_text_items_ratio) if self.use_picture else MarkdownPictureSerializer(), params=params)
            full_text = serializer.serialize().text
            page_list = full_text.split(page_break_placeholder)
            record.media[0].metadata["pdf_metadata"].update({
                "num_pages": len(page_list),
                "page_offsets": np.cumsum([len(t) for t in page_list]).tolist(),
            })
            record.text = "".join(page_list)
            yield record

