from typing import Any, Optional, Union
from typing_extensions import override
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, BaseListSerializer
import textwrap
from docling_core.transforms.serializer.common import create_ser_result
import numpy as np
from pathlib import Path
from tabulate import tabulate
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
    ListItem,
    PageNumberItem,
)
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    _PageBreakSerResult,
    create_ser_result,
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
        tag = "docling_picture_annotation"
        if self.filter_non_text_items_ratio and alpha_count < len(text_res) * self.filter_non_text_items_ratio:
            tag = "docling_picture_annotation_non_text"
        return create_ser_result(text=f"<{tag}>{text_res}</{tag}>", span_source=item)


from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseTextSerializer,
    BaseTableSerializer,
    BasePictureSerializer,
    SerializationResult,
)
from docling_core.types.doc.document import (
    CodeItem,
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
        text_part = item.text.strip()

        if isinstance(item, PageNumberItem):
            text_part = f"<docling_page_number>{text_part}</docling_page_number>"
        elif isinstance(item, CodeItem):
            text_part = (
                f"`{text_part}`" if is_inline_scope else f"```\n{text_part}\n```"
            )
            escape_html = False
            escape_underscores = False
        elif isinstance(item, FormulaItem):
            if text_part:
                text_part = f"${text_part}$" if is_inline_scope else f"$${text_part}$$"
            text_part = f"<docling_formula>{text_part}</docling_formula>"
            escape_html = False
            escape_underscores = False
        elif params.wrap_width:
            text_part = textwrap.fill(item.text, width=params.wrap_width)
        elif isinstance(item, ListItem):
            text_part = text_part

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

class TextListSerializer(BaseModel, BaseListSerializer):
    """Markdown-specific list serializer."""

    @override
    def serialize(
        self,
        *,
        item: Union[UnorderedList, OrderedList],
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)
        my_visited = visited if visited is not None else set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        sep = "\n"
        my_parts: list[SerializationResult] = []
        for p in parts:
            if p.text and p.text[0] == " " and my_parts:
                my_parts[-1].text = sep.join([my_parts[-1].text, p.text])  # update last
                my_parts[-1].spans.extend(p.spans)
            else:
                my_parts.append(p)

        indent_str = list_level * params.indent * " "
        is_ol = isinstance(item, OrderedList)
        text_res = sep.join(
            [
                # avoid additional marker on already evaled sublists
                (
                    c.text
                    if c.text and c.text[0] == " "
                    else (
                        f"{indent_str}"
                        f"{'' if isinstance(c, _PageBreakSerResult) else (f'{i + 1}. ' if is_ol else '')}"  # noqa: E501
                        f"{c.text}"
                    )
                )
                for i, c in enumerate(my_parts)
            ]
        )
        return create_ser_result(text=text_res, span_source=my_parts)

class TextTableSerializer(BaseTableSerializer):
    """Markdown-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            rows = [
                [
                    # make sure that md tables are not broken
                    # due to newline chars in the text
                    col.text.replace("\n", " ")
                    for col in row
                ]
                for row in item.data.grid
            ]
            if len(rows) > 1 and len(rows[0]) > 0:
                try:
                    table_text = tabulate(rows[1:], headers=rows[0], tablefmt="github")
                except ValueError:
                    table_text = tabulate(
                        rows[1:],
                        headers=rows[0],
                        tablefmt="github",
                        disable_numparse=True,
                    )
            else:
                table_text = ""

            table_text = f"<docling_table>{table_text}</docling_table>"
            if table_text:
                res_parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in res_parts])

        return create_ser_result(text=text_res, span_source=res_parts)


class TextDocSerializer(MarkdownDocSerializer):
    """Markdown-specific document serializer."""

    text_serializer: BaseTextSerializer = TextSerializer()
    list_serializer: BaseListSerializer = TextListSerializer()
    table_serializer: BaseTableSerializer = TextTableSerializer()
    picture_serializer: BasePictureSerializer = ContentPictureSerializer(filter_non_text_items_ratio=0.4)

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
