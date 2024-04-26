import heapq
import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


POLICY_SUBSTRINGS = {
    "german": [
        "benutzungsbedingungen",
        "nutzungsbedingungen",
        "nutzungsbestimmungen",
        "datenschutzerklärung",
        "datenschutzbestimmungen",
        "datenschutzrichtlinie",
        "cookie-richtlinie",
        "verwendet cookies",
        "benutzt cookies",
        "cookies verwendet",
        "verwendung von cookies",
        "einsatz von cookies",
        "nutzung von cookies",
        "verwenden cookies",
        "benutzen cookies"
    ]
}




class MultilingualPolicyFilter(BaseFilter):
    """Applies C4 Policy filter for other languages

    - Remove lines with cookies and terms of use keywords

    Reference implementation: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/c4_utils.py#L197
    Args:
        exclusion_writer: optionally pass in a writer that will save the dropped documents
        language: used to determine policy strings and for language specific punkt tokenizer from nltk
        split_paragraph: by default (as in the paper) split on "\n".
            Set to "False" to apply the filters to each sentence instead of to each line
        min_num_sentences: remove documents that do not have at least this number of sentences (after line filtering).
            set to -1 to disable
    """

    name = "⛰ C4 Quality"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        language: str = "german",
        split_paragraph: bool = True,  # default as used on c4. Set to "False" to split with sent_tokenize
        min_num_sentences: int = 5,  # set to -1 to disableQ
        policy_strings: str = None
    ):
        super().__init__(exclusion_writer)
        self.language = language
        self.split_paragraph = split_paragraph
        self.min_num_sentences = min_num_sentences
        self.policy_strings = policy_strings if policy_strings is not None else POLICY_SUBSTRINGS[self.language]


    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        from nltk.tokenize import sent_tokenize

        lines = (
            doc.text.splitlines()
            if self.split_paragraph
            else sent_tokenize(doc.text, language=self.language)
        )

        num_sentences = 0
        kept_lines = []

        for line in lines:
            line = line.strip()
            words = line.split()
            self.stat_update("line-total")
            # check line has too long word
            line_l = line.lower()
            # lorem ipsum
            if any(p in line_l for p in self.policy_strings):
                self.stat_update("line-filter-policy")
                continue
            num_sentences += len(sent_tokenize(line, language=self.language)) if self.split_paragraph else 1
            kept_lines.append(line)
            self.stat_update("line-kept")
        if num_sentences < self.min_num_sentences:
            return False, "too_few_sentences"

        doc.text = ("\n" if self.split_paragraph else " ").join(kept_lines).strip()
        return True


class C4ParagraphFilter(BaseFilter):
    """Applies paragraph filtering from mC4

    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/c4_utils.py#L551
    """

    name = "⛰ C4 Paragraph"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

        self.min_paragraphs = 3
        self.min_paragraph_len = 200
        self.line_delimiter = "\n"

    def paragraph_filter(self, page):
        """Returns False iff a page has too few or too short paragraphs."""
        lines = page.split(self.line_delimiter)
        # Filter out docs that don't have at least three "paragraphs"
        # (lines >= `min_paragraph_len` chars).
        if (
            len(lines) < self.min_paragraphs
            or min(heapq.nlargest(3, [len(line) for line in lines])) < self.min_paragraph_len
        ):
            return False
        return True

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if not self.paragraph_filter(doc.text):
            return False, f"< {self.min_paragraphs} paragraphs"
        return True
