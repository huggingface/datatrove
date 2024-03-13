import heapq
import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


CITATION_REGEX = re.compile(r"\[\d*]|\[edit]|\[citation needed]")
END_PUNCTUATION = (".", "?", "!", '"', "'")
ELLIPSIS = "..."
POLICY_SUBSTRINGS = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
]


class C4QualityFilter(BaseFilter):
    """Applies heuristic rules from C4 https://jmlr.org/papers/volume21/20-074/20-074.pdf

    - We only retained lines that ended in a terminal punctuation mark (! . " ?)
    - We discarded any page with fewer than 5 sentences and only retained lines that contained at least 3 words
    - [NOT IMPLEMENTED] We removed any page that contained any word on the “List of Dirty, Naughty, Obscene or Otherwise Bad Words”
    - We removed any line with the word Javascript.
    - We removed any page where the phrase “lorem ipsum” appeared
    - We removed any pages that contained a curly bracket
    Additional filters not mentioned on the list from the paper but on the code:
    - Remove lines with one word over 1000 chars
    - Remove lines with cookies and terms of use keywords



    Reference implementation: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/c4_utils.py#L197
    """

    name = "⛰ C4 Quality"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        tokenizer_language: str = "english",
        split_paragraph: bool = True,  # default as used on c4. Set to "False" to split with sent_tokenize
        remove_citations: bool = True,
        filter_no_terminal_punct: bool = True,
        min_num_sentences: int = 5,  # set to -1 to disable
        min_words_per_line: int = 3,  # set to -1 to disable
        max_word_length: int = 1000,  # set to -1 to disable
        filter_lorem_ipsum: bool = True,
        filter_javascript: bool = True,
        filter_curly_bracket: bool = True,
        filter_policy: bool = True,
    ):
        super().__init__(exclusion_writer)
        self.tokenizer_language = tokenizer_language
        self.split_paragraph = split_paragraph
        self.remove_citations = remove_citations
        self.filter_no_terminal_punct = filter_no_terminal_punct
        self.min_num_sentences = min_num_sentences
        self.min_words_per_line = min_words_per_line
        self.max_word_length = max_word_length
        self.filter_lorem_ipsum = filter_lorem_ipsum
        self.filter_javascript = filter_javascript
        self.filter_curly_bracket = filter_curly_bracket
        self.filter_policy = filter_policy

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        from nltk.tokenize import sent_tokenize

        lines = (
            doc.text.splitlines()
            if self.split_paragraph
            else sent_tokenize(doc.text, language=self.tokenizer_language)
        )

        num_sentences = 0
        kept_lines = []

        for line in lines:
            line = line.strip()
            words = line.split()
            self.stat_update("line-total")
            # check line has too long word
            if self.max_word_length != -1 and any(len(word) > self.max_word_length for word in words):
                self.stat_update("line-filter-too_long_word")
                continue
            # remove citation
            if self.remove_citations:
                line = CITATION_REGEX.sub("", line)
            # end punctuation
            if self.filter_no_terminal_punct and (not line.endswith(END_PUNCTUATION) or line.endswith(ELLIPSIS)):
                self.stat_update("line-filter-no_terminal_punc")
                continue
            # min words per line
            if len(words) < self.min_words_per_line:
                self.stat_update("line-filter-too_few_words")
                continue
            line_l = line.lower()
            # lorem ipsum
            if self.filter_lorem_ipsum and "lorem ipsum" in line_l:
                return False, "lorem_ipsum"  # drop entire doc
            # javascript
            if self.filter_javascript and "javascript" in line_l:
                self.stat_update("line-filter-javascript")
                continue
            # bracket
            if self.filter_curly_bracket and "{" in line:
                return False, "curly_bracket"  # drop entire doc
            # policy
            if self.filter_policy and any(p in line_l for p in POLICY_SUBSTRINGS):
                self.stat_update("line-filter-policy")
                continue
            num_sentences += len(sent_tokenize(line, language=self.tokenizer_language)) if self.split_paragraph else 1
            kept_lines.append(line)
            self.stat_update("line-kept")
        if num_sentences < self.min_num_sentences:
            return False, "too_few_sentences"

        doc.text = ("\n" if self.split_paragraph else " ").join(kept_lines).strip()


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
