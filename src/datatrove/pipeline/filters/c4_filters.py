import heapq
import re

from numpy.random import default_rng

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


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
    Args:
        exclusion_writer: optionally pass in a writer that will save the dropped documents
        tokenizer_language: load a diff language specific punkt tokenizer from nltk
        split_paragraph: by default (as in the paper) split on "\n".
            Set to "False" to apply the filters to each sentence instead of to each line
        remove_citations: remove wikipedia style citations from the text
        filter_no_terminal_punct: remove lines without terminal punctuation marks
        min_num_sentences: remove documents that do not have at least this number of sentences (after line filtering).
            set to -1 to disable
        min_words_per_line: drop lines without this min number of words
        max_word_length: drop lines where at least one word has more than this number of characters
        filter_lorem_ipsum: drop documents that contain "lorem ipsum"
        filter_javascript: drop lines mentioning "javascript"
        filter_curly_bracket: drop documents containing {
        filter_policy: drop lines containing any of the phrases in POLICY_SUBSTRINGS
    """

    name = "⛰ C4 Quality"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
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
        language: str = Languages.english,
    ):
        super().__init__(exclusion_writer)
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
        self.tokenizer = load_word_tokenizer(language)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lines = doc.text.splitlines() if self.split_paragraph else self.tokenizer.sent_tokenize(doc.text)

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
            num_sentences += len(self.tokenizer.sent_tokenize(line)) if self.split_paragraph else 1
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


_EN_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/25e679f03d96baa721cde20db9944649e8d0a844/en"
_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/5faf2ba42d7b1c0977169ec3611df25a3c08eb13/"
_BADWORDS_LANGS = [
    "ar",
    "cs",
    "da",
    "de",
    "en",
    "eo",
    "es",
    "fa",
    "fi",
    "fil",
    "fr",
    "fr-CA-u-sd-caqc",
    "hi",
    "hu",
    "it",
    "ja",
    "kab",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "th",
    "tlh",
    "tr",
    "zh",
]
# Words that are allowed since they are common subwords in languages without
# spaces. These each filter >10% of documents of their language when disallowed.
_BADWORDS_ALLOWLIST = {"ja": {"sm", "グロ", "女の子"}, "zh": {"性"}}


class C4BadWordsFilter(BaseFilter):
    """
    Badwords filter from C4.
    Args:
        keep_fraction (float): what percentage of pages containing bad words should be kept
        fail_on_missing_language (bool) whether to fail when a document has an unknown language
        seed (int): used for the uniform distribution generator for use with keep_fraction
        default_language (str): what language for samples without language in their metadata
    """

    name = "⛰ C4 Badwords"

    def __init__(
        self,
        keep_fraction: float = 0.0,
        fail_on_missing_language: bool = True,
        seed: int = None,
        default_language: str = "en",
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.keep_fraction = keep_fraction
        self.fail_on_missing_language = fail_on_missing_language
        self._badwords_regex: dict[str, re.Pattern] = {}
        self.uniform = default_rng(seed).uniform
        self.default_language = default_language

    def _get_badwords(self, lang: str):
        if lang not in self._badwords_regex:
            if lang not in _BADWORDS_LANGS:
                if self.fail_on_missing_language:
                    raise ValueError(
                        f'There is not badwords list available for "{lang}". '
                        f"Set fail_on_missing_language=False to continue anyway."
                    )
                else:
                    return None
            local_path = cached_asset_path_or_download(
                _BADWORDS_URL + lang if lang != "en" else _EN_BADWORDS_URL,
                namespace="filters",
                subfolder="c4_badwords",
            )
            badwords: set[str] = set()
            # load from file
            with open(local_path, "rt") as f:
                badwords.update(line.strip() for line in f)
            for lang, allowlist in _BADWORDS_ALLOWLIST.items():
                badwords -= allowlist

            words = [re.escape(w) for w in badwords]
            self._badwords_regex[lang] = (
                # For Japanese, Thai, and Chinese, do not require word separations.
                re.compile("|".join(words))
                if lang in ("ja", "th", "zh")
                # For other languages, match only when flanked by non-word chars.
                else re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
            )
        return self._badwords_regex[lang]

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lang: str = doc.metadata.get("language", self.default_language)
        badwords_regex = self._get_badwords(lang)
        if badwords_regex is None:
            self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
            return True
        badwords_found = badwords_regex.search(doc.text.lower())
        if badwords_found is not None:
            self.stat_update("documents_with_badwords", f"documents_with_badwords_{lang}")
            if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                self.stat_update("document_kept_with_badwords", f"document_kept_with_badwords_{lang}")
                return True
            self.stat_update(f"document_removed_with_badwords_{lang}")
            return False, "document_removed_with_badwords"
        return True
