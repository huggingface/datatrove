import re

from datatrove.data import Document
from datatrove.pipeline.enrishers.base_enrisher import BaseEnrisher
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


class C4QualityEnrisher(BaseEnrisher):
    """Applies heuristic rules from C4 https://jmlr.org/papers/volume21/20-074/20-074.pdf

    Check for the following filters and store the results in the metadata instead of dropping the documents:

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
        tokenizer_language: load a diff language specific punkt tokenizer from nltk
        split_paragraph: by default (as in the paper) split on "\n".
            Set to "False" to apply the filters to each sentence instead of to each line
        remove_citations: remove wikipedia style citations from the text
        no_terminal_punct: remove lines without terminal punctuation marks
        num_sentences: number of sentences
        words_per_line: drop lines without this min number of words
        max_word_length: drop lines where at least one word has more than this number of characters
        check_lorem_ipsum: check if documents contains "lorem ipsum"
        check_javascript: Check if lines are mentioning "javascript"
        check_curly_bracket: check documents contain {
        check_policy: check if lines contain any of the phrases in POLICY_SUBSTRINGS
    """

    name = "⛰ C4 Quality Enrisher"

    def __init__(
        self,
        split_paragraph: bool = True,  # default as used on c4. Set to "False" to split with sent_tokenize
        remove_citations: bool = True,
        no_terminal_punct: bool = True,
        num_sentences: bool = True,
        words_per_line: bool = True,
        max_word_length: bool = True,
        check_lorem_ipsum: bool = True,
        check_javascript: bool = True,
        check_curly_bracket: bool = True,
        check_policy: bool = True,
        language: str = Languages.english,
        store_lines: bool = False,
    ):
        super().__init__()
        self.split_paragraph = split_paragraph
        self.remove_citations = remove_citations
        self.no_terminal_punct = no_terminal_punct
        self.num_sentences = num_sentences
        self.words_per_line = words_per_line
        self.max_word_length = max_word_length
        self.check_lorem_ipsum = check_lorem_ipsum
        self.check_javascript = check_javascript
        self.check_curly_bracket = check_curly_bracket
        self.check_policy = check_policy
        self.tokenizer = load_word_tokenizer(language)
        self.store_lines = store_lines

    def enrish(self, doc: Document) -> Document:
        lines = doc.text.splitlines() if self.split_paragraph else self.tokenizer.sent_tokenize(doc.text)

        num_sentences = 0
        lines_metadata = []
        for line in lines:
            line = line.strip()
            words = line.split()

            self.stat_update("line-total")

            line_dict = {}
            if not words:
                if self.store_lines:
                    line_dict["line"] = line

                lines_metadata.append(line_dict)
                continue

            # check line has too long words
            if self.max_word_length:
                line_dict["max_word_length"] = max(map(len, words))

            # remove citation
            if self.remove_citations:
                line = CITATION_REGEX.sub("", line)

            # end punctuation
            if self.no_terminal_punct:
                line_dict["no_terminal_punct"] = not line.endswith(END_PUNCTUATION) or line.endswith(ELLIPSIS)

            # length of words
            if self.words_per_line:
                line_dict["words_per_line"] = len(words)

            line_lower = line.lower()
            # lorem ipsum
            if self.check_lorem_ipsum:
                line_dict["has_lorem_ipsum"] = "lorem ipsum" in line_lower

            # javascript
            if self.check_javascript:
                line_dict["has_javascript"] = "javascript" in line_lower

            # curly bracket
            if self.check_curly_bracket:
                line_dict["has_curly_bracket"] = "{" in line

            # policy
            if self.check_policy:
                line_dict["has_policy"] = any(substring in line_lower for substring in POLICY_SUBSTRINGS)

            if self.store_lines:
                line_dict["line"] = line

            lines_metadata.append(line_dict)

            if self.num_sentences:
                num_sentences += len(self.tokenizer.sent_tokenize(line)) if self.split_paragraph else 1

        doc.metadata["c4_quality"] = {
            "lines": lines_metadata,
        }
        if self.num_sentences:
            doc.metadata["c4_quality"]["num_sentences"] = num_sentences

        self.stat_update("doc-total")

        return doc
