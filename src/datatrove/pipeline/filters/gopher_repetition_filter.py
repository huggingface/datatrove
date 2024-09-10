import re
from collections import Counter

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import PRECALCULATED_STATS, BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


"""
Table A1 from https://arxiv.org/pdf/2112.11446.pdf
    duplicate line fraction                 0.30
    duplicate paragraph fraction            0.30
    duplicate line character fraction       0.20
    duplicate paragraph character fraction  0.20

    top 2-gram character fraction           0.20
    top 3-gram character fraction           0.18
    top 4-gram character fraction           0.16

    duplicate 5-gram character fraction     0.15
    duplicate 6-gram character fraction     0.14
    duplicate 7-gram character fraction     0.13
    duplicate 8-gram character fraction     0.12
    duplicate 9-gram character fraction     0.11
    duplicate 10-gram character fraction    0.10
"""


def get_n_grams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def find_duplicates(x: list[str]) -> tuple[int, int]:
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in x:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1

        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars


def find_top_duplicate(x: list[str]) -> int:
    counter = Counter()
    for element in x:
        counter[element] += 1
    top_n_gram = counter.most_common(1)[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def find_all_duplicate(words: list[str], n: int) -> int:
    n_words = len(words)
    unique = set()
    repeated_chars, idx = 0, 0
    while idx < n_words - n + 1:
        n_gram = "".join(words[idx : idx + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)
            idx += n
        else:
            unique.add(n_gram)
            idx += 1
    assert repeated_chars <= len("".join(words))
    return repeated_chars


class GopherRepetitionFilter(BaseFilter):
    name = "👯 Gopher Repetition"

    def __init__(
        self,
        precalculated_stats: PRECALCULATED_STATS = PRECALCULATED_STATS.re_calculate_if_missing,
        dup_line_frac: float | None = 0.3,
        dup_para_frac: float | None = 0.3,
        dup_line_char_frac: float | None = 0.2,
        dup_para_char_frac: float | None = 0.2,
        top_n_grams: tuple[tuple[int, float]] = ((2, 0.2), (3, 0.18), (4, 0.16)),
        dup_n_grams: tuple[tuple[int, float]] = (
            (5, 0.15),
            (6, 0.14),
            (7, 0.13),
            (8, 0.12),
            (9, 0.11),
            (10, 0.10),
        ),
        exclusion_writer: DiskWriter = None,
        language: str = Languages.english,
    ):
        """

        Args:
            dup_line_frac:
            dup_para_frac:
            dup_line_char_frac:
            dup_para_char_frac:
            top_n_grams:
            dup_n_grams:
            exclusion_writer:
        """
        super().__init__(exclusion_writer)

        self.precalculated_stats = precalculated_stats
        self.dup_line_frac = dup_line_frac
        self.dup_para_frac = dup_para_frac
        self.dup_line_char_frac = dup_line_char_frac
        self.dup_para_char_frac = dup_para_char_frac
        self.top_n_grams = top_n_grams
        self.dup_n_grams = dup_n_grams
        self.paragraph_exp = re.compile(r"\n{2,}")
        self._line_splitter = re.compile("\n+")
        self.tokenizer = load_word_tokenizer(language)

    def _filter_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        if self.dup_para_frac:
            dup_para_frac = doc.metadata.get("gopher", {}).get("dup_para_frac")
            if dup_para_frac is None:
                logger.warning(
                    f"Missing 'dup_para_frac' in doc metadata for {doc.id}"
                    "Ensure that the previous enrisher war run with `dup_para_frac` enabled."
                )
                return False, "missing_dup_para_frac"
            if dup_para_frac > self.dup_para_frac:
                return False, "dup_para_frac"

        if self.dup_para_char_frac:
            dup_para_char_frac = doc.metadata.get("gopher", {}).get("dup_para_char_frac")
            if dup_para_char_frac is None:
                logger.warning(
                    f"Missing 'dup_para_char_frac' in doc metadata for {doc.id}"
                    "Ensure that the previous enrisher war run with `dup_para_char_frac` enabled."
                )
                return False, "missing_dup_para_char_frac"
            if dup_para_char_frac > self.dup_para_char_frac:
                return False, "dup_para_char_frac"

        if self.dup_line_frac:
            dup_line_frac = doc.metadata.get("gopher", {}).get("dup_line_frac")
            if dup_line_frac is None:
                logger.warning(
                    f"Missing 'dup_line_frac' in doc metadata for {doc.id}"
                    "Ensure that the previous enrisher war run with `dup_line_frac` enabled."
                )
                return False, "missing_dup_line_frac"
            if dup_line_frac > self.dup_line_frac:
                return False, "dup_line_frac"

        if self.dup_line_char_frac:
            dup_line_char_frac = doc.metadata.get("gopher", {}).get("dup_line_char_frac")
            if dup_line_char_frac is None:
                logger.warning(
                    f"Missing 'dup_line_char_frac' in doc metadata for {doc.id}"
                    "Ensure that the previous enrisher war run with `dup_line_char_frac` enabled."
                )
                return False, "missing_dup_line_char_frac"
            if dup_line_char_frac > self.dup_line_char_frac:
                return False, "dup_line_char_frac"

        if self.top_n_grams:
            for n, n_frac in self.top_n_grams:
                top_n_gram = doc.metadata.get("gopher", {}).get(f"top_{n}_gram")
                if top_n_gram is None:
                    logger.warning(
                        f"Missing 'top_{n}_gram' in doc metadata for {doc.id}"
                        "Ensure that the previous enrisher war run with `top_n_gram` enabled."
                    )
                    return False, "missing_top_n_gram"
                if top_n_gram > n_frac:
                    return False, f"top_{n}_gram"

        if self.dup_n_grams:
            for n, n_frac in self.dup_n_grams:
                dup_n_gram = doc.metadata.get("gopher", {}).get(f"duplicated_{n}_n_grams")
                if dup_n_gram is None:
                    logger.warning(
                        f"Missing 'duplicated_{n}_n_grams' in doc metadata for {doc.id}"
                        "Ensure that the previous enrisher war run with `duplicated_{n}_n_grams` enabled."
                    )
                    return False, "missing_duplicated_n_grams"
                if dup_n_gram > n_frac:
                    return False, f"duplicated_{n}_n_grams"

        return True

    def _filter_maybe_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        text = doc.text

        _force_recalc = False
        if self.precalculated_stats == PRECALCULATED_STATS.re_calculate:
            _force_recalc = True

        paragraphs = None
        paragraphs_duplicates = None
        para_char_duplicates = None

        if self.dup_para_frac:
            if "dup_para_frac" not in doc.metadata.get("gopher", {}) or _force_recalc:
                paragraphs = self.paragraph_exp.split(text.strip())
                paragraphs_duplicates, para_char_duplicates = find_duplicates(paragraphs)
                dup_para_frac = paragraphs_duplicates / len(paragraphs)
            else:
                dup_para_frac = doc.metadata["gopher"]["dup_para_frac"]
            if dup_para_frac > self.dup_para_frac:
                return False, "dup_para_frac"

        if self.dup_para_char_frac:
            if "dup_para_char_frac" not in doc.metadata.get("gopher", {}) or _force_recalc:
                if paragraphs is None:
                    paragraphs = self.paragraph_exp.split(text.strip())
                    paragraphs_duplicates, para_char_duplicates = find_duplicates(paragraphs)
                dup_para_char_frac = para_char_duplicates / len(text)
            else:
                dup_para_char_frac = doc.metadata["gopher"]["dup_para_char_frac"]
            if dup_para_char_frac > self.dup_para_char_frac:
                return False, "dup_para_char_frac"

        lines = None
        line_duplicates = None
        char_duplicates = None
        if self.dup_line_frac:
            if "dup_line_frac" not in doc.metadata.get("gopher", {}) or _force_recalc:
                lines = self._line_splitter.split(text)
                line_duplicates, char_duplicates = find_duplicates(lines)
                dup_line_frac = line_duplicates / len(lines)
            else:
                dup_line_frac = doc.metadata["gopher"]["dup_line_frac"]
            if dup_line_frac > self.dup_line_frac:
                return False, "dup_line_frac"

        if self.dup_line_char_frac:
            if "dup_line_char_frac" not in doc.metadata.get("gopher", {}) or _force_recalc:
                if lines is None:
                    lines = self._line_splitter.split(text)
                    line_duplicates, char_duplicates = find_duplicates(lines)
                dup_line_char_frac = char_duplicates / len(text)
            else:
                dup_line_char_frac = doc.metadata["gopher"]["dup_line_char_frac"]
            if dup_line_char_frac > self.dup_line_char_frac:
                return False, "dup_line_char_frac"

        words = None
        if self.top_n_grams:
            for n, n_frac in self.top_n_grams:
                if f"top_{n}_gram" not in doc.metadata.get("gopher", {}) or _force_recalc:
                    if words is None:
                        words = self.tokenizer.word_tokenize(text)
                    n_grams = get_n_grams(words, n)
                    if not n_grams:
                        continue
                    top_char_length = find_top_duplicate(n_grams)
                    ngram_ratio = top_char_length / len(text)
                else:
                    ngram_ratio = doc.metadata["gopher"][f"top_{n}_gram"]
                if ngram_ratio > n_frac:
                    return False, f"top_{n}_gram"

        if self.dup_n_grams:
            for n, n_frac in self.dup_n_grams:
                if f"duplicated_{n}_n_grams" not in doc.metadata.get("gopher", {}) or _force_recalc:
                    if words is None:
                        words = self.tokenizer.word_tokenize(text)
                    n_duplicates_char = find_all_duplicate(words, n)
                    ngram_ratio = n_duplicates_char / len(text)
                else:
                    ngram_ratio = doc.metadata["gopher"][f"duplicated_{n}_n_grams"]
                if ngram_ratio > n_frac:
                    return False, f"duplicated_{n}_n_grams"

        return True

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if (
            self.precalculated_stats == PRECALCULATED_STATS.re_calculate
            or self.precalculated_stats == PRECALCULATED_STATS.re_calculate_if_missing
        ):
            return self._filter_maybe_from_existing_stats(doc)
        elif self.precalculated_stats == PRECALCULATED_STATS.re_use:
            if "gopher" not in doc.metadata:
                logger.warning(f"gopher not found in metadata for {doc.id}.")
                return False, "missing_gopher_field"
            return self._filter_from_existing_stats(doc)
        else:
            raise ValueError(f"Unknown precalculated_stats: {self.precalculated_stats}")
