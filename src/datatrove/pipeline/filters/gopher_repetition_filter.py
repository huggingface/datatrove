from collections import Counter

from nltk.tokenize import word_tokenize
import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter

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
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def find_duplicates(x: list[str]) -> tuple[int, int]:
    unique_x = set()
    duplicate_chars = 0
    for element in x:
        if element in unique_x:
            duplicate_chars += len(element)
        else:
            unique_x.add(element)
    return len(unique_x), duplicate_chars


def find_top_duplicate(x: list[str]) -> int:
    counter = Counter()
    for element in x:
        counter[element] += 1
    top_n_gram = counter.most_common()[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def find_all_duplicate(words: list[str], n: int) -> int:
    n_words = len(words)
    unique = set()
    repeated_chars, idx = 0, 0
    while idx < n_words - n + 1:
        n_gram = " ".join(words[idx:idx + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)  # TODO check
            idx += n
        else:
            unique.add(n_gram)
            idx += 1
    return repeated_chars


class GopherRepetitionFilter(BaseFilter):
    def __init__(
            self,
            dup_line_frac: float | None = 0.3,
            dup_para_frac: float | None = 0.3,
            dup_line_char_frac: float | None = 0.2,
            dup_para_char_frac: float | None = 0.2,
            top_n_grams: tuple[tuple[int, float]] = ((2, 0.2), (3, 0.18), (4, 0.16)),
            dup_n_grams: tuple[tuple[int, float]] = ((5, 0.15), (6, 0.14), (7, 0.13),
                                                     (8, 0.12), (9, 0.11), (10, 0.10)),

            **kwargs
    ):
        """

        @param kwargs:
        """
        super().__init__(**kwargs)

        self.dup_line_frac = dup_line_frac
        self.dup_para_frac = dup_para_frac
        self.dup_line_char_frac = dup_line_char_frac
        self.dup_para_char_frac = dup_para_char_frac
        self.top_n_grams = top_n_grams
        self.dup_n_grams = dup_n_grams
        self.paragraph_exp = re.compile(r"\n{2,}")
        self.name = "👯 Gopher Repetition"

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        """
        text = doc.content
        lines = text.splitlines()
        line_duplicates, char_duplicates = find_duplicates(lines)
        if self.dup_line_frac and line_duplicates / len(lines) > self.dup_line_frac:
            return False, "dup_line_frac"
        if self.dup_line_char_frac and char_duplicates / len(text) > self.dup_line_char_frac:
            return False, "dup_line_char_frac"

        paragraphs = self.paragraph_exp.split(text)
        paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)
        if self.dup_para_frac and paragraphs_duplicates / len(paragraphs) > self.dup_para_frac:
            return False, "dup_para_frac"
        if char_duplicates / len(text) > self.dup_para_char_frac:
            return False, "dup_para_char_frac"

        words = word_tokenize(text, language="english")  # TODO we should use language id filter

        for n, n_frac in self.top_n_grams:
            top_char_length = find_top_duplicate(get_n_grams(words, n))
            if top_char_length / len(text) > n_frac:
                return False, f"top_{n}_gram"

        for n, n_frac in self.dup_n_grams:
            n_duplicates_char = find_all_duplicate(words, n)
            if n_duplicates_char > n_frac:
                return False, f"duplicated_{n}_n_grams"

        return True
