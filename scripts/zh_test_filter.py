from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import TERMINAL_PUNCTUATION, split_into_words
from datatrove.utils.typeshelper import Languages


class FineWebQualityFilter(BaseFilter):
    name = "FineWeb Quality"

    def __init__(
            self,
            exclusion_writer: DiskWriter = None,
            line_punct_thr: float = 0.12,
            line_punct_exclude_zero: bool = False,
            stop_chars: tuple[str] | None = None,
            short_line_thr: float = 0.67,
            short_line_length: int = 30,
            char_duplicates_ratio: float = 0.01,
            new_line_ratio: float = 0.3,
            language: str = Languages.english,
    ):
        super().__init__(exclusion_writer)
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.stop_chars = stop_chars if stop_chars is not None else tuple(TERMINAL_PUNCTUATION)
        self.short_line_threshold = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio
        self.language = language

    def filter(self, doc) -> bool | tuple[bool, str]:
        lines = doc.text.split("\n")
        lines = [line for line in lines if  line.strip()]
        if len(lines) == 0:
            return False, "empty"

        ratio = sum(1 for line in lines if line.endswith(self.stop_chars)) / len(lines)
        if ratio < self.line_punct_thr and not (ratio == 0 and self.line_punct_exclude_zero):
            print(f"本文档行结束标点符号阈值{ratio}，标准阈值{self.line_punct_thr}")
            return False, "line_punct_ratio"

        ratio = sum(1 for line in lines if len(line) <= self.short_line_length) / len(lines)
        if ratio > self.short_line_threshold:
            return False, "short_line_ratio"

        ratio = find_duplicates(lines)[1] / len(doc.text.replace("\n", ""))
        print(f"重复率：{ratio*100}%，标准：1%")
        if ratio > self.char_duplicates_ratio:
            return False, "char_dup_ratio"

        words = split_into_words(doc.text, self.language)
        new_line = doc.text.count("\n")
        if new_line / len(words) > self.new_line_ratio:
            return False, "list_ratio"

        return True


if __name__ == '__main__':
    from datatrove.data import Document
    from datatrove.pipeline.filters.preprocess_beta2_filter import RepeatingRowsFilter
    rp_filter = RepeatingRowsFilter()
    
    docx = Document
    origin_text = """You are here: ALUMNI & COMMUNITY
Saturday, 18. November 2017

Notes & Dates

International Planning Sessions Winter 2017/18

?

Upcoming Holiday at TU Dortmund University:
25.12.2017-05.01.2018 - Christmas Break

_______________________

Master Programmes in Germany on Urban-, Regional-, and? Infrastructure Planning

_______________________

_______________________

Alumni & Community

SIADP - SPRING International Association of Development Planners

SIADP - SPRING International Association for Development Planners

Please also visit the SIADP-Website

Since the first SPRING batch completed the course almost 700 educated planners from more than 70 countries returned to their home countries and most of them work in the field of development.

To keep the ties to each other and to facilitate communication among all we founded SIADP in 1991. It provides the opportunity to discuss news and to spread developments which are interesting for third world planners.

Alumni meetings have been organised in Indonesia, Philippines, Tanzania, Ghana, and Ethiopia. SIADP helps the members to look for jobs, keep contacts and with general information by distributing the Newsletter once a year.

All requests about SIADP Organisation to:

Dr. Anne Weber:

?"""
    docx.text = origin_text
    flag, doc_text = rp_filter.filter(docx)
    print(len(doc_text.split('\n')))
    print(len(origin_text.split('\n')))
    fb_filters = FineWebQualityFilter()
    print(fb_filters.filter(docx))


