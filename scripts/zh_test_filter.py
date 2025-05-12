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
        lines = [line for line in lines if line.strip() != ""]
        if len(lines) == 0:
            return False, "empty"
        ratio = sum(1 for line in lines if line.endswith(self.stop_chars)) / len(lines)
        if ratio < self.line_punct_thr and not (ratio == 0 and self.line_punct_exclude_zero):
            return False, "line_punct_ratio"

        ratio = sum(1 for line in lines if len(line) <= self.short_line_length) / len(lines)
        if ratio > self.short_line_threshold:
            return False, "short_line_ratio"

        ratio = find_duplicates(lines)[1] / len(doc.text.replace("\n", ""))

        if ratio > self.char_duplicates_ratio:
            return False, "char_dup_ratio"

        words = split_into_words(doc.text, self.language)
        new_line = doc.text.count("\n")
        if new_line / len(words) > self.new_line_ratio:
            return False, "list_ratio"

        return True


if __name__ == '__main__':
    from datatrove.data import Document
    docx = Document

    docx.text = """• Chevy Chase
  • Tysons Corner

Abstract

Slipman CW, Jackson HB, Lipetz JS, Chan KT, Lenrow D, Vresilovic EJ. Sacroiliac joint pain referral zones. Arch Phys Med Rehabil 2000;81:334-8. Objective: To determine the patterns of pain referral from the sacroiliac joint. Study Design: Retrospective. Participants/Methods: Fifty consecutive patients who satisfied clinical criteria and demonstrated a positive diagnostic response to a fluoroscopically guided sacroiliac joint injection were included. Each patient’s preinjection pain description was used to determine areas of pain referral, and 18 potential pain-referral zones were established. Outcome Measures: Observed areas of pain referral. Results: Eighteen men (36.0%) and 32 women (64.0%) were included with a mean age of 42.5 years (range, 20 to 75 yrs) and a mean symptom duration of 18.2 months (range, 1 to 72 mo). Forty-seven patients (94.0%) described buttock pain, and 36 patients (72.0%) described lower lumbar pain. Groin pain was described in 7 patients (14.0%). Twenty-five patients (50.0%) described associated lower-extremity pain. Fourteen patients (28.0%) described leg pain distal to the knee, and 6 patients (14.0%) reported foot pain. Eighteen patterns of pain referral were observed. A statistically significant relationship was identified between pain location and age, with younger patients more likely to describe pain distal to the knee.Conclusions: Pain referral from the sacroiliac joint does not appear to be limited to the lumbar region and buttock. The variable patterns of pain referral observed may arise for several reasons, including the joint’s complex innervation, sclerotomal pain referral, irritation of adjacent structures, and varying locations of injury with the sacroiliac joint. © 2000 by the American Congress of Rehabilitation Medicine and the American Academy of Physical Medicine and Rehabilitation

Source: http://www.archives-pmr.org/article/S0003-9993(00)90080-7/fulltext"""
    filters = FineWebQualityFilter()
    print(filters.filter(docx))


