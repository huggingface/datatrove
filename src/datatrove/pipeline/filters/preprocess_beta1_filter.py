from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.text import split_into_words, check_non_alpha_ratio, add_modifiers_to_meta
from datatrove.utils.typeshelper import Languages
from datatrove.pipeline.writers.disk_base import DiskWriter


def check_line_word_num(words, min_word_num: int = 3):
    return len(words) >= min_word_num


def is_line_valid(
        line: str,
        max_non_alpha_words_ratio,
        whitelist_chars,
        use_whitelist,
        min_word_num,
) -> bool:
    if line == '':
        return True
    words = split_into_words(line, Languages.english)
    if len(words) == 0:
        return False
    return check_line_word_num(words, min_word_num=min_word_num) \
        and check_non_alpha_ratio(words,
                                  max_non_alpha_words_ratio=max_non_alpha_words_ratio,
                                  whitelist_chars=whitelist_chars,
                                  use_whitelist=use_whitelist)
        # and check_pure_alpha_word_ratio(words)


def modify_doc_by_paragraph(
        doc: Document,
        valid_line_in_paragraph_ratio,
        max_non_alpha_words_ratio,
        whitelist_chars, 
        use_whitelist,
        min_word_num,        
):
    text = doc.text
    paras = text.split('\n\n')
    new_paras = []
    for para in paras:
        lines = para.split('\n')
        total_num = len(lines)
        invalid_line_num = 0
        for line in lines:
            if not is_line_valid(line,
                                 max_non_alpha_words_ratio=max_non_alpha_words_ratio,
                                 whitelist_chars=whitelist_chars,
                                 use_whitelist=use_whitelist,
                                 min_word_num=min_word_num):
                invalid_line_num += 1

        if (len(lines)-invalid_line_num) / total_num >= valid_line_in_paragraph_ratio:
            new_paras.append(para)
        

    new_text = '\n\n'.join(new_paras)
    doc.text = new_text
    if text != new_text:
        add_modifiers_to_meta(doc, 'preprocess_beta1')


class PreprocessBeta1Filter(BaseFilter):
    name = "🌍 Preprocess Beta1"
    _requires_dependencies = []
    
    def __init__(
            self,
            valid_line_in_paragraph_ratio: float = 0.5,
            max_non_alpha_words_ratio: float = 0.8,
            whitelist_chars=('(', ')', '%'),
            use_whitelist = True,
            min_word_num = 3,
            exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.valid_line_in_paragraph_ratio = valid_line_in_paragraph_ratio
        self.max_non_alpha_words_ratio = max_non_alpha_words_ratio
        self.whitelist_chars = whitelist_chars
        self.use_whitelist = use_whitelist
        self.min_word_num = min_word_num

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        modify_doc_by_paragraph(doc,
                                valid_line_in_paragraph_ratio=self.valid_line_in_paragraph_ratio,
                                max_non_alpha_words_ratio=self.max_non_alpha_words_ratio,
                                whitelist_chars=self.whitelist_chars,
                                use_whitelist=self.use_whitelist,
                                min_word_num=self.min_word_num)
        if len(doc.text) == 0:
            return False, 'preprocess_beta1_filter'
        return True
