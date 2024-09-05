from abc import ABC, abstractmethod
from typing import Callable, Iterator

from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.typeshelper import Languages


def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]


def simple_span_tokenize(text: str, sents: list[str]) -> Iterator[tuple[int, int]]:
    start_index = 0
    for sent in sents:
        start_char = text.index(sent, start_index)
        end_char = start_char + len(sent)
        start_index = end_char
        yield start_char, end_char


class WordTokenizer(ABC):
    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def sent_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        pass


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        super().__init__()
        check_required_dependencies(f"{punkt_language} word tokenizer", ["nltk"])
        self.punkt_language = punkt_language
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from nltk import load

            self._tokenizer = load(f"tokenizers/punkt/{self.punkt_language}.pickle")
        return self._tokenizer

    def word_tokenize(self, text) -> list[str]:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text, language=self.punkt_language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text, language=self.punkt_language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return list(self.tokenizer.span_tokenize(text))


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, spacy_language: str, config=None):
        super().__init__()
        check_required_dependencies(f"{spacy_language} word tokenizer", ["spacy"])
        if spacy_language == "vi":
            check_required_dependencies(f"{spacy_language} word tokenizer", ["pyvi"])
        elif spacy_language == "zh":
            check_required_dependencies(f"{spacy_language} word tokenizer", ["jieba"])
        self.spacy_language = spacy_language
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import spacy

            if self.config is None:
                self._tokenizer = spacy.blank(self.spacy_language)
            else:
                self._tokenizer = spacy.blank(self.spacy_language, config=self.config)
            self._tokenizer.add_pipe("sentencizer")
        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        tokens = [token.text for token in self.tokenizer(text, disable=["parser", "tagger", "ner"])]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        sents = [sent.text for sent in self.tokenizer(text, disable=["parser", "tagger", "ner"]).sents]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [
            (sent.start_char, sent.end_char)
            for sent in self.tokenizer(text, disable=["parser", "tagger", "ner"]).sents
        ]


class StanzaTokenizer(WordTokenizer):
    def __init__(self, stanza_language: str, **stanza_kwargs):
        super().__init__()
        check_required_dependencies(f"{stanza_language} word tokenizer", ["stanza"])
        self.stanza_language = stanza_language
        self.stanza_kwargs = stanza_kwargs
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import stanza
            from stanza.pipeline.core import DownloadMethod

            self._tokenizer = stanza.Pipeline(
                self.stanza_language,
                processors="tokenize",
                download_method=DownloadMethod.REUSE_RESOURCES,
                **self.stanza_kwargs,
            )

        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        sents = [sentence.text for sentence in doc.sentences]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        doc = self.tokenizer(text)
        return [(sent.tokens[0].start_char, sent.tokens[-1].end_char) for sent in doc.sentences]


class ThaiTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("th word tokenizer", ["pythainlp"])

    def word_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize as th_word_tokenize

        tokens = th_word_tokenize(text, keep_whitespace=False, engine="newmm-safe")
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import sent_tokenize as th_sent_tokenize

        sents = th_sent_tokenize(text)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class IndicNLPTokenizer(WordTokenizer):
    def __init__(self, language: str):
        super().__init__()
        self.language = language
        check_required_dependencies(f"{language} word tokenizer", [("indicnlp", "indic-nlp-library")])

    def word_tokenize(self, text) -> list[str]:
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize as indicnlp_trivial_tokenize

        tokens = indicnlp_trivial_tokenize(text, self.language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        sents = sentence_split(text, lang=self.language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class KiwiTokenizer(WordTokenizer):
    def __init__(self, model_type="sbg"):
        super().__init__()
        check_required_dependencies("ko word tokenizer", ["kiwipiepy"])
        self.model_type = model_type
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from kiwipiepy import Kiwi

            self._tokenizer = Kiwi(model_type=self.model_type)
        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        tokens = [token.form for token in self.tokenizer.tokenize(text)]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        sents = [sent.text for sent in self.tokenizer.split_into_sents(text)]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [(sent.start, sent.end) for sent in self.tokenizer.split_into_sents(text)]


"""
    If you know a better tokenizer or better proxy language, please submit a PR

    Notes:
        - SpaCy is generally preferred over NLTK as it is faster
        - Stanza is at least one order of magnitude slower than NLTK/SpaCy
"""

WORD_TOKENIZER_FACTORY: dict[str, Callable[[], WordTokenizer]] = {
    Languages.english: lambda: SpaCyTokenizer("en"),
    Languages.korean: lambda: KiwiTokenizer(),
    Languages.german: lambda: SpaCyTokenizer("de"),
    Languages.french: lambda: SpaCyTokenizer("fr"),
    Languages.czech: lambda: SpaCyTokenizer("cz"),
    Languages.danish: lambda: SpaCyTokenizer("da"),
    Languages.dutch: lambda: SpaCyTokenizer("nl"),
    Languages.estonian: lambda: SpaCyTokenizer("et"),
    Languages.finnish: lambda: SpaCyTokenizer("fi"),
    Languages.greek: lambda: SpaCyTokenizer("el"),
    Languages.italian: lambda: SpaCyTokenizer("it"),
    Languages.malayalam: lambda: SpaCyTokenizer("ml"),
    Languages.norwegian: lambda: SpaCyTokenizer("nb"),
    Languages.polish: lambda: SpaCyTokenizer("pl"),
    Languages.portuguese: lambda: SpaCyTokenizer("pt"),
    Languages.russian: lambda: SpaCyTokenizer("ru"),
    Languages.slovenian: lambda: SpaCyTokenizer("sl"),
    Languages.spanish: lambda: SpaCyTokenizer("es"),
    Languages.swedish: lambda: SpaCyTokenizer("sv"),
    Languages.turkish: lambda: SpaCyTokenizer("tr"),
    Languages.chinese: lambda: SpaCyTokenizer("zh", {"nlp": {"tokenizer": {"segmenter": "jieba"}}}),
    Languages.japanese: lambda: SpaCyTokenizer("ja"),  # note that there are some issues for >50k chars text
    Languages.vietnamese: lambda: SpaCyTokenizer("vi"),
    Languages.indonesian: lambda: SpaCyTokenizer("id"),
    Languages.persian: lambda: SpaCyTokenizer("fa"),
    Languages.arabic: lambda: SpaCyTokenizer("ar"),
    Languages.hindi: lambda: SpaCyTokenizer("hi"),
    Languages.tamil: lambda: SpaCyTokenizer("ta"),
    Languages.urdu: lambda: SpaCyTokenizer("ur"),
    Languages.marathi: lambda: SpaCyTokenizer("mr"),
    Languages.telugu: lambda: SpaCyTokenizer("te"),
    Languages.hungarian: lambda: SpaCyTokenizer("hu"),
    Languages.romanian: lambda: SpaCyTokenizer("ro"),
    Languages.ukrainian: lambda: SpaCyTokenizer("uk"),
    Languages.slovak: lambda: SpaCyTokenizer("sk"),
    Languages.bulgarian: lambda: SpaCyTokenizer("bg"),
    Languages.catalan: lambda: SpaCyTokenizer("ca"),
    Languages.croatian: lambda: SpaCyTokenizer("hr"),
    Languages.latin: lambda: SpaCyTokenizer("la"),
    Languages.serbian: lambda: SpaCyTokenizer("sr"),
    Languages.lithuanian: lambda: SpaCyTokenizer("lt"),
    Languages.hebrew: lambda: SpaCyTokenizer("he"),
    Languages.latvian: lambda: SpaCyTokenizer("lv"),
    Languages.icelandic: lambda: SpaCyTokenizer("is"),
    Languages.armenian: lambda: SpaCyTokenizer("hy"),
    Languages.basque: lambda: SpaCyTokenizer("eu"),
    Languages.thai: lambda: ThaiTokenizer(),
    Languages.tagalog: lambda: SpaCyTokenizer("tl"),
    Languages.albanian: lambda: SpaCyTokenizer("sq"),
    Languages.macedonian: lambda: SpaCyTokenizer("mk"),
    Languages.azerbaijani: lambda: SpaCyTokenizer("az"),
    Languages.amharic: lambda: SpaCyTokenizer("am"),
    Languages.bengali: lambda: SpaCyTokenizer("bn"),
    Languages.malay: lambda: SpaCyTokenizer("ms"),
    Languages.nepali: lambda: SpaCyTokenizer("ne"),
    Languages.kazakh: lambda: StanzaTokenizer("kk"),
    Languages.gujarati: lambda: SpaCyTokenizer("gu"),
    Languages.kannada: lambda: SpaCyTokenizer("kn"),
    Languages.welsh: lambda: StanzaTokenizer("cy"),
    Languages.norwegian_nynorsk: lambda: SpaCyTokenizer("nn"),
    Languages.sinhala: lambda: SpaCyTokenizer("si"),
    Languages.tatar: lambda: SpaCyTokenizer("tt"),
    Languages.afrikaans: lambda: SpaCyTokenizer("af"),
    Languages.kirghiz: lambda: SpaCyTokenizer("ky"),
    Languages.irish: lambda: SpaCyTokenizer("ga"),
    Languages.luxembourgish: lambda: SpaCyTokenizer("lb"),
    Languages.maltese: lambda: StanzaTokenizer("mt"),
    Languages.sanskrit: lambda: SpaCyTokenizer("sa"),
    Languages.yoruba: lambda: SpaCyTokenizer("yo"),
    Languages.serbocroatian: lambda: SpaCyTokenizer("sr"),
    Languages.oriya: lambda: IndicNLPTokenizer("or"),
    Languages.punjabi: lambda: IndicNLPTokenizer("pa"),
    Languages.assamese: lambda: IndicNLPTokenizer("as"),
    Languages.sindhi: lambda: IndicNLPTokenizer("sd"),
    Languages.belarusian: lambda: StanzaTokenizer("be"),
    Languages.tigrinya: lambda: SpaCyTokenizer("ti"),
    Languages.uyghur: lambda: StanzaTokenizer("ug"),
    Languages.tswana: lambda: SpaCyTokenizer("tn"),
    Languages.wolof: lambda: StanzaTokenizer("wo"),
    Languages.ganda: lambda: SpaCyTokenizer("lg"),
    Languages.gaelic: lambda: StanzaTokenizer("gd"),
    Languages.manx: lambda: StanzaTokenizer("gv"),
    Languages.galician: lambda: StanzaTokenizer("gl"),
    Languages.northern_sami: lambda: StanzaTokenizer("se"),
    Languages.old_bulgarian: lambda: StanzaTokenizer("cu"),
    Languages.faroese: lambda: SpaCyTokenizer("fo"),
    Languages.norwegian_bokmal: lambda: SpaCyTokenizer("nb"),
    # proxies
    Languages.bosnian: lambda: SpaCyTokenizer("hr"),  # Proxy
    Languages.esperanto: lambda: SpaCyTokenizer("en"),  # Proxy
    Languages.occitan: lambda: SpaCyTokenizer("ca"),  # Proxy
    Languages.swahili: lambda: SpaCyTokenizer("en"),  # Proxy
    Languages.javanese: lambda: SpaCyTokenizer("en"),  # Proxy
    Languages.uzbek: lambda: SpaCyTokenizer("tr"),  # Proxy, alternative ru
    Languages.tajik: lambda: SpaCyTokenizer("ru"),  # Proxy
    Languages.kurdish: lambda: SpaCyTokenizer("en"),  # Proxy, multiple scripts!
    Languages.bashkir: lambda: SpaCyTokenizer("tt"),  # Proxy
    Languages.western_frisian: lambda: SpaCyTokenizer("nl"),  # Proxy
    Languages.breton: lambda: StanzaTokenizer("cy"),  # Proxy
    Languages.malagasy: lambda: SpaCyTokenizer("en"),  # Proxy
    Languages.yiddish: lambda: SpaCyTokenizer("he"),  # Proxy
    Languages.somali: lambda: SpaCyTokenizer("en"),  # Proxy
    Languages.turkmen: lambda: SpaCyTokenizer("tr"),  # Proxy
    Languages.pashto: lambda: SpaCyTokenizer("xx"),  # Proxy. xx is "multi-language"
    # iso 639-3
    Languages.war: lambda: IndicNLPTokenizer("war"),
    Languages.northern_kurdish: lambda: StanzaTokenizer("kmr"),
    Languages.nigerian_pidgin: lambda: StanzaTokenizer("pcm"),
    Languages.gothic: lambda: StanzaTokenizer("got"),
    Languages.western_armenian: lambda: StanzaTokenizer("hyw"),
    Languages.coptic: lambda: StanzaTokenizer("cop"),
    Languages.ancient_hebrew: lambda: StanzaTokenizer("hbo"),
    Languages.old_russian: lambda: StanzaTokenizer("orv"),
    Languages.russia_buriat: lambda: StanzaTokenizer("bxr"),
    Languages.old_french: lambda: StanzaTokenizer("fro"),
    Languages.lower_sorbian: lambda: SpaCyTokenizer("dsb"),
    Languages.ancient_greek: lambda: SpaCyTokenizer("grc"),
    Languages.literary_chinese: lambda: StanzaTokenizer("lzh"),
    Languages.upper_sorbian: lambda: SpaCyTokenizer("hsb"),
    Languages.erzya: lambda: StanzaTokenizer("myv"),
    Languages.ligurian: lambda: SpaCyTokenizer("lij"),
    Languages.konkani: lambda: IndicNLPTokenizer("kok"),
    # proxies
    Languages.cebuano: lambda: SpaCyTokenizer("en"),  # Proxy
    Languages.south_azerbaijani: lambda: SpaCyTokenizer("fa"),  # Proxy
    Languages.sorani: lambda: SpaCyTokenizer("fa"),  # Proxy
}

WORD_TOKENIZER_CACHE: dict[str, WordTokenizer] = {}


def load_word_tokenizer(language: str) -> WordTokenizer:
    if language not in WORD_TOKENIZER_CACHE:
        if language not in WORD_TOKENIZER_FACTORY:
            raise ValueError(f"Language '{language}' doesn't have a tokenizer.")
        tokenizer = WORD_TOKENIZER_FACTORY[language]()
        WORD_TOKENIZER_CACHE[language] = tokenizer
    return WORD_TOKENIZER_CACHE[language]
