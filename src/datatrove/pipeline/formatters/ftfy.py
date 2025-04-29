from typing import Literal

from .base import BaseFormatter


class FTFYFormatter(BaseFormatter):
    """
    General approach: fixing unreadable/wrong encoding text is good. Enforcing a specific/strict formatting isn't.
    We want models to be able to recognize a wide variety of characters and formats, and not have a bunch of
    unseen tokens because of strict normalization.
    """

    name = "😎 FTFY"
    _requires_dependencies = ["ftfy"]

    def __init__(
        self,
        # these two border a bit on the strict formatting, but they're mostly display related
        unescape_html: str | bool = "auto",
        remove_terminal_escapes: bool = True,
        # this is the main thing we are interested in
        fix_encoding: bool = True,
        restore_byte_a0: bool = True,
        replace_lossy_sequences: bool = True,
        decode_inconsistent_utf8: bool = True,
        fix_c1_controls: bool = True,
        # all remaining ones will enforce a specific text format and models will not recognize non normalized text
        # properly
        fix_latin_ligatures: bool = False,
        fix_character_width: bool = False,  # this enforces the WRONG punctuation for chinese and japanese
        # (half-width instead of full-width)
        uncurl_quotes: bool = False,  # while it makes sense, models then don't recognize curly quotes at all
        fix_line_breaks: bool = False,  # on the fence about this one
        fix_surrogates: bool = True,  # encoding related so ok
        remove_control_chars: bool = True,  # they're literally useless
        normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = None,  # no normalization
    ):
        super().__init__()
        from ftfy import TextFixerConfig

        self.config = TextFixerConfig(
            unescape_html=unescape_html,
            remove_terminal_escapes=remove_terminal_escapes,
            fix_encoding=fix_encoding,
            restore_byte_a0=restore_byte_a0,
            replace_lossy_sequences=replace_lossy_sequences,
            decode_inconsistent_utf8=decode_inconsistent_utf8,
            fix_c1_controls=fix_c1_controls,
            fix_latin_ligatures=fix_latin_ligatures,
            fix_character_width=fix_character_width,
            uncurl_quotes=uncurl_quotes,
            fix_line_breaks=fix_line_breaks,
            fix_surrogates=fix_surrogates,
            remove_control_chars=remove_control_chars,
            normalization=normalization,
        )

    def format(self, text: str) -> str:
        import ftfy

        return ftfy.fix_text(text, config=self.config)
