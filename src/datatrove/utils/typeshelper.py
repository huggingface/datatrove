from dataclasses import dataclass


@dataclass
class NiceRepr:
    emoji: str
    name: str

    def __post_init__(self):
        self.name = self.name.capitalize()

    def get_name(self):
        return f"---> {self.emoji} {self.name}\n"


class Languages:
    english = "en"
    spanish = "es"
    portuguese = "pt"
    italian = "it"
    french = "fr"
    swedish = "sv"
    romanian = "ro"
    german = "de"
    latin = "la"


class StatHints:
    total = "total"
    dropped = "dropped"
    forwarded = "forwarded"


class ExtensionHelperSD:
    stage_1_signature = ".c4_sig"
    stage_2_duplicates = ".c4_dup"
    index = ".c4_index"


class ExtensionHelperES:
    stage_1_sequence = ".es_sequence"
    stage_1_sequence_size = ".es_sequence.size"
    stage_2_big_sequence = ".big_sequence"
    stage_2_bytes_offset = ".info"
    stage_3_bytes_ranges = ".bytearange"
