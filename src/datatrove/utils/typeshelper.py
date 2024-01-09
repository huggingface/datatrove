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
