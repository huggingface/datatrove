from langcodes import Language
from loguru import logger

from datatrove.data import Document


NLTK_WARNING_MSG = True


def nltk_warning_msg(doc: Document):
    global NLTK_WARNING_MSG
    if NLTK_WARNING_MSG:
        if "language" not in doc.metadata:
            logger.warning("⚠️ Some documents have no language id. English is assumed")
            NLTK_WARNING_MSG = False


def get_language(doc: Document):
    nltk_warning_msg(doc)
    language_id = doc.metadata.get("language", "en")
    return Language.make(language=language_id).display_name()
