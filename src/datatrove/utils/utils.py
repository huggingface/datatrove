from langcodes import Language
from loguru import logger

from datatrove.data import Document


def get_language(doc: Document):
    language_id = doc.metadata.get("language")
    return Language.make(language=language_id).display_name() if language_id else "english"


def nltk_warning_msg(doc: Document):
    if "language" not in doc.metadata:
        logger.warning("⚠️ Some documents have no language id. English is assumed")
        return False
