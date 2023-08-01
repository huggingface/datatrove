from langcodes import Language

from datatrove.data import Document


def get_language(doc: Document):
    language_id = doc.metadata.get("language")
    return Language.make(language=language_id).display_name() if language_id else "english"
