from datatrove.data import Document


def iter_pages(document: Document):
    start = 0
    for media in document.media:
        for offset in media.metadata.get("page_offsets", []):  # type: ignore
            yield document.text[start:offset]
            start = offset
