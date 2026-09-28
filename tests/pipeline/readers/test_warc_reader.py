import codecs
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from datatrove.pipeline.readers.warc import WarcReader, process_record


pytest.importorskip("cchardet")
pytest.importorskip("magic")
pytest.importorskip("warcio")

from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter


URL = "https://example.org/page"
DATE = "2026-09-28T00:00:00Z"
RECORD_ID = "<urn:uuid:fd771bec-06ce-4f20-a2ce-5d27e0502717>"


def make_record(
    payload: bytes,
    content_type: str | None = "text/html; charset=iso-8859-15",
    record_type: str = "response",
    mime_type: str | None = "text/html",
    record_id: str = RECORD_ID,
) -> ArcWarcRecord:
    """Create and parse a WARC member, including its real HTTP headers."""
    stream = io.BytesIO()
    writer = WARCWriter(stream)
    headers = StatusAndHeaders("200 OK", [("Content-Type", content_type)] if content_type else [], protocol="HTTP/1.1")
    warc_headers = {"WARC-Date": DATE, "WARC-Record-ID": record_id}
    if mime_type is not None:
        warc_headers["WARC-Identified-Payload-Type"] = mime_type
    record = writer.create_warc_record(
        URL,
        record_type,
        payload=io.BytesIO(payload),
        http_headers=headers if record_type == "response" else None,
        warc_headers_dict=warc_headers,
    )
    writer.write_record(record)
    stream.seek(0)
    return next(ArchiveIterator(stream))


@pytest.mark.parametrize(
    "header",
    [
        "text/html; charset=ISO-8859-15",
        'text/html; CHARSET="iso-8859-15"',
        'text/html; other="a;b"; charset=iso-8859-15',
        'text/html; charset = "iso-8859-15"',
    ],
)
def test_http_charset_precedes_heuristic_detection(header: str) -> None:
    """A valid declaration must prevent a wrong heuristic from losing or corrupting text."""
    text = "<p>Un preu de 20 € per aquesta edició.</p>"
    record = make_record(text.encode("iso8859_15"), header)
    with patch("cchardet.detect", return_value={"encoding": "VISCII"}) as detect:
        assert process_record(record) == {"text": text, "id": RECORD_ID, "url": URL, "date": DATE}
    detect.assert_not_called()


@pytest.mark.parametrize(
    "text,encoding,guess",
    [
        ("<p>Цена: 20 рублей</p>", "windows-1251", "windows-1252"),
        ("<p>m²</p>", "windows-1252", "ISO-8859-10"),
        ("<p>中文内容</p>", "gb18030", "windows-1252"),
        ("<p>日本語の文章</p>", "shift_jis", "windows-1252"),
        ("<p>العربية</p>", "windows-1256", "windows-1252"),
    ],
)
def test_declared_charset_prevents_silent_mojibake(text: str, encoding: str, guess: str) -> None:
    """Even a decodable but incorrect detector result must not replace a valid declaration."""
    with patch("cchardet.detect", return_value={"encoding": guess}) as detect:
        result = process_record(make_record(text.encode(encoding), f"text/html; charset={encoding}"))
    assert result["text"] == text
    detect.assert_not_called()


@pytest.mark.parametrize(
    "header",
    [
        None,
        "text/html",
        "text/html; charset=invalid-charset",
        "text/html; charset=utf-8",
        'text/html; charset=""',
        "text/html; charset=base64_codec",
        "text/html; charset=zlib_codec",
        "text/html; charset=windows-1252\0",
    ],
)
def test_unusable_declaration_retains_detector_fallback(header: str | None) -> None:
    """Missing, unsupported or byte-incompatible declarations keep the existing fallback."""
    text = "<p>Información</p>"
    with patch("cchardet.detect", return_value={"encoding": "windows-1252"}) as detect:
        result = process_record(make_record(text.encode("cp1252"), header))
    assert result["text"] == text
    detect.assert_called_once()


@pytest.mark.parametrize("text", ["", "<p>日本語 i català €</p>", "\ufeff<p>UTF-8 BOM</p>"])
def test_utf8_path_is_unchanged_even_with_conflicting_http(text: str) -> None:
    """Keep the established UTF-8-first behavior, including empty content and its BOM."""
    with patch("cchardet.detect") as detect:
        result = process_record(make_record(text.encode(), "text/html; charset=windows-1252"))
    assert result["text"] == text
    detect.assert_not_called()


UNICODE_BOMS = [
    ("utf-16-le", codecs.BOM_UTF16_LE, "UTF-16"),
    ("utf-16-be", codecs.BOM_UTF16_BE, "UTF-16"),
    ("utf-32-le", codecs.BOM_UTF32_LE, "UTF-32"),
    ("utf-32-be", codecs.BOM_UTF32_BE, "UTF-32"),
]


@pytest.mark.parametrize("encoding,bom,guess", UNICODE_BOMS)
@pytest.mark.parametrize("header", [None, "text/html; charset=windows-1252"])
def test_unicode_bom_retains_detector_path(encoding: str, bom: bytes, guess: str, header: str | None) -> None:
    """Transport metadata must not override the existing BOM-aware detector path."""
    text = "<p>Gràcies 日本語</p>"
    with patch("cchardet.detect", return_value={"encoding": guess}) as detect:
        result = process_record(make_record(bom + text.encode(encoding), header))
    assert result["text"] == text
    detect.assert_called_once()


@pytest.mark.parametrize("label", ["iso-8859-1", "latin1", "us-ascii"])
def test_html_legacy_labels_use_windows_1252(label: str) -> None:
    """HTML's common legacy labels must preserve C1-range punctuation and euro signs."""
    text = "<p>Prix : 20 € — édition</p>"
    with patch("cchardet.detect") as detect:
        result = process_record(make_record(text.encode("cp1252"), f"text/html; charset={label}"))
    assert result["text"] == text
    detect.assert_not_called()


@pytest.mark.parametrize("encoding,bom,guess", UNICODE_BOMS)
def test_truncated_unicode_payload_retains_detector_path(encoding: str, bom: bytes, guess: str) -> None:
    """A conflicting HTTP declaration must not mask a failed BOM-aware decode."""
    with patch("cchardet.detect", return_value={"encoding": guess}) as detect:
        assert process_record(make_record(bom + b"\0", "text/html; charset=windows-1252")) is None
    detect.assert_called_once()


def test_bom_only_payload_keeps_previous_detector_result() -> None:
    """Do not introduce a new decoding policy for BOM-only payloads without HTTP metadata."""
    with patch("cchardet.detect", return_value={"encoding": "windows-1251"}) as detect:
        result = process_record(make_record(codecs.BOM_UTF16_LE, None))
    assert result["text"] == codecs.BOM_UTF16_LE.decode("windows-1251")
    detect.assert_called_once()


def test_xhtml_keeps_literal_declared_codec() -> None:
    """The HTML Latin-1 label remapping must not be applied to XHTML/XML."""
    result = process_record(
        make_record(b"<p>\x80</p>", "application/xhtml+xml; charset=iso-8859-1", mime_type="application/xhtml+xml")
    )
    assert result["text"] == "<p>\x80</p>"


def test_wet_conversion_without_http_headers() -> None:
    """WET conversion records continue to use their existing detector fallback."""
    text = "Un preu de 20 € per aquesta edició."
    with patch("cchardet.detect", return_value={"encoding": "iso-8859-15"}):
        result = process_record(make_record(text.encode("iso8859_15"), None, "conversion", "text/plain"))
    assert result["text"] == text


@pytest.mark.parametrize("guess", [None, "UTF-8", "VISCII"])
def test_undecodable_content_still_drops(guess: str | None) -> None:
    """No declaration and no usable heuristic must still produce no document."""
    with patch("cchardet.detect", return_value={"encoding": guess}):
        assert process_record(make_record(b"\xff", None)) is None


def test_nontext_payload_filter_is_unchanged() -> None:
    """A charset declaration must not turn an image response into a text document."""
    assert process_record(make_record(b"\xff", "text/html; charset=windows-1252", mime_type="image/jpeg")) is None


@pytest.mark.parametrize("html", [True, False])
def test_older_crawl_uses_magic_before_charset(html: bool) -> None:
    """Missing WARC payload types must still use libmagic, regardless of HTTP claims."""
    text = "<!doctype html><html><body><p>m²</p></body></html>"
    payload = text.encode("cp1252") if html else b"\x89PNG\r\n\x1a\n" + b"\0" * 64
    result = process_record(make_record(payload, "text/html; charset=windows-1252", mime_type=None))
    assert result["text"] == text if html else result is None


def test_gzip_warc_reader_keeps_legacy_and_utf8_documents(tmp_path: Path) -> None:
    """Exercise file opening, WARC/HTTP parsing, decoding and Document metadata end to end."""
    expected = ["<p>Un preu de 20 € per aquesta edició.</p>", "<p>日本語</p>"]
    with (tmp_path / "sample.warc.gz").open("wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        writer.write_record(make_record(expected[0].encode("iso8859_15")))
        writer.write_record(
            make_record(
                expected[1].encode(),
                "text/html; charset=utf-8",
                record_id="<urn:uuid:0a246cbb-6f63-4bb9-9bfb-37f896878c88>",
            )
        )
    with patch("cchardet.detect", return_value={"encoding": "VISCII"}):
        docs = list(WarcReader(str(tmp_path), glob_pattern="*.warc.gz").run())
    assert [doc.text for doc in docs] == expected
    assert all(doc.metadata["url"] == URL and doc.metadata["date"] == DATE for doc in docs)
    assert len({doc.id for doc in docs}) == 2
