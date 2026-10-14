"""Checksum verification and the HTTP-range zip-member fetcher, against a local server."""

from __future__ import annotations

import hashlib
import io
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from ircot import sources
from ircot.sources import ChecksumError, Pin, ZipMemberSource, verify

MEMBER = b'{"id": "2hop__1_2", "question": "Who?"}\n' * 5000


def make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/other.jsonl", b"x" * 1000)
        zf.writestr("data/musique_ans_v1.0_dev.jsonl", MEMBER)
        zf.writestr("data/after.jsonl", b"y" * 1000)
    return buf.getvalue()


ARCHIVE = make_zip()


def handler(ranges: bool) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        served: list[str] = []  # noqa: RUF012 - per-class request log for assertions
        supports_ranges = ranges

        def do_GET(self) -> None:
            header = self.headers.get("Range")
            Handler.served.append(header or "full")
            body, status = ARCHIVE, 200
            if ranges and header:
                lo, hi = (int(v) for v in header.removeprefix("bytes=").split("-"))
                body, status = ARCHIVE[lo : hi + 1], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    return Handler


@pytest.fixture(params=[True, False], ids=["ranges", "no-ranges"])
def server(request: pytest.FixtureRequest) -> Iterator[tuple[str, type[BaseHTTPRequestHandler]]]:
    h = handler(request.param)
    srv = HTTPServer(("127.0.0.1", 0), h)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}/musique.zip", h
    srv.shutdown()
    thread.join()


def source(url: str, digest: str | None = None) -> ZipMemberSource:
    pin = Pin("dev.jsonl", len(MEMBER), digest or hashlib.sha256(MEMBER).hexdigest())
    return ZipMemberSource(url, len(ARCHIVE), "data/musique_ans_v1.0_dev.jsonl", pin)


def test_member_is_extracted_and_verified(server, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    url, h = server
    monkeypatch.setattr(sources, "MUSIQUE_DEV", source(url))
    monkeypatch.setenv("IRCOT_DATA_DIR", str(tmp_path))
    path = sources.fetch_musique(lambda _: None)
    assert path.read_bytes() == MEMBER
    assert not list(path.parent.glob("*.part"))
    # With ranges only the directory, the local header and the member travel; without, the
    # fetcher falls back to one full download.
    assert ("full" in h.served) != h.supports_ranges
    again = sources.fetch_musique(lambda _: None)  # cached: verified, not downloaded
    assert again == path


def test_a_wrong_pin_fails_and_leaves_nothing_behind(server, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    url, _ = server
    monkeypatch.setattr(sources, "MUSIQUE_DEV", source(url, digest="0" * 64))
    monkeypatch.setenv("IRCOT_DATA_DIR", str(tmp_path))
    with pytest.raises(ChecksumError):
        sources.fetch_musique(lambda _: None)
    assert not sources.musique_path().exists()
    assert not list((tmp_path / "raw").glob("*.part"))


def test_verify_checks_size_then_digest(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    verify(f, Pin("f", 3, hashlib.sha256(b"abc").hexdigest()))
    with pytest.raises(ChecksumError, match="bytes"):
        verify(f, Pin("f", 4, hashlib.sha256(b"abc").hexdigest()))
    with pytest.raises(ChecksumError, match="sha256"):
        verify(f, Pin("f", 3, "0" * 64))


def test_every_pin_is_a_sha256() -> None:
    pins = [sources.MUSIQUE_DEV.pin, sources.HOTPOTQA_DEV.pin, *sources.EVALUATOR_FILES]
    pins += [*sources.BGE_SMALL.files, *sources.MINILM_SQUAD2.files]
    for pin in pins:
        assert len(pin.sha256) == 64
        int(pin.sha256, 16)
        assert pin.size >= 0
