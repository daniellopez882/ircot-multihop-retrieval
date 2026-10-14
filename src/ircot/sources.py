"""Pinned datasets, models and the official MuSiQue evaluator, with checksum-verified fetching.

Every file is pinned by size and sha256. The values were recorded on the first download and a
mismatch is an error, never a warning, so a run manifest that names these files names their bytes.

MuSiQue is only distributed as one 272 MB zip on Google Drive (linked from the official
repository). Only the 30 MB answerable dev split is needed, so the fetcher reads the zip's central
directory with HTTP range requests and inflates just that member; if the server ignores ranges it
falls back to downloading the archive to a temporary file.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import tempfile
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ircot.paths import evaluator_dir, models_dir, raw_dir

Log = Callable[[str], None]
USER_AGENT = "ircot-multihop-retrieval/0.1"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Pin:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ZipMemberSource:
    url: str
    archive_size: int
    member: str
    pin: Pin


@dataclass(frozen=True)
class UrlSource:
    url: str
    pin: Pin


@dataclass(frozen=True)
class HubModel:
    repo: str
    revision: str
    files: tuple[Pin, ...]
    license: str

    def pin(self, name: str) -> Pin:
        return next(p for p in self.files if p.name == name)


MUSIQUE_DEV = ZipMemberSource(
    url=(
        "https://drive.usercontent.google.com/download"
        "?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"
    ),
    archive_size=272_049_578,
    member="data/musique_ans_v1.0_dev.jsonl",
    pin=Pin(
        "musique_ans_v1.0_dev.jsonl",
        30_439_728,
        "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    ),
)

HOTPOTQA_DEV = UrlSource(
    url=(
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
        "1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet"
    ),
    pin=Pin(
        "hotpotqa_distractor_validation.parquet",
        27_452_575,
        "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    ),
)

BGE_SMALL = HubModel(
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files=(
        Pin(
            "onnx/model.onnx",
            133_093_490,
            "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        ),
        Pin(
            "tokenizer.json",
            711_396,
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        ),
    ),
    license="MIT",
)

MINILM_SQUAD2 = HubModel(
    repo="deepset/minilm-uncased-squad2",
    revision="934656cdda79824eabf503ed56e15c01ddbdbe3f",
    files=(
        Pin(
            "model.safetensors",
            133_466_376,
            "240478c7251d5e55be35feb47177054b960259b9328af662ea3cee87ce9755a6",
        ),
        Pin(
            "config.json",
            477,
            "688d980e03e028843d378fb3bf9919dc4b677b24e2a6ef37c97815d5c056151b",
        ),
        Pin(
            "vocab.txt",
            231_508,
            "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        ),
    ),
    license="CC BY 4.0",
)

EVALUATOR_BASE = (
    "https://raw.githubusercontent.com/StonyBrookNLP/musique/"
    "922ac98f19a201998dbdae6d7f2887a5258dbdeb/"
)
EVALUATOR_FILES = (
    Pin(
        "evaluate_v1.0.py", 4606, "f5fe66ae61dbea5172cba9d428d9924a5811f3457edc945fc1d81369c30e74b7"
    ),
    Pin(
        "metrics/__init__.py", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    Pin(
        "metrics/answer.py",
        2837,
        "10368f619b4d5ef5d83748c05a96c0afd332a14ab5c010740c98d58dfaefe974",
    ),
    Pin(
        "metrics/metric.py", 628, "c858d1bfda2f0b005065e87a402cd2f82154eb7eed5916845ae130759cc3a299"
    ),
    Pin(
        "metrics/group.py", 1152, "b7bff5d19b426a353e50ad07db125fdb6fdd2660a229e1b11752b63e4ebd9087"
    ),
    Pin(
        "metrics/support.py",
        2315,
        "ac16c0daf458a6a4d6db97682c2340fe5b5936a947bf32c04dc3bf16406077c6",
    ),
    Pin(
        "metrics/group_answer_sufficiency.py",
        2320,
        "6fd72646055b736cfae474eba58f7d4823ee8f539c0746bb87dd5c8208d7abcc",
    ),
    Pin(
        "metrics/group_support_sufficiency.py",
        2546,
        "ab6f9aa92243190b891d28a1787781029bd808616e347cbf325526bffcc10820",
    ),
)

READER_ONNX_NAME = "minilm-uncased-squad2.onnx"


class ChecksumError(RuntimeError):
    pass


class MissingAssetError(FileNotFoundError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, pin: Pin) -> None:
    size = path.stat().st_size
    if size != pin.size:
        raise ChecksumError(f"{path}: expected {pin.size} bytes, found {size}")
    digest = sha256_file(path)
    if digest != pin.sha256:
        raise ChecksumError(f"{path}: sha256 {digest} does not match the pinned {pin.sha256}")


def _open(url: str, byte_range: tuple[int, int] | None = None):  # type: ignore[no-untyped-def]
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def _read_range(url: str, start: int, end: int) -> bytes | None:
    """Bytes ``start..end`` inclusive, or None when the server answers without range support."""
    with _open(url, (start, end)) as resp:
        if resp.status != 206:
            return None
        return resp.read()


def _download(url: str, dest: Path, pin: Pin, log: Log) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"download  {url}")
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, _open(url) as resp:
            shutil.copyfileobj(resp, out, length=CHUNK)
        verify(tmp, pin)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


@dataclass(frozen=True)
class _CentralEntry:
    method: int
    crc32: int
    compressed: int
    uncompressed: int
    header_offset: int


def _central_directory(url: str, archive_size: int) -> dict[str, _CentralEntry] | None:
    tail_len = min(archive_size, 65_557)
    tail = _read_range(url, archive_size - tail_len, archive_size - 1)
    if tail is None:
        return None
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ChecksumError("end of central directory not found; the archive changed")
    _, _, _, _, count, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[eocd : eocd + 22])
    directory = _read_range(url, cd_offset, cd_offset + cd_size - 1)
    if directory is None:
        return None
    entries: dict[str, _CentralEntry] = {}
    pos = 0
    for _ in range(count):
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", directory[pos : pos + 46])
        method, crc, csize, usize = fields[4], fields[7], fields[8], fields[9]
        name_len, extra_len, comment_len, offset = fields[10], fields[11], fields[12], fields[16]
        name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8")
        entries[name] = _CentralEntry(method, crc, csize, usize, offset)
        pos += 46 + name_len + extra_len + comment_len
    return entries


def _extract_member_by_range(source: ZipMemberSource, dest: Path, log: Log) -> bool:
    entries = _central_directory(source.url, source.archive_size)
    if entries is None:
        return False
    entry = entries.get(source.member)
    if entry is None or entry.method != zipfile.ZIP_DEFLATED:
        raise ChecksumError(f"{source.member} missing from the archive or not deflated")
    local = _read_range(source.url, entry.header_offset, entry.header_offset + 29)
    if local is None:
        return False
    name_len, extra_len = struct.unpack("<HH", local[26:30])
    start = entry.header_offset + 30 + name_len + extra_len
    size_mb, total_mb = entry.compressed / 1e6, source.archive_size / 1e6
    log(f"range     {source.member} ({size_mb:.1f} MB of a {total_mb:.0f} MB archive)")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    crc = 0
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            _open(source.url, (start, start + entry.compressed - 1)) as r,
        ):
            if r.status != 206:
                return False
            while block := r.read(CHUNK):
                data = inflater.decompress(block)
                crc = zlib.crc32(data, crc)
                out.write(data)
            tail = inflater.flush()
            crc = zlib.crc32(tail, crc)
            out.write(tail)
        if crc != entry.crc32:
            raise ChecksumError(f"{source.member}: CRC32 mismatch after inflating")
        verify(tmp, source.pin)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def _extract_member_from_full_archive(source: ZipMemberSource, dest: Path, log: Log) -> None:
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        archive = Path(scratch) / "archive.zip"
        member = Path(scratch) / "member"
        log(f"download  full archive ({source.archive_size / 1e6:.0f} MB), ranges unsupported")
        with archive.open("wb") as out, _open(source.url) as resp:
            shutil.copyfileobj(resp, out, length=CHUNK)
        with (
            zipfile.ZipFile(archive) as zf,
            zf.open(source.member) as src,
            member.open("wb") as out,
        ):
            shutil.copyfileobj(src, out, length=CHUNK)
        verify(member, source.pin)
        member.replace(dest)


def musique_path() -> Path:
    return raw_dir() / MUSIQUE_DEV.pin.name


def hotpotqa_path() -> Path:
    return raw_dir() / HOTPOTQA_DEV.pin.name


def fetch_musique(log: Log = print) -> Path:
    dest = musique_path()
    if dest.exists():
        verify(dest, MUSIQUE_DEV.pin)
        log(f"verified  {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _extract_member_by_range(MUSIQUE_DEV, dest, log):
        _extract_member_from_full_archive(MUSIQUE_DEV, dest, log)
    log(f"verified  {dest}")
    return dest


def fetch_hotpotqa(log: Log = print) -> Path:
    dest = hotpotqa_path()
    if not dest.exists():
        _download(HOTPOTQA_DEV.url, dest, HOTPOTQA_DEV.pin, log)
    verify(dest, HOTPOTQA_DEV.pin)
    log(f"verified  {dest}")
    return dest


def _hub_file(model: HubModel, pin: Pin, *, download: bool) -> Path | None:
    if download:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(model.repo, pin.name, revision=model.revision))
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(model.repo, pin.name, revision=model.revision)
    return Path(hit) if isinstance(hit, str) else None


def model_files(model: HubModel, *, download: bool = False) -> dict[str, Path]:
    """Verified local paths of every pinned file of ``model`` in the Hugging Face cache."""
    paths: dict[str, Path] = {}
    for pin in model.files:
        local = _hub_file(model, pin, download=download)
        if local is None:
            raise MissingAssetError(f"{model.repo}/{pin.name} is not cached; run `ircot fetch`")
        verify(local, pin)
        paths[pin.name] = local
    return paths


def reader_onnx_path() -> Path:
    return models_dir() / READER_ONNX_NAME


def fetch_models(log: Log = print) -> None:
    from ircot.onnx_bert import export_squad_reader

    for model in (BGE_SMALL, MINILM_SQUAD2):
        model_files(model, download=True)
        log(f"verified  {model.repo}@{model.revision[:10]}")
    dest = reader_onnx_path()
    if not dest.exists():
        weights = model_files(MINILM_SQUAD2)
        dest.parent.mkdir(parents=True, exist_ok=True)
        export_squad_reader(weights["model.safetensors"], weights["config.json"], dest)
        log(f"exported  {dest} from {MINILM_SQUAD2.repo} safetensors")
    log(f"ready     {dest} (sha256 {sha256_file(dest)[:16]})")


def fetch_evaluator(log: Log = print) -> Path:
    root = evaluator_dir()
    for pin in EVALUATOR_FILES:
        dest = root / pin.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(EVALUATOR_BASE + pin.name, dest, pin, log)
        verify(dest, pin)
    log(f"verified  official MuSiQue evaluator in {root}")
    return root


def evaluator_present() -> bool:
    root = evaluator_dir()
    return all((root / pin.name).exists() for pin in EVALUATOR_FILES)


def dataset_checksums() -> dict[str, dict[str, object]]:
    """What a run manifest records about the raw dataset files (pins, not re-hashed)."""
    return {
        "musique": {
            "file": MUSIQUE_DEV.pin.name,
            "sha256": MUSIQUE_DEV.pin.sha256,
            "bytes": MUSIQUE_DEV.pin.size,
            "source": f"{MUSIQUE_DEV.url} (zip member {MUSIQUE_DEV.member})",
            "license": "CC BY 4.0",
        },
        "hotpotqa": {
            "file": HOTPOTQA_DEV.pin.name,
            "sha256": HOTPOTQA_DEV.pin.sha256,
            "bytes": HOTPOTQA_DEV.pin.size,
            "source": HOTPOTQA_DEV.url,
            "license": "CC BY-SA 4.0",
        },
    }
