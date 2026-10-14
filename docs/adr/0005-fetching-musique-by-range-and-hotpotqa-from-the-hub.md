# ADR 0005: Read MuSiQue out of its archive by HTTP range; take HotpotQA from the Hub

Status: accepted

## Context

MuSiQue is distributed as a single Google Drive zip of about 272 MB (all splits of both the
answerable and the full variant), linked from the official repository. Only the answerable dev
split is needed, about 30 MB uncompressed. HotpotQA's original download host for the distractor
dev file did not respond when this project was built; the dataset's maintainers publish the same
data on the Hugging Face Hub as parquet.

## Decision

- MuSiQue: the fetcher reads the zip's end-of-central-directory record and central directory with
  HTTP range requests, then requests only the dev member's bytes and inflates them while computing
  CRC32 and sha256, so about 7 MB travels instead of 272 MB. If the server ignores range requests,
  the fetcher downloads the archive to a temporary directory and extracts the member there. Both
  paths verify against the same pinned sha256 before the file is moved into place.
- HotpotQA: the `distractor/validation` parquet from `hotpotqa/hotpot_qa`, pinned to a Hub
  revision and sha256.
- Every pin (size and sha256) was recorded on first download and is committed in
  `ircot/sources.py`; a mismatch is an error.

## Consequences

- `ircot fetch` stays small and fast, and an interrupted fetch leaves no partial file behind.
- The Drive URL uses the `confirm=t` download form. If Google changes it, both paths fail and the
  URL has to be updated; the pinned hash still guarantees the bytes.
- Reading parquet adds `pyarrow` as a dependency.

## Alternatives rejected

- Downloading and unpacking the whole MuSiQue zip: 272 MB for 30 MB of data.
- Community re-uploads of MuSiQue on the Hub: convenient, but not published by the dataset
  authors, and the official archive is still reachable.
- Waiting for the original HotpotQA host: not reproducible.
