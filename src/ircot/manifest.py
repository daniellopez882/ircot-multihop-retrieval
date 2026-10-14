"""Run manifests: what produced a result (code, config, data, machine) recorded next to it."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "scipy", "onnxruntime", "onnx", "tokenizers", "pyarrow", "huggingface-hub")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: object) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def git_sha(repo: Path | None = None) -> str:
    """HEAD of the working tree, with ``-dirty`` when tracked files have uncommitted changes."""
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return int(status.ullTotalPhys)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _physical_cores() -> int | None:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).NumberOfCores",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.split()
            return sum(int(v) for v in out) or None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        cores = {
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("core id")
        }
        return len(cores) or None
    except OSError:
        return None


def machine_info() -> dict[str, object]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "cpu": _cpu_model(),
        "physical_cores": _physical_cores(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(_ram_bytes() / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "accelerator": "none (CPU only)",
    }


def local_date() -> str:
    """The machine's local calendar date; results are dated where they were measured."""
    return datetime.now().astimezone().date().isoformat()


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
