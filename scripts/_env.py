#!/usr/bin/env python3
"""Environment-provenance emitter for publication runs.

Captures the interpreter, OS, hardware, key package versions, MOSEK version
(if licensed and importable), and git state, so a run's result JSON can be
tied to the exact environment it was produced in. Stdlib + already-installed
packages only — no new dependencies.

Usage:
    python scripts/_env.py                 # print JSON to stdout
    python scripts/_env.py path/to/env.json  # write JSON to that path
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "scipy", "sympy", "cvxpy", "clarabel", "scs")


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _mosek_version() -> str:
    try:
        import mosek
        return ".".join(str(v) for v in mosek.Env().getversion())
    except Exception:
        return "not installed"


def _cpu_description() -> str:
    try:
        proc = platform.processor()
        if proc:
            return proc
    except Exception:
        pass
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        return platform.machine() or "unknown"
    except Exception:
        return "unknown"


def _total_ram_bytes():
    """Total physical RAM in bytes, or None if it cannot be determined."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, check=True)
            return int(out.stdout.strip())
    except Exception:
        pass
    return None


def _total_ram_str() -> str:
    n = _total_ram_bytes()
    if n is None:
        return "unknown"
    return f"{n / (1024 ** 3):.1f} GiB"


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty():
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return None


def collect_env() -> dict:
    """Snapshot interpreter, OS, hardware, package, and git provenance."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu": _cpu_description(),
        "ram": _total_ram_str(),
        "packages": {name: _pkg_version(name) for name in PACKAGES},
        "mosek_version": _mosek_version(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }


def write_env(path: str = "env.json") -> None:
    """Write ``collect_env()`` as indented JSON to ``path``."""
    json.dump(collect_env(), open(path, "w"), indent=2)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        write_env(path)
        print(f"wrote {path}", file=sys.stderr)
    else:
        # No output path given: print to stdout rather than writing a stray
        # ./env.json file into whatever directory this happened to run in.
        print(json.dumps(collect_env(), indent=2))


if __name__ == "__main__":
    main()
