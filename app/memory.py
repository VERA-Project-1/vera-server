
from pathlib import Path

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


def _read_kb(field: str) -> int | None:
    try:
        for line in _STATUS.read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def rss_mb() -> float | None:
    kb = _read_kb("VmRSS")
    return kb / 1024 if kb is not None else None


def peak_mb() -> float | None:
    kb = _read_kb("VmHWM")
    return kb / 1024 if kb is not None else None


def reset_peak() -> bool:
    try:
        _CLEAR_REFS.write_text("5")
        return True
    except OSError:
        return False


def fmt(mb: float | None) -> str:
    return f"{mb:.0f} MB" if mb is not None else "n/a"
