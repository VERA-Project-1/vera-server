import ctypes
import ctypes.util
import sys

# glibc allocator limits, applied before numpy/onnxruntime allocate anything (this package is imported
# first). Equivalent to MALLOC_ARENA_MAX=2, MALLOC_TRIM_THRESHOLD_=65536, MALLOC_MMAP_THRESHOLD_=65536:
# freed memory goes back to the OS after each prediction (~300 MB peak instead of ~455 MB).
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3
_M_ARENA_MAX = -8

if sys.platform.startswith("linux"):
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        _libc.mallopt(_M_ARENA_MAX, 2)
        _libc.mallopt(_M_TRIM_THRESHOLD, 65536)
        _libc.mallopt(_M_MMAP_THRESHOLD, 65536)
    except (OSError, AttributeError):  # not glibc (e.g. musl/Alpine): keep the defaults
        pass
