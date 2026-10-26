"""rough-vol-nn: deep calibration of the rough Bergomi model."""

# On Windows, loading pyarrow's DLLs *after* torch and scipy can crash the interpreter
# (access violation). pandas >= 3 imports pyarrow lazily for its string dtype, so make
# sure it is loaded first, before any torch import.
try:
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - pyarrow is optional for pandas
    pass

__version__ = "0.1.0"
