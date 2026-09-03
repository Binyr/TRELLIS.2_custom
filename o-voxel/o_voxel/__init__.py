"""Top-level package exports.

Keep CUDA-only helpers lazy so CPU geometry voxelization does not import
``flex_gemm`` and initialize CUDA merely from ``import o_voxel``.
"""

from importlib import import_module

from . import convert, io, serialize

__all__ = [
    "convert",
    "io",
    "serialize",
]

_LAZY_MODULES = {"postprocess", "rasterize"}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
