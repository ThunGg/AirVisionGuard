"""Compatibility helpers for legacy MXNet data conversion tools."""

import numpy as np


def _restore_numpy_aliases():
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _patch_old_ml_dtypes_for_onnx_import():
    try:
        import ml_dtypes
    except ImportError:
        return

    # MXNet imports its optional ONNX bridge at module import time. The record
    # converters do not use ONNX, but onnx>=1.19 expects newer ml_dtypes names.
    import_only_aliases = {
        "float4_e2m1fn": "float8_e4m3fn",
        "float8_e8m0fnu": "float8_e5m2",
    }
    for name, fallback in import_only_aliases.items():
        if not hasattr(ml_dtypes, name) and hasattr(ml_dtypes, fallback):
            setattr(ml_dtypes, name, getattr(ml_dtypes, fallback))


def import_mxnet():
    _restore_numpy_aliases()
    _patch_old_ml_dtypes_for_onnx_import()

    try:
        import mxnet as mx
    except AttributeError as exc:
        if "ml_dtypes" in str(exc):
            raise RuntimeError(
                "MXNet failed while importing ONNX. Update ml_dtypes to a "
                "newer version, for example: pip install 'ml_dtypes>=0.5.0'"
            ) from exc
        raise

    return mx
