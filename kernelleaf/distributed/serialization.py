"""Message codecs for the KernelLeaf distributed communication boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
from typing import Any

import numpy as np

from ..backend import asnumpy
from .protocol import Message, ProtocolError


_ARRAY_MARKER = "__kernelleaf_ndarray__"
_ARRAY_FIELDS = {_ARRAY_MARKER, "dtype", "shape", "data"}
_SUPPORTED_DTYPE_KINDS = frozenset("biuf")


class CodecError(ValueError):
    """Raised when a message cannot be encoded or decoded."""


class MessageCodec(ABC):
    """Transport-independent message serialization interface."""

    name = "abstract"

    @abstractmethod
    def encode(self, message: Message) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def decode(self, payload: bytes) -> Message:
        raise NotImplementedError


def _is_cupy_array(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "cupy"


def _encode_value(value):
    if isinstance(value, np.ndarray) or _is_cupy_array(value):
        # Device-to-host transfer is deliberate and confined to this network
        # serialization boundary. Importing this module does not import CuPy.
        array = np.asarray(asnumpy(value))
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if array.dtype.kind not in _SUPPORTED_DTYPE_KINDS:
            raise CodecError(
                f"JSON ndarray codec does not support dtype {array.dtype}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise CodecError("JSON ndarray codec requires finite floating values")
        return {
            _ARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.reshape(-1).tolist(),
        }
    if isinstance(value, np.generic):
        return _encode_value(value.item())
    if isinstance(value, dict):
        if _ARRAY_MARKER in value:
            raise CodecError(f"payload key {_ARRAY_MARKER!r} is reserved")
        if not all(isinstance(key, str) for key in value):
            raise CodecError("JSON payload mapping keys must be strings")
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise CodecError(f"JSON codec cannot encode {type(value).__name__}")


def _decode_array(value):
    if set(value) != _ARRAY_FIELDS or value.get(_ARRAY_MARKER) is not True:
        raise CodecError("invalid ndarray JSON object")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int)
        or dimension < 0 for dimension in shape
    ):
        raise CodecError("ndarray shape must contain non-negative integers")
    try:
        dtype = np.dtype(value["dtype"])
    except (TypeError, ValueError) as error:
        raise CodecError(f"invalid ndarray dtype: {value['dtype']!r}") from error
    if dtype.kind not in _SUPPORTED_DTYPE_KINDS:
        raise CodecError(f"JSON ndarray codec does not support dtype {dtype}")
    data = value["data"]
    if not isinstance(data, list):
        raise CodecError("ndarray data must be a flat list")
    expected = math.prod(shape)
    if len(data) != expected:
        raise CodecError(
            f"ndarray data length mismatch: expected {expected}, got {len(data)}"
        )
    try:
        array = np.asarray(data, dtype=dtype).reshape(tuple(shape))
    except (TypeError, ValueError, OverflowError) as error:
        raise CodecError("ndarray data is incompatible with dtype/shape") from error
    if dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise CodecError("decoded ndarray contains non-finite values")
    return array


def _decode_value(value):
    if isinstance(value, dict):
        if _ARRAY_MARKER in value:
            return _decode_array(value)
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


class JsonCodec(MessageCodec):
    """Portable JSON codec with explicit numeric ndarray objects."""

    name = "json"

    def encode(self, message: Message) -> bytes:
        if not isinstance(message, Message):
            raise CodecError("encode expects a Message")
        try:
            text = json.dumps(
                _encode_value(message.to_dict()),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, CodecError):
                raise
            raise CodecError(f"message is not valid JSON: {error}") from error
        return text.encode("utf-8")

    def decode(self, payload: bytes) -> Message:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise CodecError("decode expects bytes-like input")
        try:
            value = json.loads(bytes(payload).decode("utf-8"))
            return Message.from_dict(_decode_value(value))
        except CodecError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, ProtocolError) as error:
            raise CodecError(f"invalid JSON message: {error}") from error
