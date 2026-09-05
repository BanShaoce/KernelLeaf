"""Small batch-first Transformer encoder built from KernelLeaf TensorOps."""

import math

import numpy as np

from ..autograd import Tensor
from .. import ops
from .nn_basic import (
    Dropout, LayerNorm, Linear, Module, ReLU, Softmax,
)

__all__ = [
    "MultiHeadSelfAttention",
    "PositionwiseFeedForward",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "SinusoidalPositionalEncoding",
]


def _sequence_linear(layer, inputs):
    """Flatten leading dimensions so V4 Linear auto dispatch remains usable."""
    if len(inputs.shape) < 2:
        raise ValueError("Linear sequence input must have at least two dimensions")
    leading = inputs.shape[:-1]
    flattened = inputs.reshape((-1, inputs.shape[-1]))
    output = layer(flattened)
    return output.reshape(leading + (output.shape[-1],))


def _padding_bias(mask, inputs, batch_size, sequence_length):
    if not isinstance(mask, Tensor):
        raise TypeError("padding_mask must be a KernelLeaf Tensor")
    if mask.device != inputs.device:
        raise ValueError(
            f"device mismatch: inputs are on {inputs.device}, "
            f"padding_mask is on {mask.device}"
        )
    if mask.requires_grad:
        raise ValueError("padding_mask must not require gradients")
    if mask.shape != (batch_size, sequence_length):
        raise ValueError(
            "padding_mask must have shape "
            f"({batch_size}, {sequence_length}), got {mask.shape}"
        )
    values = mask.realize_cached_data()
    if str(values.dtype) != "bool":
        raise TypeError("padding_mask must have boolean dtype")
    xp = inputs.device.xp
    negative = -1e4 if str(inputs.dtype) == "float16" else -1e9
    bias = xp.where(values, 0.0, negative).astype(inputs.dtype)
    return Tensor(
        bias.reshape((batch_size, 1, 1, sequence_length)),
        device=inputs.device,
        requires_grad=False,
    )


class MultiHeadSelfAttention(Module):
    """Scaled dot-product self-attention for batch-first inputs.

    ``padding_mask`` is boolean with shape ``(batch, sequence)``. True entries
    are visible as keys; False entries are excluded from every query.
    """

    def __init__(self, d_model, num_heads, dropout=0.0, *, bias=True,
                 device=None, dtype="float32", implementation="auto"):
        super().__init__()
        d_model = int(d_model)
        num_heads = int(num_heads)
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = float(dropout)
        self.implementation = implementation
        options = {
            "bias": bias, "device": device, "dtype": dtype,
            "implementation": implementation,
        }
        self.q_proj = Linear(d_model, d_model, **options)
        self.k_proj = Linear(d_model, d_model, **options)
        self.v_proj = Linear(d_model, d_model, **options)
        self.out_proj = Linear(d_model, d_model, **options)
        self.softmax = Softmax(axis=-1, implementation=implementation)
        self.attention_dropout = Dropout(dropout)

    def forward(self, inputs, padding_mask=None):
        if len(inputs.shape) != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError(
                "MultiHeadSelfAttention expects batch-first input "
                f"(batch, sequence, {self.d_model}), got {inputs.shape}"
            )
        batch, sequence, _ = inputs.shape
        if batch <= 0 or sequence <= 0:
            raise ValueError("attention batch and sequence dimensions must be positive")

        shape = (batch, sequence, self.num_heads, self.head_dim)
        queries = _sequence_linear(self.q_proj, inputs).reshape(shape).transpose((1, 2))
        keys = _sequence_linear(self.k_proj, inputs).reshape(shape).transpose((1, 2))
        values = _sequence_linear(self.v_proj, inputs).reshape(shape).transpose((1, 2))
        scores = (queries @ keys.transpose()) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores + ops.broadcast_to(
                _padding_bias(padding_mask, inputs, batch, sequence), scores.shape
            )
        probabilities = self.attention_dropout(self.softmax(scores))
        context = (probabilities @ values).transpose((1, 2)).reshape(
            (batch, sequence, self.d_model)
        )
        return _sequence_linear(self.out_proj, context)


class PositionwiseFeedForward(Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.0, *, device=None,
                 dtype="float32", implementation="auto"):
        super().__init__()
        if d_model <= 0 or dim_feedforward <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        self.linear1 = Linear(
            d_model, dim_feedforward, device=device, dtype=dtype,
            implementation=implementation,
        )
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(
            dim_feedforward, d_model, device=device, dtype=dtype,
            implementation=implementation,
        )

    def forward(self, inputs):
        hidden = self.activation(_sequence_linear(self.linear1, inputs))
        return _sequence_linear(self.linear2, self.dropout(hidden))


class TransformerEncoderLayer(Module):
    """Pre-norm Transformer encoder layer with a ReLU feed-forward network."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, *, device=None, dtype="float32",
                 implementation="auto"):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_p = float(dropout)
        self.layer_norm_eps = float(layer_norm_eps)
        self.implementation = implementation
        self.self_attn = MultiHeadSelfAttention(
            d_model, nhead, dropout, device=device, dtype=dtype,
            implementation=implementation,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model, dim_feedforward, dropout, device=device, dtype=dtype,
            implementation=implementation,
        )
        self.norm1 = LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype,
            implementation=implementation,
        )
        self.norm2 = LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype,
            implementation=implementation,
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, padding_mask=None):
        normalized = self.norm1(src)
        src = src + self.dropout1(self.self_attn(normalized, padding_mask))
        normalized = self.norm2(src)
        return src + self.dropout2(self.feed_forward(normalized))

    def clone(self):
        # Clone from the layer's current placement, including when ``to`` was
        # called before the layer was passed to TransformerEncoder.
        device = self.norm1.weight.device
        dtype = self.norm1.weight.dtype
        result = TransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout_p,
            self.layer_norm_eps,
            device=device,
            dtype=dtype,
            implementation=self.implementation,
        )
        result.load_state_dict(self.state_dict())
        return result


class TransformerEncoder(Module):
    """Stack independent copies of a ``TransformerEncoderLayer``."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        if not isinstance(encoder_layer, TransformerEncoderLayer):
            raise TypeError("encoder_layer must be a TransformerEncoderLayer")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        self.layers = [encoder_layer]
        self.layers.extend(encoder_layer.clone() for _ in range(1, self.num_layers))
        self.norm = norm

    def forward(self, src, padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, padding_mask)
        return output if self.norm is None else self.norm(output)


class SinusoidalPositionalEncoding(Module):
    """Fixed sinusoidal positions for batch-first feature sequences."""

    def __init__(self, d_model, max_length=5000, dropout=0.0, *, device=None,
                 dtype="float32"):
        super().__init__()
        d_model = int(d_model)
        max_length = int(max_length)
        if d_model <= 0 or max_length <= 0:
            raise ValueError("d_model and max_length must be positive")
        positions = np.arange(max_length, dtype=np.float32)[:, None]
        frequencies = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = np.zeros((1, max_length, d_model), dtype=np.float32)
        encoding[0, :, 0::2] = np.sin(positions * frequencies)
        odd_columns = encoding[0, :, 1::2].shape[1]
        encoding[0, :, 1::2] = np.cos(
            positions * frequencies[:odd_columns]
        )
        self.d_model = d_model
        self.max_length = max_length
        self.encoding = Tensor(
            encoding, device=device, dtype=dtype, requires_grad=False
        )
        self.dropout = Dropout(dropout)

    def forward(self, inputs):
        if len(inputs.shape) != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError(
                "positional encoding expects batch-first input with last "
                f"dimension {self.d_model}, got {inputs.shape}"
            )
        length = inputs.shape[1]
        if length > self.max_length:
            raise ValueError(
                f"sequence length {length} exceeds max_length {self.max_length}"
            )
        if self.encoding.device != inputs.device:
            raise ValueError(
                f"device mismatch: inputs are on {inputs.device}, "
                f"positional encoding is on {self.encoding.device}; "
                "move the module with .to(device)"
            )
        positions = Tensor(
            self.encoding.realize_cached_data()[:, :length, :],
            device=self.encoding.device,
            requires_grad=False,
        )
        return self.dropout(
            inputs + ops.broadcast_to(positions, inputs.shape)
        )
