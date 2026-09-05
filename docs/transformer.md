# V9 Transformer Encoder

KernelLeaf V9 provides a deliberately small, trainable Transformer Encoder.
It contains `MultiHeadSelfAttention`, `PositionwiseFeedForward`,
`TransformerEncoderLayer`, `TransformerEncoder`, and an optional fixed
`SinusoidalPositionalEncoding`.

The input layout is batch-first: `(batch, sequence, d_model)`. `d_model` must
be divisible by `num_heads`. The implementation uses separate query, key,
value, and output projections and a ReLU position-wise feed-forward network.
Encoder layers use **pre-norm** residual blocks:

```text
x = x + dropout(attention(layer_norm(x)))
x = x + dropout(ffn(layer_norm(x)))
```

`padding_mask` is an optional boolean KernelLeaf `Tensor` with shape
`(batch, sequence)`, on the same device as the input. `True` means that a
position is a visible key and `False` excludes it from every query. This is a
key-padding mask: it does not zero the output at padded query positions. Every
sample should therefore contain at least one `True` position. A causal mask is
not implemented in V9.

```python
import numpy as np
import kernelleaf as kl

device = kl.cpu()
layer = kl.nn.TransformerEncoderLayer(
    d_model=8, nhead=2, dim_feedforward=16,
    dropout=0.1, device=device, implementation="auto",
)
model = kl.nn.TransformerEncoder(layer, num_layers=2)
x = kl.Tensor(np.random.randn(3, 5, 8).astype("float32"), device=device)
mask = kl.Tensor(np.ones((3, 5), dtype=bool), device=device,
                 requires_grad=False)
y = model(x, mask)
```

Each encoder layer is an independent clone. Parameters and positional buffers
participate in recursive `to`, `state_dict`, and `load_state_dict`. Move both
the model and input explicitly when changing devices; no operator copies CUDA
arrays to CPU.

The projections flatten `(batch, sequence)` temporarily, allowing V4 Linear
auto dispatch to select its supported fused path. Attention probabilities and
normalization directly reuse the V4 Softmax and LayerNorm auto dispatch.
Backward remains the existing TensorOp gradient graph. No PyTorch dependency
or duplicate attention-specific softmax/norm implementation is introduced.

Run the self-contained synthetic example and focused tests with:

```bash
python -m examples.transformer_encoder_demo --device cpu
python -m examples.transformer_encoder_demo --device cuda
python -m pytest tests/test_transformer.py -q
```

V9 intentionally does not include token embeddings, a decoder, cross
attention, GPT, FlashAttention, KV caching, GQA, MoE, or a text tokenizer.
