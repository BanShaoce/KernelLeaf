import numpy as np
import pytest

import kernelleaf as kl


def _available_devices():
    devices = [kl.cpu()]
    if kl.is_cuda_available():
        devices.append(kl.cuda(0))
    return devices


def _tensor(values, device, requires_grad=True):
    return kl.Tensor(values, device=device, requires_grad=requires_grad)


def test_encoder_shape_parameter_recursion_and_independent_layers():
    layer = kl.nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0)
    encoder = kl.nn.TransformerEncoder(layer, 2, norm=kl.nn.LayerNorm(8))
    x = kl.Tensor(np.zeros((2, 5, 8), dtype=np.float32))
    mask = kl.Tensor(np.ones((2, 5), dtype=bool), requires_grad=False)

    assert encoder(x, mask).shape == x.shape
    assert len(encoder.parameters()) == 34
    assert len(encoder.state_dict()) == 34
    assert encoder.layers[0] is not encoder.layers[1]
    assert encoder.layers[0].self_attn.q_proj.weight is not (
        encoder.layers[1].self_attn.q_proj.weight
    )
    names = dict(encoder.named_parameters())
    assert "layers.0.self_attn.q_proj.weight" in names
    assert "layers.1.feed_forward.linear2.bias" in names
    assert "norm.weight" in names


def test_encoder_reuses_v4_auto_dispatch_modules():
    layer = kl.nn.TransformerEncoderLayer(
        8, 2, 16, dropout=0.0, implementation="auto"
    )
    layer(kl.Tensor(np.ones((2, 3, 8), dtype=np.float32)))

    assert layer.self_attn.q_proj.last_implementation == "eager"
    assert layer.self_attn.softmax.last_implementation == "eager"
    assert layer.norm1.last_implementation == "eager"
    assert layer.norm2.last_implementation == "eager"


def test_padding_mask_hides_keys_from_valid_query_positions():
    np.random.seed(12)
    attention = kl.nn.MultiHeadSelfAttention(8, 2, dropout=0.0)
    attention.eval()
    rng = np.random.default_rng(12)
    base = rng.normal(size=(1, 4, 8)).astype(np.float32)
    changed = base.copy()
    changed[:, 2:, :] += 100.0
    mask = kl.Tensor([[True, True, False, False]], requires_grad=False)

    masked_a = attention(kl.Tensor(base), mask).numpy()
    masked_b = attention(kl.Tensor(changed), mask).numpy()
    unmasked_b = attention(kl.Tensor(changed)).numpy()

    np.testing.assert_allclose(masked_a[:, :2], masked_b[:, :2], atol=1e-5)
    assert not np.allclose(masked_a[:, :2], unmasked_b[:, :2])


def test_dropout_is_disabled_recursively_in_eval_mode():
    np.random.seed(3)
    encoder = kl.nn.TransformerEncoder(
        kl.nn.TransformerEncoderLayer(8, 2, 16, dropout=0.5), 2
    )
    x = kl.Tensor(np.ones((2, 4, 8), dtype=np.float32), requires_grad=False)

    encoder.eval()
    first = encoder(x).numpy()
    second = encoder(x).numpy()
    np.testing.assert_array_equal(first, second)

    encoder.train()
    assert not np.array_equal(encoder(x).numpy(), encoder(x).numpy())


@pytest.mark.parametrize("device", _available_devices(), ids=str)
def test_encoder_forward_backward_on_available_devices(device):
    np.random.seed(9)
    rng = np.random.default_rng(9)
    encoder = kl.nn.TransformerEncoder(
        kl.nn.TransformerEncoderLayer(
            8, 2, 16, dropout=0.0, device=device, implementation="auto"
        ),
        2,
        norm=kl.nn.LayerNorm(8, device=device, implementation="auto"),
    )
    x = _tensor(rng.normal(size=(2, 4, 8)).astype(np.float32), device)
    mask = _tensor(
        np.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool),
        device,
        requires_grad=False,
    )

    output = encoder(x, mask)
    (output * output).sum().backward()

    assert output.shape == x.shape
    assert x.grad is not None and x.grad.device == device
    assert np.isfinite(x.grad.numpy()).all()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad.device == device for parameter in encoder.parameters())
    assert all(np.isfinite(parameter.grad.numpy()).all()
               for parameter in encoder.parameters())


@pytest.mark.skipif(not kl.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_encoder_cpu_cupy_forward_backward_parity():
    np.random.seed(23)
    rng = np.random.default_rng(23)

    def make_model(device):
        return kl.nn.TransformerEncoder(
            kl.nn.TransformerEncoderLayer(
                8, 2, 16, dropout=0.0, device=device, implementation="auto"
            ),
            1,
            norm=kl.nn.LayerNorm(8, device=device, implementation="auto"),
        )

    cpu_model = make_model(kl.cpu())
    cuda_model = make_model(kl.cuda(0))
    cuda_model.load_state_dict(cpu_model.state_dict())
    values = rng.normal(size=(2, 4, 8)).astype(np.float32)
    upstream = rng.normal(size=(2, 4, 8)).astype(np.float32)
    visible = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool)

    results = []
    for device, model in ((kl.cpu(), cpu_model), (kl.cuda(0), cuda_model)):
        x = _tensor(values, device)
        mask = _tensor(visible, device, requires_grad=False)
        grad_output = _tensor(upstream, device, requires_grad=False)
        output = model(x, mask)
        (output * grad_output).sum().backward()
        results.append((
            output.numpy(),
            x.grad.numpy(),
            [parameter.grad.numpy() for parameter in model.parameters()],
        ))

    np.testing.assert_allclose(results[0][0], results[1][0], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=3e-4, atol=3e-5)
    for cpu_grad, cuda_grad in zip(results[0][2], results[1][2]):
        np.testing.assert_allclose(cpu_grad, cuda_grad, rtol=3e-4, atol=3e-5)


def test_layer_clone_uses_current_device_and_preserves_values():
    layer = kl.nn.TransformerEncoderLayer(4, 2, 8, dropout=0.0)
    target = kl.cuda(0) if kl.is_cuda_available() else kl.cpu()
    layer.to(target)
    encoder = kl.nn.TransformerEncoder(layer, 2)

    assert all(parameter.device == target for parameter in encoder.parameters())
    np.testing.assert_array_equal(
        encoder.layers[0].norm1.weight.numpy(),
        encoder.layers[1].norm1.weight.numpy(),
    )


def test_positional_encoding_is_a_persistent_buffer():
    positional = kl.nn.SinusoidalPositionalEncoding(5, max_length=7, dropout=0.0)
    x = kl.Tensor(np.zeros((2, 4, 5), dtype=np.float32))

    result = positional(x)
    assert result.shape == x.shape
    assert positional.parameters() == []
    assert list(positional.state_dict()) == ["encoding"]
    np.testing.assert_allclose(result.numpy()[0, 0, 0::2], 0.0, atol=1e-7)
    np.testing.assert_allclose(result.numpy()[0, 0, 1::2], 1.0, atol=1e-7)


def test_transformer_input_and_mask_validation():
    attention = kl.nn.MultiHeadSelfAttention(8, 2)
    x = kl.Tensor(np.zeros((2, 3, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="divisible"):
        kl.nn.MultiHeadSelfAttention(7, 2)
    with pytest.raises(ValueError, match="batch-first"):
        attention(kl.Tensor(np.zeros((2, 8), dtype=np.float32)))
    with pytest.raises(TypeError, match="KernelLeaf Tensor"):
        attention(x, np.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        attention(x, kl.Tensor(np.ones((2, 2), dtype=bool), requires_grad=False))
    with pytest.raises(TypeError, match="boolean"):
        attention(x, kl.Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=False))
    with pytest.raises(ValueError, match="must not require"):
        attention(x, kl.Tensor(np.ones((2, 3), dtype=bool)))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        kl.nn.Dropout(1.0)


class _TinySequenceClassifier(kl.nn.Module):
    def __init__(self):
        super().__init__()
        self.position = kl.nn.SinusoidalPositionalEncoding(4, max_length=4)
        self.encoder = kl.nn.TransformerEncoder(
            kl.nn.TransformerEncoderLayer(4, 2, 8, dropout=0.0),
            1,
            norm=kl.nn.LayerNorm(4),
        )
        self.head = kl.nn.Linear(4, 2)

    def forward(self, features, padding_mask):
        encoded = self.encoder(self.position(features), padding_mask)
        pooled = kl.ops.summation(encoded, axes=(1,)) / features.shape[1]
        return self.head(pooled)


def test_tiny_transformer_can_overfit_a_sequence_task():
    np.random.seed(7)
    rng = np.random.default_rng(2)
    values = np.zeros((8, 4, 4), dtype=np.float32)
    labels = np.array([0] * 4 + [1] * 4, dtype=np.int32)
    values[:4, :, 0] = -1.0
    values[4:, :, 0] = 1.0
    values += rng.normal(0, 0.05, values.shape).astype(np.float32)
    features = kl.Tensor(values, requires_grad=False)
    targets = kl.Tensor(labels, requires_grad=False)
    mask = kl.Tensor(np.ones((8, 4), dtype=bool), requires_grad=False)
    model = _TinySequenceClassifier()
    optimizer = kl.optim.Adam(model.parameters(), lr=0.02)
    criterion = kl.nn.SoftmaxLoss()

    initial = float(criterion(model(features, mask), targets).numpy())
    for _ in range(40):
        optimizer.reset_grad()
        loss = criterion(model(features, mask), targets)
        loss.backward()
        optimizer.step()

    logits = model(features, mask).numpy()
    final = float(criterion(model(features, mask), targets).numpy())
    assert final < initial * 0.05
    np.testing.assert_array_equal(np.argmax(logits, axis=1), labels)
