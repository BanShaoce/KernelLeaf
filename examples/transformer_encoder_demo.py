"""Train a tiny Transformer encoder on a synthetic sequence task."""

import argparse

import numpy as np

import kernelleaf as kl


class SequenceClassifier(kl.nn.Module):
    def __init__(self, d_model, num_heads, num_layers, sequence_length, device):
        super().__init__()
        self.position = kl.nn.SinusoidalPositionalEncoding(
            d_model, max_length=sequence_length, dropout=0.0, device=device
        )
        layer = kl.nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            device=device,
            implementation="auto",
        )
        self.encoder = kl.nn.TransformerEncoder(
            layer,
            num_layers,
            norm=kl.nn.LayerNorm(d_model, device=device, implementation="auto"),
        )
        self.classifier = kl.nn.Linear(
            d_model, 2, device=device, implementation="auto"
        )

    def forward(self, features, padding_mask):
        encoded = self.encoder(self.position(features), padding_mask)
        pooled = kl.ops.summation(encoded, axes=(1,)) / features.shape[1]
        return self.classifier(pooled)


def make_problem(samples, sequence_length, d_model, seed):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, sequence_length, d_model)).astype(np.float32)
    # The final position is padding. It contains distracting values which must
    # not be used as keys by attention.
    features[:, -1, :] = rng.normal(0, 3, size=(samples, d_model))
    labels = (features[:, :-1, 0].sum(axis=1) > 0).astype(np.int32)
    padding_mask = np.ones((samples, sequence_length), dtype=bool)
    padding_mask[:, -1] = False
    return features, labels, padding_mask


def run(device, steps=80, samples=32, sequence_length=6, d_model=8,
        num_heads=2, num_layers=1, seed=7):
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    np.random.seed(seed)
    features, labels, padding_mask = make_problem(
        samples, sequence_length, d_model, seed
    )
    x = kl.Tensor(features, device=device, requires_grad=False)
    y = kl.Tensor(labels, device=device, requires_grad=False)
    mask = kl.Tensor(padding_mask, device=device, requires_grad=False)
    model = SequenceClassifier(
        d_model, num_heads, num_layers, sequence_length, device
    )
    optimizer = kl.optim.Adam(model.parameters(), lr=0.01)
    criterion = kl.nn.SoftmaxLoss()
    losses = []

    for step in range(steps):
        optimizer.reset_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.numpy()))
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == steps:
            accuracy = float((np.argmax(logits.numpy(), axis=1) == labels).mean())
            print(
                f"step={step + 1:03d} loss={losses[-1]:.6f} "
                f"accuracy={accuracy:.3f}"
            )
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    device = kl.cpu() if args.device == "cpu" else kl.cuda(0)
    run(
        device,
        steps=args.steps,
        samples=args.samples,
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
