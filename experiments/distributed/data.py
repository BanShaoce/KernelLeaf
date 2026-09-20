"""Deterministic lightweight datasets for distributed algorithm demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class SparseClassificationData:
    indices: List[np.ndarray]
    values: List[np.ndarray]
    labels: np.ndarray


def make_sparse_classification(
    *,
    samples: int,
    features: int,
    nnz: int,
    seed: int,
    task_seed: Optional[int] = None,
) -> SparseClassificationData:
    """Create a high-dimensional sparse binary classification problem."""
    if samples <= 0 or features <= 0 or nnz <= 0:
        raise ValueError("samples, features, and nnz must be positive")
    if nnz > features:
        raise ValueError("nnz cannot exceed the feature dimension")
    rng = np.random.default_rng(seed)
    weight_rng = np.random.default_rng(seed if task_seed is None else task_seed)
    signal_features = min(features, max(8, min(32, features // 16)))
    signal_count = max(1, min(nnz // 2, signal_features))
    true_weights = np.zeros(features, dtype=np.float32)
    true_weights[:signal_features] = weight_rng.normal(
        0.0, 1.0, size=signal_features
    ).astype(np.float32)
    true_weights /= np.sqrt(float(signal_count))
    rows = []
    values = []
    labels = np.empty(samples, dtype=np.int64)
    for row in range(samples):
        signal_indices = rng.choice(signal_features, size=signal_count, replace=False)
        noise_candidates = np.setdiff1d(
            np.arange(features, dtype=np.int32), signal_indices, assume_unique=True
        )
        noise_indices = rng.choice(
            noise_candidates, size=nnz - signal_count, replace=False
        )
        indices = np.sort(np.concatenate((signal_indices, noise_indices))).astype(
            np.int32
        )
        row_values = rng.lognormal(mean=0.0, sigma=0.35, size=nnz).astype(np.float32)
        signal = float(np.dot(true_weights[indices], row_values))
        labels[row] = int(signal + rng.normal(0.0, 0.45) > 0.0)
        rows.append(indices)
        values.append(row_values)
    return SparseClassificationData(rows, values, labels)


def make_lda_corpus(
    *,
    documents: int,
    vocabulary: int,
    topics: int,
    doc_length: int,
    seed: int,
    alpha: float = 0.1,
    beta: float = 0.01,
):
    """Sample a synthetic LDA corpus and return documents plus true phi."""
    if documents <= 0 or vocabulary <= 0 or topics <= 0 or doc_length <= 0:
        raise ValueError("LDA dimensions must be positive")
    rng = np.random.default_rng(seed)
    topic_word = rng.dirichlet(np.full(vocabulary, 0.15), size=topics)
    doc_topic = rng.dirichlet(np.full(topics, alpha), size=documents)
    docs = []
    for document in range(documents):
        # Draw one dominant topic per document to preserve identifiable clusters.
        topic = int(rng.choice(topics, p=doc_topic[document] / doc_topic[document].sum()))
        words = rng.choice(
            vocabulary, size=doc_length, replace=True, p=topic_word[topic]
        ).astype(np.int32)
        docs.append(words)
    return docs, topic_word.astype(np.float64)
