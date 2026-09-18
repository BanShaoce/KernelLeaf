"""Distributed collapsed-Gibbs LDA with sparse count range updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .runtime import ParameterServerClient

from .data import make_lda_corpus


def word_topic_key(topic: int, word: int) -> str:
    return f"wt:{int(topic)}:{int(word)}"


def topic_total_key(topic: int) -> str:
    return f"tw:{int(topic)}"


@dataclass(frozen=True)
class LDAConfig:
    documents: int = 320
    vocabulary: int = 256
    topics: int = 4
    doc_length: int = 32
    sweeps: int = 8
    alpha: float = 0.1
    beta: float = 0.01
    seed: int = 23
    worker_delay: float = 0.0
    mode: str = "sync"
    switch_at: int = -1
    switch_to: str = ""


def _add_delta(delta: Dict[str, float], topic: int, word: int, amount: float) -> None:
    wt_key = word_topic_key(topic, word)
    tw_key = topic_total_key(topic)
    delta[wt_key] = delta.get(wt_key, 0.0) + float(amount)
    delta[tw_key] = delta.get(tw_key, 0.0) + float(amount)


def _pull_document_counts(
    client: ParameterServerClient,
    words: np.ndarray,
    topics: int,
) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
    unique_words = np.unique(words)
    keys = []
    for topic in range(topics):
        keys.extend(word_topic_key(topic, int(word)) for word in unique_words)
        keys.append(topic_total_key(topic))
    pulled = client.pull_scalars(keys)
    word_topic = {
        (topic, int(word)): float(
            pulled.get(word_topic_key(topic, int(word)), 0.0)
        )
        for topic in range(topics)
        for word in unique_words
    }
    totals = np.asarray(
        [pulled.get(topic_total_key(topic), 0.0) for topic in range(topics)],
        dtype=np.float64,
    )
    return word_topic, totals


def run_lda_worker(
    endpoints: Sequence[str],
    worker_id: int,
    *,
    workers: int,
    transport: str,
    config: LDAConfig,
    timeout: float,
    result_queue,
    registration_barrier=None,
) -> None:
    """Top-level process entry point for one distributed LDA worker."""
    try:
        corpus, _ = make_lda_corpus(
            documents=config.documents,
            vocabulary=config.vocabulary,
            topics=config.topics,
            doc_length=config.doc_length,
            seed=config.seed,
            alpha=config.alpha,
            beta=config.beta,
        )
        rng = np.random.default_rng(config.seed + 10_000 + int(worker_id))
        local_docs = list(range(int(worker_id), config.documents, workers))
        assignments: Dict[int, np.ndarray] = {}
        doc_counts: Dict[int, np.ndarray] = {}
        client = ParameterServerClient(
            endpoints,
            worker_id=f"lda-{worker_id}",
            transport=transport,
            timeout=timeout,
        )
        client.register()
        # Do not let an early synchronous worker submit clock 0 until every
        # worker has registered with every shard.
        if registration_barrier is not None:
            registration_barrier.wait(timeout=timeout)
        started = time.perf_counter()
        pushes = 0
        current_mode = config.mode
        async_clock = 0

        for sweep in range(config.sweeps):
            if config.switch_at >= 0 and sweep == config.switch_at:
                current_mode = config.switch_to or (
                    "async" if config.mode == "sync" else "sync"
                )
                client.set_mode(current_mode)

            pending_delta: Dict[str, float] = {}
            for doc_index in local_docs:
                words = corpus[doc_index]
                if doc_index not in assignments:
                    topics = rng.integers(0, config.topics, size=len(words))
                    assignments[doc_index] = topics.astype(np.int32, copy=False)
                    doc_counts[doc_index] = np.bincount(
                        topics, minlength=config.topics
                    ).astype(np.float64)
                    for topic, word in zip(topics, words):
                        _add_delta(pending_delta, int(topic), int(word), 1.0)
                elif sweep > 0:
                    topics = assignments[doc_index]
                    counts = doc_counts[doc_index].copy()
                    word_topic, topic_totals = _pull_document_counts(
                        client, words, config.topics
                    )
                    for position, word_value in enumerate(words):
                        word = int(word_value)
                        old_topic = int(topics[position])
                        counts[old_topic] -= 1.0
                        old_wt = max(0.0, word_topic[(old_topic, word)] - 1.0)
                        old_total = max(0.0, topic_totals[old_topic] - 1.0)
                        probabilities = (
                            (old_wt + config.beta)
                            / (old_total + config.beta * config.vocabulary)
                            * (counts + config.alpha)
                        )
                        probabilities = np.maximum(probabilities, 1e-300)
                        probabilities /= probabilities.sum()
                        new_topic = int(rng.choice(config.topics, p=probabilities))
                        counts[new_topic] += 1.0
                        topics[position] = new_topic

                        word_topic[(old_topic, word)] -= 1.0
                        topic_totals[old_topic] -= 1.0
                        word_topic[(new_topic, word)] = (
                            word_topic.get((new_topic, word), 0.0) + 1.0
                        )
                        topic_totals[new_topic] += 1.0
                        _add_delta(pending_delta, old_topic, word, -1.0)
                        _add_delta(pending_delta, new_topic, word, 1.0)

                if current_mode == "async":
                    client.push(pending_delta, clock=async_clock, op="add")
                    pushes += 1
                    async_clock += 1
                    pending_delta = {}
                    if config.worker_delay:
                        time.sleep(config.worker_delay)

            if current_mode == "sync":
                client.push(pending_delta, clock=sweep, op="add")
                pushes += 1

        client.close()
        result_queue.put(
            {
                "worker": worker_id,
                "assignments": assignments,
                "doc_counts": doc_counts,
                "pushes": pushes,
                "elapsed": time.perf_counter() - started,
            }
        )
    except Exception as error:
        import traceback

        result_queue.put(
            {
                "worker": worker_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )


def evaluate_lda(
    client: ParameterServerClient,
    worker_results: Sequence[Mapping[str, object]],
    *,
    config: LDAConfig,
) -> Dict[str, object]:
    """Compute training perplexity and top words from final server counts."""
    keys = []
    for topic in range(config.topics):
        keys.extend(word_topic_key(topic, word) for word in range(config.vocabulary))
        keys.append(topic_total_key(topic))
    counts = client.pull_scalars(keys)
    word_topic = np.empty((config.topics, config.vocabulary), dtype=np.float64)
    topic_totals = np.empty(config.topics, dtype=np.float64)
    for topic in range(config.topics):
        topic_totals[topic] = counts.get(topic_total_key(topic), 0.0)
        for word in range(config.vocabulary):
            word_topic[topic, word] = counts.get(
                word_topic_key(topic, word), 0.0
            )
    word_topic = np.maximum(word_topic, 0.0)
    topic_totals = np.maximum(topic_totals, 0.0)
    phi = (word_topic + config.beta) / (
        topic_totals[:, None] + config.beta * config.vocabulary
    )
    corpus, true_phi = make_lda_corpus(
        documents=config.documents,
        vocabulary=config.vocabulary,
        topics=config.topics,
        doc_length=config.doc_length,
        seed=config.seed,
        alpha=config.alpha,
        beta=config.beta,
    )
    total_log_likelihood = 0.0
    total_words = 0
    for result in worker_results:
        doc_counts = result.get("doc_counts", {})
        for doc_index, raw_counts in doc_counts.items():
            counts_array = np.asarray(raw_counts, dtype=np.float64)
            theta = (counts_array + config.alpha) / (
                counts_array.sum() + config.alpha * config.topics
            )
            for word in corpus[int(doc_index)]:
                probability = float(np.dot(theta, phi[:, int(word)]))
                total_log_likelihood += math.log(max(probability, 1e-300))
                total_words += 1
    top_words = [
        [int(word) for word in np.argsort(word_topic[topic])[-8:][::-1]]
        for topic in range(config.topics)
    ]
    overlap = []
    for topic in range(config.topics):
        learned = set(top_words[topic])
        truth = set(int(word) for word in np.argsort(true_phi[topic])[-8:])
        overlap.append(len(learned & truth) / 8.0)
    return {
        "perplexity": math.exp(-total_log_likelihood / max(1, total_words)),
        "top_words": top_words,
        "top_word_overlap": float(np.mean(overlap)),
    }
