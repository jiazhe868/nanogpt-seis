"""Deduplication: exact (normalized-hash) + near-duplicate (MinHash LSH).

The MinHash signature of a document is independent of every other document, so
that step — the expensive one — is computed across a process pool. The LSH
insert/query is stateful and stays serial, but it is cheap once signatures exist.
"""
from __future__ import annotations

import hashlib
import os
import re
from multiprocessing import Pool

from datasketch import MinHash, MinHashLSH

_TOKEN_RE = re.compile(r"\w+")

# Worker-shared state, set in the parent before the pool forks; worker processes
# inherit these by copy-on-write (Linux fork) and read them by index — so the
# ~GB of document text is never pickled across the process boundary.
_worker_texts: list[str] = []
_worker_num_perm = 128
_worker_shingle_k = 5


def exact_key(text: str) -> str:
    """Hash of the whitespace-collapsed lowercased text, for exact dedup."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _minhash(text: str, num_perm: int, shingle_k: int) -> MinHash:
    """MinHash signature over word shingles (k-grams of words)."""
    tokens = _TOKEN_RE.findall(text.lower())
    signature = MinHash(num_perm=num_perm)
    if len(tokens) < shingle_k:
        shingles = {" ".join(tokens)} if tokens else set()
    else:
        shingles = {" ".join(tokens[i:i + shingle_k])
                    for i in range(len(tokens) - shingle_k + 1)}
    for shingle in shingles:
        signature.update(shingle.encode("utf-8"))
    return signature


def _worker_signature(doc_index: int):
    """Pool worker: return just the MinHash hash values for one document."""
    return _minhash(_worker_texts[doc_index], _worker_num_perm,
                    _worker_shingle_k).hashvalues


def dedup(
    docs: list[dict],
    *,
    threshold: float = 0.7,
    num_perm: int = 128,
    shingle_k: int = 5,
    workers: int | None = None,
) -> tuple[list[dict], dict]:
    """Return (kept_docs, stats).

    Docs are processed longest-first so that when near-duplicates exist we keep
    the most complete version. Exact duplicates are removed first (cheap).
    """
    global _worker_texts, _worker_num_perm, _worker_shingle_k
    stats = {"input": len(docs), "exact_dups": 0, "near_dups": 0}

    # 1) exact dedup (cheap, serial).
    seen_exact: set[str] = set()
    unique: list[dict] = []
    for doc in docs:
        key = exact_key(doc["text"])
        if key in seen_exact:
            stats["exact_dups"] += 1
            continue
        seen_exact.add(key)
        unique.append(doc)

    # 2) MinHash signatures — parallel. Sort longest-first here so the signature
    #    order matches the insertion order below.
    unique.sort(key=lambda doc: len(doc["text"]), reverse=True)
    _worker_texts = [doc["text"] for doc in unique]
    _worker_num_perm, _worker_shingle_k = num_perm, shingle_k
    n_unique = len(unique)
    workers = workers or min(32, (os.cpu_count() or 2))
    if workers > 1 and n_unique >= 2000:
        with Pool(workers) as pool:                  # forks: inherits _worker_texts
            signatures = pool.map(_worker_signature, range(n_unique), chunksize=256)
    else:
        signatures = [_worker_signature(i) for i in range(n_unique)]
    _worker_texts = []                               # release the text copy

    # 3) LSH insert/query — serial, but fast (dict ops on precomputed signatures).
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    for i, doc in enumerate(unique):
        signature = MinHash(num_perm=num_perm, hashvalues=signatures[i])
        if lsh.query(signature):
            stats["near_dups"] += 1
            continue
        lsh.insert(str(i), signature)
        kept.append(doc)

    stats["kept"] = len(kept)
    return kept, stats
