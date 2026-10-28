"""A compact, directly implemented BM25 inverted index."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Iterable

import numpy as np


_TOKEN = re.compile(r"(?u)\b[^\W_]+(?:'[^\W_]+)?\b")
_MONTHS = {
    "jan": "1",
    "january": "1",
    "feb": "2",
    "february": "2",
    "mar": "3",
    "march": "3",
    "apr": "4",
    "april": "4",
    "may": "5",
    "jun": "6",
    "june": "6",
    "jul": "7",
    "july": "7",
    "aug": "8",
    "august": "8",
    "sep": "9",
    "sept": "9",
    "september": "9",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NATURAL_DATE = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*,?\s*(\d{{4}}))?\b"
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")


def _date_aliases(text: str) -> list[str]:
    aliases: list[str] = []
    for match in _NATURAL_DATE.finditer(text):
        month = int(_MONTHS[match.group(1)])
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        aliases.append(f"monthday_{month:02d}_{day:02d}")
        if year is not None:
            aliases.append(f"date_{year:04d}_{month:02d}_{day:02d}")
    for match in _ISO_DATE.finditer(text):
        year, month, day = (int(value) for value in match.groups())
        aliases.append(f"monthday_{month:02d}_{day:02d}")
        aliases.append(f"date_{year:04d}_{month:02d}_{day:02d}")
    return list(dict.fromkeys(aliases))


def tokenize(text: str) -> list[str]:
    """Tokenize text with aliases that align natural-language and ISO dates."""

    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    date_aliases = _date_aliases(normalized)
    normalized = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", normalized)
    output: list[str] = []
    for raw in _TOKEN.findall(normalized):
        token = raw.strip("_-'")
        if token.endswith("'s"):
            token = token[:-2]
        if not token:
            continue
        if token.isdigit():
            token = str(int(token))
        output.append(token)
        month_number = _MONTHS.get(token)
        if month_number is not None:
            output.append(month_number)
    return output + date_aliases


@dataclass(slots=True)
class BM25Index:
    vocabulary: dict[str, int]
    document_lengths: np.ndarray
    posting_offsets: np.ndarray
    posting_documents: np.ndarray
    posting_frequencies: np.ndarray
    inverse_document_frequencies: np.ndarray
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    @property
    def document_count(self) -> int:
        return int(self.document_lengths.shape[0])

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_lengths: list[int] = []
        for document_index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            document_lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((document_index, frequency))

        vocabulary_terms = sorted(postings)
        vocabulary = {term: index for index, term in enumerate(vocabulary_terms)}
        offsets = [0]
        posting_documents: list[int] = []
        posting_frequencies: list[int] = []
        idf: list[float] = []
        document_count = len(document_lengths)
        for term in vocabulary_terms:
            term_postings = postings[term]
            document_frequency = len(term_postings)
            idf.append(
                math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )
            posting_documents.extend(document for document, _ in term_postings)
            posting_frequencies.extend(frequency for _, frequency in term_postings)
            offsets.append(len(posting_documents))

        lengths = np.asarray(document_lengths, dtype=np.float32)
        return cls(
            vocabulary=vocabulary,
            document_lengths=lengths,
            posting_offsets=np.asarray(offsets, dtype=np.int64),
            posting_documents=np.asarray(posting_documents, dtype=np.int32),
            posting_frequencies=np.asarray(posting_frequencies, dtype=np.float32),
            inverse_document_frequencies=np.asarray(idf, dtype=np.float32),
            average_document_length=float(lengths.mean()) if len(lengths) else 0.0,
            k1=k1,
            b=b,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if top_k <= 0 or not self.document_count:
            return []
        scores = np.zeros(self.document_count, dtype=np.float32)
        query_terms = Counter(tokenize(query))
        average_length = self.average_document_length or 1.0
        for term, query_frequency in query_terms.items():
            term_id = self.vocabulary.get(term)
            if term_id is None:
                continue
            start = int(self.posting_offsets[term_id])
            end = int(self.posting_offsets[term_id + 1])
            documents = self.posting_documents[start:end]
            frequencies = self.posting_frequencies[start:end]
            length_normalization = self.k1 * (
                1.0 - self.b + self.b * self.document_lengths[documents] / average_length
            )
            contribution = (
                self.inverse_document_frequencies[term_id]
                * frequencies
                * (self.k1 + 1.0)
                / (frequencies + length_normalization)
            )
            scores[documents] += contribution * query_frequency
        candidates = np.flatnonzero(scores > 0)
        ranked = sorted(candidates.tolist(), key=lambda index: (-float(scores[index]), index))
        return [(index, float(scores[index])) for index in ranked[:top_k]]

    def save(self, directory: Path) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        vocabulary_path = directory / "bm25_vocabulary.json"
        arrays_path = directory / "bm25_arrays.npz"
        terms = [term for term, _ in sorted(self.vocabulary.items(), key=lambda item: item[1])]
        vocabulary_path.write_text(
            json.dumps(terms, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            arrays_path,
            average_document_length=np.asarray([self.average_document_length], dtype=np.float32),
            b=np.asarray([self.b], dtype=np.float32),
            document_lengths=self.document_lengths,
            inverse_document_frequencies=self.inverse_document_frequencies,
            k1=np.asarray([self.k1], dtype=np.float32),
            posting_documents=self.posting_documents,
            posting_frequencies=self.posting_frequencies,
            posting_offsets=self.posting_offsets,
        )
        return {
            "algorithm": "bm25_okapi",
            "average_document_length": self.average_document_length,
            "b": self.b,
            "document_count": self.document_count,
            "k1": self.k1,
            "tokenizer": "unicode_words_composite_date_aliases_v2",
            "vocabulary_size": len(self.vocabulary),
            "files": {
                vocabulary_path.name: sha256(vocabulary_path.read_bytes()).hexdigest(),
                arrays_path.name: sha256(arrays_path.read_bytes()).hexdigest(),
            },
        }

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        terms = json.loads((directory / "bm25_vocabulary.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "bm25_arrays.npz")
        return cls(
            vocabulary={str(term): index for index, term in enumerate(terms)},
            document_lengths=arrays["document_lengths"],
            posting_offsets=arrays["posting_offsets"],
            posting_documents=arrays["posting_documents"],
            posting_frequencies=arrays["posting_frequencies"],
            inverse_document_frequencies=arrays["inverse_document_frequencies"],
            average_document_length=float(arrays["average_document_length"][0]),
            k1=float(arrays["k1"][0]),
            b=float(arrays["b"][0]),
        )
