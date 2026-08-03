from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    # Cat SAU dau ket cau (. ! ?) khi phia sau la khoang trang hoac xuong dong.
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_END.split(text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        size = self.max_sentences_per_chunk
        for start in range(0, len(sentences), size):
            group = sentences[start : start + size]
            chunks.append(" ".join(group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        pieces = self._split(text, self.separators)
        return [p.strip() for p in pieces if p and p.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: doan da du nho.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: het separator -> cat cung theo chunk_size de khong mat du lieu.
        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        rest = remaining_separators[1:]

        parts = current_text.split(separator)
        if len(parts) == 1:
            # Separator khong xuat hien -> thu separator uu tien ke tiep.
            return self._split(current_text, rest)

        # Gop cac phan nho lai cho toi sat chunk_size; phan nao van qua lon thi de quy.
        chunks: list[str] = []
        buffer = ""
        for part in parts:
            candidate = part if not buffer else buffer + separator + part

            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""

            if len(part) <= self.chunk_size:
                buffer = part
            else:
                chunks.extend(self._split(part, rest))

        if buffer:
            chunks.append(buffer)
        return chunks


class ArticleChunker:
    """Chien luoc chunking tuy chinh cho corpus chinh sach TMDT (K4).

    Ly do thiet ke: corpus cua nhom la van ban chinh sach co cau truc theo dieu
    khoan ("## Dieu 33. ..."). Moi dieu la mot don vi ngu nghia tron ven, tra loi
    duoc mot cau hoi chinh sach. Cat theo ranh gioi dieu giu nguyen tieu de +
    toan bo noi dung dieu do trong cung mot chunk, thay vi cat ngang giua khoan
    nhu FixedSizeChunker.

    Hai quyet dinh thiet ke quan trong:
      1. Tieu de cap 1 (`# ...`) cua tai lieu duoc gan lam TIEN TO cho MOI chunk,
         de moi chunk mang du ngu canh chu de khi embed (chunk "Dieu 20" khong
         con mo coi khoi tai lieu "Quy trinh dat hang truc tuyen").
      2. Chi cat tai heading cap 2 tro xuong (`## Dieu ...`), nen mot dieu khong
         bao gio bi tach roi khoi so hieu dieu cua no.

    Neu mot dieu dai hon max_chunk_size, chunk do duoc chia tiep bang
    RecursiveChunker nhung van gan lai tien to tieu de o dau moi manh.
    """

    _DOC_TITLE = re.compile(r"^#\s+(.*)$", re.MULTILINE)
    _SECTION = re.compile(r"^#{2,6}\s+.*$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 900) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        title_match = self._DOC_TITLE.search(text)
        doc_title = title_match.group(1).strip() if title_match else ""
        prefix = f"{doc_title}\n" if doc_title else ""

        matches = list(self._SECTION.finditer(text))
        if not matches:
            body = text[title_match.end():] if title_match else text
            return [
                f"{prefix}{piece}".strip()
                for piece in RecursiveChunker(chunk_size=self.max_chunk_size).chunk(body)
            ]

        sections: list[str] = []
        for index, match in enumerate(matches):
            heading = match.group().lstrip("#").strip()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            sections.append(f"{heading}\n{body}".strip())

        chunks: list[str] = []
        for section in sections:
            full = f"{prefix}{section}".strip()
            if len(full) <= self.max_chunk_size:
                chunks.append(full)
                continue
            heading, _, body = section.partition("\n")
            budget = max(120, self.max_chunk_size - len(prefix) - len(heading) - 2)
            for piece in RecursiveChunker(chunk_size=budget).chunk(body):
                chunks.append(f"{prefix}{heading}\n{piece}".strip())
        return [c for c in chunks if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=max(1, chunk_size // 10)),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            count = len(chunks)
            comparison[name] = {
                "count": count,
                "avg_length": (sum(lengths) / count) if count else 0.0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "chunks": chunks,
            }
        return comparison
