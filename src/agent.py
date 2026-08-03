from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    NO_CONTEXT_MESSAGE = (
        "Khong tim thay ngu canh lien quan trong co so tri thuc de tra loi cau hoi nay."
    )

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def build_prompt(self, question: str, results: list[dict]) -> str:
        """Ghep cac chunk truy xuat duoc thanh context co danh so nguon."""
        blocks = []
        for index, result in enumerate(results, start=1):
            source = result.get("metadata", {}).get("source_url") or result.get(
                "metadata", {}
            ).get("doc_id", result.get("id", "unknown"))
            blocks.append(
                f"[{index}] (nguon: {source} | score={result.get('score', 0.0):.3f})\n"
                f"{result.get('content', '')}"
            )
        context = "\n\n".join(blocks)

        return (
            "Ban la tro ly tra loi dua tren tai lieu duoc cung cap.\n"
            "Chi su dung thong tin trong phan NGU CANH ben duoi. Neu ngu canh khong "
            "du thong tin, hay noi ro la khong tim thay trong tai lieu. "
            "Trich dan so hieu nguon [1], [2], ... khi tra loi.\n\n"
            "=== NGU CANH ===\n"
            f"{context}\n\n"
            "=== CAU HOI ===\n"
            f"{question}\n\n"
            "=== TRA LOI ==="
        )

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        """Truy xuat -> dung prompt -> goi LLM.

        `metadata_filter` la tuy chon: khi duoc truyen, agent truy xuat qua
        `search_with_filter` de ngu canh dua vao LLM dung bang tap chunk da loc,
        tranh tinh trang bang ket qua va cau tra loi noi ve hai chunk khac nhau.
        """
        if metadata_filter:
            results = self.store.search_with_filter(
                question, top_k=top_k, metadata_filter=metadata_filter
            )
        else:
            results = self.store.search(question, top_k=top_k)
        if not results:
            return self.NO_CONTEXT_MESSAGE

        prompt = self.build_prompt(question, results)
        response = self.llm_fn(prompt)
        return response if isinstance(response, str) else str(response)