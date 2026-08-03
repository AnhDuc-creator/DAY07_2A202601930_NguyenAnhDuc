"""
scripts/benchmark.py — chay 5 benchmark query cua nhom tren chien luoc ca nhan.

Chay:
    EMBEDDING_PROVIDER=local python3 scripts/benchmark.py            # chien luoc ArticleChunker
    EMBEDDING_PROVIDER=local python3 scripts/benchmark.py --baseline # FixedSizeChunker de so sanh

PowerShell (Windows):
    $env:EMBEDDING_PROVIDER="local"; python scripts/benchmark.py

In ra:
    - Bang so sanh 3 chien luoc chunking (Bai tap 3.1 Buoc 1)
    - Bang du doan do tuong tu cosine (Bai tap 3.3)
    - Bang ket qua 5 benchmark query (REPORT_CANHAN Phan 5)
    - So sanh search() vs search_with_filter() cho query can loc metadata
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from ingest import build_knowledge_base, load_documents
from src.agent import KnowledgeBaseAgent
from src.chunking import (
    ArticleChunker,
    ChunkingStrategyComparator,
    FixedSizeChunker,
    compute_similarity,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    LocalEmbedder,
    _mock_embed,
)

DATA_DIR = "data/k4_ecommerce"

# 5 benchmark query cua NHOM (phai trung voi REPORT_NHOM.md).
BENCHMARK = [
    {
        "id": 1,
        "query": "Khi giao hàng bị chậm trễ thì người bán phải làm gì cho khách hàng?",
        "gold": "Phải thông tin kịp thời cho khách hàng và tạo cơ hội để khách hàng có thể hủy hợp đồng nếu muốn (Điều 33 khoản 2).",
        "expect_doc": "nd52-van-chuyen-giao-nhan",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Giá niêm yết trên website không ghi rõ đã bao gồm phí vận chuyển thì được hiểu thế nào?",
        "gold": "Nếu không ghi rõ, giá niêm yết được hiểu là đã bao gồm mọi chi phí liên quan như thuế, phí đóng gói, phí vận chuyển (Điều 31 khoản 2).",
        "expect_doc": "nd52-thong-tin-hang-hoa-gia-dieu-kien",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Người bán trên sàn giao dịch thương mại điện tử có những trách nhiệm gì?",
        "gold": "Cung cấp đầy đủ chính xác thông tin định danh khi đăng ký, cung cấp thông tin hàng hóa theo Điều 30 đến 34, bảo đảm tính trung thực, tuân thủ pháp luật về thanh toán quảng cáo khuyến mại sở hữu trí tuệ bảo vệ người tiêu dùng, và thực hiện nghĩa vụ thuế (Điều 37).",
        "expect_doc": "nd52-trach-nhiem-nguoi-ban",
        "filter": {"customer_role": "seller"},
    },
    {
        "id": 4,
        "query": "Hệ thống bị tấn công làm lộ thông tin cá nhân của người tiêu dùng thì phải thông báo cho cơ quan chức năng trong bao lâu?",
        "gold": "Trong vòng 24 giờ sau khi phát hiện sự cố (Điều 72 khoản 3).",
        "expect_doc": "nd52-bao-ve-thong-tin-ca-nhan",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Người bán không công bố thời hạn trả lời đề nghị giao kết hợp đồng thì sau bao lâu đề nghị hết hiệu lực?",
        "gold": "Sau 12 giờ kể từ khi khách hàng gửi đề nghị giao kết mà không được trả lời thì đề nghị chấm dứt hiệu lực (Điều 20 khoản 2).",
        "expect_doc": "nd52-quy-trinh-dat-hang",
        "filter": None,
    },
]

# 5 cặp câu cho Bài tập 3.3 (dự đoán trước khi chạy).
SIMILARITY_PAIRS = [
    ("Tôi muốn trả lại hàng và lấy lại tiền", "Chính sách hoàn trả và hoàn tiền cho khách hàng", "cao"),
    ("Thời hạn giao hàng dự kiến là bao lâu", "Thời hạn ước tính cho việc giao hàng hoặc cung ứng dịch vụ", "cao"),
    ("Bảo vệ thông tin cá nhân của người tiêu dùng", "Điều kiện đăng ký thiết lập website bán hàng", "thấp"),
    ("Thanh toán trực tuyến phải được mã hóa", "Bảo mật giao dịch thanh toán của khách hàng", "cao"),
    ("Quy định về đấu giá trực tuyến trên sàn thương mại điện tử", "Công thức tính diện tích hình tròn", "thấp"),
]


def select_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as exc:
            print(f"[!] Local embedder khong san sang ({exc}); tam dung mock.")
            return _mock_embed
    return _mock_embed


def extractive_llm(prompt: str) -> str:
    """LLM thay the: tra ve cau dau tien cua chunk co diem cao nhat trong ngu canh.

    Dung de danh gia grounding ma khong phu thuoc API key. Neu co LLM that,
    thay ham nay bang lenh goi model.
    """
    marker = "=== NGU CANH ==="
    if marker not in prompt:
        return "Khong co ngu canh."
    body = prompt.split(marker, 1)[1].split("=== CAU HOI ===", 1)[0]
    first_block = body.split("\n[2]", 1)[0]
    lines = [
        line.strip()
        for line in first_block.splitlines()
        if line.strip() and not line.strip().startswith("[1]") and not line.strip().startswith("(nguon")
    ]
    text = " ".join(lines)
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    return ". ".join(sentences[:3])[:400]


def short(text: str, width: int = 90) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("..." if len(flat) > width else "")


def print_strategy_comparison() -> None:
    docs = load_documents(DATA_DIR)[:3]
    print("\n## Bang 1 - So sanh 3 chien luoc chunking (baseline, Bai tap 3.1)\n")
    print("| Tai lieu | Chien luoc | So chunk | Do dai TB | Min | Max |")
    print("|---|---|---|---|---|---|")
    for doc in docs:
        result = ChunkingStrategyComparator().compare(doc.content, chunk_size=500)
        for name, stats in result.items():
            print(
                f"| {doc.id} | {name} | {stats['count']} | {stats['avg_length']:.0f} "
                f"| {stats['min_length']} | {stats['max_length']} |"
            )
        custom = ArticleChunker().chunk(doc.content)
        lengths = [len(c) for c in custom] or [0]
        print(
            f"| {doc.id} | article (custom) | {len(custom)} | {sum(lengths)/len(lengths):.0f} "
            f"| {min(lengths)} | {max(lengths)} |"
        )


def print_similarity_table(embedder) -> None:
    print("\n## Bang 2 - Du doan do tuong tu cosine (Bai tap 3.3)\n")
    print("| Cap | Cau A | Cau B | Du doan | Diem thuc te |")
    print("|---|---|---|---|---|")
    for index, (a, b, prediction) in enumerate(SIMILARITY_PAIRS, start=1):
        score = compute_similarity(embedder(a), embedder(b))
        print(f"| {index} | {short(a, 45)} | {short(b, 45)} | {prediction} | {score:.3f} |")


def run_benchmark(embedder, chunker, label: str) -> None:
    store = build_knowledge_base(DATA_DIR, embedding_fn=embedder, chunker=chunker)
    agent = KnowledgeBaseAgent(store=store, llm_fn=extractive_llm)

    print(f"\n## Bang 3 - Ket qua 5 benchmark query (chien luoc: {label})\n")
    print(f"So chunk trong store: {store.get_collection_size()}\n")
    print("| # | Cau hoi | Top-1 chunk | Score | Relevant | Cau tra loi cua Agent |")
    print("|---|---|---|---|---|---|")

    hits_top1 = 0
    hits_top3 = 0
    for case in BENCHMARK:
        if case["filter"]:
            results = store.search_with_filter(case["query"], top_k=3, metadata_filter=case["filter"])
        else:
            results = store.search(case["query"], top_k=3)

        if not results:
            print(f"| {case['id']} | {short(case['query'], 60)} | (khong co ket qua) | - | KHONG | - |")
            continue

        top1 = results[0]
        top3_docs = [r["metadata"].get("doc_id") for r in results]
        is_top1 = top1["metadata"].get("doc_id") == case["expect_doc"]
        is_top3 = case["expect_doc"] in top3_docs
        hits_top1 += int(is_top1)
        hits_top3 += int(is_top3)

        answer = agent.answer(case["query"], top_k=3, metadata_filter=case["filter"])
        print(
            f"| {case['id']} | {short(case['query'], 60)} | {short(top1['content'], 70)} "
            f"| {top1['score']:.3f} | {'CO' if is_top3 else 'KHONG'} | {short(answer, 90)} |"
        )

    print(f"\nSo query co chunk lien quan trong top-3: {hits_top3} / {len(BENCHMARK)}")
    print(f"So query co chunk lien quan o top-1: {hits_top1} / {len(BENCHMARK)}")

    # So sanh co loc / khong loc cho query 3.
    case = BENCHMARK[2]
    print("\n## Bang 4 - Metadata filter co giup khong? (query 3)\n")
    print("| Che do | Top-1 doc_id | Score | Trung gold doc |")
    print("|---|---|---|---|")
    plain = store.search(case["query"], top_k=3)
    filtered = store.search_with_filter(case["query"], top_k=3, metadata_filter=case["filter"])
    for mode, res in (("search()", plain), ("search_with_filter(seller)", filtered)):
        if not res:
            print(f"| {mode} | (rong) | - | KHONG |")
            continue
        doc_id = res[0]["metadata"].get("doc_id")
        print(f"| {mode} | {doc_id} | {res[0]['score']:.3f} | {'CO' if doc_id == case['expect_doc'] else 'KHONG'} |")


def main() -> int:
    use_baseline = "--baseline" in sys.argv
    embedder = select_embedder()
    backend = getattr(embedder, "_backend_name", type(embedder).__name__)
    print(f"Backend nhung: {backend}")
    if backend == "mock embeddings fallback":
        print("[!] Dang chay bang MOCK. Dat EMBEDDING_PROVIDER=local truoc khi lay so cho bao cao.")

    print_strategy_comparison()
    print_similarity_table(embedder)

    if use_baseline:
        run_benchmark(embedder, FixedSizeChunker(chunk_size=500, overlap=50), "FixedSizeChunker(500/50) - baseline")
    else:
        run_benchmark(embedder, ArticleChunker(max_chunk_size=900), "ArticleChunker (custom, cat theo Dieu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())