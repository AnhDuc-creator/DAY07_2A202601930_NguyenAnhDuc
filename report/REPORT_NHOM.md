# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Pilot
**Thành viên:** Nguyễn Văn Dương, Nguyễn Văn Tấn, Nguyễn Anh Đức
**Ngày:** 2026-08-03

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Phạm vi bộ tài liệu (Scope)

**Chủ đề (cố định theo lớp K4):** Chính sách thương mại điện tử / hỗ trợ khách hàng cho người mua và người bán.

**Phạm vi cụ thể nhóm tập trung:**
> Chọn 5 tài liệu về đổi trả, đăng bán, thanh toán, khiếu nại và quyền riêng tư để hỗ trợ benchmark retrieval có metadata lọc rõ ràng.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | returns-policy | https://help.shopee.vn/portal/4/article/77244 | 2026-08-02 / 2026.08 | ~600 | customer_role=buyer, category=returns, department=support, language=vi |
| 2 | seller-listing | https://help.shopee.vn/portal/4/article/77246 | 2026-08-02 / 2026.08 | ~550 | customer_role=seller, category=listing, department=seller-operations, language=vi |
| 3 | payment-terms | https://help.shopee.vn/portal/4/article/77223 | 2026-08-02 / 2026.08 | ~600 | customer_role=buyer, category=payment, department=finance, language=vi |
| 4 | seller-appeal | https://help.shopee.vn/portal/4/article/79191 | 2026-08-02 / 2026.08 | ~650 | customer_role=seller, category=dispute, department=seller-operations, language=vi |
| 5 | privacy-and-data | https://help.shopee.vn/portal/4/article/77212 | 2026-08-02 / 2026.08 | ~650 | customer_role=both, category=privacy, department=trust-and-safety, language=vi |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| customer_role | string | buyer / seller / both | Cho phép lọc theo vai trò của người dùng trong benchmark. |
| category | string | returns / payment / privacy | Giúp phân biệt các nhóm chính sách và nhắm đúng tài liệu. |
| department | string | support / seller-operations | Cung cấp ngữ cảnh tổ chức và giúp giảm nhiễu khi query rộng. |
| language | string | vi | Đảm bảo corpus có ngữ cảnh ngôn ngữ nhất quán. |
| effective_date | string | 2026-08-01 | Cho biết thời điểm hiệu lực của chính sách để tránh nhầm lẫn. |
| source_url | string | https://help.shopee.vn/... | Là nền tảng provenance và minh bạch nguồn. |
| retrieved_at | string | 2026-08-02 | Ghi lại thời điểm thu thập, tăng tính minh bạch. |
| document_version | string | 2026.08 / not-stated | Cho biết phiên bản hoặc trạng thái tài liệu. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| returns-policy | FixedSizeChunker (`fixed_size`) | 3 | ~180 ký tự | Có, nhưng có thể cắt giữa ý. |
| returns-policy | SentenceChunker (`by_sentences`) | 3 | ~190 ký tự | Có, giữ cấu trúc câu tốt hơn. |
| seller-listing | RecursiveChunker (`recursive`) | 2 | ~220 ký tự | Có, phù hợp với tiêu đề và đoạn ngắn. |

### Chiến lược của từng thành viên

**Nguyễn Văn Dương**
- **Loại chiến lược:** Sentence chunking
- **Mô tả & lý do chọn cho chủ đề này:** Chia theo câu giúp giữ ý nghĩa từng quy định và giữ được ngữ cảnh chính sách ngắn. 
- **Code snippet (nếu custom):**
```python
from src.chunking import SentenceChunker
chunker = SentenceChunker(max_sentences_per_chunk=2)
```

**Nguyễn Văn Tấn**
- **Loại chiến lược:** Recursive chunking
- **Mô tả & lý do chọn:** Chia theo tiêu đề và đoạn văn ngắn làm tăng độ liên kết giữa các phần liên quan. 
- **Code snippet (nếu custom):**
```python
from src.chunking import RecursiveChunker
chunker = RecursiveChunker(chunk_size=220, separators=["\n\n", "\n", "."])
```

**Nguyễn Anh Đức**
- **Loại chiến lược:** Fixed-size chunking
- **Mô tả & lý do chọn:** Dùng cho mục đích baseline và so sánh tốc độ, phù hợp khi corpus nhỏ và nội dung có cấu trúc rõ. 
- **Code snippet (nếu custom):**
```python
from src.chunking import FixedSizeChunker
chunker = FixedSizeChunker(chunk_size=180, overlap=20)
```

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Văn Dương | Sentence | 8.5 | Giữ câu và điều kiện rõ | Có thể tạo chunk quá ngắn ở một số tài liệu |
| Nguyễn Văn Tấn | Recursive | 9.0 | Phù hợp với cấu trúc heading và paragraph | Cần cấu trúc đầu vào tốt hơn |
| Nguyễn Anh Đức | Fixed-size | 8.0 | Dễ triển khai | Dễ cắt giữa câu và làm mất ngữ cảnh |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Recursive chunking phù hợp nhất vì corpus có cấu trúc đoạn và tiêu đề rõ ràng, trong khi sentence chunking cũng tốt cho chính sách ngắn. Với bộ dữ liệu này, recursive giúp tìm đúng quy định hơn khi query có nhiều điều kiện.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Shopee cho phép người bán đổi trả sản phẩm trong bao lâu? | Chính sách đổi trả cho phép người bán gửi yêu cầu trong vòng 15 ngày kể từ khi nhận hàng, tùy theo điều kiện cụ thể của từng trường hợp. | returns-policy |
| 2 | Ai là đối tượng được áp dụng chính sách bảo mật dữ liệu cá nhân? | Chính sách này áp dụng cho cả người bán và người mua đang sử dụng dịch vụ, trừ khi có tuyên bố rõ ràng ngược lại. | privacy-and-data |
| 3 | Nếu khách hàng thanh toán thất bại thì quy trình xử lý như thế nào? | Đơn hàng sẽ bị khóa hoặc chờ xử lý lại cho đến khi thanh toán được khôi phục hoặc hủy theo quy định. | payment-terms |
| 4 | Người bán có thể khiếu nại quyết định của nền tảng bằng cách nào? | Người bán có thể gửi khiếu nại kèm bằng chứng và lời giải thích để nền tảng rà soát lại quyết định. | seller-appeal |
| 5 | Những điều kiện nào khiến sản phẩm bị từ chối đăng bán? | Sản phẩm bị từ chối đăng bán khi vi phạm quy định về nội dung, hàng hóa cấm, thương hiệu hoặc giấy tờ liên quan. | seller-listing |

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Shopee cho phép người bán đổi trả sản phẩm trong bao lâu? | Recursive + metadata | Không rõ | Top-3 có dấu hiệu nhiễu; `returns-policy` xuất hiện nhưng không có keyword evidence đủ mạnh. |
| 2 | Ai là đối tượng được áp dụng chính sách bảo mật dữ liệu cá nhân? | Recursive + metadata | Có | Metadata `category=privacy` giúp đưa `privacy-and-data` vào top-3. |
| 3 | Nếu khách hàng thanh toán thất bại thì quy trình xử lý như thế nào? | Recursive | Có | Đây là câu hỏi có kết quả ổn định nhất, giữ `payment-terms` ở đầu. |
| 4 | Người bán có thể khiếu nại quyết định của nền tảng bằng cách nào? | Recursive + metadata | Không | Cả unfiltered và filtered đều không đưa `seller-appeal` vào top-3; đây là case thất bại chính. |
| 5 | Những điều kiện nào khiến sản phẩm bị từ chối đăng bán? | Recursive + metadata | Có | Filter theo `customer_role=seller` giúp đưa đúng `seller-listing` vào top-3. |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> Có. Với câu hỏi về quyền riêng tư và đăng bán, việc dùng metadata `category=privacy` hoặc `customer_role=seller` giúp giảm nhiễu và đưa đúng tài liệu vào top-3. Tuy nhiên, với câu hỏi về đổi trả và khiếu nại, embedding similarity vẫn còn bị nhiễu nên cần cải thiện chunking hoặc thêm dữ liệu bản ghi có từ khóa rõ hơn.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> Metadata `customer_role` và `category` là chìa khóa để phân biệt tài liệu cho buyer và seller; chunking theo cấu trúc đoạn văn tốt hơn chunking cố định khi query hỏi về điều kiện cụ thể. Benchmark thực tế cho thấy filter giúp cải thiện rõ rệt ở các câu hỏi về quyền riêng tư và đăng bán.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng một corpus nhưng chiến lược chunking khác nhau dẫn đến khác biệt về sự rõ ràng của câu trả lời và khả năng giữ ngữ cảnh. Recursive và sentence chunking tốt hơn fixed-size, nhưng khi query không khớp từ khóa chính xác, cả hai vẫn có thể bị nhiễu. Đây là lý do metadata-based filtering nên đi cùng với chunk quality tốt hơn.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Nhóm sẽ bổ sung thêm tài liệu có nhiều từ khóa cụ thể về đổi trả và khiếu nại để tăng độ phủ cho các benchmark query, đồng thời chuẩn hóa metadata trước khi nạp vào kho vector để filter hiệu quả hơn.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 9/10 |
| Thiết kế chiến lược (Strategy Design) | 14/15 |
| Chất lượng truy xuất (Retrieval Quality) | 9/10 |
| Thuyết trình (Demo) | 4/5 |
| **Tổng phần nhóm** | **36/40** |
