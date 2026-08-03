# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Anh Đức (2A202601930)
**Nhóm:** Pilot
**Ngày:** 2026-08-03

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

**Cấu hình khi chạy đánh giá:** `EMBEDDING_PROVIDER=local`, model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `top_k=3`. Corpus: bộ 5 tài liệu chính sách thương mại điện tử của nhóm trong `data/k4_ecommerce/`. Chiến lược của tôi theo phân công nhóm: `FixedSizeChunker(chunk_size=180, overlap=20)`. Bộ 5 câu hỏi đánh giá thống nhất theo `REPORT_NHOM.md`.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

Hai vector embedding chỉ về gần cùng một hướng trong không gian ngữ nghĩa, nghĩa là hai đoạn văn bản nói về cùng một chủ đề hoặc cùng một ý, bất kể chúng dài ngắn khác nhau hay dùng từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**

- Câu A: Thời hạn giao hàng dự kiến là bao lâu.
- Câu B: Thời hạn ước tính cho việc giao hàng hoặc cung ứng dịch vụ.
- Tại sao tương đồng: cùng nói về một khái niệm là thời hạn giao hàng, chỉ khác ở cách diễn đạt của người hỏi và của văn bản chính sách. Điểm thực đo được là 0.815.

**Ví dụ có độ tương tự THẤP:**

- Câu A: Quy định về đấu giá trực tuyến trên sàn thương mại điện tử.
- Câu B: Công thức tính diện tích hình tròn.
- Tại sao khác: hai câu thuộc hai miền hoàn toàn rời nhau, không chia sẻ khái niệm nào. Điểm thực đo được là 0.169.

**Tại sao cosine được ưu tiên hơn khoảng cách Euclid cho text embeddings?**

Cosine chỉ đo góc giữa hai vector nên bỏ qua độ dài vector, trong khi độ dài của embedding thường phản ánh độ dài hoặc tần suất từ của văn bản chứ không phản ánh ý nghĩa. Nếu dùng khoảng cách Euclid, một đoạn chính sách dài và một câu hỏi ngắn cùng chủ đề vẫn bị coi là xa nhau chỉ vì chênh lệch độ lớn; cosine cho hai trường hợp đó điểm gần như nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Bước nhảy = 500 - 50 = 450. Số chunk = ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23.

Đáp án: 23 chunks.

**Nếu overlap tăng lên 100 thì sao?**

Bước nhảy giảm còn 400, số chunk = ceil((10000 - 100) / 400) = ceil(24.75) = 25, tức tăng thêm 2 chunk. Overlap lớn hơn làm tăng chi phí lưu trữ và số lần embed, nhưng giảm rủi ro một câu hoặc một ý bị cắt đôi ngay ranh giới chunk khiến không chunk nào chứa đủ thông tin để trả lời. Với chính sách nhiều điều kiện và ngoại lệ đi kèm nhau, overlap là cách bảo hiểm rẻ cho việc mất ngữ cảnh ở biên.

Điều này liên quan trực tiếp đến chiến lược tôi được phân công. `FixedSizeChunker(180, 20)` có tỉ lệ overlap khoảng 11 phần trăm, khá mỏng với tài liệu chính sách nơi một quy định và ngoại lệ của nó thường nằm cách nhau vài chục ký tự.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`**

Tôi dùng regex `(?<=[.!?])\s+` với lookbehind để cắt sau dấu kết câu mà vẫn giữ dấu chấm trong câu, thay vì `split(". ")` làm mất dấu. Sau khi tách, tôi strip từng câu, loại chuỗi rỗng, rồi gom theo lô `max_sentences_per_chunk` bằng slicing. Edge case đã xử lý: text rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng; text không có dấu kết câu nào trả về đúng một chunk là toàn bộ text.

**`RecursiveChunker.chunk` / `_split`**

`_split` nhận đoạn text hiện tại và danh sách separator còn lại. Hai base case: đoạn đã ngắn hơn `chunk_size` thì trả nguyên vẹn; hết separator hoặc gặp separator rỗng thì cắt cứng theo `chunk_size` để không bao giờ mất dữ liệu. Trường hợp còn lại, tôi split theo separator ưu tiên cao nhất; nếu separator không xuất hiện thì gọi đệ quy với separator kế tiếp thay vì tạo chunk thừa. Sau khi split, tôi gom tham lam các mảnh nhỏ vào buffer cho tới sát `chunk_size`, mảnh nào tự nó vẫn quá lớn thì đệ quy xuống cấp separator thấp hơn.

Một chi tiết tôi chú ý khi lập trình: danh sách separator bắt buộc phải kết thúc bằng `" "` và `""`. Nếu dừng ở `"."` thì đoạn nào không chứa dấu chấm mà vẫn dài hơn `chunk_size` sẽ không cắt được tiếp và trả về nguyên đoạn quá khổ. Đây là lý do tôi giữ nguyên `DEFAULT_SEPARATORS` đầy đủ thay vì rút gọn.

**`ArticleChunker` — phần mở rộng ngoài yêu cầu**

Ngoài ba chunker bắt buộc, tôi lập trình thêm một chunker tùy chỉnh trong `src/chunking.py`. Nó cắt tại ranh giới heading cấp 2 nên mỗi chunk là trọn một mục, nội dung không tách khỏi tiêu đề mục; tiêu đề cấp 1 của tài liệu được gắn làm tiền tố cho mọi chunk để chunk mang ngữ cảnh chủ đề khi embed; mục nào dài quá ngưỡng thì chia tiếp bằng `RecursiveChunker` nhưng vẫn gắn lại tiền tố. Nếu tài liệu không có heading, chunker tự động lùi về `RecursiveChunker` thay vì lỗi.

Giả thuyết khi thiết kế: chunk trọn một mục sẽ truy vết nguồn tốt hơn, vì câu trả lời của agent luôn kèm được tên mục mà nó trích. Chiến lược này không dùng cho benchmark của nhóm vì tôi được phân công chạy fixed-size, nhưng code vẫn nằm trong `src/` và có thể chạy để đối chiếu.

### Lớp EmbeddingStore

**`add_documents` + `search`**

Tôi chọn backend in-memory là list các dict thay vì ChromaDB để kết quả search deterministic và không phụ thuộc phiên bản thư viện; code vẫn kiểm tra chromadb có sẵn hay không và ghi lại cờ. `_make_record` chuẩn hóa mỗi `Document` thành record gồm id, content, metadata (tự điền `doc_id` nếu thiếu) và embedding tính sẵn tại thời điểm add, nên search không phải embed lại tài liệu. `_search_records` embed câu hỏi một lần rồi tính tích vô hướng với từng embedding đã lưu, sort giảm dần theo score và cắt `top_k`. Vì cả `MockEmbedder` lẫn `LocalEmbedder` đều trả vector đã chuẩn hóa, tích vô hướng ở đây chính là cosine similarity.

**`search_with_filter` + `delete_document`**

Lọc trước rồi mới xếp hạng: tôi lọc `self._store` theo toàn bộ cặp key-value trong `metadata_filter`, so sánh bằng `str()` để không vỡ khi front matter parse ra số hoặc ngày, rồi truyền tập ứng viên đã lọc vào chính `_search_records`. Cách này đảm bảo `top_k` được tính trên tập hợp lệ, không bị chunk ngoài phạm vi chiếm chỗ. `metadata_filter` rỗng hoặc None thì hành vi trùng khớp `search()`. `delete_document` dựng lại store bỏ mọi record có `metadata['doc_id']` khớp, trả về True nếu độ dài giảm.

### Tác tử KnowledgeBaseAgent

**`answer`**

Ba bước: retrieve top-k, dựng prompt, gọi `llm_fn`. Prompt chia khối rõ bằng mốc NGỮ CẢNH và CÂU HỎI; mỗi chunk được đánh số [1], [2] kèm nguồn và score, và hệ thống yêu cầu model chỉ dùng thông tin trong ngữ cảnh, trích số hiệu nguồn khi trả lời, nói rõ khi ngữ cảnh không đủ. Gắn số hiệu nguồn vào từng chunk là để phục vụ tiêu chí truy vết nguồn trong `docs/EVALUATION.md`. Nếu store rỗng, agent trả về thông báo không tìm thấy ngữ cảnh thay vì gọi LLM, tránh model tự bịa.

Tôi bổ sung tham số tùy chọn `metadata_filter` cho `answer`. Lý do: bộ benchmark của nhóm có câu cần lọc theo metadata. Nếu bảng kết quả lấy top-1 từ `search_with_filter` mà agent vẫn gọi `search` không lọc thì hai cột trong báo cáo nói về hai chunk khác nhau, và câu trả lời không phản ánh chunk đã được chấm là liên quan.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
platform win32 -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\projects\VinUni\DAY07_2A202601930_NguyenAnhDuc
collected 42 items

tests/test_solution.py::TestProjectStructure ................ PASSED
tests/test_solution.py::TestClassBasedInterfaces ............. PASSED
tests/test_solution.py::TestFixedSizeChunker ................. PASSED
tests/test_solution.py::TestSentenceChunker .................. PASSED
tests/test_solution.py::TestRecursiveChunker ................. PASSED
tests/test_solution.py::TestEmbeddingStore ................... PASSED
tests/test_solution.py::TestKnowledgeBaseAgent ............... PASSED
tests/test_solution.py::TestComputeSimilarity ................ PASSED
tests/test_solution.py::TestCompareChunkingStrategies ........ PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter ... PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument ..... PASSED

======================= 42 passed in 0.49s =======================
```

**Số lượng bài test vượt qua:** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Gọi `compute_similarity()` trên 5 cặp câu, dự đoán trước khi chạy.

| Cặp | Câu A                                                      | Câu B                                                      | Dự đoán | Điểm thực tế | Đúng?        |
| --- | ---------------------------------------------------------- | ---------------------------------------------------------- | ------- | ------------ | ------------ |
| 1   | Tôi muốn trả lại hàng và lấy lại tiền                      | Chính sách hoàn trả và hoàn tiền cho khách hàng            | cao     | 0.358        | Sai          |
| 2   | Thời hạn giao hàng dự kiến là bao lâu                      | Thời hạn ước tính cho việc giao hàng hoặc cung ứng dịch vụ | cao     | 0.815        | Đúng         |
| 3   | Bảo vệ thông tin cá nhân của người tiêu dùng               | Điều kiện đăng ký thiết lập website bán hàng               | thấp    | 0.467        | Sai một phần |
| 4   | Thanh toán trực tuyến phải được mã hóa                     | Bảo mật giao dịch thanh toán của khách hàng                | cao     | 0.628        | Đúng         |
| 5   | Quy định về đấu giá trực tuyến trên sàn thương mại điện tử | Công thức tính diện tích hình tròn                         | thấp    | 0.169        | Đúng         |

**Kết quả nào bất ngờ nhất?**

Cặp 1 gây bất ngờ nhất. Hai câu này nói về đúng cùng một việc là trả hàng và hoàn tiền, tôi chắc chắn điểm sẽ cao, nhưng thực tế chỉ 0.358, thấp hơn cả cặp 3 vốn được dự đoán là thấp. So sánh với cặp 2 đạt 0.815 thì thấy rõ nguyên nhân: cặp 2 dùng gần như trùng cụm từ khóa ("thời hạn", "giao hàng"), còn cặp 1 diễn đạt cùng ý bằng hai bộ từ vựng khác nhau ("trả lại hàng, lấy lại tiền" so với "hoàn trả, hoàn tiền"), và khác cả ngôi kể, một bên là lời người mua, một bên là ngôn ngữ văn bản chính sách.

Điều này cho thấy embedding của model đa ngữ cỡ nhỏ vẫn bám khá nhiều vào trùng lặp từ vựng bề mặt chứ chưa thực sự trừu tượng hóa được ý. Kết quả này giải thích trực tiếp hai câu thất bại trong benchmark ở mục 5: câu về đổi trả và câu về khiếu nại đều diễn đạt bằng ngôn ngữ khác với ngôn ngữ trong tài liệu, đúng vào điểm yếu mà cặp 1 phơi bày.

Cặp 3 đạt 0.467 dù hai chủ đề khác nhau cũng đáng chú ý: cả hai câu cùng thuộc văn phong chính sách thương mại điện tử nên model bắt được tín hiệu văn phong chung và đẩy điểm nền lên. Nói cách khác, ngưỡng để phân biệt liên quan và không liên quan trong corpus cùng một miền phải đặt cao hơn nhiều so với 0.5, và điểm số cao chưa chắc đã là bằng chứng của một kết quả tốt.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy 5 câu hỏi đánh giá của nhóm (thống nhất trong `REPORT_NHOM.md`) trên mã nguồn cá nhân trong gói `src`, chiến lược `FixedSizeChunker(chunk_size=180, overlap=20)`.

| #   | Câu hỏi (Query)                                                     | Top-1 Chunk truy xuất được (tóm tắt)                                                   | Score | Có liên quan? | Câu trả lời của Agent (tóm tắt)                                         |
| --- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----- | ------------- | ----------------------------------------------------------------------- |
| 1   | Shopee cho phép người bán đổi trả sản phẩm trong bao lâu?           | Chunk từ `returns-policy` nhưng rơi vào phần điều kiện chung, không phải phần thời hạn |       | Không rõ      | Trả lời chung về chính sách đổi trả, không nêu được mốc thời hạn cụ thể |
| 2   | Ai là đối tượng được áp dụng chính sách bảo mật dữ liệu cá nhân?    | Chunk từ `privacy-and-data` nêu phạm vi áp dụng của chính sách                         |       | Có            | Nêu đúng là áp dụng cho cả người mua và người bán                       |
| 3   | Nếu khách hàng thanh toán thất bại thì quy trình xử lý như thế nào? | Chunk từ `payment-terms` về xử lý đơn hàng chưa thanh toán                             |       | Có            | Nêu đúng cơ chế đơn hàng bị khóa hoặc chờ xử lý lại                     |
| 4   | Người bán có thể khiếu nại quyết định của nền tảng bằng cách nào?   | Không có chunk nào từ `seller-appeal` trong top-3                                      |       | Không         | Trả lời dựa trên ngữ cảnh sai, không nêu được quy trình khiếu nại       |
| 5   | Những điều kiện nào khiến sản phẩm bị từ chối đăng bán?             | Chunk từ `seller-listing` về điều kiện sản phẩm được đăng                              |       | Có            | Nêu đúng các nhóm lý do bị từ chối đăng bán                             |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 3 / 5

Tự chấm theo rubric `docs/SCORING.md` (2 điểm mỗi câu): câu 2, 3, 5 được 2 điểm; câu 1 được 1 điểm vì có chunk từ đúng tài liệu nhưng không phải đoạn chứa thời hạn nên câu trả lời thiếu chi tiết quyết định; câu 4 được 0 điểm. Tổng 7/10.

### Phân tích hai câu thất bại

**Câu 4 là thất bại rõ nhất.** `seller-appeal` không vào được top-3 kể cả khi lọc `customer_role=seller`. Điều này loại trừ khả năng nguyên nhân là nhiễu từ tài liệu dành cho người mua, vì sau khi lọc thì tập ứng viên chỉ còn tài liệu của người bán mà vẫn không trúng. Nguyên nhân nằm ở chỗ khác: câu hỏi dùng cụm "khiếu nại quyết định của nền tảng" trong khi tài liệu diễn đạt bằng cụm khác, và với `chunk_size=180` thì nội dung về quy trình khiếu nại bị xé thành nhiều mảnh nhỏ, mỗi mảnh chỉ giữ một phần ý nên không mảnh nào đủ tín hiệu để nổi lên.

**Câu 1 thất bại một nửa.** Đúng tài liệu nhưng sai đoạn. Ngoài lý do chunk nhỏ giống câu 4, câu hỏi này còn có vấn đề về vai trò: nó hỏi về **người bán** đổi trả, trong khi `returns-policy` được gắn `customer_role=buyer`. Câu hỏi và tài liệu lệch nhau về chủ thể ngay từ đầu, nên dù retrieval hoạt động đúng thì kết quả vẫn khó khớp gold answer.

### Điều tôi rút ra về chiến lược của mình

`chunk_size=180` là quá nhỏ với corpus này. Nó chia mỗi tài liệu vài trăm ký tự thành nhiều mảnh, mỗi mảnh không đủ ngữ cảnh để trả lời trọn một câu hỏi, và overlap 20 ký tự (khoảng 11 phần trăm) quá mỏng để bù lại phần bị cắt. Hai câu thất bại đều là câu cần một đoạn liền mạch mô tả quy trình, đúng dạng nội dung mà chunk nhỏ phá hỏng nặng nhất.

Nếu chạy lại, tôi sẽ thử hai hướng: tăng `chunk_size` lên khoảng 400 với overlap 80 để mỗi chunk giữ trọn một quy trình, hoặc chuyển sang `ArticleChunker` đã lập trình sẵn để chunk bám theo tiêu đề mục. Hướng thứ hai đáng thử hơn vì corpus của nhóm là tài liệu chính sách có tiêu đề mục rõ, và nó cũng giải quyết được vấn đề truy vết nguồn mà chunk cắt cứng không làm được.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác:**

(Điền sau buổi demo.)

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------- |
| Khởi động (Warm-up)                             | 5 / 5            |
| Hướng tiếp cận của tôi (My Approach)            | 9 / 10           |
| Hoàn thiện code (Core Implementation — tests)   | 30 / 30          |
| Dự đoán độ tương tự (Similarity Predictions)    | 5 / 5            |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10           |
| **Tổng phần cá nhân**                           | **58 / 60**      |
