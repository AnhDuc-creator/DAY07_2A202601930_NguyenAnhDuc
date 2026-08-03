# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Anh Đức (2A202601930)
**Nhóm:** Pilot
**Ngày:** 2026-08-03

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

**Cấu hình khi chạy đánh giá:** `EMBEDDING_PROVIDER=local`, model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Corpus `data/k4_ecommerce/` gồm 9 tài liệu trích Nghị định 52/2013/NĐ-CP cộng 2 tài liệu khởi động của lớp.

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

Cosine chỉ đo góc giữa hai vector nên bỏ qua độ dài vector, trong khi độ dài của embedding thường phản ánh độ dài hoặc tần suất từ của văn bản chứ không phản ánh ý nghĩa. Nếu dùng khoảng cách Euclid, một điều luật dài và một câu hỏi ngắn cùng chủ đề vẫn bị coi là xa nhau chỉ vì chênh lệch độ lớn; cosine cho hai trường hợp đó điểm gần như nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Bước nhảy = 500 - 50 = 450. Số chunk = ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23.

Đáp án: 23 chunks.

**Nếu overlap tăng lên 100 thì sao?**

Bước nhảy giảm còn 400, số chunk = ceil((10000 - 100) / 400) = ceil(24.75) = 25, tức tăng thêm 2 chunk. Overlap lớn hơn làm tăng chi phí lưu trữ và số lần embed, nhưng giảm rủi ro một câu hoặc một ý bị cắt đôi ngay ranh giới chunk khiến không chunk nào chứa đủ thông tin để trả lời. Với văn bản chính sách nhiều câu dài, overlap là cách bảo hiểm rẻ cho việc mất ngữ cảnh ở biên.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`**

Tôi dùng regex `(?<=[.!?])\s+` với lookbehind để cắt sau dấu kết câu mà vẫn giữ dấu chấm trong câu, thay vì `split(". ")` làm mất dấu. Sau khi tách, tôi strip từng câu, loại chuỗi rỗng, rồi gom theo lô `max_sentences_per_chunk` bằng slicing. Edge case đã xử lý: text rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng; text không có dấu kết câu nào trả về đúng một chunk là toàn bộ text.

**`RecursiveChunker.chunk` / `_split`**

`_split` nhận đoạn text hiện tại và danh sách separator còn lại. Hai base case: đoạn đã ngắn hơn `chunk_size` thì trả nguyên vẹn; hết separator hoặc gặp separator rỗng thì cắt cứng theo `chunk_size` để không bao giờ mất dữ liệu. Trường hợp còn lại, tôi split theo separator ưu tiên cao nhất; nếu separator không xuất hiện thì gọi đệ quy với separator kế tiếp thay vì tạo chunk thừa. Sau khi split, tôi gom tham lam các mảnh nhỏ vào buffer cho tới sát `chunk_size`, mảnh nào tự nó vẫn quá lớn thì đệ quy xuống cấp separator thấp hơn.

**`ArticleChunker` — chiến lược tùy chỉnh của tôi (yêu cầu riêng K4)**

Corpus của nhóm là văn bản chính sách có cấu trúc điều khoản. Tôi cắt tại ranh giới heading cấp 2 nên mỗi chunk là trọn một điều, không bao giờ tách nội dung khỏi số hiệu điều của nó. Điểm thứ hai: tiêu đề cấp 1 của tài liệu được gắn làm tiền tố cho mọi chunk, để chunk "Điều 20" vẫn mang ngữ cảnh "Quy trình đặt hàng trực tuyến" khi đem đi embed. Điều nào dài hơn `max_chunk_size` thì chia tiếp bằng `RecursiveChunker` nhưng vẫn được gắn lại tiền tố ở đầu mỗi mảnh.

### Lớp EmbeddingStore

**`add_documents` + `search`**

Tôi chọn backend in-memory là list các dict thay vì ChromaDB để kết quả search deterministic và không phụ thuộc phiên bản thư viện; code vẫn kiểm tra chromadb có sẵn hay không và ghi lại cờ. `_make_record` chuẩn hóa mỗi `Document` thành record gồm id, content, metadata (tự điền `doc_id` nếu thiếu) và embedding tính sẵn tại thời điểm add, nên search không phải embed lại tài liệu. `_search_records` embed câu hỏi một lần rồi tính tích vô hướng với từng embedding đã lưu, sort giảm dần theo score và cắt `top_k`. Vì cả `MockEmbedder` lẫn `LocalEmbedder` đều trả vector đã chuẩn hóa, tích vô hướng ở đây chính là cosine similarity.

**`search_with_filter` + `delete_document`**

Lọc trước rồi mới xếp hạng: tôi lọc `self._store` theo toàn bộ cặp key-value trong `metadata_filter`, so sánh bằng `str()` để không vỡ khi front matter parse ra số hoặc ngày, rồi truyền tập ứng viên đã lọc vào chính `_search_records`. Cách này đảm bảo `top_k` được tính trên tập hợp lệ, không bị chunk ngoài phạm vi chiếm chỗ. `metadata_filter` rỗng hoặc None thì hành vi trùng khớp `search()`. `delete_document` dựng lại store bỏ mọi record có `metadata['doc_id']` khớp, trả về True nếu độ dài giảm.

### Tác tử KnowledgeBaseAgent

**`answer`**

Ba bước: retrieve top-k, dựng prompt, gọi `llm_fn`. Prompt chia khối rõ bằng mốc NGỮ CẢNH và CÂU HỎI; mỗi chunk được đánh số [1], [2] kèm nguồn và score, và hệ thống yêu cầu model chỉ dùng thông tin trong ngữ cảnh, trích số hiệu nguồn khi trả lời, nói rõ khi ngữ cảnh không đủ. Gắn số hiệu nguồn vào từng chunk là để phục vụ tiêu chí truy vết nguồn trong `docs/EVALUATION.md`. Nếu store rỗng, agent trả về thông báo không tìm thấy ngữ cảnh thay vì gọi LLM, tránh model tự bịa.

Tôi bổ sung tham số tùy chọn `metadata_filter` cho `answer`. Lý do: khi benchmark có câu cần lọc theo `customer_role`, nếu bảng kết quả lấy top-1 từ `search_with_filter` mà agent vẫn gọi `search` không lọc thì hai cột trong báo cáo nói về hai chunk khác nhau, và câu trả lời không phản ánh chunk đã được chấm là liên quan.

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

| Cặp | Câu A                                                      | Câu B                                                      | Dự đoán | Điểm thực tế | Đúng?        |
| --- | ---------------------------------------------------------- | ---------------------------------------------------------- | ------- | ------------ | ------------ |
| 1   | Tôi muốn trả lại hàng và lấy lại tiền                      | Chính sách hoàn trả và hoàn tiền cho khách hàng            | cao     | 0.358        | Sai          |
| 2   | Thời hạn giao hàng dự kiến là bao lâu                      | Thời hạn ước tính cho việc giao hàng hoặc cung ứng dịch vụ | cao     | 0.815        | Đúng         |
| 3   | Bảo vệ thông tin cá nhân của người tiêu dùng               | Điều kiện đăng ký thiết lập website bán hàng               | thấp    | 0.467        | Sai một phần |
| 4   | Thanh toán trực tuyến phải được mã hóa                     | Bảo mật giao dịch thanh toán của khách hàng                | cao     | 0.628        | Đúng         |
| 5   | Quy định về đấu giá trực tuyến trên sàn thương mại điện tử | Công thức tính diện tích hình tròn                         | thấp    | 0.169        | Đúng         |

**Kết quả nào bất ngờ nhất?**

Cặp 1 gây bất ngờ nhất. Hai câu này nói về đúng cùng một việc là trả hàng và hoàn tiền, tôi chắc chắn điểm sẽ cao, nhưng thực tế chỉ 0.358, thấp hơn cả cặp 3 vốn được dự đoán là thấp. So sánh với cặp 2 đạt 0.815 thì thấy rõ nguyên nhân: cặp 2 dùng gần như trùng cụm từ khóa ("thời hạn", "giao hàng"), còn cặp 1 diễn đạt cùng ý bằng hai bộ từ vựng khác nhau ("trả lại hàng, lấy lại tiền" so với "hoàn trả, hoàn tiền"), và khác cả ngôi kể, một bên là lời người mua, một bên là ngôn ngữ văn bản chính sách.

Điều này cho thấy embedding của model đa ngữ cỡ nhỏ vẫn bám khá nhiều vào trùng lặp từ vựng bề mặt chứ chưa thực sự trừu tượng hóa được ý. Kéo theo hệ quả trực tiếp cho hệ RAG: nếu người dùng hỏi bằng ngôn ngữ đời thường mà corpus viết bằng ngôn ngữ pháp lý, retrieval sẽ yếu. Cặp 3 đạt 0.467 dù hai chủ đề khác nhau cũng củng cố nhận định này, cả hai câu cùng thuộc văn phong pháp lý thương mại điện tử nên model bắt được tín hiệu văn phong chung và đẩy điểm nền lên. Nói cách khác, ngưỡng để phân biệt liên quan và không liên quan trong corpus cùng một miền phải đặt cao hơn nhiều so với 0.5.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy 5 câu hỏi đánh giá của nhóm trên mã nguồn cá nhân, chiến lược `ArticleChunker` (chia theo điều khoản, `max_chunk_size=900`). Số chunk trong store: 44.

| #   | Câu hỏi                                                                        | Top-1 Chunk truy xuất được                                              | Score | Có liên quan?                | Câu trả lời của Agent                                                                           |
| --- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ----- | ---------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | Khi giao hàng bị chậm trễ thì người bán phải làm gì cho khách hàng?            | Quy trình giao kết hợp đồng, Điều 20 Chấm dứt đề nghị giao kết hợp đồng | 0.623 | Không                        | Trả lời về thời hạn chấm dứt đề nghị giao kết, lạc đề so với câu hỏi                            |
| 2   | Giá niêm yết không ghi rõ đã bao gồm phí vận chuyển thì hiểu thế nào?          | Thông tin về hàng hóa giá cả, Điều 31 Thông tin về giá cả               | 0.636 | Có                           | Nêu đúng quy tắc giá được hiểu là đã bao gồm mọi chi phí liên quan                              |
| 3   | Người bán trên sàn có những trách nhiệm gì? (lọc `customer_role=seller`)       | Trách nhiệm của sàn và người bán, Điều 37 Trách nhiệm của người bán     | 0.840 | Có                           | Liệt kê đúng nghĩa vụ cung cấp thông tin, bảo đảm trung thực, tuân thủ pháp luật, nghĩa vụ thuế |
| 4   | Hệ thống bị tấn công làm lộ thông tin thì báo cơ quan chức năng trong bao lâu? | Bảo vệ thông tin cá nhân, Điều 70 Xin phép người tiêu dùng khi thu thập | 0.625 | Có (đúng tài liệu, sai điều) | Nói về cơ chế xin phép thu thập thông tin, thiếu mốc 24 giờ trong Điều 72                       |
| 5   | Không công bố thời hạn trả lời đề nghị giao kết thì bao lâu hết hiệu lực?      | Quy trình giao kết hợp đồng, Điều 20 Chấm dứt đề nghị giao kết hợp đồng | 0.818 | Có                           | Nêu đúng mốc 12 giờ kể từ khi gửi đề nghị                                                       |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 (top-1 đúng tài liệu: 4/5)

Tự chấm theo rubric `docs/SCORING.md` (2 điểm mỗi câu): câu 2, 3, 5 được 2 điểm; câu 4 được 1 điểm vì đúng tài liệu nhưng sai điều nên câu trả lời thiếu chi tiết quyết định là mốc 24 giờ; câu 1 được 0 điểm. Tổng 7/10.

### So sánh với đường cơ sở

Chạy lại đúng 5 câu đó với `FixedSizeChunker(chunk_size=500, overlap=50)`, số chunk tăng lên 59:

| Chiến lược                         | Số chunk | Top-3 có chunk liên quan | Top-1 đúng | Score trung bình |
| ---------------------------------- | -------- | ------------------------ | ---------- | ---------------- |
| ArticleChunker (của tôi)           | 44       | 4 / 5                    | 4 / 5      | 0.708            |
| FixedSizeChunker 500/50 (baseline) | 59       | 5 / 5                    | 5 / 5      | 0.795            |

Chiến lược tùy chỉnh của tôi thua đường cơ sở. Tôi giữ nguyên kết quả này thay vì tinh chỉnh tham số cho đẹp, vì nguyên nhân thua giải thích được và chính là bài học đáng giá nhất tôi rút ra từ lab.

**Tại sao chunk mạch lạc hơn lại truy xuất kém hơn.** `ArticleChunker` tạo chunk trung bình 609 đến 673 ký tự, mỗi chunk là trọn một điều. Điều 33 về vận chuyển và giao nhận có phần lớn nội dung là danh sách các thông tin phải công bố (phương thức giao hàng, thời hạn ước tính, giới hạn địa lý), còn quy định về chậm trễ chỉ là một khoản ngắn ở cuối. Khi embed cả điều thành một vector, khoản về chậm trễ bị pha loãng trong toàn bộ nội dung còn lại, nên câu hỏi số 1 không kéo được chunk đó lên. Baseline cắt cứng 500 ký tự thì tình cờ có một chunk mà khoản về chậm trễ chiếm tỉ trọng lớn, mật độ tín hiệu cao hơn nên thắng.

Đây là đánh đổi giữa tính mạch lạc của chunk và mật độ tín hiệu khi embed: một vector duy nhất không biểu diễn tốt một đoạn dài chứa nhiều ý rời nhau. Chunk mạch lạc về mặt con người đọc không đồng nghĩa với chunk tối ưu cho embedding.

**Nhưng baseline thắng có cái giá.** Nhìn chunk mà baseline trả về: câu 1 bắt đầu bằng "về mặt địa lý cho việc giao hàng", câu 5 bắt đầu bằng "ều 20. Chấm dứt đề nghị giao kết hợp đồng" — cắt giữa từ, mất luôn số hiệu điều. Câu trả lời của agent vì thế cũng bắt đầu giữa chừng và không truy vết được là căn cứ vào điều nào. Theo tiêu chí grounding quality và source traceability trong `docs/EVALUATION.md`, baseline yếu hơn rõ rệt: nó tìm đúng chỗ nhưng không nói được đang trích từ đâu. `ArticleChunker` khi trúng thì câu trả lời luôn kèm đúng số hiệu điều.

**Metadata filter có giúp không.** Với `ArticleChunker`, câu 3 vốn đã đúng nên lọc không đổi gì (0.840 cả hai chế độ). Nhưng với baseline thì lọc cứu được kết quả: `search()` trả về Điều 76 về giải quyết tranh chấp (score 0.774, sai), còn `search_with_filter(customer_role=seller)` trả về đúng Điều 37 (0.773). Hai điểm số gần như bằng nhau, nghĩa là embedding hoàn toàn không phân biệt được hai chunk này; chỉ có metadata mới tách được. Đây là bằng chứng cụ thể cho việc metadata filter không phải trang trí mà là tín hiệu độc lập với embedding, đặc biệt hữu ích khi khoảng cách score giữa kết quả đúng và kết quả nhiễu quá nhỏ.

**Hướng cải thiện tôi sẽ thử nếu có thêm thời gian.** Chia nhỏ tới cấp khoản thay vì cấp điều, mỗi khoản một chunk nhưng vẫn gắn tiền tố tiêu đề tài liệu và số hiệu điều. Cách này giữ được ưu điểm truy vết nguồn của `ArticleChunker` mà vẫn có mật độ tín hiệu cao như baseline.

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
