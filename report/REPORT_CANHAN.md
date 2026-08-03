# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Anh Đức (2A202601930)
**Nhóm:** Pilot
**Ngày:** 2026-08-03

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

**Cấu hình khi chạy đánh giá:** `EMBEDDING_PROVIDER=local`, model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `top_k=3`. Corpus `data/k4_ecommerce/`: 9 tài liệu chính sách thương mại điện tử trích Nghị định 52/2013/NĐ-CP. Chiến lược của tôi: `ArticleChunker` (tùy chỉnh, cắt theo heading/section). Mọi số liệu lấy từ output của `bench.py`.

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

Bước nhảy giảm còn 400, số chunk = ceil((10000 - 100) / 400) = ceil(24.75) = 25, tức tăng thêm 2 chunk. Overlap lớn hơn làm tăng chi phí lưu trữ và số lần embed, nhưng giảm rủi ro một câu hoặc một ý bị cắt đôi ngay ranh giới chunk khiến không chunk nào chứa đủ thông tin để trả lời. Với văn bản pháp lý nhiều câu liệt kê dài, overlap là cách bảo hiểm rẻ cho việc mất ngữ cảnh ở biên.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`**

Tôi dùng regex `(?<=[.!?])\s+` với lookbehind để cắt sau dấu kết câu mà vẫn giữ dấu chấm trong câu, thay vì `split(". ")` làm mất dấu. Sau khi tách, tôi strip từng câu, loại chuỗi rỗng, rồi gom theo lô `max_sentences_per_chunk` bằng slicing. Edge case đã xử lý: text rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng; text không có dấu kết câu nào trả về đúng một chunk là toàn bộ text.

**`RecursiveChunker.chunk` / `_split`**

`_split` nhận đoạn text hiện tại và danh sách separator còn lại. Hai base case: đoạn đã ngắn hơn `chunk_size` thì trả nguyên vẹn; hết separator hoặc gặp separator rỗng thì cắt cứng theo `chunk_size` để không bao giờ mất dữ liệu. Trường hợp còn lại, tôi split theo separator ưu tiên cao nhất; nếu separator không xuất hiện thì gọi đệ quy với separator kế tiếp thay vì tạo chunk thừa. Sau khi split, tôi gom tham lam các mảnh nhỏ vào buffer cho tới sát `chunk_size`, mảnh nào tự nó vẫn quá lớn thì đệ quy xuống cấp separator thấp hơn.

Một chi tiết tôi chú ý: danh sách separator bắt buộc phải kết thúc bằng `" "` và `""`. Nếu dừng ở `"."` thì đoạn nào không chứa dấu chấm mà vẫn dài hơn `chunk_size` sẽ không cắt được tiếp và trả về nguyên đoạn quá khổ.

**`ArticleChunker` — chiến lược tùy chỉnh của tôi (yêu cầu riêng K4)**

Corpus của nhóm là văn bản chính sách có cấu trúc điều khoản với heading rõ ràng. Tôi cắt tại ranh giới heading cấp 2 nên mỗi chunk là trọn một điều, nội dung không bao giờ tách khỏi số hiệu điều. Điểm thứ hai: tiêu đề cấp 1 của tài liệu được gắn làm tiền tố cho mọi chunk, để chunk "Điều 20" vẫn mang ngữ cảnh "Quy trình đặt hàng trực tuyến" khi đem đi embed thay vì mồ côi khỏi tài liệu gốc. Điều nào dài hơn `max_chunk_size` thì chia tiếp bằng `RecursiveChunker` nhưng vẫn gắn lại tiền tố ở đầu mỗi mảnh. Nếu tài liệu không có heading, chunker tự động lùi về `RecursiveChunker` thay vì lỗi.

Giả thuyết khi thiết kế: chunk trọn một điều sẽ truy vết nguồn tốt hơn, vì câu trả lời của agent luôn kèm được số hiệu điều mà nó trích.

### Lớp EmbeddingStore

**`add_documents` + `search`**

Tôi chọn backend in-memory là list các dict thay vì ChromaDB để kết quả search deterministic và không phụ thuộc phiên bản thư viện; code vẫn kiểm tra chromadb có sẵn hay không và ghi lại cờ. `_make_record` chuẩn hóa mỗi `Document` thành record gồm id, content, metadata (tự điền `doc_id` nếu thiếu) và embedding tính sẵn tại thời điểm add, nên search không phải embed lại tài liệu. `_search_records` embed câu hỏi một lần rồi tính tích vô hướng với từng embedding đã lưu, sort giảm dần theo score và cắt `top_k`. Vì cả `MockEmbedder` lẫn `LocalEmbedder` đều trả vector đã chuẩn hóa, tích vô hướng ở đây chính là cosine similarity.

**`search_with_filter` + `delete_document`**

Lọc trước rồi mới xếp hạng: tôi lọc `self._store` theo toàn bộ cặp key-value trong `metadata_filter`, so sánh bằng `str()` để không vỡ khi front matter parse ra số hoặc ngày, rồi truyền tập ứng viên đã lọc vào chính `_search_records`. Cách này đảm bảo `top_k` được tính trên tập hợp lệ, không bị chunk ngoài phạm vi chiếm chỗ. `metadata_filter` rỗng hoặc None thì hành vi trùng khớp `search()`. `delete_document` dựng lại store bỏ mọi record có `metadata['doc_id']` khớp, trả về True nếu độ dài giảm.

### Tác tử KnowledgeBaseAgent

**`answer`**

Ba bước: retrieve top-k, dựng prompt, gọi `llm_fn`. Prompt chia khối rõ bằng mốc NGỮ CẢNH và CÂU HỎI; mỗi chunk được đánh số [1], [2] kèm nguồn và score, và hệ thống yêu cầu model chỉ dùng thông tin trong ngữ cảnh, trích số hiệu nguồn khi trả lời, nói rõ khi ngữ cảnh không đủ. Gắn số hiệu nguồn vào từng chunk là để phục vụ tiêu chí truy vết nguồn trong `docs/EVALUATION.md`. Nếu store rỗng, agent trả về thông báo không tìm thấy ngữ cảnh thay vì gọi LLM, tránh model tự bịa.

Tôi bổ sung tham số tùy chọn `metadata_filter` cho `answer`. Lý do: bộ benchmark có câu 3 cần lọc theo `customer_role`. Nếu bảng kết quả lấy top-1 từ `search_with_filter` mà agent vẫn gọi `search` không lọc thì hai cột trong báo cáo nói về hai chunk khác nhau, và câu trả lời không phản ánh chunk đã được chấm là liên quan.

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

Điều này cho thấy embedding của model đa ngữ cỡ nhỏ vẫn bám khá nhiều vào trùng lặp từ vựng bề mặt chứ chưa thực sự trừu tượng hóa được ý. Kết quả này dự báo đúng thất bại ở câu benchmark số 1 tại mục 5: câu hỏi diễn đạt bằng ngôn ngữ đời thường trong khi điều luật diễn đạt bằng ngôn ngữ pháp lý.

Cặp 3 đạt 0.467 dù hai chủ đề khác nhau cũng đáng chú ý: cả hai câu cùng thuộc văn phong pháp lý thương mại điện tử nên model bắt được tín hiệu văn phong chung và đẩy điểm nền lên. Nói cách khác, ngưỡng để phân biệt liên quan và không liên quan trong corpus cùng một miền phải đặt cao hơn nhiều so với 0.5, và một điểm số cao chưa chắc đã là bằng chứng của kết quả tốt.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy 5 câu hỏi đánh giá của nhóm trên mã nguồn cá nhân trong gói `src`, chiến lược `ArticleChunker(max_chunk_size=900)`. Số chunk trong store: 42.

| #   | Câu hỏi (Query)                                                                | Top-1 Chunk truy xuất được                                              | Score | Có liên quan?                | Câu trả lời của Agent                                                                           |
| --- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ----- | ---------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | Khi giao hàng bị chậm trễ thì người bán phải làm gì cho khách hàng?            | Quy trình giao kết hợp đồng, Điều 20 Chấm dứt đề nghị giao kết hợp đồng | 0.623 | Không                        | Trả lời về thời hạn chấm dứt đề nghị giao kết, lạc đề so với câu hỏi                            |
| 2   | Giá niêm yết không ghi rõ đã bao gồm phí vận chuyển thì hiểu thế nào?          | Thông tin về hàng hóa giá cả, Điều 31 Thông tin về giá cả               | 0.636 | Có                           | Nêu đúng quy tắc giá được hiểu là đã bao gồm mọi chi phí liên quan                              |
| 3   | Người bán trên sàn có những trách nhiệm gì? (lọc `customer_role=seller`)       | Trách nhiệm của sàn và người bán, Điều 37 Trách nhiệm của người bán     | 0.840 | Có                           | Liệt kê đúng nghĩa vụ cung cấp thông tin, bảo đảm trung thực, tuân thủ pháp luật, nghĩa vụ thuế |
| 4   | Hệ thống bị tấn công làm lộ thông tin thì báo cơ quan chức năng trong bao lâu? | Bảo vệ thông tin cá nhân, Điều 70 Xin phép người tiêu dùng khi thu thập | 0.625 | Có (đúng tài liệu, sai điều) | Nói về cơ chế xin phép thu thập thông tin, thiếu mốc 24 giờ trong Điều 72                       |
| 5   | Không công bố thời hạn trả lời đề nghị giao kết thì bao lâu hết hiệu lực?      | Quy trình giao kết hợp đồng, Điều 20 Chấm dứt đề nghị giao kết hợp đồng | 0.818 | Có                           | Nêu đúng mốc 12 giờ kể từ khi gửi đề nghị                                                       |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 (top-1 đúng tài liệu: 4/5)

Tự chấm theo rubric `docs/SCORING.md` (2 điểm mỗi câu): câu 2, 3, 5 được 2 điểm; câu 4 được 1 điểm vì đúng tài liệu nhưng sai điều nên câu trả lời thiếu chi tiết quyết định là mốc 24 giờ; câu 1 được 0 điểm. Tổng 7/10.

### Phân tích thất bại

**Câu 1 là thất bại rõ nhất.** Gold answer nằm ở Điều 33 khoản 2, nhưng top-1 trả về Điều 20 với score 0.623 và chunk đúng không xuất hiện trong top-3.

Nguyên nhân là mật độ tín hiệu khi embed. Tài liệu vận chuyển giao nhận chỉ dài 721 ký tự và `ArticleChunker` gộp trọn Điều 33 thành một chunk. Phần lớn nội dung điều này là danh sách các thông tin phải công bố (phương thức giao hàng, thời hạn ước tính, giới hạn địa lý), còn quy định về chậm trễ chỉ là một khoản ngắn ở cuối. Khi embed cả điều thành một vector duy nhất, khoản về chậm trễ bị pha loãng trong toàn bộ nội dung còn lại. Trong khi đó Điều 20 chứa nhiều cụm từ về thời hạn và nghĩa vụ trả lời của người bán, trùng với câu hỏi ở mức từ vựng bề mặt nên thắng.

Đây là đánh đổi giữa tính mạch lạc của chunk và mật độ tín hiệu: một vector không biểu diễn tốt một đoạn dài chứa nhiều ý rời nhau. Chunk mạch lạc với người đọc không đồng nghĩa chunk tối ưu cho embedding.

**Câu 4 thất bại một nửa.** Top-1 là Điều 70 chứ không phải Điều 72 chứa mốc 24 giờ, nên câu trả lời của agent thiếu đúng con số mà câu hỏi cần. Nó vẫn được tính là liên quan vì tôi chấm ở cấp `doc_id`, và điều này bộc lộ một hạn chế của cách đo: với corpus tổ chức theo điều khoản, chấm ở cấp tài liệu là quá rộng. Lần sau nên chấm ở cấp chunk, tức đúng điều mới tính.

### Metadata filter có giúp không

| Chế độ                                     | Top-1 doc_id               | Score | Trúng gold doc |
| ------------------------------------------ | -------------------------- | ----- | -------------- |
| `search()`                                 | nd52-trach-nhiem-nguoi-ban | 0.840 | Có             |
| `search_with_filter(customer_role=seller)` | nd52-trach-nhiem-nguoi-ban | 0.840 | Có             |

Với chiến lược của tôi, lọc không thay đổi gì ở câu 3 vì kết quả vốn đã đúng. Tôi ghi lại nguyên trạng thay vì chọn một câu khác cho đẹp hơn: trên corpus này, embedding đã đủ tách bạch nên metadata chưa có đất phát huy. Giá trị thật của `search_with_filter` chỉ xuất hiện khi khoảng cách score giữa kết quả đúng và kết quả nhiễu quá nhỏ, tình huống chưa xảy ra ở câu 3 (0.840 cách khá xa các kết quả sau).

### Điều tôi sẽ làm khác

Chia tới cấp khoản thay vì cấp điều, mỗi khoản một chunk nhưng vẫn gắn tiền tố tiêu đề tài liệu và số hiệu điều. Cách này giữ được ưu điểm truy vết nguồn của `ArticleChunker` mà vẫn có mật độ tín hiệu cao, và giải quyết trực tiếp cả hai ca hỏng ở trên: câu 1 hỏng vì khoản bị pha loãng trong điều, câu 4 hỏng vì đúng tài liệu nhưng sai điều.

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
