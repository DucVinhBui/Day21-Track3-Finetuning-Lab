# Lab 21 — Evaluation Report

**Họ tên**: <điền>  **MSSV**: <điền>  **Ngày**: 2026-10-07
**Tier**: `LAPTOP`  **Base model**: `Qwen/Qwen3.5-2B`  **Phần cứng thực tế**: Apple M3 Pro, 18 GB unified memory, PyTorch MPS (fp16), macOS 27

> Mọi con số dưới đây lấy từ file trong `results/` (`runs.csv`, `verdict.json`,
> `autopsy.json`, `baselines_frozen.json`, `side_by_side.json`, …). Log đầy đủ ở `logs/`.

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `Qwen/Qwen3.5-2B` (`BASE_MODEL` trong `.env`) | Máy không có GPU NVIDIA. Mình chạy toàn bộ NB1–NB5 trên Apple MPS. 2B là model lớn nhất của họ Qwen3.5 vừa bộ nhớ (6,1 GB lúc train) mà vẫn chừa chỗ cho hệ điều hành; 4B fp16 riêng trọng số đã ~9 GB. Cùng một base cho mốc (NB2) lẫn mọi adapter. |
| Tier | `LAPTOP` (batch 1 × grad_accum 8 = 8, `max_length` 1024) | Tier dành cho 2B; batch hiệu dụng 8 < 32 (deck §11.4). |
| Dataset | Corpus mặc định: 250 ticket CSKH → JSON 4 trường | Chạy một lượt với corpus chuẩn trước, theo gợi ý README. Không sửa tập eval (checksum giữ nguyên). |
| Prompt (b) | Giữ nguyên `OPTIMIZED_PROMPT` gốc (SHA `719e74d3b6232053`) | Không làm yếu (b). |

**Thay đổi code để chạy trên Mac** (`src/labkit/generate.py`): `device_map="auto"` nạp
thẳng lên MPS bị treo ở "Loading weights 0/320" (đo được >10 phút ở 400% CPU, trong khi
nạp lên CPU chỉ mất 4 s). Vì vậy `load_base()` trên MPS nạp model vào CPU rồi `.to("mps")`;
riêng đường 4-bit dùng `device_map="mps"`, vì bitsandbytes 0.50 lượng tử hoá được trên MPS.
`peak_vram_gb()` trên MPS đọc `torch.mps.driver_allocated_memory()`: MPS không có bộ
đếm đỉnh, nhưng allocator giữ mọi block cho tới `empty_cache()` (chỉ gọi *giữa* các run),
nên giá trị cuối run xấp xỉ mức đỉnh. Cột VRAM vì thế là **bộ nhớ MPS**, không phải
`torch.cuda.max_memory_allocated()`, và không so trực tiếp được với số đo trên T4.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 ticket target + 15 câu regression |
| `max_length` | 1024 (theo tier). p95 đo được là **98**, max 101, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **58 step** (⌈225/8⌉ × 2), dùng chung cho cả 4 run |
| Precision | fp16 + GradScaler (MPS không nhận bf16) |

**Về `max_length` (rubric 1.3).** Mình giữ 1024 của tier thay vì hạ xuống 256. Chuỗi dài
nhất trong corpus là 101 token, nên ở cả 1024 lẫn 256 **không có token nào bị cắt**. Với
`per_device_batch=1` cũng không có padding, nên `max_length` không đổi được dù một token
nào trong batch hay một byte bộ nhớ nào. Đổi thành 256 chỉ có ý nghĩa với dataset có đuôi
dài hơn, hoặc với batch > 1. Nếu đổi sang dataset riêng, mình sẽ đặt `max_length` ≈ p95
được làm tròn lên.

**Template có giữ khối `<think>` không?** **Có.** `results/template_check.json`:
`ok: true`, `open_tag_present: true`, `body_present: true`. Chuỗi render vẫn giữ nguyên
`<think>\nbuoc 1: …\n</think>`. Với corpus này, câu trả lời huấn luyện là JSON trần, và
chat template tự chèn một khối `<think>\n\n</think>` rỗng vào *generation prompt*. Khối
rỗng đó nằm ở phần bị mask (xem §2), nên `assistant-only`, `masked-think` và `response-only`
cho ra cùng một mask. Đó cũng là lý do `valid_trace_rate = 0.0` trong `verdict.json`.

`scripts/check_mask_agreement.py` trên Qwen3.5-2B: template **không** có marker
`{% generation %}`. Mask của TRL `assistant_masks` = 0/31 token, đường "TRL vá template"
= 13/31 token (gồm cả `<think></think>` rỗng), còn mask của labkit = 9/31 token (chỉ JSON
+ `<|im_end|>`). Lab dùng mask của labkit qua dataset đã tokenize sẵn.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.3936** (37/94 token) |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

Đoạn **được** tính loss:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị** mask (system + user + mở đầu assistant):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

Để đối chứng, `MASK_MODE=everything` cho 94/94 = 100% token được tính loss. Đó chính là
lỗi kinh điển, và `supervised_fraction ≥ 0.95` sẽ bị bắt.

---

## 3. Ba baseline (NB2 đóng băng TRƯỚC khi train; (c) đo ở NB5)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.600 | 0.000 | 2631.6 |
| (b) base + optimized prompt | 0.600 | 0.600 | 1.000 | 1110.2 |
| (c) LoRA fine-tune (`correct`) | **1.000** | **0.067** | 1.000 | 907.5 |

*(results/baselines_frozen.json, results/verdict.json. `eval_limit: null`, tức là chạy trên
tập eval đầy đủ chứ không phải chế độ smoke.)*

**(b) có thật sự mạnh hơn (a) không?** **Có**: 0.600 so với 0.000 trên target, 1.000 so
với 0.000 trên format. Với prompt naive, base 2B không trả JSON đúng schema lần nào. Prompt
(b), gồm schema, danh sách nhãn và một ví dụ, kéo format lên 100%. Regression của (a) và
(b) bằng nhau vì nhóm regression luôn được hỏi **không có system prompt**, đúng như thiết
kế của NB2.

Mình **không** sửa `OPTIMIZED_PROMPT`.

**(b) sai ở đâu?** (`results/side_by_side.json → summary.b_field_acc`): `product` 0.94,
`intent` 0.58, `urgency` 0.56, `sentiment` **0.32**. Ba kiểu lỗi lặp lại nhiều nhất:
- (b) gần như luôn trả `sentiment: tich_cuc`, kể cả với ticket "Bực mình", "Quá tệ".
- (b) nhầm `doi_tra` ↔ `hoan_tien` khi ticket viết "Hoàn lại" hoặc "Cho tôi trả lại".
- Ba lần (b) lấy **mã đơn** làm `product` (ví dụ `"product": "OD684661"`).

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

| Run | vị trí | r | trainable | LR | train loss (NB4, trung bình) | loss step cuối | **target (NB5 §4)** | format | VRAM GB (MPS) | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 16,819,200 | 1e-4 | 0.4489 | 0.0012 | **1.000** | 1.000 | 6.08 | 426.0 |
| `attn_only` | q,v (2 module) | 322 *(matched)* | 16,816,128 | 1e-4 | **0.4406** | 0.0053 | 0.975 | 1.000 | 6.78 | 368.5 |
| `wrong_lr` | text-linear | 16 | 16,819,200 | 1e-5 | 1.1690 | 0.4324 | 0.420 | 0.995 | 6.01 | 428.1 |
| `qlora` | text-linear, 4-bit NF4 | 16 | 16,819,200 | 1e-4 | 0.4500 | 0.0014 | 0.990 | 1.000 | **4.93** | 549.4 |

*Cột "train loss" là `final_loss` trong `runs.csv` (= `train_loss` của Trainer, tức trung
bình trên mọi step). Cột "loss step cuối" lấy từ log lần ghi cuối (step 55).*

Cả bốn run có **cùng `max_steps` = 58**. `attn_only` lệch `correct` 3,072 tham số
(**0.018%**, dưới ngưỡng 5%). Mỗi run đối chứng chỉ đổi **một** biến:

| Run | Biến duy nhất bị đổi |
|---|---|
| `attn_only` | **vị trí** gắn adapter (q,v thay vì mọi linear của decoder text); rank được giải ra cho khớp ngân sách |
| `wrong_lr` | **learning rate** (1e-5, thang full-FT, thay vì 1e-4) |
| `qlora` | **độ chính xác của base** (4-bit NF4 thay vì fp16) |

**Xếp hạng theo target (NB5):** `correct` 1.000 > `qlora` 0.990 > `attn_only` 0.975 ≫ (b) 0.600 > `wrong_lr` 0.420
**Xếp hạng theo train loss trung bình (NB4):** `attn_only` 0.4406 < `correct` 0.4489 < `qlora` 0.4500 ≪ `wrong_lr` 1.169

Hai thứ tự **khác nhau** ở vị trí số 1.

**4.1 — Vị trí và rank.** `attn_only` có cùng ngân sách tham số với `correct` (16,8 triệu),
nhưng dồn hết vào 2 ma trận với rank 322 thay vì rải đều 12 loại module với rank 16. Trên
tập target nó **thua nhẹ**: 0.975 so với 1.000, tức khoảng 5 trường sai trên 200, toàn bộ
format vẫn đúng. Nếu xếp hạng bằng train loss trung bình, `attn_only` lại **đứng đầu**
(0.4406 < 0.4489). Lý do là nó giảm loss nhanh hơn ở mấy step đầu (step 10: 1.79 so với
2.03), nhưng loss ở các step cuối thì cao hơn `correct` khoảng 4 lần (0.0053 so với
0.0012). Nhìn bằng chỉ số thay thế sẽ kết luận sai: "q,v với rank cao là đủ". Rank 322 là
gấp 20 lần, mà vẫn không bù được việc bỏ trống các lớp MLP. Vậy với cùng ngân sách, đòn
bẩy là **vị trí**, không phải rank. Mình nói chừng mực: trên một tác vụ hẹp và dễ như thế
này, khoảng cách chỉ 2,5 điểm, và 50 mẫu eval thì chưa đủ để nói khoảng cách đó chắc chắn
khác 0. Kết luận mạnh hơn mà mình bảo vệ được là: **tăng rank không thay thế được vị trí**.
Nếu rank mới là đòn bẩy, `attn_only` với r=322 đáng lẽ phải thắng, và nó đã không thắng.

**4.2 — `wrong_lr` chỉ khác đúng một con số.** Đường loss của nó vẫn giảm đều và trông
"khỏe mạnh": 2.60 → 2.50 → 1.98 → 1.49 → 1.18 → … → 0.43. Không có dấu hiệu bị kẹt.
Nhưng đến step 58 nó mới xuống tới 0.43, mức mà `correct` đã xuống dưới ngay từ step 15 (0.37). Trên tập
target, `wrong_lr` đạt **0.420, thấp hơn cả base model được prompt tử tế (0.600)**. Nếu chỉ
nhìn đường loss mà không biết LR, mình sẽ kết luận một trong hai điều sai: "LoRA học kém
hơn full fine-tune", hoặc "cần nhiều data/epoch hơn". Thật ra chỉ cần nhân LR lên 10 lần.
Đây là kết quả mình thấy đáng giá nhất: một fine-tune cấu hình sai **tệ hơn việc không
fine-tune**, và loss không báo điều đó.

**4.3 — `qlora`: tiết kiệm bao nhiêu, trả giá bằng gì?** Bộ nhớ MPS giảm từ 6.08 xuống
4.93 GB (**−19%**). Target gần như không mất (0.990 so với 1.000), đường loss gần như trùng
với `correct`. Cái giá nằm ở tốc độ: train chậm hơn **29%** (549 s so với 426 s) và latency
suy luận chậm hơn **67%** (1513 so với 908 ms/mẫu), vì mỗi lần nhân ma trận phải giải lượng
tử. Số đo của mình **không** ủng hộ mạnh khuyến nghị "đừng dùng QLoRA cho Qwen3.5" khi xét
**chất lượng** trên tác vụ này: thiệt hại chỉ 1 điểm, nằm trong mức nhiễu của 50 mẫu. Có
hai lưu ý:
- Mức tiết kiệm bộ nhớ ở 2B nhỏ hơn nhiều so với 41% đo trên T4 với 4B
  (`docs/MEASURED-T4-2026-08-20.md`), vì ở 2B activation và optimizer chiếm tỉ lệ lớn hơn
  trọng số.
- Đây là kernel bitsandbytes trên MPS, nên tốc độ không đại diện cho CUDA.

Kết luận thực dụng: ở 2B trên máy 18 GB, 16-bit vừa bộ nhớ, nên QLoRA chỉ đổi lấy chậm hơn
mà không được gì. QLoRA chỉ đáng dùng khi 16-bit không vừa.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.400` · `regression Δ = −0.533` (ngưỡng −0.020) · `valid_trace_rate = 0.00`
· format 1.000 · latency 907.5 ms (so với 1110.2 ms của (b))

**Diễn giải.** Trên đúng tác vụ được train, fine-tune thắng tuyệt đối:
- target từ 0.600 lên 1.000, 47/50 ticket tốt hơn (b) và 0/50 ticket tệ hơn;
- nhanh hơn 18% vì không cần prompt dài.

Nhưng nó **đánh mất gần như toàn bộ năng lực phổ thông**: regression từ 0.600 xuống 0.067.
12/15 câu hỏi kiến thức, như "Thủ đô của Việt Nam?" hay "Ai là tác giả Truyện Kiều?", giờ
được trả lời bằng một object JSON triage, ví dụ
`{"intent": "hoi_thong_tin", "urgency": "thap", …, "product": "thành phố"}`. Ba câu còn
lại cũng chỉ là vài chữ hoặc là văn bản sai. Chỉ câu dịch sang tiếng Anh còn ghi điểm. Đây không phải mô hình "quên" kiến thức theo nghĩa trọng số bị
xoá. Nó đã học một **ánh xạ vô điều kiện**: *mọi* lượt user → JSON 4 khoá. Lý do là 225/225
mẫu train có cùng một dạng output, không có một mẫu nào dạy nó khi nào *không* phải phân
loại. Trong khi đó, LR 10x với 58 step đẩy loss xuống ~0.001: adapter thuộc lòng định dạng
tới mức định dạng lấn át cả câu hỏi.

Vì cổng yêu cầu regression không tụt quá 0.02, phán quyết đúng là **không deploy adapter
này làm model đa dụng**. Nếu endpoint chỉ phục vụ đúng việc triage và không bao giờ nhận
câu hỏi khác, nó vượt trội (b). Nhưng đó là một quyết định sản phẩm cần nói rõ, không phải
một điều để giấu sau con số target 1.000. Theo deck §6.3, bước sửa tiếp theo là trộn 1–5%
dữ liệu phổ thông (replay) vào tập train và chạy lại cổng; mình chưa làm bước này. Lưu ý
thêm: target = 1.000 cũng là một tín hiệu cần cảnh giác. Corpus được sinh từ template
(cùng cấu trúc "mình đặt X mã đơn Y. Z."), nên con số đó đo khả năng thuộc template chứ
chưa đo khả năng tổng quát hoá sang ticket viết tự do.

**Lưu ý về thước đo regression** (để không đổ hết cho fine-tune): baseline chỉ đạt 0.600
một phần vì metric `keyword_recall` khắt khe. (b) trả lời đúng "**1.000 mét**" nhưng bị
chấm 0 vì keyword là `1000`. Câu 2¹⁰ bị cắt ở 96 token trước khi tới đáp số. Ngoài ra base
2B có vài lỗi kiến thức thật ("sông Hồng" dài nhất, TP.HCM "trước đây là Quảng Đông"). Dù
vậy, khoảng cách 0.600 → 0.067 không phải do metric: đầu ra của fine-tune là JSON, không
chứa câu trả lời nào để chấm.

---

## 6. Định tính: có cả ca THUA

Nguồn: `results/side_by_side.json` (sinh bởi `scripts/side_by_side.py`, cùng thiết lập
greedy như NB2/NB5). `results/qualitative.json` của NB5 chỉ xếp hạng các dự đoán target
của fine-tune, mà cả 50 đều đạt 1.0. Vì vậy các ca thua phải tìm ở nhóm regression.

| # | Đầu vào (rút gọn) | Nhãn / keyword | (b) base + prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | *target #6* "…balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua…" | `doi_tra · thap · balo laptop · tieu_cuc` | `hoi_thong_tin · thap · … · tich_cuc` (0.50) | `doi_tra · thap · balo laptop · tieu_cuc` (1.00) | ✅ **FT thắng**: (b) bị "Hỏi cho biết thôi" đánh lừa intent và bỏ qua "Lần cuối mua" khi chấm sentiment |
| 2 | *target #32* "…máy xay sinh tố mã đơn OD906403. Hoàn lại. Mong shop phản hồi…" | `doi_tra · trung_binh · máy xay sinh tố · tich_cuc` | `hoan_tien · thap · "OD906403" · tieu_cuc` (0.00) | đúng cả 4 trường (1.00) | ✅ **FT thắng**: (b) lấy mã đơn làm product. FT học được quy ước "Hoàn lại" → `doi_tra` của corpus |
| 3 | *target #16* "…máy xay sinh tố mã đơn OD684661. Không hoạt động. Khẩn…" | `san_pham_loi · cao · máy xay sinh tố · trung_tinh` | product = `"OD684661"`, sentiment = `tich_cuc` (0.50) | đúng cả 4 trường (1.00) | ✅ **FT thắng** |
| 4 | *regression #0* "Thủ đô của Việt Nam là thành phố nào?" | `Hà Nội` | "Thủ đô của Việt Nam hiện nay là thành phố **Hà Nội**…" (1.00) | `{"intent": "hoi_thong_tin", "urgency": "thap", …, "product": "thành phố", "location": "Việt Nam"}` (0.00) | ❌ **FT thua**: trả JSON triage cho một câu hỏi kiến thức, thậm chí tự bịa thêm khoá `location` |
| 5 | *regression #8* "Ai là tác giả của Truyện Kiều?" | `Nguyễn Du` | "…được viết bởi nhà thơ **Nguyễn Du**…" (1.00) | `{"intent": "hoi_thong_tin", …, "topic": "truyện Kiều", "product": "truyện Kiều", …}` (0.00) | ❌ **FT thua** |
| 6 | *regression #3* "Viết một câu chúc mừng sinh nhật bằng tiếng Việt." | `sinh nhật` | "Chúc mừng sinh nhật bạn! …" (1.00) | `{"intent": "van_chuyen", …, "product": "trung_tinh", "sentiment": "tich_cuc"}` (0.00) | ❌ **FT thua**: JSON vô nghĩa, giá trị nhãn lọt sai trường |
| 7 | *regression #4* "Dịch sang tiếng Anh: 'Tôi thích đọc sách'." | `read, book` | "I like reading books." (1.00) | "\"I like reading books.\"" (1.00) | Hoà: câu duy nhất fine-tune giữ được |

Tổng: target **47 thắng / 0 thua / 3 hoà**; regression **0 thắng / 10 thua / 5 hoà**
(`side_by_side.json → summary`).

**Mẫu chung của các ca FT thua:** mọi ca thua đều là đầu vào **không phải ticket**. Fine-tune
không phân biệt được "đây có phải ticket không" vì tập train không có mẫu phủ định. Khi một
đầu vào không có sản phẩm, nó nhét danh từ bất kỳ vào `product` ("thành phố", "sắt",
"đơn vị đo lường"), và đôi khi tạo thêm khoá ngoài schema (`location`, `topic`). Điều này
cho thấy định dạng được học thuộc chứ không được hiểu.

---

## 7. Kết luận và điều tôi học được

**Kết luận.** Bản fine-tune `correct` (LoRA mọi linear của decoder, r=16, LR 1e-4,
58 step, Qwen3.5-2B) **không nên deploy** làm model dùng chung, dù nó đạt target 1.000 so
với 0.600 của base được prompt tử tế. Lý do nằm ở cơ chế, không chỉ ở con số: tập train
chỉ gồm một loại output (JSON 4 khoá), không có replay và không có mẫu phủ định. Vì vậy
gradient chỉ dạy đúng một điều: "luôn trả JSON". Với LR ở thang LoRA đúng, adapter học điều
đó rất nhanh (loss 2.6 → 0.001), và hành vi đó tràn sang mọi đầu vào. Regression rơi từ
0.600 xuống 0.067, vượt xa ngưỡng 0.02.

Đòn bẩy thật sự trong lab, theo thứ tự đo được:
1. **Learning rate.** Chỉ đổi đúng một con số, target rơi từ 1.000 xuống 0.420, thấp hơn
   cả không fine-tune.
2. **Phân phối dữ liệu.** Nó quyết định việc fine-tune thắng tác vụ nhưng thua cổng hồi
   quy.
3. **Vị trí adapter.** Có tác dụng nhưng nhỏ trên tác vụ dễ này: 2,5 điểm, dù rank đã nâng
   20 lần cho khớp ngân sách.
4. **Độ chính xác 4-bit.** Gần như không ảnh hưởng chất lượng, nhưng làm chậm 29–67%.

Mask đúng là điều kiện cần cho mọi thứ trên: nếu tính loss cả trên prompt, mọi so sánh
phía sau đều vô nghĩa.

Bước tiếp theo đúng là trộn replay phổ thông vào tập train rồi đo lại *cả* target lẫn
regression. Nếu sau đó fine-tune vẫn không giữ được regression mà vẫn hơn (b) đáng kể,
lựa chọn hợp lý là chỉ gọi adapter qua một bộ định tuyến "đây là ticket", còn câu hỏi
chung đi thẳng vào base.

**Ba điều tôi học được:**
1. *Đường loss "đẹp" không có nghĩa là cấu hình đúng.* Run `wrong_lr` có loss giảm đều
   từng bước, không một dấu hiệu bất thường nào, nhưng kết quả lại thua chính base model.
   Mình chỉ biết điều đó nhờ chấm trên tác vụ.
2. *Xếp hạng bằng train loss có thể đảo kết luận.* Theo train loss trung bình, `attn_only`
   đứng đầu; trên target thì `correct` đứng đầu. Nếu dừng ở NB4, mình sẽ viết rằng "q,v với
   rank cao là đủ".
3. *Target 1.000 là một tín hiệu để nghi ngờ, không phải để ăn mừng.* Cùng một adapter vừa
   đạt điểm tuyệt đối trên tác vụ vừa trả JSON cho câu "Thủ đô của Việt Nam?". Thiếu nhóm
   regression thì mình đã báo cáo một model hỏng như một thành công.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- trộn 5% dữ liệu hỏi–đáp phổ thông (không trùng 15 câu eval) vào tập train và chạy lại
  NB3 + NB5, xem regression hồi phục được bao nhiêu và target mất bao nhiêu;
- giảm xuống `EPOCHS=1`, xem có điểm cân bằng nào mà target > 0.9 trong khi regression
  vẫn giữ được.

---

## Phụ lục: tái lập

```bash
cp .env.example .env   # rồi đặt COMPUTE_TIER=LAPTOP, BASE_MODEL=Qwen/Qwen3.5-2B
make nb1 && make nb2 && make nb3 && make nb4 && make nb5
.venv/bin/python scripts/side_by_side.py     # bảng định tính §6
make verify
```

Phiên bản: torch 2.14.1 (MPS), transformers 5.19.0, bitsandbytes 0.50.2. Log từng
notebook nằm trong `logs/`.

## Phụ lục: thưởng đã làm

- [x] **B1 NB6 merge + hot-swap.** `results/merge_check.json`: target trước merge 1.000,
  sau `merge_and_unload()` 1.000 (Δ +0.000, ngưỡng 0.01, n=50). Hot-swap **3 adapter**
  (`correct`, `attn_only`, `qlora`) trên cùng một base đã nạp; cả ba đều trả JSON đúng cho
  cùng một ticket (`logs/nb6.log`). Model đã merge (~3,7 GB) được xoá sau khi kiểm tra vì
  máy hết dung lượng đĩa; chạy lại `make nb6` để sinh lại.
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
