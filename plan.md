# Implementation Plan — MacroNews Portfolio Lab

> Trạng thái: **DRAFT, đang thảo luận.** Chưa chốt cụm/thứ tự.
> Cập nhật: 2026-10-30 (tuần 7). Làm một mình.

## Mốc thời gian cứng

| Mốc | Ngày | Còn lại |
|---|---|---|
| Progress report | 13/11/2026 | ~2 tuần |
| Final presentation | 08/01/2027 | ~10 tuần |
| Final report | 15/01/2027 | ~11 tuần |

## Nguyên tắc

1. **Đi đường giá trước, đường text sau.** Đường giá (prices → technical views → BL → QP →
   backtest) cho ra một hệ thống chạy được end-to-end mà không cần NLP. Đây là thứ cần có
   cho progress report.
2. **Engine nhận dữ liệu qua interface, không quan tâm nguồn.** View generator nhận một
   ma trận feature bất kỳ (technical hoặc news), nên đường news chỉ việc cắm vào sau.
3. **Không look-ahead** là bất biến, phải có unit test cho nó ở mọi cụm.
4. Test set 2018–2023 chỉ chạy **một lần** khi có kết quả cuối.

## Các cụm (đề xuất)

### C0 — Nền tảng
Khung repo, config (YAML + `.env`), logging, lớp storage (Parquet khi dev, MySQL theo
proposal), lịch giao dịch / các mốc thứ Sáu 16:00 ET.

### C1 — Dữ liệu thị trường
Giá điều chỉnh hằng ngày (yfinance) cho 11 tài sản + XLRE, XLC, SPY; DTB3 và vài macro
series từ FRED; tính lợi suất vượt trội ngày/tuần; kiểm tra chất lượng dữ liệu.

### C2 — Portfolio engine (lõi toán)
Ledoit–Wolf covariance, prior π = δΣw_ref, Black–Litterman (Woodbury), QP long-only với
turnover penalty (CVXPY/Clarabel), closed-form Lagrange và KKT để test. Các baseline:
EW, MinVar, InvVol, Markowitz, BL no-view. Test trên dữ liệu synthetic.

### C3 — Technical features và view generator
MOM / REV / VOL, chuẩn hoá cross-sectional; Ridge view generator dùng chung cho mọi nguồn
(purged expanding window, ngưỡng |q| > 0.5σ, Ω từ residual out-of-fold); dựng P, q, Ω.

### C4 — Backtest engine và metrics
Walk-forward hằng tuần, khớp lệnh ở giá đóng cửa phiên kế tiếp, weight drift, phí giao
dịch; metrics (return, vol, Sharpe, Sortino, MDD, CVaR, turnover); lưu các lần chạy.
**→ Sau C4: hệ thống technical-only chạy được, đủ cho progress report.**

### C5 — Dữ liệu text
Scrape FOMC (statements, minutes, press conf); tải và lọc FNSPID; làm sạch, MinHash dedup,
chuẩn hoá timestamp sang giờ ET, map ticker sang GICS sector.

### C6 — NLP
Gán nhãn topic, baseline Loughran–McDonald và TF-IDF, FinBERT multi-task (masked loss),
chấm điểm toàn bộ corpus và lưu vào `doc_scores`.

### C7 — News indices và news views
Chỉ số tuần với decay half-life 14 ngày; S^(k), H, S^sec, U; z-score cuốn chiếu 156 tuần;
cắm vào view generator ở C3.

### C8 — Đánh giá đầy đủ
Ablation (none / tech / news / both), placebo 200 lần, stationary bootstrap, sensitivity,
giai đoạn stress 2020/2022, so sánh chất lượng NLP với kết quả danh mục.

### C9 — Platform
FastAPI (`/backtest`, `/runs`, `/compare`, `/week/{date}`), React frontend, Docker Compose.

### C10 — Báo cáo
Progress report, slide, final report.

## Thứ tự và lịch (đề xuất)

| Tuần | Ngày | Cụm |
|---|---|---|
| 7–8 | 30/10 – 13/11 | C0, C1, C2, C3, C4 (bản tối thiểu) → progress report |
| 9–10 | 16/11 – 27/11 | C5, C6 (baseline trước, FinBERT sau) |
| 11 | 30/11 – 04/12 | C6 (fine-tune + chấm điểm), C7 |
| 12–13 | 07/12 – 18/12 | C8 |
| 14–15 | 21/12 – 01/01 | C9, viết báo cáo |
| 16–17 | 04/01 – 15/01 | Thuyết trình, final report |

## Vấn đề cần chốt khi làm một mình

- [ ] **Cohen's κ** cần ít nhất 2 người gán nhãn. Phương án: dùng một LLM làm annotator thứ
      hai, hoặc nhờ một bạn gán 200 mẫu.
- [ ] **Gán nhãn 1.500 văn bản** một mình: giảm xuống khoảng 600–800, hoặc dùng LLM gán
      nhãn trước rồi mình sửa lại.
- [ ] **FNSPID** (15.7 triệu bài): chỉ lấy title + đoạn đầu, giới hạn 2010–2023, lấy mẫu
      theo ngành để giảm chi phí chấm điểm.
- [ ] **MySQL hay Parquet/SQLite**: dùng Parquet khi dev cho nhanh, chuyển sang MySQL ở C9?
- [ ] **Phạm vi frontend**: drill-down theo tuần có thể để sau cùng.
- [ ] **GPU** cho fine-tune: Colab, Kaggle hay cloud?
