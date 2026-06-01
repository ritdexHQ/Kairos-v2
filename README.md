<div align="center">

# 🤖 KAIROS QUANT SYSTEM v2.0

**HỆ SINH THÁI GIAO DỊCH ĐỊNH LƯỢNG TỰ ĐỘNG & PHÂN TÍCH DỮ LIỆU CHUYÊN SÂU CHO THỊ TRƯỜNG CRYPTOCURRENCY**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Polars](https://img.shields.io/badge/Polars-High_Performance-CD792C?style=flat-square)](https://pola.rs/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-Desktop_GUI-41CD52?style=flat-square&logo=qt)](https://www.riverbankcomputing.com/software/pyqt/)
[![CCXT](https://img.shields.io/badge/CCXT-Multi_Exchange-2EA043?style=flat-square)](https://github.com/ccxt/ccxt)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## 📖 MỤC LỤC TỔNG QUAN

1. [Tầm nhìn & Triết lý hệ thống](#1-tầm-nhìn--triết-lý-hệ-thống-vision--philosophy)
2. [Giới thiệu chung](#2-giới-thiệu-chung-overview)
3. [Tính năng cốt lõi & Công nghệ đột phá](#3-tính-năng-cốt-lõi--công-nghệ-đột-phá)
4. [Kiến trúc hệ thống](#4-kiến-trúc-hệ-thống-system-architecture)
5. [Logic Chiến lược & Hệ thống Bỏ phiếu](#5-logic-chiến-lược-giao-dịch-core-engine--voting-system)
6. [Hệ thống AI & Machine Learning](#6-hệ-thống-trí-tuệ-nhân-tạo--machine-learning-aiml)
7. [Backtest & Dashboard PyQt6](#7-hệ-thống-backtest--trực-quan-hóa-pyqt6-dashboard)
8. [Quản trị rủi ro & Quản lý vốn](#8-quản-trị-rủi-ro--quản-lý-vốn-risk-management)
9. [Cấu trúc thư mục](#9-cấu-trúc-thư-mục-directory-tree)
10. [Yêu cầu & Cài đặt](#10-yêu-cầu--hướng-dẫn-cài-đặt-setup-instructions)
11. [Hướng dẫn cấu hình & Sử dụng](#11-hướng-dẫn-sử-dụng--cấu-hình-configuration--usage)
12. [Lộ trình phát triển](#12-lộ-trình-phát-triển-tương-lai-roadmap)
13. [Cảnh báo rủi ro](#13-cảnh-báo-rủi-ro-disclaimer)

---

## 1. TẦM NHÌN & TRIẾT LÝ HỆ THỐNG (VISION & PHILOSOPHY)

**KAIROS QUANT SYSTEM** không chỉ là một trading bot — mà là một **Hệ sinh thái giao dịch định lượng (Quantitative Trading Ecosystem)** được thiết kế để vận hành theo nguyên tắc **data-driven, xác suất và tự động hóa hoàn toàn**.

Tầm nhìn của hệ thống là xây dựng một **cỗ máy ra quyết định giao dịch không cảm xúc**, nơi mọi hành động đều được dẫn dắt bởi dữ liệu, mô hình thống kê và logic định lượng — không bao giờ bởi FOMO, Panic Sell hay thiên kiến nhận thức của con người.

> **Triết lý cốt lõi:**
> *"Tư duy chiến lược của con người — tốc độ và kỷ luật tuyệt đối của máy móc."*

### Kiến trúc vận hành

- **System Architect — Bộ não chiến lược:** Thiết kế kiến trúc tổng thể của hệ thống: từ logic giao dịch, pipeline dữ liệu (ETL), mô hình quản trị rủi ro, đến cơ chế chấm điểm và tối ưu hóa chiến lược. Mọi quyết định đều được chuẩn hóa thành thuật toán có thể kiểm chứng bằng dữ liệu lịch sử.
- **AI & Automation — Bộ máy thực thi:** AI và Automation được sử dụng để tăng tốc quá trình phát triển, tối ưu hóa mã nguồn và áp dụng Machine Learning vào việc phân loại tín hiệu thị trường, nhận diện mẫu hành vi giá và cải thiện hiệu suất chiến lược theo thời gian thực.

### Mục tiêu hệ thống

| Mục tiêu | Cách tiếp cận |
|----------|---------------|
| Loại bỏ cảm xúc khỏi giao dịch | Chuẩn hóa mọi quyết định thành thuật toán chấm điểm |
| Quyết định dựa trên xác suất thống kê | Ensemble Voting System — không phụ thuộc vào bất kỳ chỉ báo đơn lẻ nào |
| Tự động hóa hoàn toàn | 2 thread song song: quét tín hiệu + quản lý vị thế |
| Tối ưu hóa liên tục | Online retraining từ kinh nghiệm thực chiến (`trading_memory.csv`) |
| Khả năng mở rộng | Modular architecture — thêm chiến lược/sàn mà không viết lại core |

**KAIROS được tạo ra để biến giao dịch từ "nghệ thuật cảm tính" thành "khoa học định lượng"** — nơi kỷ luật, dữ liệu và thuật toán thay thế hoàn toàn trực giác.

---

## 2. GIỚI THIỆU CHUNG (OVERVIEW)

**KAIROS** là một **End-to-End Algorithmic Trading System** được thiết kế chuyên sâu cho thị trường tiền điện tử, bao phủ toàn bộ vòng đời của một giao dịch — từ thu thập dữ liệu, xử lý, ra quyết định đến backtest và trực quan hóa.

Hệ thống được xây dựng theo kiến trúc **pipeline định lượng hoàn chỉnh**, đảm bảo tốc độ xử lý cao, khả năng mở rộng và tính nhất quán trong ra quyết định.

### Các thành phần chính

**ETL Pipeline — Thu thập & chuẩn hóa dữ liệu**
Tự động thu thập dữ liệu đa khung thời gian (Multi-Timeframe: 1m → 1d) từ API Binance, OKX, Bybit qua CCXT. Dữ liệu được gộp nến (resample) bằng Polars `group_by_dynamic`, chuẩn hóa và lưu trữ theo cấu trúc time-series phục vụ phân tích định lượng.

**High-Performance Processing — Xử lý hiệu năng cao**
Ứng dụng **vectorization** kết hợp **Polars** (cho bar-to-bar) và **Pandas** (cho vectorized backtest) để xử lý hàng triệu dòng dữ liệu thị trường. Pipeline được tối ưu để giảm latency: tính toán 8 khung thời gian song song với `joblib`, không lookahead với kỹ thuật `merge_asof` và `forward-fill`.

**Intelligent Decision Engine — Bộ máy ra quyết định**
Kết hợp phân tích kỹ thuật truyền thống với **TradingMLP** (PyTorch ResBlock) dự báo trạng thái thị trường. Đồng thời tích hợp tín hiệu vi mô (OrderBook, CVD, Liquidations) từ WebSocket để nâng cao chất lượng quyết định.

**Visual Backtesting Platform — Nền tảng kiểm thử trực quan**
Ứng dụng Desktop PyQt6 + PyQtGraph, hoạt động như một nền tảng Quant chuyên nghiệp: tương tác với dữ liệu lịch sử, kiểm thử chiến lược, phân tích hiệu suất (Equity Curve, Drawdown, Win Rate) và tối ưu tham số.

### 4 Chế độ hoạt động (The 4-Mode Execution Framework)

| Mode | Mô tả | Đặc điểm kỹ thuật |
|------|-------|-------------------|
| **Live Trading** | Giao dịch tiền thật trực tiếp | Multi-thread, CCXT, WebSocket real-time |
| **Paper Trading** | Giao dịch giả lập với data thật | Mô phỏng slippage + phí, PnL chính xác |
| **Bar-to-Bar Backtest** | Mô phỏng từng cây nến (Event-driven) | Loại bỏ hoàn toàn Look-ahead bias |
| **Vectorized Backtest** | Quét hàng triệu nến siêu tốc | Pandas/Polars matrix ops, ML-guided routing |

---

## 3. TÍNH NĂNG CỐT LÕI & CÔNG NGHỆ ĐỘT PHÁ

### 🚀 Core Trading & Execution

- **Multi-Exchange Integration:** Tích hợp liền mạch Binance, OKX, Bybit qua chuẩn `ccxt`. Xử lý đặc thù từng sàn (Hedge Mode của OKX/Bybit với `posSide`, One-way Mode của Binance với `reduceOnly`).
- **Dynamic Position Sizing:** Tự động tính khối lượng lệnh từ USDT → số coin theo giá hiện tại. Đòn bẩy động điều chỉnh theo ATR (1x–50x).
- **Market Snapshot (WebSocket):** `KairosDataManager` singleton quản lý nhiều WebSocket đến Binance Futures, theo dõi đồng thời: CVD (Cumulative Volume Delta), Order Book Imbalance, Liquidations (Long/Short riêng biệt), Funding Rate.

### ⚡ Xử lý dữ liệu High-Performance (Dual-Engine)

- **Dual-Backtesting Pipeline:** Vectorized Backtest để tìm kiếm ý tưởng nhanh → Bar-to-Bar để thẩm định chính xác với slippage và độ trễ.
- **Polars Vectorization:** Tính toán 8 khung thời gian (1m → 1d) bằng `group_by_dynamic` với `start_by="window"` để khớp mốc thời gian chẵn. Nhanh hơn Pandas ~3–5x trên dataset lớn.
- **No Lookahead Guarantee:** `build_htf_candle()` dùng kỹ thuật shift index + ffill để đảm bảo nến đang chạy (live candle) không biết tương lai. Validated trên backtest bar-to-bar.

### 📡 Hệ thống Monitor & Thông báo

- **Logging System đa cấp độ:** `DynamicTimeFormatter` — trong backtest, timestamp log phản chiếu thời gian giả lập thay vì giờ thật, giúp debug rất trực quan.
- **PyQt6 Signal Bridge:** `SignalManager` (QObject) emit `data_changed` signal mỗi khi state thay đổi → GUI tự cập nhật thread-safe mà không cần polling.
- **Telegram / Email:** Push thông báo ngay khi mở/đóng lệnh, chạm SL/TP hoặc phát hiện rủi ro bất thường.

### 🌐 Multi-Exchange Support

| Sàn | Loại | Đặc thù xử lý |
|-----|------|----------------|
| Binance Futures | One-way Mode | `reduceOnly=True` khi đóng |
| OKX Swap | Hedge Mode | `posSide='long'/'short'` |
| Bybit Linear | Hedge Mode | `posSide='long'/'short'` |

---

## 4. KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

Hệ thống được thiết kế theo chuẩn **Modular Architecture**, phân tách rõ ràng trách nhiệm từng lớp (Separation of Concerns):

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA (ETL Pipeline)                                        │
│  Binance WS ──► KairosDataManager ──► CVD, Imbalance, Liq, Funding  │
│  CCXT REST ───► lay_ohlcv ──────────► 8 TF: 1m/3m/5m/15m/30m/1h/4h/1d│
│  Alternative.me ► Fear & Greed │ Binance Futures ► Open Interest      │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ Clean Multi-TF DataFrames
┌─────────────────────────────────▼────────────────────────────────────┐
│  LAYER 2 — AI/ML ENGINE (PyTorch)                                     │
│  Feature Engineering ──► 80-dim input (18 indicators × 4 TF)        │
│  TradingMLP: Input → 3×ResBlock(256) → Output(8 classes)            │
│  Online Retraining: trading_memory.csv ──► Fine-tuning weights       │
│  Output: state_id (0-7) + confidence + strategy_name                 │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ regime + strategy routing
┌─────────────────────────────────▼────────────────────────────────────┐
│  LAYER 3 — STRATEGY ENGINE (5 Strategies + Voting)                    │
│  Squeeze │ Breakout │ Trend Following │ Mean Reversion │ Scalping     │
│  Parallel Scoring (joblib, n_jobs=-1) across 8 timeframes            │
│  OrderFlow confirmation (CVD + Imbalance từ WebSocket)               │
│  Dynamic Leverage: ATR/ATR_mean ratio → auto-adjust [1x–50x]        │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ signal + score + leverage
┌─────────────────────────────────▼────────────────────────────────────┐
│  LAYER 4 — RISK + EXECUTION                                           │
│  Portfolio Risk: max orders │ capital allocation │ total risk %      │
│  Dynamic SL/TP: ATR-based, R:R=2:1, clamp [1.0–6.0]                │
│  CCXT Execution: market order + set_leverage → Binance/OKX/Bybit    │
│  State Persistence: JSON ──► PyQt6 signal ──► GUI update            │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ trade history + live data
┌─────────────────────────────────▼────────────────────────────────────┐
│  LAYER 5 — MONITORING & VISUALIZATION (PyQt6 Dashboard)              │
│  Tab 1: Realtime Trading │ Tab 2: Paper Trading                      │
│  Tab 3: Backtest & Strategy │ Tab 4: Vector Chart Analysis           │
│  Market Heatmap │ Equity Curve │ Trade Calendar │ Scatter Plot       │
└──────────────────────────────────────────────────────────────────────┘
```

### Mô tả từng lớp

**Lớp Dữ liệu (`/lay_du_lieu` — ETL Layer)**
Đóng vai trò hệ thống ETL (Extract-Transform-Load), kết nối trực tiếp với API các sàn để kéo OHLCV và thông tin tài khoản. Thực hiện gộp nến (Resampling) đa khung thời gian bằng Polars `group_by_dynamic`, xử lý lỗ hổng dữ liệu (Fill NA) và chuẩn bị dữ liệu đầu vào sạch cho các bộ lọc phía sau.

**Lớp Chiến lược (`/chien_luoc` — Core Logic Layer)**
Bộ não logic chứa các công thức toán học và hệ thống chỉ báo kỹ thuật (EMA, RSI, ADX, Bollinger Bands, ATR, Breakout...). Kết hợp phân tích kỹ thuật đa khung, tín hiệu vi mô (OrderBook, CVD) và macro (Fear & Greed, Open Interest) để chấm điểm tín hiệu theo cơ chế Voting System.

**Lớp Trí tuệ Nhân tạo (`/ml` — AI Core Engine)**
Trung tâm tri giác của hệ thống. **TradingMLP** (PyTorch MLP + ResBlock + GELU) phân loại thị trường thành 8 trạng thái vi mô. Điểm sáng nhất: cơ chế **tự tiến hóa (Self-Evolution)** — liên tục Fine-tuning trọng số từ kinh nghiệm thực chiến trong `trading_memory.csv`.

**Lớp Thực thi (`/thuc_thi_lenh` — Execution Layer)**
Execution Engine đảm bảo an toàn tài sản: quản lý API Keys, xử lý Rate-limits, tính Position Sizing, kiểm soát rủi ro danh mục và thực thi lệnh đóng/mở tối ưu trên 3 sàn.

**Lớp Giám sát (`/hien_thi` — Telemetry & Command Center)**
"Trung tâm chỉ huy" của toàn hệ sinh thái, biến hàng triệu điểm dữ liệu thành Insight trực quan: Equity Curve, Drawdown Chart, Market Heatmap, Trade Scatter Plot — phục vụ phân tích hiệu suất và tối ưu chiến lược.

---

## 5. LOGIC CHIẾN LƯỢC GIAO DỊCH (CORE ENGINE & VOTING SYSTEM)

**KAIROS** từ chối việc đi tìm "Holy Grail" hay phụ thuộc vào bất kỳ chỉ báo đơn lẻ nào. Điểm tạo nên sự khác biệt là **Cơ chế Bỏ phiếu Chiến lược Tổ hợp (Ensemble Strategy Voting System)**.

### 5 Chiến lược giao dịch tích hợp

| # | Chiến lược | ML Regime | Logic cốt lõi | Khung thời gian |
|---|------------|-----------|---------------|-----------------|
| 1 | **Breakout** | Đầu xu hướng (state 2) | Price action phá vỡ đỉnh/đáy N nến + ADX xác nhận + Volume tăng | 1H (weight 5), 15M (3), 4H (2) |
| 2 | **Trend Following** | Xu hướng mạnh (state 3) | EMA alignment 1D/4H/1H/15M đồng thuận + ADX > 20 + ATR đủ biến động | 1D (10), 4H (8), 1H (5), 15M (3) |
| 3 | **Mean Reversion** | Cao trào/Hồi quy (state 4,5) | RSI < 30 hoặc > 70 tại biên BB + Volume đột biến + Macro filter 4H | 1H (weight 5), 15M (3) |
| 4 | **Squeeze** | Nén chặt (state 1) | BB bandwidth < mean (BOP) → bứt phá + EMA hướng + Volume xác nhận | 15M (5), 30M (4), 1H (3) |
| 5 | **Scalping** | Nhiễu động (state 6) | RSI extreme tại biên BB + Volume spike trên khung ngắn | 1M (4), 3M (3), 5M (2) |

### Quy trình chấm điểm và Trọng số động

Mỗi chiến lược tính điểm theo từng khung thời gian với trọng số riêng, chạy **song song** bằng `joblib.Parallel(n_jobs=-1, prefer="threads")`:

```
Total Score = Σ (diem_khung × weight × he_so_bo_sung)
```

Trong đó `he_so_bo_sung` đến từ:
- **ATR**: biến động cao → tăng điểm xác nhận
- **ADX**: xu hướng đủ mạnh (>25) → cộng thêm
- **Volume**: `DOT_BIEN` → nhân đôi điểm
- **OrderFlow**: CVD và Order Book Imbalance từ WebSocket xác nhận cuối

**Ngưỡng xác nhận (Threshold = 20 điểm):** Chỉ khi `Total Score ≥ 20` hệ thống mới mở lệnh, giảm thiểu tín hiệu giả (False Signal).

### Điều hướng chiến lược theo ML Regime

```python
STRATEGY_MAP = {
    0: None,               # Đóng_Băng  → không trade
    1: "Squeeze",          # Nén_Chặt   → bứt phá sau nén
    2: "Breakout",         # Đầu_XH     → vào sớm theo breakout
    3: "Trend_following",  # XH_Mạnh    → follow trend đa khung
    4: "Mean_reversion",   # Cao_Trào   → đánh ngược khi kiệt sức
    5: "Mean_reversion",   # Hồi_Quy    → đánh ngược về trung bình
    6: "Scalping",         # Nhiễu_Động → scalp biên độ hẹp
    7: None,               # Quét_TK    → quá nguy hiểm, bỏ qua
}
```

---

## 6. HỆ THỐNG TRÍ TUỆ NHÂN TẠO & MACHINE LEARNING (AI/ML)

Module ML độc lập (`/ml/trang_thai_thi_truong_ml`) đóng vai trò **màng lọc rủi ro tối thượng** và bộ điều phối chiến lược tự động.

### Kiến trúc TradingMLP (PyTorch)

```
Input (80-dim): 18 features × 4 TF (5M/15M/1H/4H) + 8 ctx_last_state
      ↓
Linear(80→256) + BatchNorm + GELU + Dropout(0.15)
      ↓
ResBlock × 3: [Linear(256→256) + BN + GELU + Dropout(0.3)] + residual
      ↓
Linear(256→64) + BatchNorm + GELU + Dropout(0.3)
      ↓
Linear(64→8)  →  8 Market Regimes
```

**Kaiming He Initialization** giúp hội tụ nhanh từ epoch đầu. **GELU thay ReLU** cho gradient mượt hơn. **ResBlock** tránh vanishing gradient.

### 18 Features Lõi (tính trên mỗi khung thời gian)

| Feature | Ý nghĩa | Công thức |
|---------|---------|-----------|
| D | Khoảng cách giá–EMA50 | `(close - EMA50) / EMA50` |
| S | Độ dốc EMA50 | `(EMA50 - EMA50.shift(5)) / EMA50.shift(5)` |
| ADX | Sức mạnh xu hướng | Wilder's ADX (SMA-based, 14 periods) |
| RSI | Động lượng | Cutler's RSI (SMA gain/loss, 14 periods) |
| RSIslope | Độ dốc RSI | `RSI - RSI.shift(3)` |
| ROC | Tốc độ thay đổi giá | `(close - close.shift(10)) / close.shift(10)` |
| ATRn | ATR chuẩn hóa | `ATR / close` |
| VOLz | Z-score Volume | `(volume - vol_SMA) / vol_std` |
| SpreadATR | Tỷ lệ spread/ATR | `(high - low) / ATR` |
| BBwidth | Độ rộng BB | `(bb_upper - bb_lower) / bb_mid` |
| SQZ | Squeeze signal | `1.0` nếu BB nằm trong KC |
| CHOP | Choppiness Index | `100 × log10(ΣTR / (high_max - low_min)) / log10(N)` |
| ER | Efficiency Ratio | `|price_change| / Σ|individual_changes|` |
| BBpctB | Vị trí trong BB | `(close - bb_lower) / (bb_upper - bb_lower)` |
| VWAPd | VWAP deviation | `(close - VWAP) / ATR` |
| WickUpProp | Tỷ lệ bóng trên | `(high - body_top) / (high - low)` |
| WickDnProp | Tỷ lệ bóng dưới | `(body_bottom - low) / (high - low)` |
| BodyProp | Tỷ lệ thân nến | `(body_top - body_bottom) / (high - low)` |

### 8 Trạng thái thị trường (Market Regimes)

| ID | Tên | Đặc điểm | Chiến lược |
|----|-----|----------|-----------|
| 0 | Đóng_Băng | Volume cạn kiệt, không có momentum | Không trade |
| 1 | Nén_Chặt | BB squeeze, tích lũy năng lượng | Squeeze breakout |
| 2 | Đầu_Xu_Hướng | Xu hướng chớm hình thành trên M15 | Breakout entry sớm |
| 3 | Xu_Hướng_Mạnh | H4/H1/M15 đồng thuận | Trend Following |
| 4 | Cao_Trào | Giá chạy quá xa, sắp đảo chiều | Mean Reversion |
| 5 | Hồi_Quy | Giật ngược về trung bình | Mean Reversion |
| 6 | Nhiễu_Động | Đi ngang biên độ hẹp, giật liên tục | Scalping |
| 7 | Quét_Thanh_Khoản | Vol đột biến, tin tức mạnh, risk-off | Không trade |

### Cơ chế Tự tiến hóa (Self-Evolution Loop)

```
Mỗi lệnh đóng
    ↓
danh_gia_ml(packet, pnl, drawdown)
    ↓
reward = pnl × 1.0 (thắng) | pnl × 2.0 (thua)  ← Penalty gấp đôi khi thua
    ↓
Ghi vào trading_memory.csv (state, confidence, features_snapshot, reward)
    ↓
tu_dong_hoc_tu_log()  ← Fine-tune weights khi đủ dữ liệu (≥10 mẫu)
    ↓
Model mới lưu vào model_pytorch.pth  ← Sẵn sàng cho lần trade tiếp theo
```

**Epsilon-greedy Exploration:** Khi `confidence < 0.2`, hệ thống ngẫu nhiên khám phá (random regime) thay vì luôn chọn class có xác suất cao nhất — tránh bị mắc kẹt trong local optimum.

---

## 7. HỆ THỐNG BACKTEST & TRỰC QUAN HÓA (PYQT6 DASHBOARD)

KAIROS sở hữu một phần mềm Desktop chuyên dụng xây dựng bằng **PyQt6 + PyQtGraph**, biến dữ liệu khô khan thành Insight có giá trị.

### Backtest Dashboard — Kết quả kiểm thử chiến lược

![Backtest Dashboard](assets/backtest_dashboard.png)

*Backtest tháng 01/2026: 559 lệnh được thực thi giả lập. Dashboard hiển thị lãi/lỗ hàng ngày (Daily PnL Bar), lịch giao dịch theo tuần, phân phối lệnh theo khung thời gian giữ lệnh, và các chỉ số hiệu suất chuyên sâu.*

### Demo Dashboard — Giao dịch giả lập realtime

![Demo Dashboard](assets/demo_dashboard.png)

*Paper Trading đang chạy với 14 vị thế mở đồng thời. Market Heatmap hiển thị tín hiệu 7 khung thời gian (1m → 1d) cho 40+ coin. Lịch sử lệnh với lý do đóng chi tiết (SL giá, TP giá, tín hiệu đảo chiều).*

### Các tính năng Dashboard

**📊 Tab Backtest & Phân tích Chiến lược**

| Widget | Chức năng |
|--------|----------|
| Daily PnL Chart | Biểu đồ nến lãi/lỗ hàng ngày — click vào ngày để xem chi tiết lệnh |
| Trade Calendar | Heatmap lịch — màu sắc biểu thị cường độ PnL theo ngày trong tháng |
| Phân phối lệnh | Histogram thời gian giữ lệnh (1m, 5m, 15m, 1h, 4h, 1d, >1d) |
| Chỉ số hiệu suất | Win Rate, Profit Factor, R:R, Kỳ vọng/lệnh, Lợi nhuận TB, Thua lỗ TB |
| Equity Curve | Đường tài sản tổng hợp + max drawdown visualization |
| Trade Scatter | PnL vs Hold Duration — phát hiện "gồng lỗ" hay "chốt lời sớm" |
| Long vs Short | So sánh hiệu suất Long/Short riêng biệt |
| Kết quả theo coin | Bảng xếp hạng coin theo tổng PnL |

**🟢 Tab Realtime & Demo**

| Widget | Chức năng |
|--------|----------|
| Market Heatmap | Bảng nhiệt 40+ coin × 7 timeframe — màu xanh/đỏ theo tín hiệu ML |
| Vị thế đang mở | Bảng real-time: Symbol, Side, Entry Price, Size, Thời gian |
| Tổng quan tài khoản | Equity, PnL ròng, Win Rate, Tổng lệnh |
| Lịch sử giao dịch | Bảng cuộn: Symbol, PnL, Thời lượng giữ lệnh, Lý do đóng |

---

## 8. QUẢN TRỊ RỦI RO & QUẢN LÝ VỐN (RISK MANAGEMENT)

Được thiết kế với tư duy của Tài chính định lượng, rủi ro là yếu tố được đặt lên hàng đầu:

### 3 Lớp kiểm soát rủi ro

**Lớp 1 — Portfolio Risk (trước khi mở lệnh)**
- Giới hạn số lệnh đồng thời (`max_lenh_cho_phep` = 20)
- Chia vốn đều theo số coin theo dõi: `von_moi_coin = tong_von / so_coin`
- Kiểm tra tổng rủi ro: `Σ |entry - SL| × amount < max_rui_ro_tong (10%)`

**Lớp 2 — Dynamic SL/TP (per trade)**
```
vol_ratio     = ATR / ATR_mean            # đo độ biến động tương đối
he_so_sl      = base_sl(2.7) / vol_ratio  # clamp [1.0, 6.0]
SL = entry ∓ ATR × he_so_sl
TP = entry ± ATR × he_so_sl × RR(2.0)
```
Khi thị trường volatile cao (ATR > mean): SL rộng hơn để tránh noise quét stop.
Khi thị trường yên tĩnh: SL thắt chặt để bảo vệ vốn tốt hơn.

**Lớp 3 — Dynamic Leverage (per signal)**
```
vol_ratio = ATR_15m / ATR_mean_15m
leverage  = int(don_bay_goc / vol_ratio).clip(1, 50)
```

| Trạng thái ATR | Hành động Leverage |
|----------------|-------------------|
| ATR cao + Volume đột biến | Giảm 3x — tin mạnh, rủi ro cao |
| ADX thấp + BB nén | Giới hạn 10x — tích lũy, sideway |
| Breakout rõ + ADX > 25 | Giữ nguyên đòn bẩy gốc |
| ATR > mean 5% | Giảm 1x |
| ATR < mean 5% | Tăng 1x |

### Time Filter
Lọc giờ 5h sáng VN (UTC+7) — giờ giãn spread trên Futures Binance. Loại nến volume < 5% trung bình 200 nến (thị trường "chết").

---

## 9. CẤU TRÚC THƯ MỤC (DIRECTORY TREE)

```
KAIROS_QUANT_SYSTEM_v2.0/
│
├── main.py                          # Entry point: PyQt6 app, 4-tab dashboard
├── requirements.txt                 # Python dependencies
│
├── config/                          # QUẢN LÝ CẤU HÌNH
│   ├── cau_hinh_giao_dich.yaml      # Coins (40+), leverage, capital, SL/TP %
│   ├── thong_tin_san.yaml           # Phí, min_notional, leverage limits per exchange
│   ├── cau_hinh_ao_config.json      # Paper trading: vốn ảo, phí, slippage
│   └── tai_khoan_api.json           # API Keys (gitignore — không commit)
│
├── ml/                              # KHỐI MACHINE LEARNING (AI Core)
│   ├── main.py                      # Luồng ML: Train, Dashboard, Auto-learning
│   └── trang_thai_thi_truong_ml/
│       ├── ml_model.py              # TradingMLP + ResBlock (PyTorch)
│       ├── ml_predict.py            # Inference + STRATEGY_MAP + reward logging
│       ├── tao_feature.py           # Feature engineering 18×4 TF (Polars vectorized)
│       ├── ml_compare.py            # So sánh hiệu suất các phiên bản model
│       ├── ml_deploy.py             # Đóng gói model sẵn sàng thực chiến
│       └── du_lieu_ml/
│           ├── model_pytorch.pth    # Trọng số mạng Neural đã train
│           ├── model_info.json      # Cấu hình input_dim, output_dim, feature_names
│           ├── scaler_params.json   # Z-score scaler (mean, std)
│           └── trading_memory.csv   # Nhật ký kinh nghiệm thực chiến
│
├── lay_du_lieu/                     # KHỐI ETL & DATA ACQUISITION
│   ├── lay_ohlcv.py                 # Fetch OHLCV + resample 8 TF (Polars)
│   ├── lay_marketsnapshot.py        # WebSocket: CVD, OBI, Liq, Funding (Singleton)
│   ├── lay_macro.py                 # Fear & Greed Index + Open Interest
│   └── lay_thong_tin_tai_khoan.py  # Balance, open positions query
│
├── chien_luoc/                      # KHỐI CORE LOGIC (Strategy Layer)
│   │
│   ├── logic_bar_to_bar/            # Bar-to-bar (Polars) — dùng cho Live/Demo/Backtest
│   │   ├── phan_tich_ky_thuat/      # 7 indicator modules
│   │   │   ├── xu_huong.py          # EMA trend, ADX (Wilder's smoothing)
│   │   │   ├── bien_dong.py         # ATR, Bollinger Bands + Squeeze detection
│   │   │   ├── dong_luong_dao_chieu.py # RSI (Cutler's, EWM)
│   │   │   ├── khoi_luong.py        # Volume rolling mean + spike detection
│   │   │   ├── cau_truc_gia.py      # Breakout: rolling high/low + price action
│   │   │   ├── vi_the.py            # Sentiment: Funding Rate, CVD, Liquidations
│   │   │   └── chu_ky.py            # Time filter: dayofweek, hour (5h spread)
│   │   │
│   │   ├── chien_luoc/              # 5 strategy implementations
│   │   │   ├── chien_luoc_breakout.py        # joblib parallel, 8 TF, OrderFlow confirm
│   │   │   ├── chien_luoc_theo_trend_following.py  # EMA+ADX+ATR multi-TF
│   │   │   ├── chien_luoc_mean_reversion.py  # RSI extreme + BB + Volume spike
│   │   │   ├── chien_luoc_squeeze.py         # BB BOP + breakout direction
│   │   │   └── chien_luoc_scalping.py        # RSI+BB on 1M/3M/5M, ADX filter
│   │   │
│   │   ├── quan_ly_chien_luoc.py    # ML routing → strategy dispatch
│   │   ├── chien_luoc_trang_thai_thi_truong.py  # Gate: time + data + ML check
│   │   ├── chien_luoc_don_bay.py    # Dynamic leverage: ATR-based [1x–50x]
│   │   └── stoploss_takeprofit.py   # ATR SL/TP: base_sl=2.7, RR=2.0
│   │
│   └── logic_vectorized/            # Vectorized (Pandas MTF) — dùng cho Backtest
│       ├── phan_tich_ky_thuat/      # MTF indicators (no lookahead bias)
│       │   ├── xu_huong.py          # EMA MTF: live candle + closed candle
│       │   ├── bien_dong.py         # ATR MTF + BB Squeeze MTF
│       │   ├── dong_luong_dao_chieu.py # RSI MTF (Wilder's smoothing)
│       │   ├── khoi_luong.py        # Volume MTF: live cumsum
│       │   └── cau_truc_gia.py      # Breakout MTF: ffill anchoring
│       ├── chien_luoc/              # 5 vectorized strategies (Pandas boolean masking)
│       ├── quan_ly_chien_luoc.py    # Run all → ML regime mask → combine signals
│       ├── stoploss_takeprofit.py   # ATR SL/TP vectorized (per-candle)
│       ├── chien_luoc_don_bay.py    # Leverage vectorized column
│       └── chien_luoc_trang_thai_thi_truong.py  # Time filter + volume filter
│
├── thuc_thi_lenh/                   # KHỐI EXECUTION ENGINE
│   ├── bo_may_thuc_thi.py           # CCXT pool (Singleton) — Binance/OKX/Bybit
│   ├── mo_lenh.py                   # Open orders: market/limit + set_leverage
│   ├── dong_lenh.py                 # Close: reduceOnly + posSide (Hedge Mode)
│   ├── theo_doi_lenh.py             # Position monitor: PnL%, entry, size
│   ├── quan_ly_lenh.py              # State manager + PyQt6 data_changed signal
│   ├── chon_san_giao_dich.py        # Exchange selector from config
│   ├── quan_ly_danh_muc.py          # Portfolio risk: max orders, capital allocation
│   └── ket_noi_san/
│       ├── binance_api.py           # Binance CCXT wrapper
│       ├── okx_api.py               # OKX CCXT wrapper
│       └── bybit_api.py             # Bybit CCXT wrapper
│
├── chuc_nang/                       # CÁC CHẾ ĐỘ VẬN HÀNH
│   ├── chay_realtime.py             # Live: 2 threads (Scanner + Manager)
│   ├── chay_demo.py                 # Paper: slippage + fee simulation
│   ├── vectorized_backtest.py       # Vectorized: ML + all strategies + SL/TP
│   ├── backtest_donluong.py         # Bar-to-bar: single coin deep test
│   └── backtest_daluong.py          # Bar-to-bar: multi-coin parallel
│
├── hien_thi/                        # KHỐI DASHBOARD (PyQt6 + PyQtGraph)
│   ├── dashboard_realtime.py        # Live: heatmap + positions + history
│   ├── dashboard_demo.py            # Demo: same layout as realtime
│   ├── dashboard_backtest.py        # Backtest: equity curve + calendar + scatter
│   └── dashboard_vectorized.py      # Vector: advanced candlestick + indicators
│
├── du_lieu/                         # KHO DỮ LIỆU
│   ├── du_lieu_vectorized/          # CSV backtest results (per symbol)
│   └── thong_tin_lenh/              # JSON state: open orders + trade history
│
├── thong_bao/                       # ALERTS
│   ├── gui_telegram.py              # Telegram bot notifications
│   └── gui_email.py                 # Email reports
│
└── utils/                           # TIỆN ÍCH HỖ TRỢ
    ├── ham_tien_ich.py              # MTF merge (merge_asof), PnL calc, build_htf_candle
    ├── log.py                       # DynamicTimeFormatter (backtest-aware)
    ├── doc_cau_hinh.py              # YAML/JSON loaders
    ├── thoi_gian.py                 # Timestamp utilities
    ├── chuyen_doi_don_vi.py         # USDT → coin quantity, round lot
    └── save_dataflie.py             # CSV export (Pandas + Polars)
```

---

## 10. YÊU CẦU & HƯỚNG DẪN CÀI ĐẶT (SETUP INSTRUCTIONS)

### Yêu cầu hệ thống

| Thành phần | Phiên bản tối thiểu | Ghi chú |
|-----------|--------------------|----|
| Python | 3.11+ | Khuyến nghị 3.11 |
| RAM | 4 GB | 8 GB nếu chạy backtest lớn |
| OS | Windows 10/11, Ubuntu 20.04+ | PyQt6 cần GUI environment |
| GPU | Tùy chọn | PyTorch dùng CPU nếu không có GPU |

### Thư viện chính

```
torch >= 2.0          # Deep Learning (TradingMLP)
polars >= 0.19        # High-performance DataFrames
pandas >= 2.0         # Vectorized backtest + MTF indicators
PyQt6 >= 6.4          # Desktop GUI dashboard
pyqtgraph             # Real-time charts
ccxt >= 4.0           # Multi-exchange API
joblib >= 1.3         # Parallel processing (multi-TF scoring)
ta >= 0.10            # Technical Analysis (RSI, EMA, ADX)
scikit-learn          # ML utilities (train/test split)
websocket-client      # Binance WebSocket streams
requests              # Fear & Greed, Open Interest APIs
PyYAML                # Config file parsing
rich                  # Terminal logging
```

### Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/PVinh05-Quant/Kairos-Quant-System.git
cd Kairos-Quant-System

# 2. Tạo và kích hoạt virtual environment
python -m venv env
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình API keys (bắt buộc cho Live/Demo)
cp config/tai_khoan_api.json.example config/tai_khoan_api.json
# Mở file và điền API keys của bạn

# 5. Khởi động ứng dụng
python main.py
```

> ⚠️ **Bảo mật:** File `config/tai_khoan_api.json` đã được thêm vào `.gitignore`. **Tuyệt đối không commit file này lên GitHub.** Chỉ cấp quyền Trade (không cấp Withdraw) cho API keys.

---

## 11. HƯỚNG DẪN SỬ DỤNG & CẤU HÌNH (CONFIGURATION & USAGE)

### 11.1 Cấu hình giao dịch (`config/cau_hinh_giao_dich.yaml`)

```yaml
# Sàn giao dịch chính (hỗ trợ: binance, okx, bybit)
san_giao_dich_chinh: "okx"

# Danh sách cặp coin theo dõi (40+ cặp mặc định)
cap_giao_dich:
  - "BTC/USDT"
  - "ETH/USDT"
  - "SOL/USDT"
  # ... thêm tùy ý

# Quản lý vốn
von_moi_lenh_usdt: 100      # Vốn mỗi lệnh (USDT)
don_bay: 7                   # Đòn bẩy gốc (trung vị 5-9, được điều chỉnh động)
max_lenh_cho_phep: 20        # Tối đa lệnh mở đồng thời

# Quản lý rủi ro cơ bản
cat_lo_percent: 0.1          # Cắt lỗ 10% (backup nếu ATR SL không kích hoạt)
chot_loi_percent: 0.15       # Chốt lời 15% (backup nếu ATR TP không kích hoạt)
```

### 11.2 Cấu hình Paper Trading (`config/cau_hinh_ao_config.json`)

```json
{
  "so_du_ban_dau": 10000,
  "ngay_bat_dau": "2025-01-01",
  "ngay_ket_thuc": "2025-12-31",
  "phi_giao_dich": 0.0004,
  "do_truot_gia": 0.0001
}
```

### 11.3 API Keys (`config/tai_khoan_api.json`)

```json
{
  "binance": {
    "apiKey": "YOUR_BINANCE_API_KEY",
    "secret": "YOUR_BINANCE_SECRET",
    "options": { "defaultType": "future" }
  },
  "okx": {
    "apiKey": "YOUR_OKX_API_KEY",
    "secret": "YOUR_OKX_SECRET",
    "password": "YOUR_OKX_PASSPHRASE",
    "options": { "defaultType": "swap" }
  },
  "bybit": {
    "apiKey": "YOUR_BYBIT_API_KEY",
    "secret": "YOUR_BYBIT_SECRET",
    "options": { "defaultType": "linear" }
  }
}
```

### 11.4 Workflow sử dụng đề xuất

```
Bước 1: Train AI Model
  └─► python ml/main.py  (hoặc dùng Tab ML trong app)
  
Bước 2: Vectorized Backtest (tìm ý tưởng nhanh)
  └─► Tab "Chiến thuật & Backtest" → chọn ngày → Run
  
Bước 3: Bar-to-Bar Backtest (thẩm định chi tiết)
  └─► python chuc_nang/backtest_donluong.py
  
Bước 4: Paper Trading (forward test an toàn)
  └─► Tab "Tài khoản Demo" → Start Demo
  
Bước 5: Live Trading (khi đã đủ tự tin)
  └─► Tab "Giao dịch Realtime" → Start Live
```

---

## 12. LỘ TRÌNH PHÁT TRIỂN TƯƠNG LAI (ROADMAP)

### v2.1 — Alternative Data Integration
- **NLP Sentiment Analysis:** Quét mạng xã hội (X/Twitter) và báo cáo kinh tế để đo tâm lý đám đông, dùng làm trọng số bổ sung trong Voting System.
- **On-chain Clustering:** Theo dõi dòng tiền cá mập (Whale wallets) và Exchange Flows để phát hiện tích lũy/phân phối.
- **Orderbook Computer Vision:** AI thị giác máy tính phân tích Heatmap sổ lệnh để phát hiện Spoofing và Iceberg Orders.

### v2.2 — Reinforcement Learning
Nâng cấp module "Thầy giáo" thành AI có khả năng **tự chơi hàng triệu kịch bản thị trường** để tìm ra bộ hyperparameters tối ưu — những con số mà bộ não người không thể tự tính được. Áp dụng PPO/SAC từ `stable-baselines3`.

### v2.3 — Portfolio Optimization
Thay vì phân bổ vốn đều, thuật toán **Modern Portfolio Theory** tự động chọn tổ hợp coin có tương quan thấp (low correlation), tạo lá chắn tự nhiên (natural hedge) cho tổng tài khoản. Sử dụng `scipy.optimize` để tìm Efficient Frontier.

### v3.0 — Options & Derivatives Expansion
Mở rộng sang thị trường Phái sinh Quyền chọn (Options): kiếm lợi nhuận từ **theta decay** (thời gian hợp đồng hao mòn) và **volatility trading** — ngay cả khi giá đi ngang, tài khoản vẫn sinh lời.

### v3.1 — Hybrid Low-level Performance
Kết hợp **Rust/C++** cho các hot-path tính toán nặng (feature engineering, order execution), giảm latency xuống mức microseconds. Python đóng vai "kiến trúc sư", Rust đóng vai "vận động viên" thực thi.

---

## 13. CẢNH BÁO RỦI RO (DISCLAIMER)

> ⚠️ **QUAN TRỌNG — ĐỌC TRƯỚC KHI SỬ DỤNG**

**1. Rủi ro thị trường:** Cryptocurrency là thị trường cực kỳ biến động. KAIROS phân tích dựa trên **xác suất thống kê từ dữ liệu quá khứ** — không có khả năng dự đoán tương lai chính xác 100%. Lợi nhuận trong quá khứ **không đảm bảo** cho tương lai.

**2. Rủi ro kỹ thuật:** Dù đã được kiểm thử, phần mềm vẫn có thể tồn tại bug không lường trước, hoặc chịu ảnh hưởng từ: độ trễ mạng lưới, lỗi API sàn giao dịch, sự cố infrastructure, hay thay đổi đột ngột trong chính sách của sàn.

**3. Trách nhiệm sử dụng:** Người dùng **hoàn toàn chịu trách nhiệm** về các quyết định cấu hình vốn, đòn bẩy và API Keys của mình. Tác giả không chịu trách nhiệm cho bất kỳ tổn thất tài chính nào phát sinh trong quá trình sử dụng hệ thống.

**4. Khuyến nghị:** Luôn bắt đầu với **Paper Trading** ít nhất 30 ngày trước khi chuyển sang Live Trading. Không bao giờ sử dụng vốn mà bạn không thể chấp nhận mất.

---

<div align="center">

## 👨‍💻 Thông tin tác giả

**P. Vinh** — Quantitative Developer & Data Engineer

*"Biến giao dịch từ nghệ thuật cảm tính thành khoa học định lượng."*

[![Email](https://img.shields.io/badge/Email-ppvinh1513%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:ppvinh1513@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-PVinh05--Quant-181717?style=flat-square&logo=github)](https://github.com/PVinh05-Quant)

---

<sub>Built with ❤️ · Python · PyTorch · Polars · PyQt6 · CCXT · Powered by Data</sub>

</div>
