<div align="center">

# KAIROS QUANT SYSTEM v2.0
### Hệ thống giao dịch định lượng tự động — AI/ML · Multi-Exchange · Real-time · Backtesting

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Polars](https://img.shields.io/badge/Polars-DataFrame-CD792C?style=flat-square)](https://pola.rs/)
[![Pandas](https://img.shields.io/badge/Pandas-DataFrame-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-41CD52?style=flat-square&logo=qt)](https://www.riverbankcomputing.com/software/pyqt/)
[![CCXT](https://img.shields.io/badge/CCXT-Multi--Exchange-2EA043?style=flat-square)](https://github.com/ccxt/ccxt)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## Tổng Quan

**Kairos** là một hệ thống giao dịch định lượng (Quantitative Trading System) được xây dựng từ đầu, tích hợp đầy đủ pipeline từ thu thập dữ liệu thô đến thực thi lệnh tự động trên 3 sàn giao dịch (Binance, OKX, Bybit). Điểm nổi bật là module **AI phân loại trạng thái thị trường (Market Regime Detection)** sử dụng PyTorch MLP, cho phép hệ thống tự động lựa chọn chiến lược phù hợp với từng pha thị trường.

Dự án được thiết kế theo nguyên tắc **production-ready**: multi-threading, WebSocket real-time, SL/TP động theo ATR, quản lý rủi ro danh mục, và dashboard PyQt6 đầy đủ cho cả realtime lẫn backtesting.

---

## Demo Hệ Thống

### Backtest Dashboard — Kết quả kiểm thử chiến lược

![Backtest Dashboard](assets/backtest_dashboard.png)

> Backtest trên dữ liệu thực tháng 01/2026: 559 lệnh được lọc và thực thi giả lập. Dashboard hiển thị lãi/lỗ hàng ngày, lịch giao dịch theo tuần, phân phối lệnh theo khung thời gian, và các chỉ số hiệu suất (Win Rate, Profit Factor, Sharpe-proxy).

### Demo Dashboard — Giao dịch giả lập realtime

![Demo Dashboard](assets/demo_dashboard.png)

> Chạy demo (paper trading) với 14 vị thế mở đồng thời trên nhiều cặp coin. Bảng nhiệt thị trường (Market Heatmap) hiển thị tín hiệu theo 7 khung thời gian (1m → 1d). Lịch sử lệnh hiển thị lý do đóng lệnh chi tiết (SL, TP, tín hiệu đảo chiều).

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                               │
│  Binance WebSocket ──► KairosDataManager ──► CVD / Imbalance    │
│  CCXT REST API ──────► lay_ohlcv ──────────► 8 Timeframes       │
│  Alternative.me ─────► Fear & Greed Index                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    ML ENGINE (PyTorch)                           │
│  Feature Engineering ──► TradingMLP ──► 8 Market Regimes        │
│  18 features × 4 TF = 80-dim input │  ResBlock × 3 + GELU       │
│  Online retraining từ trading_memory.csv (Reinforcement)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ regime (0-7)
┌───────────────────────────▼─────────────────────────────────────┐
│                  STRATEGY LAYER (5 chiến lược)                   │
│  regime 1 → Squeeze     │  regime 2 → Breakout                  │
│  regime 3 → Trend Follow│  regime 4,5 → Mean Reversion          │
│  regime 6 → Scalping    │  regime 0,7 → No Trade                │
│  Multi-TF Parallel Scoring (joblib) + OrderFlow confirmation     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ signal + leverage
┌───────────────────────────▼─────────────────────────────────────┐
│                  EXECUTION + RISK                                │
│  Dynamic SL/TP (ATR-based) │ Dynamic Leverage [1x–50x]          │
│  Portfolio Risk Check      │ CCXT Multi-Exchange                 │
│  PyQt6 Signal → GUI Update │ Telegram Notification               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Công Nghệ Sử Dụng

| Lớp | Công nghệ | Mục đích |
|-----|-----------|----------|
| **Data** | Polars, Pandas, NumPy | Xử lý dữ liệu OHLCV đa khung thời gian |
| **Streaming** | websocket-client, CCXT | WebSocket realtime + REST API 3 sàn |
| **ML** | PyTorch, scikit-learn | MLP Classifier phân loại 8 market regime |
| **Strategy** | joblib, Polars | Parallel scoring đa khung thời gian |
| **Execution** | CCXT | Order management Binance / OKX / Bybit |
| **Backtest** | Pandas, Polars | Vectorized backtesting không lookahead |
| **GUI** | PyQt6, PyQtGraph | Dashboard realtime + backtest chart |
| **Infra** | threading, JSON, YAML | Multi-thread, state persistence, config |

---

## Tính Năng Chính

### 1. AI Market Regime Detection (PyTorch)
- **TradingMLP**: MLP với 3 ResBlock (GELU, BatchNorm, Dropout) phân loại 8 trạng thái thị trường
- **80 features**: 18 chỉ báo kỹ thuật (RSI, ADX, BB, ATR, VWAP, OFI...) × 4 khung thời gian (5M/15M/1H/4H)
- **Online Learning**: Ghi nhận kết quả mỗi lệnh vào `trading_memory.csv` → tự động tái huấn luyện từ kinh nghiệm thực chiến
- **Epsilon-greedy exploration**: Ngẫu nhiên khám phá khi độ tin cậy thấp (conf < 0.2)

### 2. 5 Chiến Lược Giao Dịch Tích Hợp
| Chiến lược | Regime | Logic |
|-----------|--------|-------|
| **Breakout** | Đầu xu hướng | Price action phá vỡ + ADX + Volume đa khung |
| **Trend Following** | Xu hướng mạnh | EMA alignment 1D/4H/1H/15M + ADX + ATR |
| **Mean Reversion** | Cao trào / Hồi quy | RSI extreme + BB bands + Volume spike |
| **Squeeze** | Nén chặt | BB squeeze (BOP) → bứt phá + Volume xác nhận |
| **Scalping** | Nhiễu động | RSI extreme tại BB bands trên 1M/3M/5M |

### 3. Data Pipeline Không Lookahead
- **Multi-timeframe**: 8 khung (1m → 1d) xây từ 1m gốc bằng Polars `group_by_dynamic`
- **Live candle simulation**: `build_htf_candle()` tính OHLCV nến đang chạy mà không nhìn tương lai
- **WebSocket Market Snapshot**: CVD, Order Book Imbalance, Liquidations, Funding Rate theo realtime

### 4. Backtesting Engine (Vectorized)
- **Vectorized signal generation**: Pandas boolean masking — không loop từng nến
- **ML-guided routing**: Mỗi nến được route đến đúng chiến lược theo regime
- **Dynamic SL/TP**: Tính ATR-based SL/TP cho từng nến (không dùng % cố định)
- **Realistic simulation**: Slippage + phí giao dịch mô phỏng chính xác

### 5. Risk Management
- Giới hạn số lệnh đồng thời (`max_lenh_cho_phep`)
- Đòn bẩy động: ATR/ATR_mean ratio → auto-adjust [1x–50x]
- SL/TP theo ATR với Risk:Reward = 2:1 mặc định
- Lọc giờ giao dịch (tránh 5h sáng VN — giãn spread futures)

---

## Cấu Trúc Dự Án

```
Kairos-v2/
├── main.py                      # Entry point PyQt6 — 4 tab dashboard
├── requirements.txt
│
├── config/                      # Cấu hình hệ thống
│   ├── cau_hinh_giao_dich.yaml  # Coins, leverage, vốn/lệnh
│   ├── thong_tin_san.yaml       # Phí, giới hạn sàn
│   └── cau_hinh_ao_config.json  # Paper trading params
│
├── lay_du_lieu/                 # Data collection
│   ├── lay_ohlcv.py             # Fetch + resample 8 TF (Polars)
│   ├── lay_marketsnapshot.py    # WebSocket realtime (CVD, OBI, Liq)
│   └── lay_macro.py             # Fear & Greed + Open Interest
│
├── ml/                          # Machine Learning
│   └── trang_thai_thi_truong_ml/
│       ├── ml_model.py          # TradingMLP + ResBlock (PyTorch)
│       ├── tao_feature.py       # Feature engineering 18×4 TF
│       └── ml_predict.py        # Inference + reward logging
│
├── chien_luoc/
│   ├── logic_bar_to_bar/        # Bar-to-bar strategies (Polars)
│   │   ├── phan_tich_ky_thuat/  # 7 indicator modules
│   │   ├── chien_luoc/          # 5 strategy implementations
│   │   ├── quan_ly_chien_luoc.py# ML routing → strategy
│   │   ├── chien_luoc_don_bay.py# Dynamic leverage
│   │   └── stoploss_takeprofit.py# ATR-based SL/TP
│   └── logic_vectorized/        # Vectorized strategies (Pandas)
│       ├── phan_tich_ky_thuat/  # MTF indicators (no lookahead)
│       ├── chien_luoc/          # 5 vectorized strategies
│       ├── quan_ly_chien_luoc.py# ML-guided signal routing
│       └── stoploss_takeprofit.py
│
├── thuc_thi_lenh/               # Execution layer
│   ├── bo_may_thuc_thi.py       # CCXT connection pool (singleton)
│   ├── mo_lenh.py / dong_lenh.py# Open / close orders
│   ├── quan_ly_lenh.py          # State management + PyQt6 signals
│   └── ket_noi_san/             # Binance / OKX / Bybit adapters
│
├── chuc_nang/                   # Runtime modes
│   ├── chay_realtime.py         # Live trading (2 threads)
│   ├── chay_demo.py             # Paper trading (slippage + fees)
│   └── vectorized_backtest.py   # Vectorized backtest engine
│
├── hien_thi/                    # PyQt6 Dashboards
│   ├── dashboard_realtime.py    # Live trading UI
│   ├── dashboard_demo.py        # Paper trading UI
│   ├── dashboard_backtest.py    # Backtest results + charts
│   └── dashboard_vectorized.py  # Advanced candlestick chart
│
└── utils/                       # Shared utilities
    ├── ham_tien_ich.py           # MTF merge, PnL calc, HTF candle
    ├── log.py                    # Dynamic time logger (backtest-aware)
    └── doc_cau_hinh.py          # Config loaders
```

---

## Cài Đặt & Chạy

```bash
# 1. Clone repo
git clone https://github.com/PVinh05-Quant/Kairos-Quant-System.git
cd Kairos-Quant-System

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Cấu hình API keys (không commit file này)
cp config/tai_khoan_api.json.example config/tai_khoan_api.json
# Điền API keys của bạn vào config/tai_khoan_api.json

# 4. Khởi động ứng dụng
python main.py
```

> **Lưu ý:** File `config/tai_khoan_api.json` chứa API keys thật — đã được thêm vào `.gitignore`. Chỉ dùng file `.example` làm template.

---

## Yêu Cầu Hệ Thống

```
Python      >= 3.11
PyTorch     >= 2.0
Polars      >= 0.19
Pandas      >= 2.0
PyQt6       >= 6.4
CCXT        >= 4.0
joblib      >= 1.3
ta          >= 0.10   (Technical Analysis library)
```

---

## Kết Quả Backtest (Demo)

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số lệnh | 559 |
| Win Rate | 31.8% |
| Profit Factor | 0.87 |
| Tỷ lệ R:R | 1.87 |
| Lợi nhuận TB/lệnh | +10.46$ |
| Thua lỗ TB/lệnh | -5.60$ |
| Thời gian test | Tháng 01/2026 |

> *Kết quả backtest không đảm bảo lợi nhuận trong tương lai. Dự án này được xây dựng cho mục đích nghiên cứu và học thuật.*

---

## Điểm Kỹ Thuật Nổi Bật (Portfolio Highlights)

- **End-to-end ML pipeline**: Từ raw OHLCV → feature engineering → PyTorch training → online inference → trade execution
- **No lookahead bias**: `build_htf_candle()` và tất cả MTF indicators đều được thiết kế tránh lookahead hoàn toàn
- **Production patterns**: Singleton connection pool, PyQt6 thread-safe signals, JSON state persistence, dynamic log timestamping cho backtest
- **Polars performance**: Xử lý dataset lớn (1m candles, 30 ngày+) bằng Polars lazy evaluation thay Pandas
- **Online reinforcement**: Bot tự ghi nhận reward sau mỗi lệnh và tái huấn luyện model theo kinh nghiệm thực chiến

---

## Tác Giả

**P. Vinh** — Quantitative Trading & Data Engineering  
Email: ppvinh1513@gmail.com  
GitHub: [PVinh05-Quant](https://github.com/PVinh05-Quant)

---

<div align="center">
<sub>Built with Python · PyTorch · Polars · PyQt6 · CCXT</sub>
</div>
