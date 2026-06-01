"""TÂM LÝ & VỊ THẾ (Sentiment / Positioning) – vectorized version ⭐⭐
👉 Đám đông đang nghiêng về đâu?
- CVD (Cumulative Volume Delta) – xấp xỉ từ OHLCV
- Buyer Pressure – tỷ lệ nến bullish rolling
- Volume Surge   – smart money confirmation
📌 Backtest không có WebSocket hay Funding Rate lịch sử → dùng proxy từ giá.
📌 Dùng để:
- Lọc tín hiệu ngược chiều dòng tiền
- Phát hiện divergence giá vs CVD
- Hệ thống pro level """

import numpy as np
import pandas as pd


def pt_vi_the(df, time_frame, window_cvd=50, window_pressure=20):
    """
    Thêm các cột sentiment/positioning xấp xỉ vào DataFrame.

    Tham số
    -------
    df         : Pandas DataFrame với cột OHLCV (open/high/low/close/volume).
    time_frame : str – nhãn khung (e.g. '1m', '1h') dùng để đặt tên cột.
    window_cvd      : chu kỳ EMA làm mượt CVD (phát hiện xu hướng dòng tiền).
    window_pressure : cửa sổ rolling đếm % nến bullish.

    Cột được thêm
    -------------
    vol_delta_{tf}       – (close-open)/(high-low) * volume   (áp lực mua/bán mỗi nến)
    cvd_{tf}             – Cumulative Volume Delta tích lũy từ đầu chuỗi
    cvd_ema_{tf}         – EMA của CVD (xu hướng dài hạn)
    cvd_bull_{tf}        – 1 nếu CVD đang trên EMA (dòng tiền xu hướng tăng)
    buyer_pressure_{tf}  – rolling % nến close > open
    vol_surge_{tf}       – vol hiện tại / vol trung bình 100 nến
    vi_the_signal_{tf}   – tín hiệu tổng hợp: 1 / -1 / 0

    Cách dùng trong chiến lược
    --------------------------
    df_1h = pt_vi_the(df_1h, '1h')
    # Lọc: chỉ lấy BUY khi sentiment 1h đồng thuận
    df['signal'] = np.where(
        (df['signal'] == 1) & (df['vi_the_signal_1h'] >= 0), 1,
        np.where((df['signal'] == -1) & (df['vi_the_signal_1h'] <= 0), -1, 0)
    )
    """
    tf = time_frame.lower().replace(' ', '')
    df = df.copy()

    # 1. Volume Delta – (close-open)/(high-low) * volume
    #    Chuẩn hóa theo range nến để nến pin-bar không bị bias
    body   = df['close'] - df['open']
    range_ = (df['high'] - df['low']).clip(lower=1e-9)
    vol_delta = (body / range_) * df['volume']
    df[f'vol_delta_{tf}'] = vol_delta

    # 2. CVD – Cumulative Volume Delta (running sum)
    cvd = vol_delta.cumsum()
    df[f'cvd_{tf}']     = cvd
    cvd_ema             = cvd.ewm(span=window_cvd, adjust=False).mean()
    df[f'cvd_ema_{tf}'] = cvd_ema
    df[f'cvd_bull_{tf}'] = (cvd > cvd_ema).astype(int)

    # 3. Buyer Pressure – rolling % nến bullish (close > open)
    is_bull = (df['close'] > df['open']).astype(float)
    df[f'buyer_pressure_{tf}'] = is_bull.rolling(
        window=window_pressure, min_periods=max(5, window_pressure // 4)
    ).mean()

    # 4. Volume Surge – nến có vol >> trung bình → smart money activity
    vol_mean = df['volume'].rolling(window=100, min_periods=10).mean()
    df[f'vol_surge_{tf}'] = df['volume'] / (vol_mean + 1e-9)

    # 5. Tín hiệu tổng hợp
    #    BUY khi:  CVD xu hướng tăng  VÀ buyer pressure > 55%
    #    SELL khi: CVD xu hướng giảm  VÀ buyer pressure < 45%
    #    Volume surge >= 1.5 xác nhận tín hiệu mạnh hơn (smart money)
    cvd_bull      = df[f'cvd_bull_{tf}']
    pressure      = df[f'buyer_pressure_{tf}']
    vol_surge_col = df[f'vol_surge_{tf}']

    strong_buy  = (cvd_bull == 1) & (pressure > 0.55) & (vol_surge_col >= 1.5)
    weak_buy    = (cvd_bull == 1) & (pressure > 0.55)
    strong_sell = (cvd_bull == 0) & (pressure < 0.45) & (vol_surge_col >= 1.5)
    weak_sell   = (cvd_bull == 0) & (pressure < 0.45)

    df[f'vi_the_signal_{tf}'] = np.select(
        [strong_buy, strong_sell, weak_buy, weak_sell],
        [1,          -1,          1,        -1],
        default=0
    )

    return df
