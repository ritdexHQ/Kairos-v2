"""BIẾN ĐỘNG (Volatility)
👉 Giá chạy mạnh hay yếu?
- ATR
- Bollinger Bands
- Keltner Channel
- Donchian Channel
📌 Dùng cho SL / TP / leverage """

import polars as pl

def pt_atr(df, window=14, mean_window=100):
    """Phân tích ATR bằng Polars Native Expression"""
    
    # 1. Tính True Range và ATR ngay trong DF
    df_result = df.with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ).ewm_mean(span=window).alias("atr_series")
    ]).with_columns([
        pl.col("atr_series").rolling_mean(window_size=mean_window).alias("atr_mean_series")
    ])

    # 2. Trích xuất giá trị dòng cuối cùng
    # Polars dùng .tail(1) và chuyển thành dict
    last_row = df_result.tail(1).to_dicts()[0]
    
    atr_now = last_row['atr_series']
    atr_mean = last_row['atr_mean_series']

    # Kiểm tra null (đề phòng dữ liệu quá ngắn)
    if atr_now is None or atr_mean is None:
        return None

    trang_thai = 'BIẾN_ĐỘNG_CAO' if atr_now > atr_mean else 'BIẾN_ĐỘNG_THẤP'

    return {
        'atr_val': atr_now,
        'atr_mean': atr_mean,
        'trang_thai': trang_thai,
        'muc_do': 'CAO' if atr_now > atr_mean else 'THAP'
    }

def pt_bollinger_squeeze(df, window=20, window_dev=2):
    
    # 1. Tính toán các đường BB ngay trong DataFrame
    df_bb = df.with_columns([
        pl.col("close").rolling_mean(window_size=window).alias("mid_band"),
        pl.col("close").rolling_std(window_size=window).alias("std_dev")
    ]).with_columns([
        (pl.col("mid_band") + (window_dev * pl.col("std_dev"))).alias("upper_band"),
        (pl.col("mid_band") - (window_dev * pl.col("std_dev"))).alias("lower_band")
    ]).with_columns([
        # Tính Bandwidth: (High - Low) / Mid
        ((pl.col("upper_band") - pl.col("lower_band")) / (pl.col("mid_band") + 1e-9)).alias("w_band")
    ]).with_columns([
        # Tính trung bình Bandwidth để so sánh độ nén
        pl.col("w_band").rolling_mean(window_size=50).alias("w_band_mean")
    ])

    # 2. Trích xuất dòng cuối cùng
    last = df_bb.tail(1).to_dicts()[0]
    
    w_band_val = last['w_band']
    w_band_mean = last['w_band_mean']

    # 3. Logic phân loại
    trang_thai = 'BOP' if w_band_val < w_band_mean * 0.8 else 'MO_RONG'
    muc_do = 'CHAT' if w_band_val < w_band_mean * 0.6 else 'THUONG'

    return {
        'upper_band': last['upper_band'],
        'lower_band': last['lower_band'],
        'mid_band': last['mid_band'],
        'bandwidth': w_band_val,
        'trang_thai': trang_thai,
        'muc_do': muc_do
    }