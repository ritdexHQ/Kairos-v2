"""
chien_luoc/logic_vectorized/chien_luoc_trang_thai_thi_truong.py – Bộ lọc thị trường vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thêm cột `trade_allowed` (bool) vào DataFrame. Backtest chỉ thực thi lệnh khi True.
Các điều kiện lọc:
  1. Giờ giao dịch: loại 5h sáng VN (giãn spread futures)
  2. Ngày: có thể bật lọc cuối tuần nếu cần
  3. Volume tối thiểu: loại nến có volume cực thấp (thị trường chết)
"""
import pandas as pd


def loc_trang_thai_thi_truong(df, loc_cuoi_tuan=False, loc_gio_spread=True):
    """
    df phải có cột `timestamp`. Trả về df với cột `trade_allowed` (bool).
    """
    df = df.copy()

    if 'timestamp' not in df.columns:
        df['trade_allowed'] = True
        return df

    ts = pd.to_datetime(df['timestamp'])
    trade_allowed = pd.Series(True, index=df.index)

    # Loại giờ 5h sáng – spread giãn mạnh trên futures Binance
    if loc_gio_spread:
        trade_allowed &= (ts.dt.hour != 5)

    # Loại cuối tuần (crypto không cần, bật nếu muốn)
    if loc_cuoi_tuan:
        trade_allowed &= ts.dt.dayofweek.between(0, 4)

    # Loại nến volume quá thấp (< 5% trung bình 200 nến)
    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(window=200, min_periods=10).mean()
        trade_allowed &= df['volume'] > (vol_mean * 0.05)

    df['trade_allowed'] = trade_allowed
    return df
