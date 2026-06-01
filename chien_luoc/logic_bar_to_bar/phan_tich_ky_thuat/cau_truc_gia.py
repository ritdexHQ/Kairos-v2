"""CẤU TRÚC GIÁ (Market Structure / Price Action) ⭐
👉 Thị trường đang ở pha nào?
- Không phải indicator cổ điển, mà là logic
- Higher High / Higher Low
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Support / Resistance
- Supply / Demand
📌 Dùng để:
- Xác định trend thật
- Tránh nhiễu indicator
- Bot chuyên nghiệp luôn có nhóm này"""

import polars as pl

def pt_breakout(df, window=20):
    """Phân tích Price Action bằng Polars: Hiệu suất cao"""
      
    df_calc = df.select([
        pl.col("close"),
        pl.col("high").shift(1).rolling_max(window_size=window).alias("high_max"),
        pl.col("low").shift(1).rolling_min(window_size=window).alias("low_min")
    ])

    # 2. Lấy giá trị của dòng cuối cùng (last row)
    last = df_calc.tail(1).to_dicts()[0]
    
    close_now = last['close']
    high_max = last['high_max']
    low_min = last['low_min']

    # 3. Logic xác định trạng thái
    # Kiểm tra None để tránh lỗi khi dữ liệu chưa đủ độ dài window
    if high_max is None or low_min is None:
        trang_thai = 'KHONG'
    elif close_now > high_max:
        trang_thai = 'BREAK_OUT'
    elif close_now < low_min:
        trang_thai = 'BREAK_DOWN'
    else:
        trang_thai = 'KHONG'

    return {
        'close': close_now,
        'high_max': high_max,
        'low_min': low_min,
        'trang_thai': trang_thai
    }