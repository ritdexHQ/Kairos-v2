""" KHỐI LƯỢNG (Volume / Participation)
👉 Có tiền thật vào không?
- Volume
- Volume MA
- OBV
- VWAP
- Volume Profile
📌 Dùng để xác nhận tín hiệu """

import polars as pl

def pt_volume(df, window=20):
    # 1. Tính toán Volume trung bình (Rolling Mean)
    df_vol = df.select([
        pl.col("volume"),
        pl.col("volume").rolling_mean(window_size=window).alias("vol_mean")
    ])

    # 2. Trích xuất dòng cuối cùng
    last = df_vol.tail(1).to_dicts()[0]
    
    vol_now = last['volume']
    vol_mean = last['vol_mean']

    # Kiểm tra None nếu dữ liệu chưa đủ độ dài cửa sổ (window)
    if vol_mean is None or vol_mean == 0:
        return {
            'vol_now': vol_now,
            'vol_mean': 0,
            'trang_thai': 'KHONG_XAC_DINH'
        }

    # 3. Logic phân loại trạng thái
    if vol_now > vol_mean * 2:
        trang_thai = 'DOT_BIEN'
    elif vol_now > vol_mean:
        trang_thai = 'TANG'
    else:
        trang_thai = 'THAP'
    
    return {
        'vol_now': vol_now,
        'vol_mean': vol_mean,
        'trang_thai': trang_thai
    }