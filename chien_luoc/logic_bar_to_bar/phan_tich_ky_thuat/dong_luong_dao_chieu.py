""" ĐẢO CHIỀU / ĐỘNG LƯỢNG (Momentum / Reversal)
👉 Lực đang yếu đi hay mạnh lên?
- RSI
- Stochastic
- CCI
- Williams %R
- ROC
📌 Dùng để timing entry / exit """

import polars as pl

def pt_rsi(df, window=14):
    # 1. Tính toán RSI trực tiếp (Công thức tối ưu hóa cho Polars)
    df_rsi = df.with_columns([
        (pl.col("close").diff().alias("diff"))
    ]).with_columns([
        (
            100 - (100 / (1 + (
                pl.when(pl.col("diff") >= 0).then(pl.col("diff")).otherwise(0).ewm_mean(span=window) /
                pl.when(pl.col("diff") < 0).then(pl.col("diff").abs()).otherwise(1e-9).ewm_mean(span=window)
            )))
        ).alias("rsi_series")
    ])

    # 2. Trích xuất giá trị cuối cùng
    last_row = df_rsi.tail(1).to_dicts()[0]
    rsi_val = last_row['rsi_series']

    # Kiểm tra giá trị null nếu dữ liệu quá ngắn
    if rsi_val is None:
        return {
            'rsi_val': 50.0,
            'trang_thai': 'TRUNG_TÍNH',
            'muc_do': 'THƯỜNG'
        }

    # 3. Phân loại trạng thái
    if rsi_val > 60: 
        trang_thai = 'MẠNH'
    elif rsi_val < 40: 
        trang_thai = 'YẾU'
    else: 
        trang_thai = 'TRUNG_TÍNH'
    
    # Phân loại mức độ quá mua/quá bán
    if rsi_val > 70:
        muc_do = 'QUÁ_MUA'
    elif rsi_val < 30:
        muc_do = 'QUÁ_BÁN'
    else:
        muc_do = 'THƯỜNG'

    return {
        'rsi_val': rsi_val,
        'trang_thai': trang_thai,
        'muc_do': muc_do
    }