""" XU HƯỚNG (Trend)
👉 Giá đang đi hướng nào?
- EMA, SMA
- MACD
- ADX
- Ichimoku
- SuperTrend
📌 Dùng để chọn phe BUY / SELL """

import polars as pl

def pt_ema_trend(df, window=20):
    """Phân tích xu hướng dựa trên EMA bằng Polars: Hiệu suất cao"""
    
    # 1. Tính toán EMA trực tiếp trong DataFrame
    # Polars sử dụng ewm_mean (Exponential Weighted Moving Average)
    df_ema = df.select([
        pl.col("close"),
        pl.col("close").ewm_mean(span=window).alias("ema_val")
    ])

    # 2. Trích xuất dòng cuối cùng
    last = df_ema.tail(1).to_dicts()[0]
    
    price_now = last['close']
    ema_val = last['ema_val']

    # Kiểm tra None (đề phòng dữ liệu quá ngắn)
    if ema_val is None:
        return {
            'ema_val': price_now,
            'price_now': price_now,
            'trang_thai': 'KHÔNG_XÁC_ĐỊNH',
            'muc_do': 'YẾU'
        }

    # 3. Logic phân loại trạng thái
    trang_thai = 'TĂNG' if price_now > ema_val else 'GIẢM'
    
    # Tính khoảng cách tương đối giữa giá và EMA
    diff_ratio = abs(price_now - ema_val) / (ema_val + 1e-9)
    muc_do = 'TỐT' if diff_ratio > 0.005 else 'YẾU'
    
    return {
        'ema_val': ema_val,
        'price_now': price_now,
        'trang_thai': trang_thai,
        'muc_do': muc_do
    }

def pt_adx(df, window=28):
    """Phân tích lực xu hướng qua ADX bằng Polars Native"""
    # FIX: Kiểm tra rỗng kiểu Polars (height thay cho len)
    if df is None or df.height < window * 2: # ADX cần nhiều dữ liệu hơn cửa sổ để làm mượt
        return {'adx_val': 0, 'trang_thai': 'SIDEWAY', 'muc_do': 'THAP'}

    try:
        # 1. Tính toán True Range (TR) và Directional Movement (DM)
        df_calc = df.with_columns([
            (pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ).alias("tr")),
            (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
            (pl.col("low").shift(1) - pl.col("low")).alias("down_move")
        ])

        # 2. Tính toán +DM, -DM và làm mượt bằng EWM (tương đương Wilder's)
        df_dm = df_calc.with_columns([
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
              .then(pl.col("up_move")).otherwise(0).alias("plus_dm"),
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
              .then(pl.col("down_move")).otherwise(0).alias("minus_dm")
        ]).with_columns([
            pl.col("tr").ewm_mean(span=window, adjust=False).alias("atr_smooth"),
            pl.col("plus_dm").ewm_mean(span=window, adjust=False).alias("plus_di_smooth"),
            pl.col("minus_dm").ewm_mean(span=window, adjust=False).alias("minus_di_smooth")
        ])

        # 3. Tính DI+ , DI- và DX
        df_dx = df_dm.with_columns([
            (100 * pl.col("plus_di_smooth") / (pl.col("atr_smooth") + 1e-9)).alias("plus_di"),
            (100 * pl.col("minus_di_smooth") / (pl.col("atr_smooth") + 1e-9)).alias("minus_di")
        ]).with_columns([
            (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / 
             (pl.col("plus_di") + pl.col("minus_di") + 1e-9)).alias("dx")
        ])

        # 4. Cuối cùng tính ADX bằng cách làm mượt DX
        df_adx = df_dx.with_columns([
            pl.col("dx").ewm_mean(span=window, adjust=False).alias("adx")
        ])

        # FIX: Trích xuất giá trị cuối cùng an toàn
        last_row = df_adx.tail(1).to_dicts()[0]
        adx_val = last_row['adx']

        if adx_val is None: return {'adx_val': 0, 'trang_thai': 'SIDEWAY', 'muc_do': 'THAP'}

        trang_thai = 'CO_XU_HUONG' if adx_val > 25 else 'SIDEWAY'
        return {
            'adx_val': adx_val,
            'trang_thai': trang_thai,
            'muc_do': 'MANH' if adx_val > 40 else 'TRUNG_BINH'
        }
    except Exception:
        return {'adx_val': 0, 'trang_thai': 'SIDEWAY', 'muc_do': 'THAP'}
