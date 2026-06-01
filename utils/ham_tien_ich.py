"""
utils/ham_tien_ich.py – Hàm tiện ích đa năng
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tập hợp các hàm được dùng ở nhiều tầng khác nhau:
  • Chuẩn hóa symbol giữa các sàn (BTC/USDT vs BTCUSDT vs BTC-USDT-SWAP)
  • Tính PnL theo side
  • Merge multi-timeframe data không lookahead
  • Build HTF candle live (5m/15m/1h/4h) từ dữ liệu 1m
"""
import pandas as pd
import numpy as np
import polars as pl

def chuyen_doi_symbol_chuan(san, symbol):
    """
    Chuẩn hóa symbol về dạng BASE/QUOTE (VD: 'BTCUSDT' -> 'BTC/USDT').
    Xử lý đặc biệt cho OKX (BTC-USDT-SWAP -> BTC/USDT).
    """

    if not symbol: return symbol
   
    symbol = symbol.upper().strip()
    san = san.lower().strip() if san else 'binance'

    if '/' in symbol:
        return symbol

    if san == 'okx':
        clean_symbol = symbol.replace('-SWAP', '').replace('-', '/')
        return clean_symbol

    quote_currencies = ['USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'BTC', 'ETH', 'BNB']
    
    for quote in quote_currencies:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}"
    return f"{symbol}/USDT"

def tinh_pnl(entry, current_price, side, amount):
    """Tính PnL tuyệt đối (USDT) và tương đối (%) cho cả long lẫn short."""
    if side == 'buy':
        pnl_usdt = (current_price - entry) * amount
        pnl_percent = (current_price - entry) / entry
    else:
        pnl_usdt = (entry - current_price) * amount
        pnl_percent = (entry - current_price) / entry
    return pnl_usdt, pnl_percent

def gop_va_dong_bo_data(dfs_dict):
    """
    Merge nhiều DataFrame khác khung thời gian thành 1 DataFrame gốc 1m.
    Dùng merge_asof (backward) để tránh lookahead – indicator HTF chỉ forward-fill.
    """
    def chuẩn_hóa_df(df, name):
        # 1. Xóa cột không tên (Unnamed) - Tránh rác từ file CSV cũ
        df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
        
        # 2. Xóa cột trống
        if "" in df.columns:
            df = df.drop(columns=[""])

        # Chuẩn hóa tên cột về chữ thường
        df.columns = [col.strip().lower() for col in df.columns]
        
        # 3. Đảm bảo timestamp là một cột để có thể merge_asof
        if 'timestamp' not in df.columns:
            if df.index.name and df.index.name.lower() == 'timestamp':
                df = df.reset_index()
            else:
                raise KeyError(f"Không tìm thấy cột 'timestamp' trong khung {name}.")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sắp xếp để merge_asof không lỗi
        return df.sort_values('timestamp').reset_index(drop=True)

    # Lấy khung 1m làm gốc
    if '1m' not in dfs_dict:
        return None
        
    main_df = chuẩn_hóa_df(dfs_dict['1m'].copy(), '1m')

    for tf, df in dfs_dict.items():
        if tf == '1m': continue
        
        df_tf = chuẩn_hóa_df(df.copy(), tf)
        
        # Chỉ lấy timestamp và các chỉ báo (không lấy OHLC khung lớn)
        cols_to_exclude = ['open', 'high', 'low', 'close', 'volume']
        cols_to_use = [col for col in df_tf.columns if col not in cols_to_exclude or col == 'timestamp']
        
        main_df = pd.merge_asof(
            main_df, 
            df_tf[cols_to_use],
            on='timestamp',
            direction='backward',
            suffixes=('', f'_{tf}') # Thêm hậu tố để phân biệt RSI_5m, RSI_15m...
        )
    
    # 4. BƯỚC QUAN TRỌNG NHẤT: Đưa timestamp làm Index
    # Việc này sẽ thay thế hoàn toàn cột số 0, 1, 2...
    main_df.set_index('timestamp', inplace=True)
    
    # Xóa các cột rác phát sinh (nếu có)
    if 'index' in main_df.columns:
        main_df.drop(columns=['index'], inplace=True)
        
    return main_df

def build_htf_candle(df, time_frame):
    """
    Build HTF candle chuẩn:
    - Không lookahead
    - Có closed + live
    - Fix warmup null
    - Hỗ trợ Pandas + Polars
    """

    # ===== Detect type =====
    is_polars = pl and isinstance(df, pl.DataFrame)

    # ===== Convert về pandas =====
    if is_polars:
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # ===== Chuẩn hóa timestamp =====
    if 'timestamp' in pdf.columns:
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        pdf = pdf.set_index('timestamp')

    tf_pd = time_frame.replace('m', 'min')

    # ===== 1. CLOSED CANDLE =====
    df_htf = pdf.resample(tf_pd, label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Shift để tránh lookahead
    df_htf.index = df_htf.index + pd.to_timedelta(tf_pd)

    # Align xuống LTF
    df_htf_ffill = df_htf.reindex(pdf.index, method='ffill')

    # ===== 2. LIVE CANDLE =====
    group = pdf.index.floor(tf_pd)

    live_open = pdf.groupby(group)['open'].transform('first')
    live_high = pdf.groupby(group)['high'].cummax()
    live_low = pdf.groupby(group)['low'].cummin()
    live_volume = pdf.groupby(group)['volume'].cumsum()

    # ===== 3. FIX WARMUP (NULL đầu) =====
    high_base = df_htf_ffill['high'].fillna(live_high)
    low_base = df_htf_ffill['low'].fillna(live_low)
    volume_base = df_htf_ffill['volume'].fillna(0)

    # ===== 4. COMBINE =====
    result = pd.DataFrame(index=pdf.index)

    result['open'] = live_open
    result['high'] = np.maximum(high_base, live_high)
    result['low'] = np.minimum(low_base, live_low)
    result['close'] = pdf['close']
    result['volume'] = volume_base + live_volume

    # ===== Reset index =====
    result = result.reset_index()

    # ===== Return đúng type =====
    if is_polars:
        return pl.from_pandas(result)

    return result