""" ĐẢO CHIỀU / ĐỘNG LƯỢNG (Momentum / Reversal)
👉 Lực đang yếu đi hay mạnh lên?
- RSI
- Stochastic
- CCI
- Williams %R
- ROC
📌 Dùng để timing entry / exit """

from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np
import re

def pt_rsi(df, time_frame, window=14):
    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    df = df.copy()
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    col_name = f'rsi_{time_frame}'

    if time_frame == '1m':
        rsi_ind = RSIIndicator(df['close'], window=window)
        df[col_name] = rsi_ind.rsi()
    else:
        # Sử dụng pd_time_frame ('3min') thay vì time_frame ('3m')
        df_closed = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Dịch chuyển mốc thời gian sang tương lai 1 nến
        df_closed.index = df_closed.index + pd.to_timedelta(pd_time_frame)
        
        # Bước B: Tính Gain/Loss lịch sử
        delta = df_closed.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Seed giá trị đầu tiên bằng SMA
        sma_gain = gain.rolling(window=window, min_periods=window).mean()
        sma_loss = loss.rolling(window=window, min_periods=window).mean()
        
        gain.iloc[:window] = sma_gain.iloc[:window]
        loss.iloc[:window] = sma_loss.iloc[:window]
        
        # Tính Trung bình theo công thức Wilder's Smoothing
        avg_gain_closed = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss_closed = loss.ewm(alpha=1/window, adjust=False).mean()
        
        # Bước C: Rải "Mỏ neo" lên biểu đồ 1 phút (Forward fill)
        avg_gain_ffill = avg_gain_closed.reindex(df.index, method='ffill')
        avg_loss_ffill = avg_loss_closed.reindex(df.index, method='ffill')
        close_ffill = df_closed.reindex(df.index, method='ffill')
        
        # Bước D: Tính biến động LIVE
        change_live = df['close'] - close_ffill
        gain_live = np.where(change_live > 0, change_live, 0.0)
        loss_live = np.where(change_live < 0, -change_live, 0.0)
        
        # Bước E: Cập nhật RSI Live
        avg_gain_live = (avg_gain_ffill * (window - 1) + gain_live) / window
        avg_loss_live = (avg_loss_ffill * (window - 1) + loss_live) / window
        
        # Tránh chia cho 0
        rs_live = avg_gain_live / np.where(avg_loss_live == 0, 1e-9, avg_loss_live)
        rsi_live = np.where(avg_loss_live == 0, 100, 100 - (100 / (1 + rs_live)))
        
        df[col_name] = rsi_live

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_stochastic(df, time_frame, k_window=14, d_window=3):
    """
    Phân tích Stochastic Oscillator dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Cung cấp %K (Nhanh) và %D (Chậm) cập nhật theo thời gian thực (Live).
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số
    def parse_window(w, default):
        if isinstance(w, str):
            num_str = re.sub(r'\D', '', w)
            return int(num_str) if num_str else default
        return int(w)
        
    k_window = parse_window(k_window, 14)
    d_window = parse_window(d_window, 3)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_k = f'stoch_k_{time_frame}'
    c_d = f'stoch_d_{time_frame}'

    # 2. Tính toán Logic MTF Stochastic
    if time_frame == '1m':
        # --- Khung 1 phút (Tính trực tiếp) ---
        low_min = df['low'].rolling(window=k_window).min()
        high_max = df['high'].rolling(window=k_window).max()
        
        # Mẫu số: Đỉnh cao nhất - Đáy thấp nhất (Tránh chia cho 0)
        denominator = np.where((high_max - low_min) == 0, 1e-9, high_max - low_min)
        
        df[c_k] = 100 * (df['close'] - low_min) / denominator
        df[c_d] = df[c_k].rolling(window=d_window).mean()

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính toán quá khứ để làm mỏ neo
        # 1. Đỉnh/Đáy của (K-1) nến đóng (để so sánh với nến Live)
        htf_max_k1 = htf_high.rolling(window=k_window - 1).max()
        htf_min_k1 = htf_low.rolling(window=k_window - 1).min()
        
        # 2. Tính %K quá khứ
        htf_max_k = htf_high.rolling(window=k_window).max()
        htf_min_k = htf_low.rolling(window=k_window).min()
        htf_denom = np.where((htf_max_k - htf_min_k) == 0, 1e-9, htf_max_k - htf_min_k)
        htf_k_closed = 100 * (htf_close - htf_min_k) / htf_denom
        
        # 3. Tính Tổng %K của (D-1) nến đóng (để tính Live SMA cho %D)
        htf_k_sum_d1 = htf_k_closed.rolling(window=d_window - 1).sum()

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        max_k1_ffill = htf_max_k1.reindex(df.index, method='ffill')
        min_k1_ffill = htf_min_k1.reindex(df.index, method='ffill')
        k_sum_d1_ffill = htf_k_sum_d1.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng nến Live
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        live_close = df['close']

        # BƯỚC E: Tính Live Đỉnh/Đáy của chu kỳ K
        live_max_k = pd.concat([max_k1_ffill, live_high], axis=1).max(axis=1)
        live_min_k = pd.concat([min_k1_ffill, live_low], axis=1).min(axis=1)

        # BƯỚC F: Tính %K Live và %D Live
        live_denom = np.where((live_max_k - live_min_k) == 0, 1e-9, live_max_k - live_min_k)
        live_k = 100 * (live_close - live_min_k) / live_denom
        
        live_d = (k_sum_d1_ffill + live_k) / d_window

        df[c_k] = live_k
        df[c_d] = live_d

    # 3. Phân loại trạng thái (Signals)
    
    # Giao cắt (Crossover): %K cắt lên %D (Tín hiệu mua)
    df[f'stoch_bull_cross_{time_frame}'] = df[c_k] > df[c_d]
    
    # Quá Mua / Quá Bán (Overbought / Oversold)
    # Thường dùng mốc 80 - 20 cho Stochastic
    df[f'stoch_overbought_{time_frame}'] = df[c_k] > 80
    df[f'stoch_oversold_{time_frame}'] = df[c_k] < 20
    
    # Tín hiệu Mua cực mạnh: Giá cắt lên %D VÀ đang nằm trong vùng Quá Bán (<20)
    df[f'stoch_strong_buy_{time_frame}'] = (df[c_k] > df[c_d]) & (df[c_k] < 20)
    df[f'stoch_strong_sell_{time_frame}'] = (df[c_k] < df[c_d]) & (df[c_k] > 80)

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_cci(df, time_frame, window=20, constant=0.015):
    """
    Phân tích Commodity Channel Index (CCI) dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Sử dụng kỹ thuật Streaming Approximation để tính MAD Live siêu tốc.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 20
    else:
        window = int(window)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_cci = f'cci_{time_frame}'

    # Hàm phụ trợ tính Mean Absolute Deviation (MAD) dùng numpy cho tốc độ cao
    def calculate_mad(x):
        return np.mean(np.abs(x - np.mean(x)))

    # 2. Tính toán Logic MTF CCI
    if time_frame == '1m':
        # --- Khung 1 phút (Tính trực tiếp) ---
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=window).mean()
        
        # Dùng rolling.apply với raw=True để tăng tốc độ xử lý mảng numpy
        mad = tp.rolling(window=window).apply(calculate_mad, raw=True)
        
        # Tránh lỗi chia cho 0
        mad = np.where(mad == 0, 1e-9, mad)
        df[c_cci] = (tp - sma_tp) / (constant * mad)

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính toán Typical Price (TP) và MAD trên nến HTF quá khứ
        htf_tp_closed = (htf_high + htf_low + htf_close) / 3
        htf_sma_tp_closed = htf_tp_closed.rolling(window=window).mean()
        htf_mad_closed = htf_tp_closed.rolling(window=window).apply(calculate_mad, raw=True)

        # Cần Tổng TP của (N-1) nến trước để tính Live SMA
        htf_tp_sum_n1 = htf_tp_closed.rolling(window=window - 1).sum()

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        tp_sum_n1_ffill = htf_tp_sum_n1.reindex(df.index, method='ffill')
        mad_ffill = htf_mad_closed.reindex(df.index, method='ffill').fillna(1e-9)

        # BƯỚC D: Dựng nến Live và TP Live
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        live_close = df['close']
        
        live_tp = (live_high + live_low + live_close) / 3

        # BƯỚC E: Tính Live SMA và Streaming Live MAD
        live_sma_tp = (tp_sum_n1_ffill + live_tp) / window
        
        # Streaming MAD Approximation: (N-1)*MAD_closed + Độ lệch nến Live hiện tại
        live_mad = (mad_ffill * (window - 1) + np.abs(live_tp - live_sma_tp)) / window
        live_mad = np.where(live_mad == 0, 1e-9, live_mad)

        # BƯỚC F: Tính CCI Live
        live_cci = (live_tp - live_sma_tp) / (constant * live_mad)
        
        df[c_cci] = live_cci

    # 3. Phân loại trạng thái tín hiệu (Signals)
    
    # Kênh xu hướng truyền thống (Cắt qua mức 0)
    df[f'cci_bull_trend_{time_frame}'] = df[c_cci] > 0
    
    # Vùng Cực đoan (+100 và -100)
    # Khác với RSI (quá mua là bán), trong CCI, vượt +100 lại thường là dấu hiệu của BÙNG NỔ XU HƯỚNG MẠNH
    df[f'cci_break_up_{time_frame}'] = df[c_cci] > 100
    df[f'cci_break_down_{time_frame}'] = df[c_cci] < -100
    
    # Vùng Kiệt sức (+200 và -200) - Thường dùng để chốt lời hoặc đánh Mean Reversion (Bắt đỉnh/đáy)
    df[f'cci_exhausted_up_{time_frame}'] = df[c_cci] > 200
    df[f'cci_exhausted_down_{time_frame}'] = df[c_cci] < -200

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_williams_r(df, time_frame, window=14):
    """
    Phân tích Williams %R dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Phản ứng siêu nhạy với các điểm đảo chiều chớp nhoáng (Spike/Rejection).
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 14
    else:
        window = int(window)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_wr = f'williams_r_{time_frame}'

    # 2. Tính toán Logic MTF Williams %R
    if time_frame == '1m':
        # --- Khung 1 phút (Tính trực tiếp) ---
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        
        # Mẫu số: Đỉnh cao nhất - Đáy thấp nhất (Tránh chia cho 0)
        denominator = np.where((high_max - low_min) == 0, 1e-9, high_max - low_min)
        
        # Công thức Williams %R (Từ 0 đến -100)
        df[c_wr] = (high_max - df['close']) / denominator * -100

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        
        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tìm Đỉnh/Đáy của (N-1) nến đóng (Mỏ neo tĩnh)
        htf_max_n1 = htf_high.rolling(window=window - 1).max()
        htf_min_n1 = htf_low.rolling(window=window - 1).min()

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        max_n1_ffill = htf_max_n1.reindex(df.index, method='ffill')
        min_n1_ffill = htf_min_n1.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng nến Live
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        live_close = df['close']

        # BƯỚC E: Tính Live Đỉnh/Đáy của chu kỳ
        live_max = pd.concat([max_n1_ffill, live_high], axis=1).max(axis=1)
        live_min = pd.concat([min_n1_ffill, live_low], axis=1).min(axis=1)

        # BƯỚC F: Tính Live Williams %R
        live_denom = np.where((live_max - live_min) == 0, 1e-9, live_max - live_min)
        live_wr = (live_max - live_close) / live_denom * -100
        
        df[c_wr] = live_wr

    # 3. Phân loại trạng thái (Signals)
    
    # Mức Quá Mua (0 đến -20) và Quá Bán (-80 đến -100)
    df[f'wr_overbought_{time_frame}'] = df[c_wr] >= -20
    df[f'wr_oversold_{time_frame}'] = df[c_wr] <= -80
    
    # Động lượng bứt phá (Momentum Thrust)
    # Khác với tư duy "quá mua thì bán", nếu %R ghim chặt ở mức 0 đến -20 trong thời gian dài
    # Nó chứng tỏ phe Mua đang áp đảo hoàn toàn (Giá liên tục đóng cửa ở sát đỉnh)
    wr_roc = df[c_wr].diff(3) # Đo độ dốc của %R trong 3 nến
    df[f'wr_bull_thrust_{time_frame}'] = (df[c_wr] >= -20) & (wr_roc > 0)
    df[f'wr_bear_thrust_{time_frame}'] = (df[c_wr] <= -80) & (wr_roc < 0)

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_roc(df, time_frame, window=9):
    """
    Phân tích Rate of Change (ROC) dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Đo lường gia tốc giá (Momentum) theo thời gian thực (Live) mà không có độ trễ.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 9
    else:
        window = int(window)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_roc = f'roc_{time_frame}'

    # 2. Tính toán Logic MTF ROC
    if time_frame == '1m':
        # --- Khung 1 phút (Tính trực tiếp bằng hàm pct_change của Pandas) ---
        df[c_roc] = df['close'].pct_change(periods=window) * 100

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Dịch mốc thời gian sang tương lai 1 nến để tránh Look-ahead bias
        # Lúc này, htf_close tại nến Live T chính là giá đóng cửa của nến T-1
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tìm mốc giá tham chiếu trong quá khứ
        # Để so sánh với nến T-window, ta cần shift htf_close thêm (window - 1) nến
        ref_close_closed = htf_close.shift(window - 1)

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        ref_close_ffill = ref_close_closed.reindex(df.index, method='ffill')

        # BƯỚC D: Tính Live ROC
        # Tránh lỗi chia cho 0 trong những trường hợp dữ liệu dị biệt
        safe_ref_close = np.where(ref_close_ffill == 0, 1e-9, ref_close_ffill)
        
        # So sánh giá 1m Live với mốc tĩnh HTF trong quá khứ
        live_roc = (df['close'] - safe_ref_close) / safe_ref_close * 100
        
        df[c_roc] = live_roc

    # 3. Phân loại trạng thái tín hiệu (Signals)
    
    # Xu hướng của Động lượng (Cắt qua đường 0)
    df[f'roc_bullish_{time_frame}'] = df[c_roc] > 0
    df[f'roc_bearish_{time_frame}'] = df[c_roc] < 0
    
    # Gia tốc của Động lượng (Momentum Acceleration)
    # Rất hữu ích để đánh Breakout: ROC phải LỚN HƠN 0 VÀ ĐANG HƯỚNG LÊN
    roc_roc = df[c_roc].diff(3) # Đo độ dốc của ROC trong 3 nến 1m gần nhất
    df[f'roc_accelerating_up_{time_frame}'] = (df[c_roc] > 0) & (roc_roc > 0)
    df[f'roc_accelerating_down_{time_frame}'] = (df[c_roc] < 0) & (roc_roc < 0)

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df