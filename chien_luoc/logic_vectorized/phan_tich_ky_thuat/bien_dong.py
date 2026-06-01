"""BIẾN ĐỘNG (Volatility)
👉 Giá chạy mạnh hay yếu?
- ATR
- Bollinger Bands
- Keltner Channel
- Donchian Channel
📌 Dùng cho SL / TP / leverage """

import numpy as np
import pandas as pd
import re

def pt_atr(df, time_frame, window=14, mean_window=100):
    """
    Phân tích ATR: Mức độ biến động thị trường (Vectorized - MTF).
    Đã chuẩn hóa theo Wilder's Smoothing (RMA) cho ATR và SMA cho ATR_Mean.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window và mean_window
    def parse_window(w, default):
        if isinstance(w, str):
            num_str = re.sub(r'\D', '', w)
            return int(num_str) if num_str else default
        return int(w)
        
    window = parse_window(window, 14)
    mean_window = parse_window(mean_window, 100)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    col_atr = f'atr_{time_frame}'
    col_atr_mean = f'atr_mean_{time_frame}'

    # 2. Tính toán Logic MTF ATR
    if time_frame == '1m':
        # TR Khung 1 phút
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr = pd.concat([
            high - low, 
            (high - close_prev).abs(), 
            (low - close_prev).abs()
        ], axis=1).max(axis=1)

        # ATR chuẩn TradingView (Wilder's RMA)
        df[col_atr] = tr.ewm(alpha=1/window, adjust=False).mean()
        df[col_atr_mean] = df[col_atr].rolling(window=mean_window).mean()
        
    else:
        # BƯỚC A: Dựng nến HTF đã đóng
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Shift index sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính True Range (TR) trên nến đã đóng
        prev_close = htf_close.shift(1)
        tr1 = htf_high - htf_low
        tr2 = (htf_high - prev_close).abs()
        tr3 = (htf_low - prev_close).abs()
        tr_closed = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR quá khứ (Wilder's Smoothing)
        atr_closed = tr_closed.ewm(alpha=1/window, adjust=False).mean()
        
        # Để tính SMA Live cho atr_mean (100 nến), ta cần tổng của 99 nến ATR quá khứ
        atr_sum_99_closed = atr_closed.rolling(window=mean_window - 1).sum()

        # BƯỚC C: Rải mỏ neo (Forward Fill) về khung 1 phút
        atr_ffill = atr_closed.reindex(df.index, method='ffill')
        atr_sum_99_ffill = atr_sum_99_closed.reindex(df.index, method='ffill')
        close_closed_ffill = htf_close.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng nến LIVE HTF và Tính TR Live
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()

        tr1_live = live_high - live_low
        tr2_live = (live_high - close_closed_ffill).abs()
        tr3_live = (live_low - close_closed_ffill).abs()
        tr_live = pd.concat([tr1_live, tr2_live, tr3_live], axis=1).max(axis=1)

        # BƯỚC E: Tính ATR Live và ATR Mean Live
        # 1. Cập nhật ATR Live bằng Wilder's math
        atr_live = (atr_ffill * (window - 1) + tr_live) / window
        
        # 2. Cập nhật ATR Mean Live (Toán học của đường SMA)
        # SMA = (Tổng 99 nến quá khứ + Giá trị Live hiện tại) / 100
        atr_mean_live = (atr_sum_99_ffill + atr_live) / mean_window

        df[col_atr] = atr_live
        df[col_atr_mean] = atr_mean_live

    # 3. Phân loại biến động (Vectorized)
    # Tối ưu hóa: Dùng trực tiếp np.where thay vì np.select cho 1 điều kiện
    df[f'atr_status_{time_frame}'] = np.where(
        df[col_atr] > df[col_atr_mean], 
        'BIEN_DONG_CAO', 
        'BIEN_DONG_THAP'
    )
    
    df[f'atr_muc_do_{time_frame}'] = np.where(
        df[col_atr] > df[col_atr_mean], 
        'CAO', 
        'THAP'
    )

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_bollinger_squeeze(df, time_frame, window=20, window_dev=2):
    """
    Phân tích Bollinger Bands & Squeeze dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Sử dụng thuật toán tổng bình phương để tính Live Standard Deviation chính xác tuyệt đối.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    def parse_number(val, default):
        if isinstance(val, str):
            num_str = re.sub(r'[^\d.]', '', val) # Cho phép dấu thập phân nếu là float
            return float(num_str) if '.' in num_str else int(num_str) if num_str else default
        return val

    window = int(parse_number(window, 20))
    window_dev = float(parse_number(window_dev, 2))
    
    bw_mean_window = 50 # Mặc định cho đường trung bình Bandwidth

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    # Khởi tạo tên cột
    c_upper = f'bb_upper_{time_frame}'
    c_lower = f'bb_lower_{time_frame}'
    c_mid = f'bb_mid_{time_frame}'
    c_bw = f'bb_width_{time_frame}'

    # 2. Tính toán Logic MTF Bollinger Bands
    if time_frame == '1m':
        mid_band = df['close'].rolling(window=window).mean()
        std_dev = df['close'].rolling(window=window).std()
        
        upper_band = mid_band + (window_dev * std_dev)
        lower_band = mid_band - (window_dev * std_dev)
        bandwidth = (upper_band - lower_band) / mid_band
        bandwidth_mean = bandwidth.rolling(window=bw_mean_window).mean()
        
        df[c_upper] = upper_band
        df[c_lower] = lower_band
        df[c_mid] = mid_band
        df[c_bw] = bandwidth
        
    else:
        # BƯỚC A: Lấy nến đóng cửa của khung thời gian lớn
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính toán quá khứ để làm "mỏ neo" cho tính Live
        # 1. Tổng của N-1 nến quá khứ (để tính Live SMA)
        sum_close_closed = htf_close.rolling(window=window - 1).sum()
        
        # 2. Tổng bình phương của N-1 nến quá khứ (để tính Live Std Dev)
        sum_close_sq_closed = (htf_close ** 2).rolling(window=window - 1).sum()
        
        # 3. Tính Bandwidth quá khứ để lấy tổng 49 nến cho Bandwidth Mean Live
        mid_closed = htf_close.rolling(window=window).mean()
        std_closed = htf_close.rolling(window=window).std()
        bw_closed = (2 * window_dev * std_closed) / mid_closed
        sum_bw_49_closed = bw_closed.rolling(window=bw_mean_window - 1).sum()

        # BƯỚC C: Rải mỏ neo lên biểu đồ 1 phút
        sum_close_ffill = sum_close_closed.reindex(df.index, method='ffill')
        sum_close_sq_ffill = sum_close_sq_closed.reindex(df.index, method='ffill')
        sum_bw_49_ffill = sum_bw_49_closed.reindex(df.index, method='ffill')
        
        # BƯỚC D: Tính LIVE Bollinger Bands
        live_close = df['close']
        
        # 1. Live SMA (Mid Band)
        live_sum = sum_close_ffill + live_close
        live_mid = live_sum / window
        
        # 2. Live Variance & Standard Deviation
        live_sum_sq = sum_close_sq_ffill + (live_close ** 2)
        # Phương sai = (Tổng bình phương - (Tổng)^2 / N) / (N - 1)
        live_variance = (live_sum_sq - (live_sum ** 2) / window) / (window - 1)
        live_variance = np.maximum(live_variance, 0) # Ép >= 0 để tránh lỗi vi phân số thực
        live_std = np.sqrt(live_variance)
        
        # 3. Live Bands & Bandwidth
        live_upper = live_mid + (window_dev * live_std)
        live_lower = live_mid - (window_dev * live_std)
        live_bw = (live_upper - live_lower) / live_mid
        
        # 4. Live Bandwidth Mean (Tổng 49 nến cũ + 1 nến live) / 50
        bandwidth_mean = (sum_bw_49_ffill + live_bw) / bw_mean_window

        df[c_upper] = live_upper
        df[c_lower] = live_lower
        df[c_mid] = live_mid
        df[c_bw] = live_bw

    # 3. Phân loại trạng thái Nén (Squeeze)
    cond_squeeze = df[c_bw] < (bandwidth_mean * 0.8)
    cond_tight = df[c_bw] < (bandwidth_mean * 0.6)

    # Dùng np.where lồng nhau (hoặc np.select) để phân loại mức độ
    df[f'bb_status_{time_frame}'] = np.where(cond_squeeze, 'BOP', 'MO_RONG')
    df[f'bb_muc_do_{time_frame}'] = np.where(
        cond_tight, 'CHAT', 
        np.where(cond_squeeze, 'THUONG', 'KHONG') # Khởi tạo 'KHONG' nếu đang mở rộng
    )

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_keltner_channel(df, time_frame, window=20, atr_window=10, multiplier=2.0):
    """
    Phân tích Keltner Channel dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Kết hợp Live EMA (Đường giữa) và Live ATR (Độ rộng dải) để có dải Keltner thời gian thực.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số
    def parse_number(val, default):
        if isinstance(val, str):
            num_str = re.sub(r'[^\d.]', '', val)
            return float(num_str) if '.' in num_str else int(num_str) if num_str else default
        return val

    window = int(parse_number(window, 20))
    atr_window = int(parse_number(atr_window, 10))
    multiplier = float(parse_number(multiplier, 2.0))

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_upper = f'kc_upper_{time_frame}'
    c_lower = f'kc_lower_{time_frame}'
    c_mid = f'kc_mid_{time_frame}'

    # 2. Tính toán Logic MTF Keltner Channel
    if time_frame == '1m':
        # --- Khung 1 phút (Tính trực tiếp) ---
        # 1. Đường giữa (EMA)
        mid_band = df['close'].ewm(span=window, adjust=False).mean()
        
        # 2. ATR
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'], 
            (df['high'] - prev_close).abs(), 
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_window, adjust=False).mean() # Wilder's RMA
        
        # 3. Keltner Bands
        df[c_mid] = mid_band
        df[c_upper] = mid_band + (multiplier * atr)
        df[c_lower] = mid_band - (multiplier * atr)

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

        # BƯỚC B: Tính EMA và ATR trên nến đã đóng
        htf_ema_closed = htf_close.ewm(span=window, adjust=False).mean()
        
        htf_prev_close = htf_close.shift(1)
        tr1 = htf_high - htf_low
        tr2 = (htf_high - htf_prev_close).abs()
        tr3 = (htf_low - htf_prev_close).abs()
        htf_tr_closed = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        htf_atr_closed = htf_tr_closed.ewm(alpha=1/atr_window, adjust=False).mean()

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        ema_ffill = htf_ema_closed.reindex(df.index, method='ffill')
        atr_ffill = htf_atr_closed.reindex(df.index, method='ffill')
        prev_c_ffill = htf_close.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng nến Live và tính True Range Live
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        live_close = df['close']

        live_tr1 = live_high - live_low
        live_tr2 = (live_high - prev_c_ffill).abs()
        live_tr3 = (live_low - prev_c_ffill).abs()
        live_tr = pd.concat([live_tr1, live_tr2, live_tr3], axis=1).max(axis=1)

        # BƯỚC E: Cập nhật Live EMA và Live ATR bằng Toán học
        alpha_ema = 2 / (window + 1)
        live_ema = (live_close * alpha_ema) + (ema_ffill * (1 - alpha_ema))
        
        live_atr = (atr_ffill * (atr_window - 1) + live_tr) / atr_window

        # BƯỚC F: Tính dải Keltner Live
        df[c_mid] = live_ema
        df[c_upper] = live_ema + (multiplier * live_atr)
        df[c_lower] = live_ema - (multiplier * live_atr)

    # 3. Phân loại trạng thái tín hiệu
    # Xác định giá có phá vỡ (Breakout) ra khỏi dải Keltner không
    df[f'kc_break_upper_{time_frame}'] = df['close'] > df[c_upper]
    df[f'kc_break_lower_{time_frame}'] = df['close'] < df[c_lower]
    
    # Xác định vị thế của giá so với đường trung tâm
    df[f'kc_above_mid_{time_frame}'] = df['close'] > df[c_mid]

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_donchian_channel(df, time_frame, window=20):
    """
    Phân tích Donchian Channel dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Kênh giá sẽ mở rộng ngay lập tức (Live) nếu giá hiện tại tạo đỉnh/đáy mới.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số
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

    c_upper = f'dc_upper_{time_frame}'
    c_lower = f'dc_lower_{time_frame}'
    c_mid = f'dc_mid_{time_frame}'
    c_width = f'dc_width_{time_frame}' # Độ rộng dải (giúp đo lường nén giá)

    # 2. Tính toán Logic MTF Donchian Channel
    if time_frame == '1m':
        # Tính toán trực tiếp trên khung 1 phút
        df[c_upper] = df['high'].rolling(window=window).max()
        df[c_lower] = df['low'].rolling(window=window).min()
        df[c_mid] = (df[c_upper] + df[c_lower]) / 2
        df[c_width] = (df[c_upper] - df[c_lower]) / df[c_mid]
        
    else:
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()

        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tìm Max/Min của (N-1) nến quá khứ làm "mỏ neo tĩnh"
        htf_max_n1 = htf_high.rolling(window=window - 1).max()
        htf_min_n1 = htf_low.rolling(window=window - 1).min()

        # BƯỚC C: Rải mỏ neo về khung 1 phút (Forward fill)
        max_n1_ffill = htf_max_n1.reindex(df.index, method='ffill')
        min_n1_ffill = htf_min_n1.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng nến Live 
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()

        # BƯỚC E: Cập nhật Kênh Donchian Live
        # Nếu live_high lớn hơn đỉnh N-1 nến trước, dải Upper sẽ nhích lên theo giá
        live_upper = pd.concat([max_n1_ffill, live_high], axis=1).max(axis=1)
        live_lower = pd.concat([min_n1_ffill, live_low], axis=1).min(axis=1)
        
        live_mid = (live_upper + live_lower) / 2
        live_width = (live_upper - live_lower) / live_mid

        df[c_upper] = live_upper
        df[c_lower] = live_lower
        df[c_mid] = live_mid
        df[c_width] = live_width

    # 3. Phân loại trạng thái tín hiệu
    # Xác định xu hướng chung bằng đường Mid Band (Đường giữa)
    df[f'dc_trend_{time_frame}'] = np.where(df['close'] > df[c_mid], 'UP', 'DOWN')
    
    # Xác định xem giá đang áp sát biên nào (để chuẩn bị mồi lệnh)
    # Ví dụ: Giá nằm ở nửa trên của kênh Donchian (vùng bò kiểm soát)
    df[f'dc_bullish_zone_{time_frame}'] = df['close'] > df[c_mid]

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df




