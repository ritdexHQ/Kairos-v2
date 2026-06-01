""" XU HƯỚNG (Trend)
👉 Giá đang đi hướng nào?
- EMA, SMA
- MACD
- ADX
- Ichimoku
- SuperTrend
📌 Dùng để chọn phe BUY / SELL """

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
import re

def pt_ema_trend(df, time_frame, window=20):
    """
    Phân tích xu hướng dựa trên EMA dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Trả về 'UP' nếu giá trên EMA, 'DOWN' nếu giá dưới EMA.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window (tránh lỗi khi truyền nhầm string)
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 20
    else:
        window = int(window)

    # 2. Xử lý Index và Tên khung thời gian cho Pandas >= 2.2.0
    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    col_ema_name = f'ema_{window}_{time_frame}'

    # 3. Tính toán Logic MTF EMA
    if time_frame == '1m':
        # Tính trực tiếp trên khung 1 phút
        ema_ind = EMAIndicator(df['close'], window=window)
        df[col_ema_name] = ema_ind.ema_indicator()
    else:
        # Bước A: Lấy nến đã đóng của Khung thời gian lớn (HTF)
        df_closed = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Bước B: Tính EMA trên nến đã đóng (dùng ewm của pandas cho tiện và khớp với ta.trend)
        ema_closed = df_closed.ewm(span=window, adjust=False).mean()
        
        # Bước C: Dịch chuyển mốc thời gian sang tương lai 1 nến (Tránh Look-ahead bias)
        ema_closed.index = ema_closed.index + pd.to_timedelta(pd_time_frame)
        
        # Bước D: Rải "Mỏ neo" lên biểu đồ 1 phút (Forward fill)
        ema_ffill = ema_closed.reindex(df.index, method='ffill')
        
        # Bước E: Tính LIVE EMA cho nến lớn đang chạy
        alpha = 2 / (window + 1)
        ema_live = (df['close'] * alpha) + (ema_ffill * (1 - alpha))
        
        df[col_ema_name] = ema_live

    # 4. Phân loại xu hướng: UP nếu Close > EMA Live, ngược lại là DOWN
    df[f'is_trend_{time_frame}'] = np.where(
        df['close'] > df[col_ema_name], 
        'UP', 
        'DOWN'
    )
    
    # 5. Phân loại mức độ xu hướng (Dạng True/False để lọc sức mạnh)
    distance_pct = (df['close'] - df[col_ema_name]).abs() / df[col_ema_name]
    df[f'trend_strong_{time_frame}'] = distance_pct > 0.005
    
    # Trả lại cột timestamp như cũ nếu cần
    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_sma(df, time_frame, window=20):
    """
    Phân tích Simple Moving Average (SMA) dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Tính toán chuẩn xác Live SMA không bị Look-ahead Bias.
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

    col_sma = f'sma_{window}_{time_frame}'

    # 2. Tính toán Logic MTF SMA
    if time_frame == '1m':
        # Tính trực tiếp trên khung 1 phút
        df[col_sma] = df['close'].rolling(window=window).mean()
    else:
        # BƯỚC A: Lấy nến đã đóng của Khung thời gian lớn (HTF)
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # BƯỚC B: Dịch chuyển mốc thời gian sang tương lai 1 nến
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)
        
        # BƯỚC C: Tính tổng của (window - 1) nến HTF quá khứ
        # Đây là khối lượng dữ liệu tĩnh đã chốt sổ
        htf_sum_closed = htf_close.rolling(window=window - 1).sum()
        
        # BƯỚC D: Rải "Mỏ neo" (Forward fill) về lại khung 1 phút
        sum_ffill = htf_sum_closed.reindex(df.index, method='ffill')
        
        # BƯỚC E: Tính LIVE SMA cho từng phút
        live_sma = (sum_ffill + df['close']) / window
        
        df[col_sma] = live_sma

    # 3. Phân loại trạng thái xu hướng cơ bản
    # Trả về True nếu Giá vượt lên trên SMA (Hỗ trợ Long), False nếu ở dưới (Hỗ trợ Short)
    df[f'is_above_sma_{window}_{time_frame}'] = df['close'] > df[col_sma]
    
    # 4. Phân loại khoảng cách (Đo độ dãn của giá so với SMA để tránh mua đuổi)
    distance_pct = (df['close'] - df[col_sma]).abs() / df[col_sma]
    df[f'sma_overextended_{window}_{time_frame}'] = distance_pct > 0.02 # Giá cách SMA hơn 2%
    
    # Trả lại cột timestamp như cũ nếu cần
    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_adx(df, time_frame, window=28):
    """
    Phân tích lực xu hướng qua ADX (Vectorized - Multi-Timeframe).
    Mô phỏng chính xác thuật toán J. Welles Wilder cho nến Live.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 28
    else:
        window = int(window)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    col_adx = f'adx_{time_frame}'

    # 2. Tính toán Logic MTF ADX
    if time_frame == '1m':
        adx_ind = ADXIndicator(df['high'], df['low'], df['close'], window=window)
        df[col_adx] = adx_ind.adx()
    else:
        # BƯỚC A: Dựng nến HTF đã đóng
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Shift index sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính toán các thành phần ADX lịch sử trên nến đã đóng
        # (Dùng dịch chuyển 1 nến của chính nó để tính)
        prev_close = htf_close.shift(1)
        prev_high = htf_high.shift(1)
        prev_low = htf_low.shift(1)

        tr1 = htf_high - htf_low
        tr2 = (htf_high - prev_close).abs()
        tr3 = (htf_low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = htf_high - prev_high
        down_move = prev_low - htf_low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=htf_high.index)
        minus_dm = pd.Series(minus_dm, index=htf_low.index)

        # Làm mượt (Wilder's Smoothing alpha = 1/window)
        atr_closed = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/window, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/window, adjust=False).mean()

        plus_di_closed = 100 * plus_dm_smooth / atr_closed
        minus_di_closed = 100 * minus_dm_smooth / atr_closed
        
        dx_closed = 100 * (plus_di_closed - minus_di_closed).abs() / (plus_di_closed + minus_di_closed)
        adx_closed = dx_closed.ewm(alpha=1/window, adjust=False).mean()

        # BƯỚC C: Rải mỏ neo (Forward Fill) về khung 1 phút
        atr_ffill = atr_closed.reindex(df.index, method='ffill')
        plus_dm_ffill = plus_dm_smooth.reindex(df.index, method='ffill')
        minus_dm_ffill = minus_dm_smooth.reindex(df.index, method='ffill')
        adx_ffill = adx_closed.reindex(df.index, method='ffill')
        
        high_closed_ffill = htf_high.reindex(df.index, method='ffill')
        low_closed_ffill = htf_low.reindex(df.index, method='ffill')
        close_closed_ffill = htf_close.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng nến LIVE HTF từ các nến 1 phút đang chạy
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        live_close = df['close']

        # Tính TR, +DM, -DM của nến Live so với nến đóng trước đó
        tr1_live = live_high - live_low
        tr2_live = (live_high - close_closed_ffill).abs()
        tr3_live = (live_low - close_closed_ffill).abs()
        tr_live = pd.concat([tr1_live, tr2_live, tr3_live], axis=1).max(axis=1)

        up_move_live = live_high - high_closed_ffill
        down_move_live = low_closed_ffill - live_low
        
        plus_dm_live = np.where((up_move_live > down_move_live) & (up_move_live > 0), up_move_live, 0.0)
        minus_dm_live = np.where((down_move_live > up_move_live) & (down_move_live > 0), down_move_live, 0.0)

        # BƯỚC E: Cập nhật Smoothing cho phút hiện tại
        atr_live = (atr_ffill * (window - 1) + tr_live) / window
        plus_dm_s_live = (plus_dm_ffill * (window - 1) + plus_dm_live) / window
        minus_dm_s_live = (minus_dm_ffill * (window - 1) + minus_dm_live) / window

        # BƯỚC F: Tính toán DI, DX và ADX Live
        # Dùng np.where để tránh lỗi chia cho 0 nếu thị trường hoàn toàn không nhúc nhích
        di_sum = (plus_dm_s_live + minus_dm_s_live)
        di_sum_safe = np.where(di_sum == 0, 1e-9, di_sum)
        
        plus_di_live = 100 * plus_dm_s_live / atr_live
        minus_di_live = 100 * minus_dm_s_live / atr_live
        
        dx_live = 100 * np.abs(plus_di_live - minus_di_live) / (plus_di_live + minus_di_live)
        dx_live = dx_live.fillna(0) # Đề phòng NaN
        
        adx_live = (adx_ffill * (window - 1) + dx_live) / window
        
        df[col_adx] = adx_live

    # 3. Tạo cột Logic Xác định xu hướng
    df[f'has_trend_{time_frame}'] = df[col_adx] > 25
    df[f'is_strong_trend_{time_frame}'] = df[col_adx] > 40
    
    # Reset index nếu cần
    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_ichimoku(df, time_frame, n1=9, n2=26, n3=52):
    """
    Phân tích Ichimoku Kinko Hyo dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Tính Live Tenkan/Kijun và xử lý chuẩn xác độ trễ của Mây Kumo & Chikou Span.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số
    def parse_window(w, default):
        if isinstance(w, str):
            num_str = re.sub(r'\D', '', w)
            return int(num_str) if num_str else default
        return int(w)
        
    n1 = parse_window(n1, 9)
    n2 = parse_window(n2, 26)
    n3 = parse_window(n3, 52)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_tenkan = f'ichi_tenkan_{time_frame}'
    c_kijun = f'ichi_kijun_{time_frame}'
    c_senkou_a = f'ichi_senkou_a_{time_frame}'
    c_senkou_b = f'ichi_senkou_b_{time_frame}'
    c_chikou_ref = f'ichi_chikou_ref_{time_frame}' # Giá đóng cửa quá khứ để so sánh Chikou

    # 2. Tính toán Logic MTF Ichimoku
    if time_frame == '1m':
        # Tính toán trực tiếp trên khung 1 phút
        high_9 = df['high'].rolling(n1).max()
        low_9 = df['low'].rolling(n1).min()
        df[c_tenkan] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(n2).max()
        low_26 = df['low'].rolling(n2).min()
        df[c_kijun] = (high_26 + low_26) / 2
        
        # Mây Kumo: Tính toán giá trị và shift tiến về tương lai n2 nến
        senkou_a = ((df[c_tenkan] + df[c_kijun]) / 2).shift(n2)
        
        high_52 = df['high'].rolling(n3).max()
        low_52 = df['low'].rolling(n3).min()
        senkou_b = ((high_52 + low_52) / 2).shift(n2)
        
        df[c_senkou_a] = senkou_a
        df[c_senkou_b] = senkou_b
        
        # Chikou tham chiếu: Giá đóng cửa lùi về n2 nến
        df[c_chikou_ref] = df['close'].shift(n2)

    else:
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        
        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính toán quá khứ làm mỏ neo cho Live Tenkan & Kijun
        # Để tính đỉnh Live N nến, ta cần đỉnh của (N-1) nến quá khứ
        htf_max_n1_closed = htf_high.rolling(n1 - 1).max()
        htf_min_n1_closed = htf_low.rolling(n1 - 1).min()
        
        htf_max_n2_closed = htf_high.rolling(n2 - 1).max()
        htf_min_n2_closed = htf_low.rolling(n2 - 1).min()

        # BƯỚC C: Tính toán Đám mây Kumo hiện tại (Tĩnh)
        # 1. Vẽ mây trên HTF Closed
        htf_tenkan_closed = (htf_high.rolling(n1).max() + htf_low.rolling(n1).min()) / 2
        htf_kijun_closed = (htf_high.rolling(n2).max() + htf_low.rolling(n2).min()) / 2
        htf_senkou_a_closed = (htf_tenkan_closed + htf_kijun_closed) / 2
        htf_senkou_b_closed = (htf_high.rolling(n3).max() + htf_low.rolling(n3).min()) / 2
        
        # 2. Dịch chuyển mây tiến về n2 nến (Cloud Shift)
        htf_current_cloud_a = htf_senkou_a_closed.shift(n2)
        htf_current_cloud_b = htf_senkou_b_closed.shift(n2)
        
        # 3. Chikou Tham chiếu: Lấy giá đóng cửa n2 nến trước
        htf_chikou_ref = htf_close.shift(n2)

        # BƯỚC D: Rải (Forward Fill) về khung 1 phút
        max_n1_ffill = htf_max_n1_closed.reindex(df.index, method='ffill')
        min_n1_ffill = htf_min_n1_closed.reindex(df.index, method='ffill')
        
        max_n2_ffill = htf_max_n2_closed.reindex(df.index, method='ffill')
        min_n2_ffill = htf_min_n2_closed.reindex(df.index, method='ffill')
        
        df[c_senkou_a] = htf_current_cloud_a.reindex(df.index, method='ffill')
        df[c_senkou_b] = htf_current_cloud_b.reindex(df.index, method='ffill')
        df[c_chikou_ref] = htf_chikou_ref.reindex(df.index, method='ffill')

        # BƯỚC E: Tính LIVE Tenkan và Kijun (Nến đang chạy)
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        
        # Đỉnh n1 nến Live = Max(Đỉnh n1-1 nến cũ, Đỉnh Live hiện tại)
        live_max_n1 = pd.concat([max_n1_ffill, live_high], axis=1).max(axis=1)
        live_min_n1 = pd.concat([min_n1_ffill, live_low], axis=1).min(axis=1)
        df[c_tenkan] = (live_max_n1 + live_min_n1) / 2
        
        live_max_n2 = pd.concat([max_n2_ffill, live_high], axis=1).max(axis=1)
        live_min_n2 = pd.concat([min_n2_ffill, live_low], axis=1).min(axis=1)
        df[c_kijun] = (live_max_n2 + live_min_n2) / 2

    # 3. Tạo các Cột Logic Tín hiệu (Signals Aggregation)
    
    # Kumo Breakout (Giá vượt mây)
    # Tìm biên trên và biên dưới của mây
    kumo_top = df[[c_senkou_a, c_senkou_b]].max(axis=1)
    kumo_bottom = df[[c_senkou_a, c_senkou_b]].min(axis=1)
    
    df[f'ichi_above_cloud_{time_frame}'] = df['close'] > kumo_top
    df[f'ichi_below_cloud_{time_frame}'] = df['close'] < kumo_bottom
    
    # TK Cross (Tenkan cắt Kijun)
    df[f'ichi_tk_bullish_{time_frame}'] = df[c_tenkan] > df[c_kijun]
    
    # Chikou Span (Giá hiện tại nằm trên/dưới giá quá khứ)
    df[f'ichi_chikou_bullish_{time_frame}'] = df['close'] > df[c_chikou_ref]

    # Mây tương lai (Kumo Twist) - Xem Senkou A đang nằm trên hay dưới Senkou B
    # Lưu ý: Đây là mây ở hiện tại, không phải đám mây dự phóng 26 nến phía trước
    df[f'ichi_cloud_green_{time_frame}'] = df[c_senkou_a] > df[c_senkou_b]

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def calculate_supertrend_fast(high, low, close, window=10, multiplier=3.0):
    """
    Hàm tính toán SuperTrend nội bộ sử dụng Numpy mảng để tối ưu tốc độ (Fast Loop).
    """
    # 1. Tính ATR (Wilder's Smoothing)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0] # Xử lý nến đầu tiên
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Tính RMA (Rolling Moving Average) cho ATR
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window

    # 2. Khởi tạo mảng SuperTrend
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    in_uptrend = np.ones(n, dtype=bool)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = final_upper[0]
    in_uptrend[0] = True

    # Vòng lặp đệ quy xử lý logic SuperTrend
    for i in range(1, n):
        # Cập nhật Final Upper
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Cập nhật Final Lower
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Xác định xu hướng và đường SuperTrend chính
        if supertrend[i-1] == final_upper[i-1]:
            if close[i] > final_upper[i]:
                in_uptrend[i] = True
                supertrend[i] = final_lower[i]
            else:
                in_uptrend[i] = False
                supertrend[i] = final_upper[i]
        elif supertrend[i-1] == final_lower[i-1]:
            if close[i] < final_lower[i]:
                in_uptrend[i] = False
                supertrend[i] = final_upper[i]
            else:
                in_uptrend[i] = True
                supertrend[i] = final_lower[i]

    return supertrend, in_uptrend
def pt_supertrend(df, time_frame, window=10, multiplier=3.0):
    """
    Phân tích SuperTrend Hỗ trợ Đa Khung Thời Gian (MTF) cho nến Live.
    """
    df = df.copy()
    
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 10
    else:
        window = int(window)
        
    multiplier = float(multiplier)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_st = f'supertrend_{time_frame}'
    c_trend = f'is_st_uptrend_{time_frame}'

    if time_frame == '1m':
        # Chạy trực tiếp trên khung 1 phút
        st_val, st_up = calculate_supertrend_fast(
            df['high'].values, df['low'].values, df['close'].values, window, multiplier
        )
        df[c_st] = st_val
        df[c_trend] = st_up
        
    else:
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()

        # BƯỚC B: Tính SuperTrend trên nến quá khứ
        st_closed, uptrend_closed = calculate_supertrend_fast(
            htf_high.values, htf_low.values, htf_close.values, window, multiplier
        )
        
        st_series = pd.Series(st_closed, index=htf_close.index)
        uptrend_series = pd.Series(uptrend_closed, index=htf_close.index)

        # BƯỚC C: Dịch chuyển mốc thời gian sang tương lai 1 nến để chặn Look-ahead bias
        st_series.index = st_series.index + pd.to_timedelta(pd_time_frame)
        uptrend_series.index = uptrend_series.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC D: Rải (Forward Fill) mức cản về khung 1 phút
        st_ffill = st_series.reindex(df.index, method='ffill')
        uptrend_ffill = uptrend_series.reindex(df.index, method='ffill')

        # BƯỚC E: Tính Live Trend 
        # Nếu nến 1m đang chạy cắt qua đường cản tĩnh của SuperTrend HTF, xu hướng lập tức đảo chiều!
        live_uptrend = uptrend_ffill.copy()
        
        # Đang xu hướng TĂNG mà giá 1 phút đâm lủng cản dưới -> Đảo chiều GIẢM
        live_uptrend = np.where((uptrend_ffill == True) & (df['close'] < st_ffill), False, live_uptrend)
        
        # Đang xu hướng GIẢM mà giá 1 phút bứt phá cản trên -> Đảo chiều TĂNG
        live_uptrend = np.where((uptrend_ffill == False) & (df['close'] > st_ffill), True, live_uptrend)

        df[c_st] = st_ffill
        df[c_trend] = live_uptrend

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df





