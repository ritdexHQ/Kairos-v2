"""CẤU TRÚC GIÁ (Market Structure / Price Action) ⭐
ZigZag Indicato
Bill Williams Fractals
Pivot Points High/Low
Fair Value Gap
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

import numpy as np
import pandas as pd
import re

def pt_breakout(df, time_frame, window=20):
    """
    Phân tích Breakout (Phá vỡ Đỉnh/Đáy) dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Mức cản được cố định trong suốt nến Live, đảm bảo không có Look-ahead Bias.
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

    c_high = f'breakout_High_{time_frame}'
    c_low = f'breakout_Low_{time_frame}'
    c_status = f'breakout_{time_frame}'

    # 2. Tính toán Logic MTF Breakout
    if time_frame == '1m':
        # Tính toán trên khung 1 phút: Phải shift(1) để lấy đỉnh/đáy quá khứ
        high_max = df['high'].shift(1).rolling(window).max()
        low_min = df['low'].shift(1).rolling(window).min()
        
    else:
        # BƯỚC A: Lấy nến đã đóng của Khung thời gian lớn
        htf_high_closed = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low_closed = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()

        # BƯỚC B: Dịch chuyển mốc thời gian sang tương lai 1 nến
        # Hành động này tương đương với lệnh .shift(1) ở khung 1 phút
        htf_high_closed.index = htf_high_closed.index + pd.to_timedelta(pd_time_frame)
        htf_low_closed.index = htf_low_closed.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC C: Tìm Đỉnh cao nhất / Đáy thấp nhất của N nến đã đóng
        htf_high_max = htf_high_closed.rolling(window).max()
        htf_low_min = htf_low_closed.rolling(window).min()

        # BƯỚC D: Rải "Mỏ neo" (Forward fill) mức cản về lại khung 1 phút
        high_max = htf_high_max.reindex(df.index, method='ffill')
        low_min = htf_low_min.reindex(df.index, method='ffill')

    # 3. Định nghĩa các mảng điều kiện (So sánh giá Live 1m với Cản HTF)
    conditions = [
        (df['close'] > high_max), # Giá vượt đỉnh cũ
        (df['close'] < low_min)   # Giá thủng đáy cũ
    ]
    choices = ['BREAK_OUT', 'BREAK_DOWN']

    # 4. Gán kết quả
    df[c_high] = high_max
    df[c_low] = low_min
    df[c_status] = np.select(conditions, choices, default='None')

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def calculate_zigzag_fast(high, low, deviation=1.0):
    """
    Vòng lặp Numpy siêu tốc để tìm Đỉnh/Đáy ZigZag.
    Chỉ trả về các mốc ĐÃ ĐƯỢC XÁC NHẬN (Confirmed) để tránh Look-ahead Bias.
    """
    n = len(high)
    trend = 1  # 1: Đang tìm Đỉnh, -1: Đang tìm Đáy
    last_high = high[0]
    last_low = low[0]
    
    # Mảng lưu trữ các mốc hỗ trợ/kháng cự tĩnh
    conf_highs = np.full(n, np.nan)
    conf_lows = np.full(n, np.nan)
    
    last_conf_high = np.nan
    last_conf_low = np.nan

    for i in range(1, n):
        # Kế thừa giá trị cũ để tạo thành đường ngang liên tục (Cản tĩnh)
        conf_highs[i] = last_conf_high
        conf_lows[i] = last_conf_low
        
        if trend == 1: # Đang trong pha Tăng, tìm Đỉnh
            if high[i] > last_high:
                last_high = high[i] # Cập nhật đỉnh tạm thời
            # Nếu giá giảm sâu hơn tỷ lệ deviation tính từ đỉnh tạm -> Xác nhận Đỉnh
            elif low[i] < last_high * (1 - deviation / 100):
                trend = -1
                last_low = low[i]
                last_conf_high = last_high # CHỐT ĐỈNH!
                conf_highs[i] = last_conf_high
                
        else: # Đang trong pha Giảm, tìm Đáy
            if low[i] < last_low:
                last_low = low[i] # Cập nhật đáy tạm thời
            # Nếu giá tăng mạnh hơn tỷ lệ deviation tính từ đáy tạm -> Xác nhận Đáy
            elif high[i] > last_low * (1 + deviation / 100):
                trend = 1
                last_high = high[i]
                last_conf_low = last_low # CHỐT ĐÁY!
                conf_lows[i] = last_conf_low
                
    return conf_highs, conf_lows
def pt_zigzag(df, time_frame, deviation=1.5):
    """
    Phân tích Cấu trúc ZigZag dạng MTF.
    deviation: Tỷ lệ % đảo chiều tối thiểu để xác nhận một đoạn sóng (VD: 1.5%).
    """
    df = df.copy()
    
    # 1. Xử lý tham số
    if isinstance(deviation, str):
        num_str = re.sub(r'[^\d.]', '', deviation)
        deviation = float(num_str) if num_str else 1.5
    else:
        deviation = float(deviation)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_zz_high = f'zz_res_{time_frame}' # Resistance (Kháng cự)
    c_zz_low = f'zz_sup_{time_frame}'  # Support (Hỗ trợ)

    # 2. Tính toán Logic MTF ZigZag
    if time_frame == '1m':
        highs, lows = calculate_zigzag_fast(df['high'].values, df['low'].values, deviation)
        df[c_zz_high] = highs
        df[c_zz_low] = lows
        
    else:
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()

        # BƯỚC B: Chạy thuật toán ZigZag trên tập dữ liệu đóng
        zz_highs_closed, zz_lows_closed = calculate_zigzag_fast(
            htf_high.values, htf_low.values, deviation
        )
        
        # BƯỚC C: Gắn index và shift 1 nến sang tương lai (Tránh Look-ahead bias)
        sr_high_series = pd.Series(zz_highs_closed, index=htf_high.index)
        sr_low_series = pd.Series(zz_lows_closed, index=htf_low.index)
        
        sr_high_series.index = sr_high_series.index + pd.to_timedelta(pd_time_frame)
        sr_low_series.index = sr_low_series.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC D: Rải mỏ neo về khung 1 phút (Tạo thành các đường cản ngang kéo dài)
        df[c_zz_high] = sr_high_series.reindex(df.index, method='ffill').ffill()
        df[c_zz_low] = sr_low_series.reindex(df.index, method='ffill').ffill()

    # 3. Phân loại Cấu trúc (Tín hiệu)
    # Tín hiệu bứt phá khỏi Cản ZigZag (Xác nhận thay đổi cấu trúc thị trường thực sự)
    df[f'zz_breakout_up_{time_frame}'] = df['close'] > df[c_zz_high]
    df[f'zz_breakout_down_{time_frame}'] = df['close'] < df[c_zz_low]

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df    
        
def pt_fractals(df, time_frame, window=2):
    """
    Phân tích Bill Williams Fractals dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    window=2 nghĩa là cần 2 nến trái và 2 nến phải (Mô hình 5 nến kinh điển).
    Khắc phục triệt để Look-ahead Bias bằng cách xác nhận chậm 2 nến.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window = int(num_str) if num_str else 2
    else:
        window = int(window)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_up_frac = f'frac_res_{time_frame}' # Fractal Resistance (Kháng cự)
    c_down_frac = f'frac_sup_{time_frame}' # Fractal Support (Hỗ trợ)

    # 2. Tính toán Logic MTF Fractals
    if time_frame == '1m':
        # --- Khung 1 phút ---
        # Điểm nghi ngờ là nến T-window (shift 2)
        high_target = df['high'].shift(window)
        low_target = df['low'].shift(window)
        
        # Kiểm tra 2 nến bên trái (shift 4 và 3) và 2 nến bên phải (shift 1 và 0)
        up_cond = (
            (df['high'].shift(4) < high_target) & 
            (df['high'].shift(3) < high_target) & 
            (df['high'].shift(1) < high_target) & 
            (df['high'] < high_target)
        )
        
        down_cond = (
            (df['low'].shift(4) > low_target) & 
            (df['low'].shift(3) > low_target) & 
            (df['low'].shift(1) > low_target) & 
            (df['low'] > low_target)
        )
        
        # Lưu lại mức giá của Fractal và kéo dài (ffill) tạo thành cản tĩnh
        df[c_up_frac] = high_target.where(up_cond).ffill()
        df[c_down_frac] = low_target.where(down_cond).ffill()

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()

        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tìm Fractal trên dữ liệu HTF đã đóng
        htf_high_target = htf_high.shift(window)
        htf_low_target = htf_low.shift(window)
        
        htf_up_cond = (
            (htf_high.shift(4) < htf_high_target) & 
            (htf_high.shift(3) < htf_high_target) & 
            (htf_high.shift(1) < htf_high_target) & 
            (htf_high < htf_high_target)
        )
        
        htf_down_cond = (
            (htf_low.shift(4) > htf_low_target) & 
            (htf_low.shift(3) > htf_low_target) & 
            (htf_low.shift(1) > htf_low_target) & 
            (htf_low > htf_low_target)
        )

        # Lấy mức giá của Fractal gần nhất và ffill
        htf_up_level = htf_high_target.where(htf_up_cond).ffill()
        htf_down_level = htf_low_target.where(htf_down_cond).ffill()

        # BƯỚC C: Rải mỏ neo (Forward Fill) về khung 1 phút Live
        # Cản của Fractal 15m sẽ nằm im như một bức tường cho giá 1m đập vào
        df[c_up_frac] = htf_up_level.reindex(df.index, method='ffill').ffill()
        df[c_down_frac] = htf_down_level.reindex(df.index, method='ffill').ffill()

    # 3. Phân loại Cấu trúc (Tín hiệu Giao dịch)
    # Breakout Fractal: Đây là tín hiệu cốt lõi của Bill Williams
    df[f'frac_breakout_up_{time_frame}'] = df['close'] > df[c_up_frac]
    df[f'frac_breakout_down_{time_frame}'] = df['close'] < df[c_down_frac]
    
    # Khoảng cách đến Cản (Dùng để cài Stoploss)
    # Stoploss an toàn nhất luôn nằm dưới Fractal Support gần nhất
    df[f'frac_dist_to_sup_{time_frame}'] = (df['close'] - df[c_down_frac]) / df['close'] * 100

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_pivot_points(df, time_frame, left_bars=5, right_bars=5):
    """
    Phân tích Pivot Points High/Low (Swing Points) dạng Vectorized Đa Khung Thời Gian (MTF).
    Cực kỳ uy tín để xác định Thanh khoản (Liquidity) và Order Blocks theo chuẩn SMC.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số
    def parse_window(w, default):
        if isinstance(w, str):
            num_str = re.sub(r'\D', '', w)
            return int(num_str) if num_str else default
        return int(w)
        
    left_bars = parse_window(left_bars, 5)
    right_bars = parse_window(right_bars, 5)
    window = left_bars + right_bars + 1

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_pivot_high = f'pivot_high_{time_frame}'
    c_pivot_low = f'pivot_low_{time_frame}'

    # 2. Tính toán Logic MTF Pivot Points
    if time_frame == '1m':
        # --- Khung 1 phút ---
        # Nến mục tiêu để kiểm tra là nến bị trễ right_bars
        target_high = df['high'].shift(right_bars)
        target_low = df['low'].shift(right_bars)
        
        # Lấy Max/Min của toàn bộ cửa sổ N nến
        rolling_max = df['high'].rolling(window=window).max()
        rolling_min = df['low'].rolling(window=window).min()
        
        # Nếu nến mục tiêu chính là Max/Min của cả cửa sổ -> Xác nhận Pivot
        is_pivot_high = (target_high == rolling_max)
        is_pivot_low = (target_low == rolling_min)
        
        # Kéo dài mức giá Pivot thành các đường Cản tĩnh
        df[c_pivot_high] = target_high.where(is_pivot_high).ffill()
        df[c_pivot_low] = target_low.where(is_pivot_low).ffill()

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()

        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tìm Pivot Points trên dữ liệu HTF
        htf_target_high = htf_high.shift(right_bars)
        htf_target_low = htf_low.shift(right_bars)
        
        htf_rolling_max = htf_high.rolling(window=window).max()
        htf_rolling_min = htf_low.rolling(window=window).min()
        
        htf_is_pivot_high = (htf_target_high == htf_rolling_max)
        htf_is_pivot_low = (htf_target_low == htf_rolling_min)

        # Trích xuất mức giá và ffill trên khung HTF
        htf_pivot_high_levels = htf_target_high.where(htf_is_pivot_high).ffill()
        htf_pivot_low_levels = htf_target_low.where(htf_is_pivot_low).ffill()

        # BƯỚC C: Rải mỏ neo (Forward Fill) về khung 1 phút Live
        # Các đỉnh đáy của khung 15m/1H giờ đây sẽ nằm chờ dưới dạng mốc tĩnh trên biểu đồ 1m
        df[c_pivot_high] = htf_pivot_high_levels.reindex(df.index, method='ffill').ffill()
        df[c_pivot_low] = htf_pivot_low_levels.reindex(df.index, method='ffill').ffill()

    # 3. Phân tích Tín hiệu Hành động giá (Price Action Signals)
    
    # Sweep Liquidity (Quét thanh khoản đỉnh/đáy)
    # Giá đâm lủng qua Pivot nhưng ngay sau đó bị kéo ngược lại.
    # Đây là lúc ta kết hợp với Williams %R hoặc nến Pinbar để đánh Reversal.
    df[f'pivot_sweep_high_{time_frame}'] = df['high'] > df[c_pivot_high]
    df[f'pivot_sweep_low_{time_frame}'] = df['low'] < df[c_pivot_low]
    
    # Cấu trúc Căn bản (So sánh giá đóng cửa với Pivot để xem ai đang kiểm soát)
    df[f'pivot_break_up_{time_frame}'] = df['close'] > df[c_pivot_high]
    df[f'pivot_break_down_{time_frame}'] = df['close'] < df[c_pivot_low]

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_fvg(df, time_frame):
    """
    Phân tích Fair Value Gap (FVG / Imbalance) dạng Vectorized Đa Khung Thời Gian (MTF).
    Nhận diện khoảng trống thanh khoản và xác định xem giá đã "Lấp" (Mitigate) FVG hay chưa.
    """
    df = df.copy()
    
    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_fvg_top = f'fvg_top_{time_frame}'
    c_fvg_bot = f'fvg_bottom_{time_frame}'
    c_fvg_type = f'fvg_type_{time_frame}' # 1: Bullish FVG, -1: Bearish FVG

    # 2. Tính toán Logic MTF FVG
    if time_frame == '1m':
        # --- Khung 1 phút ---
        # Điều kiện Bullish FVG: Đáy nến hiện tại > Đỉnh nến cách đây 2 chu kỳ
        bull_fvg = df['low'] > df['high'].shift(2)
        
        # Điều kiện Bearish FVG: Đỉnh nến hiện tại < Đáy nến cách đây 2 chu kỳ
        bear_fvg = df['high'] < df['low'].shift(2)

        # Trích xuất Vùng giá trị của FVG
        fvg_top = np.where(bull_fvg, df['low'], np.where(bear_fvg, df['low'].shift(2), np.nan))
        fvg_bot = np.where(bull_fvg, df['high'].shift(2), np.where(bear_fvg, df['high'], np.nan))
        fvg_type = np.where(bull_fvg, 1, np.where(bear_fvg, -1, np.nan))

        # Kéo dài FVG gần nhất ra tương lai (Forward fill)
        df[c_fvg_top] = pd.Series(fvg_top, index=df.index).ffill()
        df[c_fvg_bot] = pd.Series(fvg_bot, index=df.index).ffill()
        df[c_fvg_type] = pd.Series(fvg_type, index=df.index).ffill()

    else:
        # --- Đa Khung Thời Gian (MTF Live) ---
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()

        # BƯỚC B: Tìm FVG trên nến HTF
        htf_bull_fvg = htf_low > htf_high.shift(2)
        htf_bear_fvg = htf_high < htf_low.shift(2)

        htf_fvg_top = np.where(htf_bull_fvg, htf_low, np.where(htf_bear_fvg, htf_low.shift(2), np.nan))
        htf_fvg_bot = np.where(htf_bull_fvg, htf_high.shift(2), np.where(htf_bear_fvg, htf_high, np.nan))
        htf_fvg_type = np.where(htf_bull_fvg, 1, np.where(htf_bear_fvg, -1, np.nan))

        # Đưa vào Series để dễ shift và ffill
        s_top = pd.Series(htf_fvg_top, index=htf_high.index).ffill()
        s_bot = pd.Series(htf_fvg_bot, index=htf_high.index).ffill()
        s_type = pd.Series(htf_fvg_type, index=htf_high.index).ffill()

        # BƯỚC C: Dịch mốc thời gian sang tương lai 1 nến
        # FVG hình thành từ nến T-2, T-1, T. Nó được CHỐT ở nến T.
        # Nên nến T+1 mới bắt đầu thấy FVG này trên chart để trade. (Tránh tuyệt đối Look-ahead bias)
        s_top.index = s_top.index + pd.to_timedelta(pd_time_frame)
        s_bot.index = s_bot.index + pd.to_timedelta(pd_time_frame)
        s_type.index = s_type.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC D: Rải mỏ neo về khung 1 phút Live
        df[c_fvg_top] = s_top.reindex(df.index, method='ffill').ffill()
        df[c_fvg_bot] = s_bot.reindex(df.index, method='ffill').ffill()
        df[c_fvg_type] = s_type.reindex(df.index, method='ffill').ffill()

    # 3. Phân loại Tín hiệu SMC với FVG
    
    # Tín hiệu: Giá ĐANG NẰM TRONG vùng FVG (Mitigation Zone)
    # Đây là Vùng Vàng (Golden Zone) để tìm kiếm lệnh Reversal
    df[f'fvg_in_zone_{time_frame}'] = (df['close'] <= df[c_fvg_top]) & (df['close'] >= df[c_fvg_bot])
    
    # Tín hiệu: Lấp FVG thành công (Mitigated)
    # Nếu là Bullish FVG, giá đâm thủng mép dưới của FVG là coi như đã lấp xong và vô hiệu hóa FVG đó
    df[f'fvg_mitigated_{time_frame}'] = np.where(
        df[c_fvg_type] == 1, df['close'] < df[c_fvg_bot], 
        np.where(df[c_fvg_type] == -1, df['close'] > df[c_fvg_top], False)
    )

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df