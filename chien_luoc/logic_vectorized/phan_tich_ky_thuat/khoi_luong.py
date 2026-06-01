""" KHỐI LƯỢNG (Volume / Participation)
👉 Có tiền thật vào không?
- Volume
- Volume MA
- OBV
- VWAP
- Volume Profile
📌 Dùng để xác nhận tín hiệu """

import numpy as np
import pandas as pd
import re

def pt_volume(df, time_frame, window=20):
    """
    Phân tích Volume dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    if isinstance(window, str):
        num_str = re.sub(r'\D', '', window)
        window_val = int(num_str) if num_str else 20
    else:
        window_val = int(window)
        
    if window_val < 2:
        window_val = 20

    # 2. Xử lý Index và Tên khung thời gian cho Pandas >= 2.2.0
    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    # 3. Tính toán Logic MTF
    if time_frame == '1m':
        # Base Timeframe: Tính bình thường
        vol_mean = df['volume'].rolling(window=window_val, min_periods=1).mean()
        vol_current = df['volume']
    else:
        # Bước A: Resample Volume bằng cách CỘNG DỒN (sum) nến đã đóng
        htf_vol = df['volume'].resample(pd_time_frame, label='left', closed='left').sum().dropna()
        
        # Bước B: Tính Volume Trung bình trên khung thời gian lớn
        htf_vol_mean = htf_vol.rolling(window=window_val, min_periods=1).mean()
        
        # Bước C: Dịch chuyển mốc thời gian sang tương lai 1 nến để tránh Look-ahead Bias
        htf_vol_mean.index = htf_vol_mean.index + pd.to_timedelta(pd_time_frame)
        
        # Bước D: Rải mỏ neo (Forward fill) đường Trung bình Volume về lại khung 1 phút
        vol_mean = htf_vol_mean.reindex(df.index, method='ffill')
        
        # Bước E: Tính LIVE Volume cho nến lớn đang chạy
        # (Cộng dồn volume của các nến 1m nằm trong nến lớn hiện tại)
        vol_current = df['volume'].groupby(pd.Grouper(freq=pd_time_frame, label='left', closed='left')).cumsum()

    # 4. Tạo các cột logic True/False
    df[f'vol_mean_{time_frame}'] = vol_mean
    df[f'vol_live_{time_frame}'] = vol_current # Lưu lại để track volume thực tế đang chạy
    
    # So sánh Volume đang chạy (vol_current) với Trung bình Volume HTF đã đóng (vol_mean)
    df[f'vol_tang_{time_frame}'] = vol_current > vol_mean
    df[f'vol_tang_manh_{time_frame}'] = vol_current > (vol_mean * 2)
    df[f'vol_giam_{time_frame}'] = vol_current < vol_mean
    df[f'vol_giam_manh_{time_frame}'] = vol_current < (vol_mean * 0.5)

    # Trả lại cột timestamp như cũ nếu cần
    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_volume_ma(df, time_frame, fast_window=5, slow_window=20):
    """
    Phân tích Volume MA Kép dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Sử dụng 2 đường MA (Nhanh và Chậm) để xác định xu hướng của dòng tiền.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    def parse_window(w, default):
        if isinstance(w, str):
            num_str = re.sub(r'\D', '', w)
            return int(num_str) if num_str else default
        return int(w)
        
    fast_window = parse_window(fast_window, 5)
    slow_window = parse_window(slow_window, 20)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_vol_live = f'vol_live_{time_frame}'
    c_vol_fast = f'vol_ma_fast_{time_frame}'
    c_vol_slow = f'vol_ma_slow_{time_frame}'

    # 2. Tính toán Logic MTF Volume MA
    if time_frame == '1m':
        df[c_vol_live] = df['volume']
        df[c_vol_fast] = df['volume'].rolling(window=fast_window, min_periods=1).mean()
        df[c_vol_slow] = df['volume'].rolling(window=slow_window, min_periods=1).mean()
        
    else:
        # BƯỚC A: Lấy nến đã đóng HTF (Sử dụng SUM thay vì LAST)
        htf_vol_closed = df['volume'].resample(pd_time_frame, label='left', closed='left').sum().dropna()

        # BƯỚC B: Dịch chuyển mốc thời gian sang tương lai 1 nến
        htf_vol_closed.index = htf_vol_closed.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC C: Tính TỔNG của (N-1) nến Volume quá khứ để làm mỏ neo tính Live SMA
        # min_periods=1 giúp hàm không bị NaN ở những nến đầu tiên
        htf_sum_fast_closed = htf_vol_closed.rolling(window=fast_window - 1, min_periods=1).sum()
        htf_sum_slow_closed = htf_vol_closed.rolling(window=slow_window - 1, min_periods=1).sum()

        # BƯỚC D: Rải mỏ neo (Forward fill) về lại khung 1 phút
        sum_fast_ffill = htf_sum_fast_closed.reindex(df.index, method='ffill').fillna(0)
        sum_slow_ffill = htf_sum_slow_closed.reindex(df.index, method='ffill').fillna(0)

        # BƯỚC E: Dựng Live Volume (Cộng dồn volume 1m trong cây nến HTF đang chạy)
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_vol = df['volume'].groupby(grouper).cumsum()

        # BƯỚC F: Tính LIVE Volume MA
        df[c_vol_live] = live_vol
        df[c_vol_fast] = (sum_fast_ffill + live_vol) / fast_window
        df[c_vol_slow] = (sum_slow_ffill + live_vol) / slow_window

    # 3. Phân loại trạng thái dòng tiền (Signals)
    
    # Dòng tiền đang có xu hướng tăng (Volume Fast cắt lên Volume Slow)
    df[f'vol_trend_up_{time_frame}'] = df[c_vol_fast] > df[c_vol_slow]
    
    # Khối lượng đột biến (Live Volume bùng nổ gấp 1.5 lần trung bình dài hạn)
    # Đây là tín hiệu xác nhận Breakout cực kỳ uy tín
    df[f'vol_surge_{time_frame}'] = df[c_vol_live] > (df[c_vol_slow] * 1.5)
    
    # Cạn cung/Cạn cầu (Khối lượng thấp hơn một nửa so với trung bình)
    # Tín hiệu tốt khi giá đang ở vùng hỗ trợ/kháng cự (Test cung)
    df[f'vol_dry_{time_frame}'] = df[c_vol_live] < (df[c_vol_slow] * 0.5)

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_obv(df, time_frame, sma_window=20):
    """
    Phân tích On-Balance Volume (OBV) dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Theo dõi dòng tiền thông minh tích lũy/phân phối theo thời gian thực.
    """
    df = df.copy()
    
    # 1. Xử lý an toàn tham số window
    if isinstance(sma_window, str):
        num_str = re.sub(r'\D', '', sma_window)
        sma_window = int(num_str) if num_str else 20
    else:
        sma_window = int(sma_window)

    pd_time_frame = time_frame.replace('m', 'min') if time_frame.endswith('m') else time_frame
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    c_obv = f'obv_{time_frame}'
    c_obv_sma = f'obv_sma_{sma_window}_{time_frame}'

    # 2. Tính toán Logic MTF OBV
    if time_frame == '1m':
        # Tính trực tiếp trên khung 1 phút
        delta = df['close'].diff().fillna(0)
        direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
        
        df[c_obv] = (direction * df['volume']).cumsum()
        df[c_obv_sma] = df[c_obv].rolling(window=sma_window).mean()
        
    else:
        # BƯỚC A: Lấy dữ liệu nến đã đóng HTF
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        htf_vol = df['volume'].resample(pd_time_frame, label='left', closed='left').sum().dropna()

        # Dịch mốc thời gian sang tương lai 1 nến
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)
        htf_vol.index = htf_vol.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính OBV trên nến đã đóng (Mốc tĩnh)
        htf_delta = htf_close.diff().fillna(0)
        htf_dir = np.where(htf_delta > 0, 1, np.where(htf_delta < 0, -1, 0))
        htf_obv_closed = (htf_dir * htf_vol).cumsum()

        # Để tính SMA Live của OBV, cần tổng OBV của (N-1) nến trước
        htf_obv_sum_n1 = htf_obv_closed.rolling(window=sma_window - 1).sum()

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        obv_ffill = htf_obv_closed.reindex(df.index, method='ffill').fillna(0)
        obv_sum_n1_ffill = htf_obv_sum_n1.reindex(df.index, method='ffill').fillna(0)
        prev_close_ffill = htf_close.reindex(df.index, method='ffill')

        # BƯỚC D: Dựng Live Volume cho nến HTF
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_vol = df['volume'].groupby(grouper).cumsum()

        # BƯỚC E: Tính LIVE OBV và LIVE OBV SMA
        # Xác định chiều của nến Live: So sánh giá hiện tại với giá đóng cửa nến trước
        live_dir = np.where(
            df['close'] > prev_close_ffill, 1, 
            np.where(df['close'] < prev_close_ffill, -1, 0)
        )
        
        # OBV Live = OBV tĩnh + (Chiều nến Live * Volume tích lũy Live)
        live_obv = obv_ffill + (live_dir * live_vol)
        
        # SMA Live = (Tổng OBV tĩnh N-1 nến + OBV Live) / N
        live_obv_sma = (obv_sum_n1_ffill + live_obv) / sma_window

        df[c_obv] = live_obv
        df[c_obv_sma] = live_obv_sma

    # 3. Phân loại trạng thái dòng tiền (Signals)
    
    # Tín hiệu Crossover: OBV cắt lên đường SMA của chính nó (Báo hiệu dòng tiền vào mạnh)
    df[f'obv_bullish_{time_frame}'] = df[c_obv] > df[c_obv_sma]
    
    # Tính gia tốc của OBV (Độ dốc của OBV so với 5 nến trước đó)
    # Rất hữu ích để bắt những cú "Pump" volume bất ngờ
    obv_roc = df[c_obv].diff(5) 
    df[f'obv_surge_{time_frame}'] = obv_roc > obv_roc.rolling(50).std() * 2

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_vwap(df, time_frame, window=20):
    """
    Phân tích Rolling VWAP (MVWAP) dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    Kết hợp hoàn hảo giữa Giá điển hình (Typical Price) và Khối lượng (Volume).
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

    c_vwap = f'vwap_{time_frame}'

    # 2. Tính toán Logic MTF Rolling VWAP
    if time_frame == '1m':
        # Tính toán trực tiếp trên khung 1 phút
        # Typical Price (TP) = (High + Low + Close) / 3
        tp = (df['high'] + df['low'] + df['close']) / 3
        pv = tp * df['volume'] # Price * Volume
        
        # Rolling VWAP = Tổng(PV) / Tổng(Volume)
        df[c_vwap] = pv.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        
    else:
        # BƯỚC A: Lấy nến đã đóng HTF
        htf_high = df['high'].resample(pd_time_frame, label='left', closed='left').max().dropna()
        htf_low = df['low'].resample(pd_time_frame, label='left', closed='left').min().dropna()
        htf_close = df['close'].resample(pd_time_frame, label='left', closed='left').last().dropna()
        htf_vol = df['volume'].resample(pd_time_frame, label='left', closed='left').sum().dropna()

        # Dịch mốc thời gian sang tương lai 1 nến
        htf_high.index = htf_high.index + pd.to_timedelta(pd_time_frame)
        htf_low.index = htf_low.index + pd.to_timedelta(pd_time_frame)
        htf_close.index = htf_close.index + pd.to_timedelta(pd_time_frame)
        htf_vol.index = htf_vol.index + pd.to_timedelta(pd_time_frame)

        # BƯỚC B: Tính Typical Price và PV cho nến đã đóng
        htf_tp_closed = (htf_high + htf_low + htf_close) / 3
        htf_pv_closed = htf_tp_closed * htf_vol

        # Để tính Live VWAP, cần tổng PV và tổng Volume của (N-1) nến trước
        htf_pv_sum_n1 = htf_pv_closed.rolling(window=window - 1).sum()
        htf_vol_sum_n1 = htf_vol.rolling(window=window - 1).sum()

        # BƯỚC C: Rải "Mỏ neo" (Forward Fill) về khung 1 phút
        pv_sum_n1_ffill = htf_pv_sum_n1.reindex(df.index, method='ffill').fillna(0)
        vol_sum_n1_ffill = htf_vol_sum_n1.reindex(df.index, method='ffill').fillna(0)

        # BƯỚC D: Dựng nến Live và Volume Live
        grouper = pd.Grouper(freq=pd_time_frame, label='left', closed='left')
        live_high = df['high'].groupby(grouper).cummax()
        live_low = df['low'].groupby(grouper).cummin()
        live_close = df['close']
        live_vol = df['volume'].groupby(grouper).cumsum()

        # BƯỚC E: Tính PV Live và VWAP Live
        live_tp = (live_high + live_low + live_close) / 3
        live_pv = live_tp * live_vol

        # VWAP Live = (Tổng PV tĩnh N-1 + PV Live) / (Tổng Volume tĩnh N-1 + Volume Live)
        live_vwap = (pv_sum_n1_ffill + live_pv) / (vol_sum_n1_ffill + live_vol)
        
        df[c_vwap] = live_vwap

    # 3. Phân loại trạng thái (Signals)
    
    # Giá đóng cửa nằm trên hay dưới VWAP (Định hướng xu hướng dòng tiền)
    df[f'vwap_bullish_{time_frame}'] = df['close'] > df[c_vwap]
    
    # Tính khoảng cách % từ giá hiện tại đến VWAP để đo độ lệch (Mean Reversion)
    # Rất hữu ích để đánh các cú Pullback (hồi về VWAP)
    distance_pct = (df['close'] - df[c_vwap]).abs() / df[c_vwap]
    df[f'vwap_overextended_{time_frame}'] = distance_pct > 0.03 # Dãn quá 3% so với VWAP

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df

def pt_volume_profile(df, time_frame='1D', price_step=10):
    """
    Phân tích Session Volume Profile dạng Vectorized Hỗ trợ Đa Khung Thời Gian (MTF).
    - time_frame: Độ dài của phiên (Khuyên dùng '1D' - Daily hoặc '4h').
    - price_step: Bước giá để gom nhóm (Binning). Ví dụ: BTC dùng 10 hoặc 50, Vàng dùng 1, Cổ phiếu dùng 0.1.
    """
    df = df.copy()
    
    # Xử lý Tên khung thời gian cho Pandas
    pd_time_frame = time_frame.replace('m', 'min').replace('D', 'd') 
    
    index_reset_needed = False
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        index_reset_needed = True

    # Khởi tạo tên cột
    c_poc = f'vp_poc_{time_frame}'
    c_vah = f'vp_vah_{time_frame}'
    c_val = f'vp_val_{time_frame}'

    # BƯỚC 1: Xác định Phiên (Session) cho từng cây nến 1 phút
    # Ví dụ: Mọi cây nến trong ngày 2023-10-01 sẽ có chung session là '2023-10-01'
    sessions = df.index.floor(pd_time_frame)
    
    # BƯỚC 2: Gom nhóm giá (Binning)
    # Thay vì dùng giá chính xác tới từng số thập phân, ta làm tròn giá về các "Vùng giá" (Price Zones)
    tp = (df['high'] + df['low'] + df['close']) / 3
    price_zones = (tp // price_step) * price_step

    # Tạo DataFrame tạm để xử lý Vectorized Groupby
    temp_df = pd.DataFrame({
        'session': sessions,
        'price_zone': price_zones,
        'volume': df['volume']
    })

    # BƯỚC 3: Tính tổng Volume cho từng Vùng giá trong từng Phiên
    vol_by_price = temp_df.groupby(['session', 'price_zone'])['volume'].sum().reset_index()

    # BƯỚC 4: Tìm POC (Point of Control) - Vùng giá có Volume lớn nhất mỗi phiên
    # idxmax() tìm index của dòng có volume lớn nhất trong từng session
    idx_poc = vol_by_price.groupby('session')['volume'].idxmax()
    poc_data = vol_by_price.loc[idx_poc].set_index('session')[['price_zone']]
    poc_data.rename(columns={'price_zone': 'POC'}, inplace=True)

    # BƯỚC 5: Tìm VAH và VAL (Value Area - Vùng chứa 70% khối lượng)
    # Sắp xếp theo session và mức giá để tính tích lũy
    vol_by_price.sort_values(['session', 'price_zone'], inplace=True)
    
    # Tính tổng volume của toàn bộ phiên
    total_vol_per_session = vol_by_price.groupby('session')['volume'].transform('sum')
    
    # Sắp xếp volume từ cao xuống thấp trong từng phiên để tìm vùng 70%
    vol_sorted = vol_by_price.sort_values(['session', 'volume'], ascending=[True, False])
    vol_sorted['cum_vol_pct'] = vol_sorted.groupby('session')['volume'].cumsum() / total_vol_per_session
    
    # Lọc ra các vùng giá nằm trong top 70% khối lượng
    value_area = vol_sorted[vol_sorted['cum_vol_pct'] <= 0.70]
    
    # VAH là giá cao nhất, VAL là giá thấp nhất trong vùng Value Area
    vah_val_data = value_area.groupby('session')['price_zone'].agg(VAH='max', VAL='min')

    # BƯỚC 6: Tổng hợp dữ liệu Profile của các phiên ĐÃ ĐÓNG
    profile_data = poc_data.join(vah_val_data)
    
    # QUAN TRỌNG: Shift 1 phiên để dùng Profile của phiên trước áp dụng cho phiên hiện tại (Live)
    # VD: Profile của Thứ 2 sẽ được làm cản tĩnh cho toàn bộ ngày Thứ 3
    profile_data_shifted = profile_data.shift(1)

    # BƯỚC 7: Rải (Map) dữ liệu về lại DataFrame 1 phút ban đầu
    # Map dựa trên session của từng dòng
    df[c_poc] = sessions.map(profile_data_shifted['POC'])
    df[c_vah] = sessions.map(profile_data_shifted['VAH'])
    df[c_val] = sessions.map(profile_data_shifted['VAL'])

    # Điền khuyết (ffill) cho những nến chưa có dữ liệu map (ví dụ đầu chuỗi)
    df[c_poc] = df[c_poc].ffill()
    df[c_vah] = df[c_vah].ffill()
    df[c_val] = df[c_val].ffill()

    # BƯỚC 8: Phân loại tín hiệu
    # Xác định giá đang nằm ở đâu so với vùng giá trị (Value Area) của phiên trước
    df[f'vp_above_vah_{time_frame}'] = df['close'] > df[c_vah] # Giá phá lên trên VAH (Báo hiệu Tăng mạnh)
    df[f'vp_below_val_{time_frame}'] = df['close'] < df[c_val] # Giá phá xuống dưới VAL (Báo hiệu Giảm mạnh)
    
    # Tín hiệu "Từ chối POC": Nếu giá chạm về POC của phiên trước và bật lên
    distance_to_poc = (df['close'] - df[c_poc]).abs() / df[c_poc]
    df[f'vp_near_poc_{time_frame}'] = distance_to_poc < 0.005 # Giá đang nằm rất sát POC (cách < 0.5%)

    if index_reset_needed:
        df.reset_index(inplace=True)
        
    return df
