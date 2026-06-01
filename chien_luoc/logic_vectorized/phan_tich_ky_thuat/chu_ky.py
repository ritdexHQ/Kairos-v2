"""THỜI GIAN & CHU KỲ (Time / Cycle) ⭐
👉 Khi nào thị trường hay phản ứng?
- Phiên Á – Âu – Mỹ
- Thời điểm ra tin
- Chu kỳ nến
- Session Range
📌 Dùng để:
- Tránh giao dịch lúc nhiễu
- Giảm leverage giờ rủi ro
- Bot futures rất cần """

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
# Giờ các phiên giao dịch (UTC)
# BTC/USDT Futures Binance có thanh khoản cao nhất London + NY
# ────────────────────────────────────────────────────────────────
# Asian session  : 00:00–08:00 UTC  (07:00–15:00 VN)
# London session : 07:00–16:00 UTC  (14:00–23:00 VN)
# New York session: 13:00–22:00 UTC (20:00–05:00 VN +1)
# London-NY overlap (highest vol): 13:00–16:00 UTC (20:00–23:00 VN)
# Funding rate settlement: 00:00, 08:00, 16:00 UTC (±15 phút biến động)
# Spread giãn (low liquidity): 22:00–00:00 UTC ≈ 05:00–07:00 VN

_ASIAN_START,  _ASIAN_END  = 0,  8
_LONDON_START, _LONDON_END = 7,  16
_NY_START,     _NY_END     = 13, 22
_OVERLAP_START, _OVERLAP_END = 13, 16   # London-NY overlap
_FUNDING_HOURS = {0, 8, 16}             # Funding rate settlement hours UTC


def _get_hour_series(df):
    """Trả về Series giờ UTC, hỗ trợ cả 'timestamp' column và DatetimeIndex."""
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp']).dt.hour
    return pd.to_datetime(df.index).hour


def _get_dayofweek_series(df):
    """Trả về Series thứ (0=T2, 6=CN), hỗ trợ cả 'timestamp' column và DatetimeIndex."""
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp']).dt.dayofweek
    return pd.to_datetime(df.index).dayofweek


# ────────────────────────────────────────────────────────────────
# Hàm cũ – giữ nguyên để backward-compatible
# ────────────────────────────────────────────────────────────────

def pt_kiem_tra_ngay(df):
    """
    Kiểm tra ngày trong tuần (vectorized).
    Thêm cột 'check_days': True (T2-CN crypto) / False nếu muốn lọc cuối tuần.
    Crypto giao dịch 24/7 nên mặc định True mọi ngày.
    """
    thu_so = _get_dayofweek_series(df)

    conditions = [
        (thu_so <= 4),   # Thứ 2 → Thứ 6
        (thu_so > 4),    # Thứ 7, Chủ Nhật
    ]
    choices = [True, True]   # Crypto 24/7 → cho phép cả cuối tuần

    df = df.copy()
    df['check_days'] = np.select(conditions, choices, default=True)
    return df


def pt_kiem_tra_gio(df):
    """
    Kiểm tra giờ giao dịch (vectorized).
    Thêm cột 'check_hours': False vào 05h sáng VN (22:00 UTC – spread giãn).
    """
    gio = _get_hour_series(df)
    df = df.copy()
    # 22:00 UTC ≈ 05:00 VN (UTC+7) – spread giãn mạnh khi đóng phiên Mỹ
    df['check_hours'] = ~gio.isin([22])
    return df


# ────────────────────────────────────────────────────────────────
# Hàm mới – phân tích phiên giao dịch đầy đủ
# ────────────────────────────────────────────────────────────────

def pt_phien_giao_dich(df, time_frame):
    """
    Phân loại mỗi nến theo phiên giao dịch chính.

    Cột được thêm
    -------------
    phien_{tf}              – tên phiên: 'OVERLAP' / 'NY' / 'LONDON' / 'ASIAN' / 'OFF_PEAK'
    is_overlap_{tf}         – bool, giờ London-NY overlap (thanh khoản cao nhất)
    is_high_vol_session_{tf}– bool, đang trong London hoặc NY (tránh Asian và off-peak)
    is_funding_hour_{tf}    – bool, ±15 phút quanh giờ settlement FR (biến động bất thường)
    session_weight_{tf}     – trọng số thanh khoản: 3 (Overlap) / 2 (London/NY) / 1 (Asian) / 0 (Off-peak)
    """
    tf  = time_frame.lower().replace(' ', '')
    df  = df.copy()
    gio = _get_hour_series(df)

    in_asian   = (gio >= _ASIAN_START)  & (gio < _ASIAN_END)
    in_london  = (gio >= _LONDON_START) & (gio < _LONDON_END)
    in_ny      = (gio >= _NY_START)     & (gio < _NY_END)
    in_overlap = (gio >= _OVERLAP_START) & (gio < _OVERLAP_END)

    # Phiên ưu tiên: Overlap > NY > London > Asian > Off-peak
    phien = np.select(
        [in_overlap,  in_ny & ~in_overlap,  in_london & ~in_ny,  in_asian & ~in_london],
        ['OVERLAP',   'NY',                 'LONDON',            'ASIAN'],
        default='OFF_PEAK'
    )
    df[f'phien_{tf}'] = phien

    df[f'is_overlap_{tf}']          = in_overlap
    df[f'is_high_vol_session_{tf}'] = in_london | in_ny   # cả hai phiên chính

    # Funding rate settlement (00:00, 08:00, 16:00 UTC) → biến động bất thường ±1h
    is_funding = gio.isin(_FUNDING_HOURS)
    # Cũng bắt giờ trước settlement 1 tiếng (các trader đóng/mở vị thế trước)
    is_pre_funding = gio.isin({h - 1 for h in _FUNDING_HOURS if h > 0} | {23})
    df[f'is_funding_hour_{tf}'] = is_funding | is_pre_funding

    weight_map = {'OVERLAP': 3, 'NY': 2, 'LONDON': 2, 'ASIAN': 1, 'OFF_PEAK': 0}
    df[f'session_weight_{tf}'] = pd.Series(phien).map(weight_map).values

    return df


def pt_session_range(df, time_frame):
    """
    Tính High/Low tích lũy của phiên giao dịch hiện tại.
    Dùng để xác định vùng support/resistance dạng Session Range.

    Yêu cầu: df phải có cột 'timestamp' hoặc DatetimeIndex với timezone UTC.

    Cột được thêm
    -------------
    session_high_{tf}      – High cao nhất từ đầu phiên hiện tại đến nến hiện tại
    session_low_{tf}       – Low thấp nhất từ đầu phiên hiện tại đến nến hiện tại
    session_range_pct_{tf} – (session_high - session_low) / session_low * 100 (% biến động phiên)
    """
    tf  = time_frame.lower().replace(' ', '')
    df  = df.copy()

    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
    else:
        ts = pd.to_datetime(df.index)

    # Xác định nhóm phiên: mỗi phiên bắt đầu tại 00:00, 07:00, 13:00 UTC
    # Ta dùng 8h block đơn giản: 00-08 / 08-16 / 16-24
    session_block = (ts.dt.hour // 8).values   # 0, 1, hoặc 2

    # Tạo key duy nhất cho mỗi phiên: date + block
    session_key = ts.dt.date.astype(str) + '_' + session_block.astype(str)
    df['_session_key'] = session_key

    # Rolling high/low trong cùng session_key (expanding)
    df[f'session_high_{tf}'] = df.groupby('_session_key')['high'].cummax()
    df[f'session_low_{tf}']  = df.groupby('_session_key')['low'].cummin()
    df[f'session_range_pct_{tf}'] = (
        (df[f'session_high_{tf}'] - df[f'session_low_{tf}'])
        / (df[f'session_low_{tf}'] + 1e-9) * 100
    )

    df.drop(columns=['_session_key'], inplace=True)
    return df
