"""
chien_luoc/logic_vectorized/chien_luoc/chien_luoc_theo_trend_following.py – Trend Following vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logic: Vào lệnh khi các khung lớn đồng thuận cùng chiều xu hướng.
  • EMA alignment: 1D + 4H + 1H + 15M đều cùng chiều
  • ADX > 20 trên 1H = xu hướng đủ mạnh (không phải sideways)
  • ATR cao = biến động có lực
  • Khung dài trọng số cao, ngắn dùng để timing
Trả về DataFrame với cột `signal` (1=BUY, -1=SELL, 0=NONE) và `entry_signal`.
"""
import numpy as np
from utils.ham_tien_ich import gop_va_dong_bo_data
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.xu_huong import pt_ema_trend, pt_adx
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.bien_dong import pt_atr


def chien_luoc_theo_trend_following(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    df_1d_calc  = pt_ema_trend(df_1d,  '1d')
    df_4h_calc  = pt_ema_trend(df_4h,  '4h')
    df_1h_calc  = pt_adx(pt_atr(pt_ema_trend(df_1h, '1h'), '1h'), '1h')
    df_15m_calc = pt_ema_trend(df_15m, '15m')

    du_lieu = {
        '1m':  df_1m,
        '15m': df_15m_calc,
        '1h':  df_1h_calc,
        '4h':  df_4h_calc,
        '1d':  df_1d_calc,
    }
    df = gop_va_dong_bo_data(du_lieu)

    df['buy_score']  = 0
    df['sell_score'] = 0

    # Trọng số: khung dài quyết định chiều, ngắn xác nhận timing
    for tf, weight in [('1d', 5), ('4h', 4), ('1h', 3), ('15m', 2)]:
        col_ema = f'is_trend_{tf}'
        col_str = f'trend_strong_{tf}'
        if col_ema not in df.columns:
            continue
        is_strong = df[col_str] if col_str in df.columns else True
        df['buy_score']  += ((df[col_ema] == 'UP')   & is_strong).astype(int) * weight
        df['sell_score'] += ((df[col_ema] == 'DOWN') & is_strong).astype(int) * weight

    # Bộ lọc: ADX 1H > 20 = có xu hướng thật (loại sideways)
    if 'adx_1h' in df.columns:
        adx_weak = df['adx_1h'] < 20
        df.loc[adx_weak, 'buy_score']  = 0
        df.loc[adx_weak, 'sell_score'] = 0

    # Bộ lọc: ATR 1H phải ở mức cao để đảm bảo có biến động đủ lấy lời
    if 'atr_status_1h' in df.columns:
        atr_low = df['atr_status_1h'] == 'BIEN_DONG_THAP'
        df.loc[atr_low, 'buy_score']  = df.loc[atr_low, 'buy_score']  // 2
        df.loc[atr_low, 'sell_score'] = df.loc[atr_low, 'sell_score'] // 2

    # Yêu cầu ít nhất 3/4 khung đồng thuận (tổng điểm tối thiểu = 1D+4H+1H = 12)
    NGUONG = 12
    df['signal'] = np.where(df['buy_score']  >= NGUONG,  1,
                   np.where(df['sell_score'] >= NGUONG, -1, 0))

    df['entry_signal'] = df['signal'].diff().fillna(0)
    return df