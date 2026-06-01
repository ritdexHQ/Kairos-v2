"""
chien_luoc/logic_vectorized/chien_luoc/chien_luoc_scalping.py – Scalping vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logic: RSI extreme + giá chạm biên Bollinger + volume xác nhận trên khung ngắn.
  • Tín hiệu chính: 1M + 3M + 5M
  • Lọc bằng 15M: không vào ngược trend mạnh (ADX > 35)
Trả về DataFrame với cột `signal` và `entry_signal`.
"""
import numpy as np
from utils.ham_tien_ich import gop_va_dong_bo_data
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.dong_luong_dao_chieu import pt_rsi
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.bien_dong import pt_bollinger_squeeze
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.khoi_luong import pt_volume
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.xu_huong import pt_adx, pt_ema_trend


def chien_luoc_scalping(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    df_1m_calc  = pt_volume(pt_bollinger_squeeze(pt_rsi(df_1m, '1m'),  '1m'),  '1m')
    df_3m_calc  = pt_volume(pt_bollinger_squeeze(pt_rsi(df_3m, '3m'),  '3m'),  '3m')
    df_5m_calc  = pt_volume(pt_bollinger_squeeze(pt_rsi(df_5m, '5m'),  '5m'),  '5m')
    df_15m_calc = pt_adx(pt_ema_trend(df_15m, '15m'), '15m')

    du_lieu = {
        '1m':  df_1m_calc,
        '3m':  df_3m_calc,
        '5m':  df_5m_calc,
        '15m': df_15m_calc,
    }
    df = gop_va_dong_bo_data(du_lieu)

    df['buy_score']  = 0
    df['sell_score'] = 0

    for tf, weight in [('1m', 4), ('3m', 3), ('5m', 2)]:
        col_rsi = f'rsi_{tf}'
        col_vol = f'vol_tang_{tf}'
        col_low = f'bb_lower_{tf}'
        col_up  = f'bb_upper_{tf}'

        if col_rsi not in df.columns:
            continue

        vol_ok   = df[col_vol] if col_vol in df.columns else True
        near_low = (df['close'] <= df[col_low] * 1.002) if col_low in df.columns else False
        near_up  = (df['close'] >= df[col_up]  * 0.998) if col_up  in df.columns else False

        dk_long  = (df[col_rsi] < 35) & near_low & vol_ok
        dk_short = (df[col_rsi] > 65) & near_up  & vol_ok

        df['buy_score']  += dk_long.astype(int)  * weight
        df['sell_score'] += dk_short.astype(int) * weight

    # Lọc: không scalp ngược trend mạnh trên 15M
    if 'adx_15m' in df.columns and 'is_trend_15m' in df.columns:
        trend_strong = df['adx_15m'] > 35
        df.loc[trend_strong & (df['is_trend_15m'] == 'UP'),   'buy_score']  = 0
        df.loc[trend_strong & (df['is_trend_15m'] == 'DOWN'), 'sell_score'] = 0

    NGUONG = 6
    df['signal'] = np.where(df['buy_score']  >= NGUONG,  1,
                   np.where(df['sell_score'] >= NGUONG, -1, 0))

    df['entry_signal'] = df['signal'].diff().fillna(0)
    return df
