"""
chien_luoc/logic_vectorized/chien_luoc/chien_luoc_squeeze.py – Squeeze vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logic: BB nén chặt (bandwidth < trung bình) → tích lũy năng lượng → bứt phá.
  • Phát hiện squeeze trên 15M + 30M
  • Xác định chiều bứt phá bằng breakout + EMA
  • Volume tăng = xác nhận lực bứt phá thật
Trả về DataFrame với cột `signal` (1=BUY, -1=SELL, 0=NONE) và `entry_signal`.
"""
import numpy as np
from utils.ham_tien_ich import gop_va_dong_bo_data
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.bien_dong import pt_bollinger_squeeze
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.cau_truc_gia import pt_breakout
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.khoi_luong import pt_volume
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.xu_huong import pt_ema_trend


def chien_luoc_squeeze(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    df_15m_calc = pt_volume(pt_breakout(pt_bollinger_squeeze(pt_ema_trend(df_15m, '15m'), '15m'), '15m'), '15m')
    df_30m_calc = pt_volume(pt_breakout(pt_bollinger_squeeze(pt_ema_trend(df_30m, '30m'), '30m'), '30m'), '30m')
    df_1h_calc  = pt_ema_trend(df_1h, '1h')

    du_lieu = {
        '1m':  df_1m,
        '15m': df_15m_calc,
        '30m': df_30m_calc,
        '1h':  df_1h_calc,
    }
    df = gop_va_dong_bo_data(du_lieu)

    df['buy_score']  = 0
    df['sell_score'] = 0

    for tf, weight in [('15m', 5), ('30m', 4)]:
        col_bb_status = f'bb_status_{tf}'
        col_bb_level  = f'bb_muc_do_{tf}'
        col_brk       = f'breakout_{tf}'
        col_ema       = f'is_trend_{tf}'
        col_vol       = f'vol_tang_{tf}'

        if col_bb_status not in df.columns:
            continue

        is_squeeze   = (df[col_bb_status] == 'BOP')
        is_tight     = (df[col_bb_level]  == 'CHAT') if col_bb_level in df.columns else False
        # Squeeze chặt hơn → nhân đôi trọng số
        he_so = np.where(is_tight, 2, 1)

        vol_ok = df[col_vol] if col_vol in df.columns else True
        ema_up = (df[col_ema] == 'UP')   if col_ema in df.columns else True
        ema_dn = (df[col_ema] == 'DOWN') if col_ema in df.columns else True

        brk_up   = (df[col_brk] == 'BREAK_OUT')  if col_brk in df.columns else False
        brk_down = (df[col_brk] == 'BREAK_DOWN') if col_brk in df.columns else False

        df['buy_score']  += (is_squeeze & brk_up   & ema_up & vol_ok).astype(int) * weight * he_so
        df['sell_score'] += (is_squeeze & brk_down & ema_dn & vol_ok).astype(int) * weight * he_so

    # Xác nhận thêm từ 1H EMA
    if 'is_trend_1h' in df.columns:
        df.loc[df['is_trend_1h'] == 'DOWN', 'buy_score']  = df.loc[df['is_trend_1h'] == 'DOWN', 'buy_score']  // 2
        df.loc[df['is_trend_1h'] == 'UP',   'sell_score'] = df.loc[df['is_trend_1h'] == 'UP',   'sell_score'] // 2

    NGUONG = 5
    df['signal'] = np.where(df['buy_score']  >= NGUONG,  1,
                   np.where(df['sell_score'] >= NGUONG, -1, 0))

    df['entry_signal'] = df['signal'].diff().fillna(0)
    return df