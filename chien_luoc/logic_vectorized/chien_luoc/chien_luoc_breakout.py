"""
chien_luoc/logic_vectorized/chien_luoc/chien_luoc_breakout.py – Breakout vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logic: Giá phá vỡ đỉnh/đáy n nến + volume tăng + ADX xác nhận xu hướng.
  • Khung 1H là tín hiệu chính (weight cao nhất)
  • 15M và 4H dùng để xác nhận đa khung
  • score >= 3 mới ra lệnh (tối thiểu 2/3 khung đồng thuận)
Trả về DataFrame với cột `signal` (1=BUY, -1=SELL, 0=NONE) và `entry_signal`.
"""
import numpy as np
from utils.ham_tien_ich import gop_va_dong_bo_data
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.cau_truc_gia import pt_breakout
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.khoi_luong import pt_volume
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.xu_huong import pt_adx


def chien_luoc_breakout(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    # Áp dụng indicator trên từng khung thời gian
    df_1h_calc  = pt_adx(pt_volume(pt_breakout(df_1h,  '1h'),  '1h'),  '1h')
    df_15m_calc = pt_adx(pt_volume(pt_breakout(df_15m, '15m'), '15m'), '15m')
    df_4h_calc  = pt_adx(pt_volume(pt_breakout(df_4h,  '4h'),  '4h'),  '4h')

    # Merge về khung 1m làm gốc (forward-fill, không lookahead)
    du_lieu = {
        '1m':  df_1m,
        '15m': df_15m_calc,
        '1h':  df_1h_calc,
        '4h':  df_4h_calc,
    }
    df = gop_va_dong_bo_data(du_lieu)

    # --- Tính điểm đồng thuận breakout ---
    df['buy_score']  = 0
    df['sell_score'] = 0

    for tf, weight in [('1h', 3), ('15m', 2), ('4h', 2)]:
        col_brk = f'breakout_{tf}'
        col_vol = f'vol_tang_{tf}'
        col_adx = f'adx_{tf}'

        if col_brk in df.columns:
            brk_up   = (df[col_brk] == 'BREAK_OUT')
            brk_down = (df[col_brk] == 'BREAK_DOWN')
            vol_ok   = df[col_vol] if col_vol in df.columns else True
            adx_ok   = (df[col_adx] > 20) if col_adx in df.columns else True

            df['buy_score']  += (brk_up   & vol_ok & adx_ok).astype(int) * weight
            df['sell_score'] += (brk_down & vol_ok & adx_ok).astype(int) * weight

    NGUONG = 4  # tối thiểu 1H + 1 khung phụ đồng thuận
    df['signal'] = np.where(df['buy_score']  >= NGUONG,  1,
                   np.where(df['sell_score'] >= NGUONG, -1, 0))

    df['entry_signal'] = df['signal'].diff().fillna(0)
    return df
