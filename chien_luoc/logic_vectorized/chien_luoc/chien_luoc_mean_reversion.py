"""
chien_luoc/logic_vectorized/chien_luoc/chien_luoc_mean_reversion.py – Đảo chiều vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logic: Giá kiệt sức ở cực trị → đảo chiều về trung bình.
  • RSI quá mua (>70) / quá bán (<30) tại dải BB
  • Volume đột biến tại cực trị = xác nhận xả/gom
  • Khung 1H chính, 4H macro filter
Trả về DataFrame với cột `signal` (1=BUY, -1=SELL, 0=NONE) và `entry_signal`.
"""
import numpy as np
from utils.ham_tien_ich import gop_va_dong_bo_data
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.dong_luong_dao_chieu import pt_rsi
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.bien_dong import pt_bollinger_squeeze
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.khoi_luong import pt_volume
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.xu_huong import pt_ema_trend


def chien_luoc_mean_reversion(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    df_1h_calc  = pt_volume(pt_bollinger_squeeze(pt_rsi(df_1h,  '1h'),  '1h'),  '1h')
    df_15m_calc = pt_volume(pt_bollinger_squeeze(pt_rsi(df_15m, '15m'), '15m'), '15m')
    df_4h_calc  = pt_ema_trend(pt_rsi(df_4h, '4h'), '4h')

    du_lieu = {
        '1m':  df_1m,
        '15m': df_15m_calc,
        '1h':  df_1h_calc,
        '4h':  df_4h_calc,
    }
    df = gop_va_dong_bo_data(du_lieu)

    df['buy_score']  = 0
    df['sell_score'] = 0

    for tf, weight in [('1h', 5), ('15m', 3)]:
        col_rsi = f'rsi_{tf}'
        col_vol = f'vol_tang_manh_{tf}'
        col_low = f'bb_lower_{tf}'
        col_up  = f'bb_upper_{tf}'

        if col_rsi not in df.columns:
            continue

        vol_spike = df[col_vol]  if col_vol in df.columns else False
        below_bb  = (df['close'] < df[col_low]) if col_low in df.columns else False
        above_bb  = (df['close'] > df[col_up])  if col_up  in df.columns else False

        # BUY: RSI quá bán + dưới dải BB + volume đột biến (gom hàng)
        df['buy_score']  += ((df[col_rsi] < 30) & below_bb & vol_spike).astype(int) * weight
        # SELL: RSI quá mua + trên dải BB + volume đột biến (xả hàng)
        df['sell_score'] += ((df[col_rsi] > 70) & above_bb & vol_spike).astype(int) * weight

    # Macro filter 4H: giảm điểm nếu trend 4H vẫn còn mạnh
    if 'is_trend_4h' in df.columns and 'rsi_4h' in df.columns:
        macro_up   = (df['is_trend_4h'] == 'UP')   & (df['rsi_4h'] > 60)
        macro_down = (df['is_trend_4h'] == 'DOWN') & (df['rsi_4h'] < 40)
        df.loc[macro_up,   'sell_score'] = df.loc[macro_up,   'sell_score'] // 2
        df.loc[macro_down, 'buy_score']  = df.loc[macro_down, 'buy_score']  // 2

    NGUONG = 5
    df['signal'] = np.where(df['buy_score']  >= NGUONG,  1,
                   np.where(df['sell_score'] >= NGUONG, -1, 0))

    df['entry_signal'] = df['signal'].diff().fillna(0)
    return df