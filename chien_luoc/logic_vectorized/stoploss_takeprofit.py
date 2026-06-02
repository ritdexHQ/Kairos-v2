"""
chien_luoc/logic_vectorized/stoploss_takeprofit.py – SL/TP vectorized theo ATR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thêm cột sl_pct và tp_pct vào DataFrame – backtest engine dùng để tính giá đóng.

  he_so_sl = base_sl / (atr / atr_mean)   clamp [1.0, 5.0]
  sl_pct   = atr * he_so_sl / close       (% so với giá vào)
  tp_pct   = sl_pct * rr                  (Risk:Reward = 2:1 mặc định)
"""
import numpy as np
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.bien_dong import pt_atr


def them_sl_tp(df, time_frame='1h', base_sl=2.5, rr=2.0):
    """
    Tính SL/TP động theo ATR và thêm vào DataFrame.
    Trả về df với cột `sl_pct` và `tp_pct`.
    """
    df = pt_atr(df.copy(), time_frame)

    col_atr  = f'atr_{time_frame}'
    col_mean = f'atr_mean_{time_frame}'

    if col_atr not in df.columns:
        df['sl_pct'] = base_sl / 100
        df['tp_pct'] = base_sl * rr / 100
        return df

    atr      = df[col_atr].ffill().fillna(0)
    atr_mean = df[col_mean].ffill().fillna(atr)

    # vol_ratio > 1 = biến động cao → SL rộng hơn để không bị noise quét stop
    vol_ratio = (atr / (atr_mean + 1e-9)).clip(0.5, 3.0)
    he_so_sl  = (base_sl / vol_ratio).clip(1.0, 5.0)
    he_so_tp  = he_so_sl * rr

    close = df['close'].replace(0, np.nan).ffill()

    df['sl_pct'] = (atr * he_so_sl / close).clip(0.005, 0.15)  # 0.5% – 15%
    df['tp_pct'] = (atr * he_so_tp / close).clip(0.01,  0.30)  # 1%   – 30%

    return df
