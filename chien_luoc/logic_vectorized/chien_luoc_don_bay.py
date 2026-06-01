"""
chien_luoc/logic_vectorized/chien_luoc_don_bay.py – Đòn bẩy động vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tính cột `leverage` theo từng nến dựa trên biến động ATR 15M:
  • ATR cao (vol_ratio > 1.2) → giảm leverage để giới hạn rủi ro
  • ATR thấp (vol_ratio < 0.8) → tăng nhẹ leverage
  • Clamp [1, max_leverage] để không vượt ngưỡng cài đặt
"""
import numpy as np
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.bien_dong import pt_atr


def them_don_bay_dong(df, don_bay_goc=5, max_leverage=20):
    """
    Tính leverage động theo ATR 15M và thêm vào DataFrame.
    Trả về df với cột `leverage`.
    """
    df = pt_atr(df.copy(), '15m')

    col_atr  = 'atr_15m'
    col_mean = 'atr_mean_15m'

    if col_atr not in df.columns:
        df['leverage'] = don_bay_goc
        return df

    atr      = df[col_atr].fillna(method='ffill').fillna(0)
    atr_mean = df[col_mean].fillna(method='ffill').fillna(atr)

    vol_ratio = (atr / (atr_mean + 1e-9)).clip(0.3, 3.0)

    # Biến động cao → giảm leverage, thấp → tăng nhẹ
    leverage = (don_bay_goc / vol_ratio).round().clip(1, max_leverage).astype(int)

    df['leverage'] = leverage
    return df
