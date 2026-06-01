"""
chien_luoc/logic_vectorized/quan_ly_chien_luoc.py – Điều phối chiến lược vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chạy tất cả 5 chiến lược song song, sau đó dùng cột `regime` từ ML để chọn
tín hiệu phù hợp cho từng nến. Kết quả là cột `signal` duy nhất.

Ánh xạ regime → chiến lược (khớp với STRATEGY_MAP trong ml_predict.py):
  1 Nén_Chặt   → Squeeze
  2 Đầu_XH     → Breakout
  3 XH_Mạnh    → Trend_following
  4 Cao_Trào   → Mean_reversion
  5 Hồi_Quy    → Mean_reversion
  6 Nhiễu_Động → Scalping
  0, 7          → không trade
"""
import numpy as np
import pandas as pd
from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_breakout        import chien_luoc_breakout
from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_squeeze          import chien_luoc_squeeze
from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_theo_trend_following import chien_luoc_theo_trend_following
from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_mean_reversion   import chien_luoc_mean_reversion
from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_scalping         import chien_luoc_scalping
from chien_luoc.logic_vectorized.chien_luoc_trang_thai_thi_truong       import loc_trang_thai_thi_truong
from chien_luoc.logic_vectorized.stoploss_takeprofit                    import them_sl_tp
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.vi_the              import pt_vi_the
from chien_luoc.logic_vectorized.phan_tich_ky_thuat.chu_ky              import pt_phien_giao_dich


REGIME_TO_STRATEGY = {
    1: 'squeeze',
    2: 'breakout',
    3: 'trend',
    4: 'reversion',
    5: 'reversion',
    6: 'scalping',
}


def chay_tat_ca_chien_luoc(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    """Chạy tất cả chiến lược, trả về dict {tên: DataFrame có cột signal}."""
    return {
        'squeeze':   chien_luoc_squeeze(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d),
        'breakout':  chien_luoc_breakout(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d),
        'trend':     chien_luoc_theo_trend_following(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d),
        'reversion': chien_luoc_mean_reversion(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d),
        'scalping':  chien_luoc_scalping(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d),
    }


def tong_hop_tin_hieu(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d, df_regime=None):
    """
    Tổng hợp tín hiệu từ tất cả chiến lược dựa trên ML regime.
    df_regime: Polars/Pandas DataFrame với cột ['timestamp', 'regime'].
               Nếu None thì chạy tất cả chiến lược và OR kết quả.
    Trả về DataFrame 1m với cột: signal, entry_signal, sl_pct, tp_pct, trade_allowed.
    """
    ket_qua = chay_tat_ca_chien_luoc(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d)

    # Lấy base df (breakout luôn có đủ cột OHLCV)
    df_base = ket_qua['breakout'][['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy() \
              if 'timestamp' in ket_qua['breakout'].columns else \
              ket_qua['breakout'].reset_index()[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    df_base['signal'] = 0

    if df_regime is not None:
        # Chuyển Polars → Pandas nếu cần
        if hasattr(df_regime, 'to_pandas'):
            df_regime = df_regime.to_pandas()

        df_regime['timestamp'] = pd.to_datetime(df_regime['timestamp'])
        df_base['timestamp']   = pd.to_datetime(df_base['timestamp'])

        df_base = df_base.merge(
            df_regime[['timestamp', 'regime']].drop_duplicates('timestamp'),
            on='timestamp', how='left'
        )
        df_base['regime'] = df_base['regime'].fillna(0).astype(int)

        for regime_id, strategy_name in REGIME_TO_STRATEGY.items():
            mask   = df_base['regime'] == regime_id
            df_sig = ket_qua[strategy_name]

            if 'timestamp' not in df_sig.columns:
                df_sig = df_sig.reset_index()

            df_sig['timestamp'] = pd.to_datetime(df_sig['timestamp'])
            sig_map = df_sig.set_index('timestamp')['signal']

            df_base.loc[mask, 'signal'] = df_base.loc[mask, 'timestamp'].map(sig_map).fillna(0)
    else:
        # Không có ML: lấy union tín hiệu từ tất cả chiến lược (ưu tiên tín hiệu khác 0)
        for name, df_s in ket_qua.items():
            if 'timestamp' not in df_s.columns:
                df_s = df_s.reset_index()
            df_s['timestamp'] = pd.to_datetime(df_s['timestamp'])
            sig_map = df_s.set_index('timestamp')['signal']
            df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])
            partial = df_base['timestamp'].map(sig_map).fillna(0)
            # Giữ tín hiệu đầu tiên khác 0
            df_base['signal'] = df_base['signal'].where(df_base['signal'] != 0, partial)

    df_base['entry_signal'] = df_base['signal'].diff().fillna(0)

    # Thêm SL/TP động theo ATR 1H
    df_base = them_sl_tp(df_base, time_frame='1h')

    # Thêm bộ lọc thị trường
    df_base = loc_trang_thai_thi_truong(df_base)

    # Thêm thông tin vị thế & tâm lý (CVD proxy từ OHLCV)
    df_base = pt_vi_the(df_base, '1m')

    # Thêm thông tin phiên giao dịch (Asian / London / NY / Overlap)
    df_base = pt_phien_giao_dich(df_base, '1m')

    # Xóa tín hiệu trong giờ không được phép
    df_base.loc[~df_base['trade_allowed'], 'signal'] = 0
    df_base.loc[~df_base['trade_allowed'], 'entry_signal'] = 0

    return df_base
