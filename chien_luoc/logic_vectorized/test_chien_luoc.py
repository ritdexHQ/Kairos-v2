"""
chien_luoc/logic_vectorized/test_chien_luoc.py – Unit tests chiến lược vectorized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chạy từ thư mục gốc dự án:
    pytest chien_luoc/logic_vectorized/test_chien_luoc.py -v
"""
import numpy as np
import pandas as pd
import pytest


# ─── Tạo dữ liệu tổng hợp ───────────────────────────────────────────────────

def tao_du_lieu_ohlcv(n: int = 3000, start: str = '2024-01-01',
                      freq: str = '1min', gia_bat_dau: float = 50000.0,
                      seed: int = 42) -> pd.DataFrame:
    """Tạo OHLCV synthetic dạng random walk. Đảm bảo H >= O/C và L <= O/C."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.0008, size=n)
    prices  = gia_bat_dau * np.exp(np.cumsum(log_ret))

    open_  = prices
    close_ = prices * np.exp(rng.normal(0, 0.0004, size=n))
    spread = prices * rng.uniform(0.0002, 0.003, size=n)
    high_  = np.maximum(open_, close_) + spread * rng.uniform(0.3, 1.0, size=n)
    low_   = np.minimum(open_, close_) - spread * rng.uniform(0.3, 1.0, size=n)
    volume = rng.uniform(200, 8000, size=n) * (1 + rng.exponential(0.3, size=n))

    return pd.DataFrame({
        'timestamp': pd.date_range(start=start, periods=n, freq=freq),
        'open':   open_,
        'high':   high_,
        'low':    low_,
        'close':  close_,
        'volume': volume,
    })


def resample_ohlcv(df_1m: pd.DataFrame, freq_str: str) -> pd.DataFrame:
    """Resample df_1m sang khung thời gian cao hơn."""
    df = df_1m.copy().set_index('timestamp')
    return (
        df.resample(freq_str).agg({
            'open': 'first', 'high': 'max',
            'low': 'min',   'close': 'last', 'volume': 'sum',
        })
        .dropna()
        .reset_index()
    )


@pytest.fixture(scope='module')
def bo_du_lieu():
    """Fixture module-level: tạo 1 lần, tái dùng cho tất cả tests."""
    df_1m = tao_du_lieu_ohlcv(n=3000, freq='1min')
    return dict(
        df_1m  = df_1m,
        df_3m  = resample_ohlcv(df_1m, '3min'),
        df_5m  = resample_ohlcv(df_1m, '5min'),
        df_15m = resample_ohlcv(df_1m, '15min'),
        df_30m = resample_ohlcv(df_1m, '30min'),
        df_1h  = resample_ohlcv(df_1m, '1h'),
        df_4h  = resample_ohlcv(df_1m, '4h'),
        df_1d  = resample_ohlcv(df_1m, '1D'),
    )


# ─── Helpers kiểm tra ────────────────────────────────────────────────────────

def _kiem_tra_cau_truc(df, ten: str):
    """Kiểm tra DataFrame trả về đúng cấu trúc chuẩn."""
    assert df is not None,                         f"[{ten}] trả về None"
    assert isinstance(df, pd.DataFrame),            f"[{ten}] không phải DataFrame"
    assert 'signal'       in df.columns,            f"[{ten}] thiếu cột 'signal'"
    assert 'entry_signal' in df.columns,            f"[{ten}] thiếu cột 'entry_signal'"
    assert len(df) > 0,                             f"[{ten}] DataFrame rỗng"
    assert not df['signal'].isna().all(),           f"[{ten}] cột signal toàn NaN"
    valid = set(df['signal'].dropna().astype(int).unique())
    assert valid.issubset({-1, 0, 1}), \
        f"[{ten}] signal ngoài {{-1,0,1}}: {valid}"


# ─── Tests 5 chiến lược ──────────────────────────────────────────────────────

def test_breakout_cau_truc(bo_du_lieu):
    from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_breakout import chien_luoc_breakout
    df = chien_luoc_breakout(**bo_du_lieu)
    _kiem_tra_cau_truc(df, 'Breakout')


def test_squeeze_cau_truc(bo_du_lieu):
    from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_squeeze import chien_luoc_squeeze
    df = chien_luoc_squeeze(**bo_du_lieu)
    _kiem_tra_cau_truc(df, 'Squeeze')


def test_trend_following_cau_truc(bo_du_lieu):
    from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_theo_trend_following import chien_luoc_theo_trend_following
    df = chien_luoc_theo_trend_following(**bo_du_lieu)
    _kiem_tra_cau_truc(df, 'TrendFollowing')


def test_mean_reversion_cau_truc(bo_du_lieu):
    from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_mean_reversion import chien_luoc_mean_reversion
    df = chien_luoc_mean_reversion(**bo_du_lieu)
    _kiem_tra_cau_truc(df, 'MeanReversion')


def test_scalping_cau_truc(bo_du_lieu):
    from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_scalping import chien_luoc_scalping
    df = chien_luoc_scalping(**bo_du_lieu)
    _kiem_tra_cau_truc(df, 'Scalping')


# ─── Tests vi_the vectorized ─────────────────────────────────────────────────

def test_vi_the_them_du_cot(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.vi_the import pt_vi_the
    df_out = pt_vi_the(bo_du_lieu['df_1m'].copy(), '1m')
    for col in ['vol_delta_1m', 'cvd_1m', 'cvd_bull_1m',
                'buyer_pressure_1m', 'vi_the_signal_1m']:
        assert col in df_out.columns, f"pt_vi_the thiếu cột '{col}'"


def test_vi_the_signal_hop_le(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.vi_the import pt_vi_the
    df_out = pt_vi_the(bo_du_lieu['df_1m'].copy(), '1m')
    valid  = set(df_out['vi_the_signal_1m'].dropna().astype(int).unique())
    assert valid.issubset({-1, 0, 1}), f"vi_the_signal có giá trị lạ: {valid}"


def test_vi_the_khong_sua_df_goc(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.vi_the import pt_vi_the
    df_orig = bo_du_lieu['df_1m']
    cols_truoc = set(df_orig.columns)
    pt_vi_the(df_orig, '1m')
    assert set(df_orig.columns) == cols_truoc, "pt_vi_the đã mutate df gốc"


def test_vi_the_khac_nhau_theo_tf(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.vi_the import pt_vi_the
    df_1m_out = pt_vi_the(bo_du_lieu['df_1m'].copy(), '1m')
    df_1h_out = pt_vi_the(bo_du_lieu['df_1h'].copy(), '1h')
    assert 'cvd_1m' in df_1m_out.columns
    assert 'cvd_1h' in df_1h_out.columns
    assert 'cvd_1m' not in df_1h_out.columns, "Tên cột 1h không được chứa '_1m'"


# ─── Tests chu_ky ────────────────────────────────────────────────────────────

def test_phien_giao_dich_them_cot(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.chu_ky import pt_phien_giao_dich
    df_out = pt_phien_giao_dich(bo_du_lieu['df_1m'].copy(), '1m')
    for col in ['phien_1m', 'is_overlap_1m', 'is_high_vol_session_1m',
                'is_funding_hour_1m', 'session_weight_1m']:
        assert col in df_out.columns, f"pt_phien_giao_dich thiếu cột '{col}'"


def test_phien_giao_dich_gia_tri_hop_le(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.chu_ky import pt_phien_giao_dich
    df_out = pt_phien_giao_dich(bo_du_lieu['df_1m'].copy(), '1m')
    gia_tri_phien = set(df_out['phien_1m'].unique())
    assert gia_tri_phien.issubset({'OVERLAP', 'NY', 'LONDON', 'ASIAN', 'OFF_PEAK'})
    assert df_out['session_weight_1m'].between(0, 3).all()


def test_session_range_them_cot(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.chu_ky import pt_session_range
    df_out = pt_session_range(bo_du_lieu['df_1m'].copy(), '1m')
    for col in ['session_high_1m', 'session_low_1m', 'session_range_pct_1m']:
        assert col in df_out.columns, f"pt_session_range thiếu cột '{col}'"


def test_session_range_high_gte_low(bo_du_lieu):
    from chien_luoc.logic_vectorized.phan_tich_ky_thuat.chu_ky import pt_session_range
    df_out = pt_session_range(bo_du_lieu['df_1m'].copy(), '1m')
    assert (df_out['session_high_1m'] >= df_out['session_low_1m']).all(), \
        "session_high phải >= session_low mọi lúc"


# ─── Tests bar_to_bar vi_the ─────────────────────────────────────────────────

def test_bar_vi_the_snapshot_none():
    from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.vi_the import pt_vi_the
    result = pt_vi_the(None)
    assert result['diem']      == 0
    assert result['trang_thai'] == 'TRUNG_TINH'


def test_bar_vi_the_snapshot_bullish():
    from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.vi_the import pt_vi_the
    snapshot = {
        'funding_rate': -0.0008,    # FR âm → short nhiều → bullish
        'buy_vol': 800, 'sell_vol': 200, 'delta': 600,
        'imbalance': 0.35, 'bid_total': 500, 'ask_total': 200,
        'liq_long': 0, 'liq_short': 0,
    }
    result = pt_vi_the(snapshot)
    assert result['diem'] > 0,              "FR âm + CVD mua + bid >> ask phải BULLISH"
    assert result['trang_thai'] == 'BULLISH'


def test_bar_vi_the_snapshot_bearish():
    from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.vi_the import pt_vi_the
    snapshot = {
        'funding_rate': 0.002,      # FR cực cao → overlong → bearish
        'buy_vol': 100, 'sell_vol': 900, 'delta': -800,
        'imbalance': -0.40, 'bid_total': 100, 'ask_total': 500,
        'liq_long': 150, 'liq_short': 20,
    }
    result = pt_vi_the(snapshot)
    assert result['diem'] < 0,              "FR cao + CVD bán + ask >> bid phải BEARISH"
    assert result['trang_thai'] == 'BEARISH'


def test_bar_funding_rate_trung_tinh():
    from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.vi_the import pt_funding_rate
    result = pt_funding_rate({'funding_rate': 0.0001})
    assert result['trang_thai'] == 'TRUNG_TINH'


def test_bar_liquidation_short_squeeze():
    from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.vi_the import pt_liquidation
    result = pt_liquidation({'liq_long': 10, 'liq_short': 200})
    assert result['trang_thai'] == 'SHORT_SQUEEZE'


# ─── Test tong_hop_tin_hieu ───────────────────────────────────────────────────

def test_tong_hop_tin_hieu_co_cau_truc_chuan(bo_du_lieu):
    from chien_luoc.logic_vectorized.quan_ly_chien_luoc import tong_hop_tin_hieu
    d = bo_du_lieu
    df_out = tong_hop_tin_hieu(
        d['df_1m'], d['df_3m'], d['df_5m'], d['df_15m'],
        d['df_30m'], d['df_1h'], d['df_4h'], d['df_1d'],
    )
    assert df_out is not None
    for col in ['signal', 'entry_signal', 'trade_allowed']:
        assert col in df_out.columns, f"tong_hop_tin_hieu thiếu cột '{col}'"
    valid = set(df_out['signal'].dropna().astype(int).unique())
    assert valid.issubset({-1, 0, 1})


if __name__ == '__main__':
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'])
