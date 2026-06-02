"""
utils/kho_du_lieu.py – Data Warehouse (DuckDB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lưu kết quả từ TẤT CẢ các chế độ vận hành vào DuckDB để phân tích SQL.

Chế độ được hỗ trợ:
  backtest_vector   – vectorized_backtest.py   (batch)
  backtest_bar      – backtest_donluong.py     (batch)
  backtest_da_luong – backtest_daluong.py      (batch, collect ở main process)
  demo              – chay_demo.py             (streaming, từng lệnh)

Schema:
  backtest_run – metadata mỗi lần chạy (run_id, mode, thời gian, config)
  lenh         – lịch sử từng lệnh (chuc_nang, regime, chien_luoc, PnL...)

Queries phân tích sẵn:
  thong_ke_theo_gio()     – winrate và PnL theo giờ trong ngày
  thong_ke_theo_thu()     – winrate và PnL theo thứ trong tuần
  thong_ke_theo_regime()  – PnL theo ML regime (0-7)
  thong_ke_theo_mode()    – so sánh kết quả giữa các chế độ
  thong_ke_tong_quat()    – summary stats cho một lần chạy
  max_drawdown()          – equity curve + underwater chart
"""
import os
import uuid
from datetime import datetime

import duckdb
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'du_lieu', 'kairos_warehouse.duckdb')

REGIME_NAME = {
    0: 'Đóng_Băng',
    1: 'Nén_Chặt',
    2: 'Đầu_Xu_Hướng',
    3: 'Xu_Hướng_Mạnh',
    4: 'Cao_Trào',
    5: 'Hồi_Quy',
    6: 'Nhiễu_Động',
    7: 'Quét_Thanh_Khoản',
}


def _ket_noi() -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)


def _tao_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_run (
            run_id      VARCHAR PRIMARY KEY,
            chuc_nang   VARCHAR,
            ngay_chay   TIMESTAMP,
            tu_ngay     VARCHAR,
            den_ngay    VARCHAR,
            symbols     VARCHAR,
            von_ban_dau DOUBLE,
            phi_gd      DOUBLE,
            slippage    DOUBLE,
            don_bay     INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS lenh (
            run_id      VARCHAR,
            chuc_nang   VARCHAR,
            symbol      VARCHAR,
            loai        VARCHAR,
            chien_luoc  VARCHAR,
            regime      INTEGER,
            regime_name VARCHAR,
            gia_vao     DOUBLE,
            gia_dong    DOUBLE,
            leverage    INTEGER,
            pnl         DOUBLE,
            thang       BOOLEAN,
            thoi_gian   TIMESTAMP,
            so_du       DOUBLE,
            gio         INTEGER,
            thu         INTEGER,
            ngay        DATE
        )
    """)


def tao_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:4]


# ─── NORMALIZE helpers ───────────────────────────────────────────────────────

def _chuan_hoa_lenh_vector(row: dict) -> dict:
    """Normalize một lệnh từ vectorized_backtest."""
    loai = str(row.get('Loại', '')).upper()
    return {
        'symbol':      row.get('Symbol', ''),
        'loai':        'LONG' if loai in ('BUY', 'LONG') else 'SHORT',
        'chien_luoc':  row.get('Strategy', ''),
        'regime':      int(row.get('Regime', -1)),
        'gia_vao':     float(row.get('Giá vào', 0)),
        'gia_dong':    float(row.get('Giá đóng', 0)),
        'leverage':    int(row.get('Leverage', 1)),
        'pnl':         float(row.get('PnL', 0)),
        'thoi_gian':   pd.to_datetime(row.get('Time')),
        'so_du':       float(row.get('Balance', 0)),
    }


def _chuan_hoa_lenh_bar(row: dict) -> dict:
    """Normalize một lệnh từ backtest_donluong / backtest_daluong."""
    side = str(row.get('side', '')).lower()
    packet = row.get('packet') or {}
    return {
        'symbol':      row.get('symbol', ''),
        'loai':        'LONG' if side == 'buy' else 'SHORT',
        'chien_luoc':  row.get('strategy', row.get('chien_luoc', '')),
        'regime':      int(packet.get('state_id', row.get('regime', -1))),
        'gia_vao':     float(row.get('entry', 0)),
        'gia_dong':    float(row.get('exit', 0)),
        'leverage':    int(row.get('leverage', 1)),
        'pnl':         float(row.get('pnl_usd', row.get('pnl', 0))),
        'thoi_gian':   pd.to_datetime(row.get('time_close', row.get('close_time'))),
        'so_du':       float(row.get('balance', row.get('so_du', 0))),
    }


def _chuan_hoa_lenh_demo(row: dict) -> dict:
    """Normalize một lệnh từ chay_demo."""
    side = str(row.get('side', '')).lower()
    packet = row.get('packet_ml') or {}
    time_str = row.get('close_time', '')
    day_str  = row.get('day', datetime.now().strftime('%Y-%m-%d'))
    try:
        thoi_gian = pd.to_datetime(f"{day_str} {time_str}")
    except Exception:
        thoi_gian = pd.Timestamp.now()
    return {
        'symbol':      row.get('symbol', ''),
        'loai':        'LONG' if side == 'buy' else 'SHORT',
        'chien_luoc':  row.get('strategy', ''),
        'regime':      int(packet.get('state_id', -1)),
        'gia_vao':     float(row.get('entry_price', 0)),
        'gia_dong':    float(row.get('exit_price', 0)),
        'leverage':    int(row.get('leverage', 1)),
        'pnl':         float(row.get('pnl', 0)),
        'thoi_gian':   thoi_gian,
        'so_du':       float(row.get('so_du', 0)),
    }


_NORMALIZER = {
    'backtest_vector':    _chuan_hoa_lenh_vector,
    'backtest_bar':       _chuan_hoa_lenh_bar,
    'backtest_da_luong':  _chuan_hoa_lenh_bar,
    'demo':               _chuan_hoa_lenh_demo,
    'realtime':           _chuan_hoa_lenh_demo,   # cùng field names với demo
}


def _build_row(chuc_nang: str, run_id: str, norm: dict) -> dict:
    """Thêm các cột derived (regime_name, gio, thu, ngay, thang) vào normalized dict."""
    regime = norm.get('regime', -1)
    ts = pd.to_datetime(norm.get('thoi_gian', pd.Timestamp.now()))
    return {
        'run_id':      run_id,
        'chuc_nang':   chuc_nang,
        'symbol':      norm['symbol'],
        'loai':        norm['loai'],
        'chien_luoc':  norm.get('chien_luoc', ''),
        'regime':      regime,
        'regime_name': REGIME_NAME.get(regime, 'unknown'),
        'gia_vao':     norm['gia_vao'],
        'gia_dong':    norm['gia_dong'],
        'leverage':    norm.get('leverage', 1),
        'pnl':         norm['pnl'],
        'thang':       norm['pnl'] > 0,
        'thoi_gian':   ts,
        'so_du':       norm.get('so_du', 0.0),
        'gio':         int(ts.hour),
        'thu':         int(ts.dayofweek),
        'ngay':        ts.date(),
    }


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def luu_run(run_id: str, chuc_nang: str, config: dict):
    """Tạo một bản ghi run mới trong bảng backtest_run."""
    con = _ket_noi()
    _tao_schema(con)
    con.execute("""
        INSERT OR REPLACE INTO backtest_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id, chuc_nang, datetime.now(),
        config.get('tu_ngay', ''),
        config.get('den_ngay', ''),
        ', '.join(config.get('symbols', [])),
        config.get('von_ban_dau', 0),
        config.get('phi_gd', 0),
        config.get('slippage', 0),
        config.get('don_bay', 1),
    ])
    con.close()


def luu_lenh_don(run_id: str, chuc_nang: str, lenh_raw: dict):
    """
    Lưu một lệnh đơn lẻ — dùng cho chế độ streaming (demo, realtime).
    lenh_raw: dict lệnh theo format của từng chế độ (tự động normalize).
    """
    normalizer = _NORMALIZER.get(chuc_nang, _chuan_hoa_lenh_bar)
    norm = normalizer(lenh_raw)
    row  = _build_row(chuc_nang, run_id, norm)
    df   = pd.DataFrame([row])
    con  = _ket_noi()
    _tao_schema(con)
    con.register('df_tmp', df)
    con.execute("INSERT INTO lenh SELECT * FROM df_tmp")
    con.close()


def luu_ket_qua_backtest(
    ds_lenh: list,
    run_id: str,
    chuc_nang: str,
    config: dict,
):
    """
    Lưu toàn bộ lịch sử lệnh — dùng cho chế độ batch (backtest).
    ds_lenh: list[dict] các lệnh đã đóng theo format của từng chế độ.
    chuc_nang: 'backtest_vector' | 'backtest_bar' | 'backtest_da_luong'
    """
    if not ds_lenh:
        return

    luu_run(run_id, chuc_nang, config)

    normalizer = _NORMALIZER.get(chuc_nang, _chuan_hoa_lenh_bar)
    rows = [_build_row(chuc_nang, run_id, normalizer(r)) for r in ds_lenh]
    df   = pd.DataFrame(rows)

    con = _ket_noi()
    _tao_schema(con)
    con.register('df_tmp', df)
    con.execute("INSERT INTO lenh SELECT * FROM df_tmp")
    con.close()


# ─── ANALYTICAL QUERIES ──────────────────────────────────────────────────────

def _where(run_id=None, chuc_nang=None) -> str:
    parts = []
    if run_id:    parts.append(f"run_id = '{run_id}'")
    if chuc_nang: parts.append(f"chuc_nang = '{chuc_nang}'")
    return ('WHERE ' + ' AND '.join(parts)) if parts else ''


def thong_ke_theo_gio(run_id=None, chuc_nang=None) -> pd.DataFrame:
    """Winrate và PnL theo giờ trong ngày (0-23)."""
    con = _ket_noi()
    df = con.execute(f"""
        SELECT gio,
               COUNT(*)                          AS so_lenh,
               ROUND(AVG(thang::INT)*100, 1)     AS winrate_pct,
               ROUND(SUM(pnl), 2)                AS tong_pnl,
               ROUND(AVG(pnl), 2)                AS tb_pnl
        FROM lenh {_where(run_id, chuc_nang)}
        GROUP BY gio ORDER BY gio
    """).df()
    con.close()
    return df


def thong_ke_theo_thu(run_id=None, chuc_nang=None) -> pd.DataFrame:
    """Winrate và PnL theo thứ trong tuần (0=Thứ Hai … 6=Chủ Nhật)."""
    thu_map = {0:'Thứ 2',1:'Thứ 3',2:'Thứ 4',3:'Thứ 5',4:'Thứ 6',5:'Thứ 7',6:'CN'}
    con = _ket_noi()
    df = con.execute(f"""
        SELECT thu,
               COUNT(*)                          AS so_lenh,
               ROUND(AVG(thang::INT)*100, 1)     AS winrate_pct,
               ROUND(SUM(pnl), 2)                AS tong_pnl,
               ROUND(AVG(pnl), 2)                AS tb_pnl
        FROM lenh {_where(run_id, chuc_nang)}
        GROUP BY thu ORDER BY thu
    """).df()
    con.close()
    df['thu_ten'] = df['thu'].map(thu_map)
    return df


def thong_ke_theo_regime(run_id=None, chuc_nang=None) -> pd.DataFrame:
    """PnL và winrate theo ML regime — regime nào hiệu quả nhất."""
    con = _ket_noi()
    df = con.execute(f"""
        SELECT regime, regime_name,
               COUNT(*)                          AS so_lenh,
               ROUND(AVG(thang::INT)*100, 1)     AS winrate_pct,
               ROUND(SUM(pnl), 2)                AS tong_pnl,
               ROUND(AVG(pnl), 2)                AS tb_pnl,
               ROUND(MIN(pnl), 2)                AS pnl_xau_nhat,
               ROUND(MAX(pnl), 2)                AS pnl_tot_nhat
        FROM lenh {_where(run_id, chuc_nang)}
        GROUP BY regime, regime_name
        ORDER BY tong_pnl DESC
    """).df()
    con.close()
    return df


def thong_ke_theo_chien_luoc(run_id=None, chuc_nang=None) -> pd.DataFrame:
    """PnL và winrate theo chiến lược."""
    con = _ket_noi()
    df = con.execute(f"""
        SELECT chien_luoc,
               COUNT(*)                          AS so_lenh,
               ROUND(AVG(thang::INT)*100, 1)     AS winrate_pct,
               ROUND(SUM(pnl), 2)                AS tong_pnl,
               ROUND(AVG(pnl), 2)                AS tb_pnl
        FROM lenh {_where(run_id, chuc_nang)}
        GROUP BY chien_luoc ORDER BY tong_pnl DESC
    """).df()
    con.close()
    return df


def thong_ke_theo_mode(run_id=None) -> pd.DataFrame:
    """So sánh kết quả giữa các chế độ (backtest_bar vs backtest_vector vs demo)."""
    con = _ket_noi()
    df = con.execute(f"""
        SELECT chuc_nang,
               COUNT(*)                          AS so_lenh,
               ROUND(AVG(thang::INT)*100, 1)     AS winrate_pct,
               ROUND(SUM(pnl), 2)                AS tong_pnl,
               ROUND(AVG(pnl), 2)                AS tb_pnl
        FROM lenh {_where(run_id)}
        GROUP BY chuc_nang ORDER BY tong_pnl DESC
    """).df()
    con.close()
    return df


def thong_ke_theo_symbol(run_id=None, chuc_nang=None) -> pd.DataFrame:
    """PnL và winrate theo từng cặp tài sản."""
    con = _ket_noi()
    df = con.execute(f"""
        SELECT symbol,
               COUNT(*)                          AS so_lenh,
               ROUND(AVG(thang::INT)*100, 1)     AS winrate_pct,
               ROUND(SUM(pnl), 2)                AS tong_pnl,
               ROUND(AVG(pnl), 2)                AS tb_pnl
        FROM lenh {_where(run_id, chuc_nang)}
        GROUP BY symbol ORDER BY tong_pnl DESC
    """).df()
    con.close()
    return df


def thong_ke_tong_quat(run_id: str) -> pd.DataFrame:
    """Summary stats đầy đủ cho một lần chạy backtest."""
    con = _ket_noi()
    df = con.execute(f"""
        WITH stats AS (
            SELECT COUNT(*)                               AS tong_lenh,
                   SUM(thang::INT)                        AS so_thang,
                   COUNT(*) - SUM(thang::INT)             AS so_thua,
                   ROUND(AVG(thang::INT)*100, 1)          AS winrate_pct,
                   ROUND(SUM(pnl), 2)                     AS tong_pnl,
                   ROUND(AVG(pnl), 2)                     AS tb_pnl_lenh,
                   ROUND(MAX(so_du), 2)                   AS dinh_von,
                   ROUND(MIN(so_du), 2)                   AS day_von,
                   MIN(thoi_gian)                         AS tu_ngay,
                   MAX(thoi_gian)                         AS den_ngay
            FROM lenh WHERE run_id = '{run_id}'
        ),
        pf AS (
            SELECT ROUND(
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) /
                NULLIF(ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 0), 2
            ) AS profit_factor
            FROM lenh WHERE run_id = '{run_id}'
        )
        SELECT s.*, p.profit_factor FROM stats s, pf p
    """).df()
    con.close()
    return df


def max_drawdown(run_id: str) -> pd.DataFrame:
    """Tính max drawdown theo equity curve."""
    con = _ket_noi()
    df = con.execute(f"""
        SELECT thoi_gian, so_du,
               MAX(so_du) OVER (ORDER BY thoi_gian ROWS UNBOUNDED PRECEDING) AS peak,
               so_du - MAX(so_du) OVER (ORDER BY thoi_gian ROWS UNBOUNDED PRECEDING) AS drawdown
        FROM lenh WHERE run_id = '{run_id}'
        ORDER BY thoi_gian
    """).df()
    con.close()
    return df


def lich_su_run() -> pd.DataFrame:
    """Danh sách tất cả các lần chạy đã lưu, kèm summary."""
    con = _ket_noi()
    _tao_schema(con)
    df = con.execute("""
        SELECT r.run_id, r.chuc_nang, r.ngay_chay,
               r.tu_ngay, r.den_ngay, r.symbols, r.von_ban_dau,
               COUNT(l.run_id)                   AS so_lenh,
               ROUND(SUM(l.pnl), 2)              AS tong_pnl,
               ROUND(AVG(l.thang::INT)*100, 1)   AS winrate_pct
        FROM backtest_run r
        LEFT JOIN lenh l ON r.run_id = l.run_id
        GROUP BY r.run_id, r.chuc_nang, r.ngay_chay,
                 r.tu_ngay, r.den_ngay, r.symbols, r.von_ban_dau
        ORDER BY r.ngay_chay DESC
    """).df()
    con.close()
    return df


def chay_sql(query: str) -> pd.DataFrame:
    """Chạy câu SQL tùy ý. Dùng để khám phá dữ liệu ad-hoc."""
    con = _ket_noi()
    _tao_schema(con)
    df = con.execute(query).df()
    con.close()
    return df
