"""
chuc_nang/vectorized_backtest.py – Vectorized Backtest Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Xử lý backtest toàn bộ dataset bằng Pandas vectorized (không loop từng nến).
Pipeline:
  1. Tải lịch sử 1m từ Binance
  2. ML regime detection vectorized
  3. Chạy chiến lược vectorized trên toàn dataset
  4. Tính metrics: winrate, PnL, max drawdown, sharpe ratio
  5. Hiển thị kết quả trên dashboard PyQt6
"""
import sys
import os
import time
import json
import matplotlib
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from utils.log import logger
    from utils.doc_cau_hinh import lay_cau_hinh_giao_dich, lay_cau_hinh_ao
    #from utils.save_dataflie import luu_du_lieu_vectorized

    from lay_du_lieu.lay_ohlcv import tai_du_lieu_lich_su, chuan_bi_du_lieu_da_khung_vectorized

    from chien_luoc.logic_vectorized.quan_ly_chien_luoc import tong_hop_tin_hieu
    from ml.trang_thai_thi_truong_ml.ml_predict import du_doan_trang_thai_ml_vector
    from utils.kho_du_lieu import luu_ket_qua_backtest, tao_run_id, thong_ke_tong_quat

except ImportError as e:
    logger.info(f"❌ Lỗi Import: {e}")
    logger.info("Vui lòng chạy script từ thư mục gốc hoặc đảm bảo cấu trúc thư mục đúng.")
    sys.exit(1)


def vectorized_backtest():
    config_backtest = lay_cau_hinh_ao()
    config_trading = lay_cau_hinh_giao_dich()
    
    VON_BAN_DAU = float(config_backtest.get('so_du_ban_dau', 10000))
    PHI_GD = float(config_backtest.get('phi_giao_dich', 0.001))
    SLIPPAGE = float(config_backtest.get('do_truot_gia', 0.001))
    START_DATE = config_backtest.get('ngay_bat_dau', '')
    END_DATE = config_backtest.get('ngay_ket_thuc', '')
    
    DON_BAY = int(config_trading.get('don_bay', 1))
    DS_SYMBOL = config_trading.get('cap_giao_dich', [])
    VON_MOI_LENH = float(config_trading.get('von_moi_lenh_usdt', 100))

    von_hien_tai = VON_BAN_DAU
    tong_lich_su_lenh = []
    run_id = tao_run_id()

    logger.info(f"BẮT ĐẦU BACKTEST TỔNG HỢP: {START_DATE} -> {END_DATE}  [run_id={run_id}]")

    for symbol in DS_SYMBOL:
        logger.info(f"🔍 Đang xử lý cặp: {symbol}")
        df_goc = tai_du_lieu_lich_su(symbol, START_DATE, END_DATE)

        if df_goc is None or df_goc.is_empty(): continue

        # Chuẩn bị dữ liệu đa khung (Vectorized)
        df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d = chuan_bi_du_lieu_da_khung_vectorized(df_goc)

        # ML regime detection trên toàn bộ dataset
        df_regime = du_doan_trang_thai_ml_vector(df_goc)

        # Tổng hợp tín hiệu từ tất cả chiến lược dựa trên regime
        df = tong_hop_tin_hieu(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d, df_regime)

        # Đảm bảo cột timestamp hợp lệ
        if 'timestamp' not in df.columns:
            df = df.reset_index()

        # --- LOGIC BACKTEST CHI TIẾT ---
        vi_the = 0
        gia_vao = 0
        co_tin_hieu = 0
        regime_vao = 0          # khởi tạo tránh NameError nếu loop không vào lệnh nào

        # Chuyển sang mảng numpy trước loop — tránh truy cập frame bên trong loop
        signals   = df['signal'].values
        opens     = df['open'].values
        closes    = df['close'].values
        highs     = df['high'].values    if 'high'     in df.columns else closes.copy()
        lows      = df['low'].values     if 'low'      in df.columns else closes.copy()
        times     = df['timestamp'].values
        sl_pcts   = df['sl_pct'].values  if 'sl_pct'   in df.columns else [0.05] * len(df)
        tp_pcts   = df['tp_pct'].values  if 'tp_pct'   in df.columns else [0.10] * len(df)
        leverages = df['leverage'].values if 'leverage' in df.columns else [DON_BAY] * len(df)
        regimes   = df['regime'].values  if 'regime'   in df.columns else [0] * len(df)

        sl_gia = 0.0
        tp_gia = 0.0

        for i in range(len(df)):
            tin_hieu_hien_tai = signals[i]
            gia_open  = opens[i]
            gia_close = closes[i]
            gia_high  = highs[i]
            gia_low   = lows[i]
            thoi_gian = times[i]
            don_bay_i = int(leverages[i])

            # A. VÀO LỆNH tại Open nến sau khi tín hiệu xuất hiện
            if co_tin_hieu != 0 and vi_the == 0:
                vi_the   = co_tin_hieu
                regime_vao = int(regimes[i - 1]) if i > 0 else 0
                phi_truot = gia_open * SLIPPAGE
                gia_vao = gia_open + phi_truot if vi_the == 1 else gia_open - phi_truot

                # SL/TP theo ATR động
                sl_pct_i = sl_pcts[i - 1] if i > 0 else 0.05
                tp_pct_i = tp_pcts[i - 1] if i > 0 else 0.10
                if vi_the == 1:
                    sl_gia = gia_vao * (1 - sl_pct_i)
                    tp_gia = gia_vao * (1 + tp_pct_i)
                else:
                    sl_gia = gia_vao * (1 + sl_pct_i)
                    tp_gia = gia_vao * (1 - tp_pct_i)

                phi_mo = (VON_MOI_LENH * don_bay_i) * PHI_GD
                von_hien_tai -= phi_mo
                co_tin_hieu   = 0

            # B. KIỂM TRA ĐÓNG LỆNH
            elif vi_the != 0:
                can_thoat = False
                loai      = 'LONG' if vi_the == 1 else 'SHORT'
                gia_dong  = gia_close

                if vi_the == 1:
                    if gia_high >= tp_gia:
                        can_thoat = True; gia_dong = tp_gia
                    elif gia_low <= sl_gia:
                        can_thoat = True; gia_dong = sl_gia
                    elif tin_hieu_hien_tai != 1:
                        can_thoat = True
                else:
                    if gia_low <= tp_gia:
                        can_thoat = True; gia_dong = tp_gia
                    elif gia_high >= sl_gia:
                        can_thoat = True; gia_dong = sl_gia
                    elif tin_hieu_hien_tai != -1:
                        can_thoat = True

                if can_thoat:
                    pnl_raw  = ((gia_dong - gia_vao) / gia_vao if vi_the == 1
                                else (gia_vao - gia_dong) / gia_vao) * (VON_MOI_LENH * don_bay_i)
                    phi_dong = (VON_MOI_LENH * don_bay_i) * (PHI_GD + SLIPPAGE)
                    pnl_net  = pnl_raw - phi_dong
                    von_hien_tai += pnl_net

                    tong_lich_su_lenh.append({
                        'Symbol':   symbol,
                        'Loại':     loai,
                        'Regime':   regime_vao,
                        'Giá vào':  gia_vao,
                        'Giá đóng': gia_dong,
                        'Leverage': don_bay_i,
                        'PnL':      pnl_net,
                        'Time':     thoi_gian,
                        'Balance':  von_hien_tai,
                        'Strategy': '',   # vectorized không track tên chiến lược per-lệnh
                    })
                    vi_the = 0

            # C. GHI NHẬN TÍN HIỆU MỚI
            if vi_the == 0 and tin_hieu_hien_tai != 0:
                co_tin_hieu = tin_hieu_hien_tai

        if not hasattr(vectorized_backtest, 'dict_du_lieu_gui'):
            vectorized_backtest.dict_du_lieu_gui = {}
            
        vectorized_backtest.dict_du_lieu_gui[symbol] = df.copy()

    # ─── LƯU VÀO DATA WAREHOUSE ─────────────────────────────────────────────
    if tong_lich_su_lenh:
        df_lenh = pd.DataFrame(tong_lich_su_lenh)
        luu_ket_qua_backtest(
            tong_lich_su_lenh, run_id, 'backtest_vector',
            config={
                'tu_ngay':     START_DATE,
                'den_ngay':    END_DATE,
                'symbols':     DS_SYMBOL,
                'von_ban_dau': VON_BAN_DAU,
                'phi_gd':      PHI_GD,
                'slippage':    SLIPPAGE,
                'don_bay':     DON_BAY,
            }
        )
        logger.info(f"Đã lưu {len(tong_lich_su_lenh)} lệnh vào warehouse [run_id={run_id}]")

    # ─── TÓM TẮT KẾT QUẢ ────────────────────────────────────────────────────
    if tong_lich_su_lenh:
        df_result = pd.DataFrame(tong_lich_su_lenh)
        tong_lenh   = len(df_result)
        so_thang    = (df_result['PnL'] > 0).sum()
        winrate     = so_thang / tong_lenh * 100 if tong_lenh else 0
        tong_pnl    = df_result['PnL'].sum()
        pnl_pct     = (von_hien_tai - VON_BAN_DAU) / VON_BAN_DAU * 100

        logger.info("=" * 60)
        logger.info(f"📋 KẾT QUẢ BACKTEST: {START_DATE} → {END_DATE}")
        logger.info(f"   Vốn ban đầu : {VON_BAN_DAU:,.0f} USDT")
        logger.info(f"   Vốn cuối    : {von_hien_tai:,.2f} USDT  ({pnl_pct:+.2f}%)")
        logger.info(f"   Tổng lệnh   : {tong_lenh}")
        logger.info(f"   Thắng / Thua: {so_thang} / {tong_lenh - so_thang}")
        logger.info(f"   Win rate    : {winrate:.1f}%")
        logger.info(f"   Tổng PnL    : {tong_pnl:+,.2f} USDT")
        logger.info("=" * 60)
    else:
        logger.warning("⚠️ Không có lệnh nào được thực hiện trong khoảng thời gian này.")

    return tong_lich_su_lenh, vectorized_backtest.dict_du_lieu_gui

if __name__ == "__main__":
    vectorized_backtest()