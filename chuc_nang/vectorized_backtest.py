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

    from chien_luoc.logic_vectorized.chien_luoc.chien_luoc_breakout import chien_luoc_breakout
   
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
    
    logger.info(f"🚀 BẮT ĐẦU BACKTEST TỔNG HỢP: {START_DATE} -> {END_DATE}")

    for symbol in DS_SYMBOL:
        logger.info(f"🔍 Đang xử lý cặp: {symbol}")
        df_goc = tai_du_lieu_lich_su(symbol, START_DATE, END_DATE)

        if df_goc is None or df_goc.is_empty(): continue

        # Chuẩn bị dữ liệu đa khung (Vectorized)
        df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d = chuan_bi_du_lieu_da_khung_vectorized(df_goc)

        df = chien_luoc_breakout(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d)

        print(df)

        # Đảm bảo cột timestamp hợp lệ
        if 'timestamp' not in df.columns:
            df = df.reset_index()

        # --- LOGIC BACKTEST CHI TIẾT ---
        vi_the = 0 
        gia_vao = 0
        co_tin_hieu = 0
        
        # Chuyển sang giá trị list để loop nhanh hơn .iloc
        signals = df['signal'].values
        opens = df['open'].values
        closes = df['close'].values
        times = df['timestamp'].values

        for i in range(len(df)):
            tin_hieu_hien_tai = signals[i]
            gia_open = opens[i]
            gia_close = closes[i]
            thoi_gian = times[i]

            # A. THỰC THI KHỚP LỆNH PENDING (Vào lệnh tại Open nến mới)
            if co_tin_hieu != 0 and vi_the == 0:
                vi_the = co_tin_hieu
                # Giá vào = Open nến hiện tại + Slippage (mô phỏng trượt giá thực tế)
                phi_truot = gia_open * SLIPPAGE
                gia_vao = gia_open + phi_truot if vi_the == 1 else gia_open - phi_truot
                
                # Trừ phí lượt MỞ ngay khi vào lệnh
                phi_mo = (VON_MOI_LENH * DON_BAY) * PHI_GD
                von_hien_tai -= phi_mo
                
                co_tin_hieu = 0 

            # B. KIỂM TRA ĐÓNG LỆNH (Dựa trên tín hiệu nến vừa đóng)
            elif vi_the != 0:
                can_thoat = False
                # Nếu đang Long (1) mà tín hiệu không còn là 1 -> Thoát
                if vi_the == 1 and tin_hieu_hien_tai != 1:
                    can_thoat = True
                    pnl_raw = (gia_close - gia_vao) / gia_vao * (VON_MOI_LENH * DON_BAY)
                    loai = 'LONG'
                # Nếu đang Short (-1) mà tín hiệu không còn là -1 -> Thoát
                elif vi_the == -1 and tin_hieu_hien_tai != -1:
                    can_thoat = True
                    pnl_raw = (gia_vao - gia_close) / gia_vao * (VON_MOI_LENH * DON_BAY)
                    loai = 'SHORT'

                if can_thoat:
                    # Trừ phí lượt ĐÓNG
                    phi_dong = (VON_MOI_LENH * DON_BAY) * PHI_GD
                    # Phí trượt giá lượt đóng (ước tính trên giá close)
                    truot_gia_dong = (VON_MOI_LENH * DON_BAY) * SLIPPAGE
                    
                    pnl_net = pnl_raw - phi_dong - truot_gia_dong
                    von_hien_tai += pnl_net
                    
                    tong_lich_su_lenh.append({
                        'Symbol': symbol, 
                        'Loại': loai,
                        'Giá vào': gia_vao,
                        'Giá đóng': gia_close, 
                        'PnL': pnl_net, 
                        'Time': thoi_gian,
                        'Balance': von_hien_tai
                    })
                    vi_the = 0

            # C. GHI NHẬN TÍN HIỆU MỚI (Chỉ khi đang trống vị thế)
            if vi_the == 0 and tin_hieu_hien_tai != 0:
                co_tin_hieu = tin_hieu_hien_tai

        if not hasattr(vectorized_backtest, 'dict_du_lieu_gui'):
            vectorized_backtest.dict_du_lieu_gui = {}
            
        vectorized_backtest.dict_du_lieu_gui[symbol] = df.copy()

    return tong_lich_su_lenh, vectorized_backtest.dict_du_lieu_gui

if __name__ == "__main__":
    vectorized_backtest()