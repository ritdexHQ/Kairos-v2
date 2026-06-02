"""
ml/trang_thai_thi_truong_ml/ml_predict.py – Inference & đánh giá ML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3 hàm chính:
  • du_doan_trang_thai_ml()        – dự đoán 1 cây nến (bar-to-bar, realtime)
  • du_doan_trang_thai_ml_vector() – dự đoán hàng loạt (batch inference, backtest)
  • danh_gia_ml()                  – ghi reward vào trading_memory.csv sau khi đóng lệnh

STATE_MAP định nghĩa 8 regime và chiến lược tương ứng:
  0 Đóng_Băng → không trade  |  1 Nén_Chặt → chờ breakout
  2 Đầu_XH → vào sớm         |  3 XH_Mạnh → follow trend
  4 Cao_Trào → chốt lời       |  5 Hồi_Quy → counter-trend
  6 Nhiễu_Động → range trade  |  7 Quét_TK → risk-off
"""
import os
import json
from datetime import datetime
import pandas as pd
import csv

# --- THƯ VIỆN LÕI ---
import torch
import numpy as np
import polars as pl

from ml.trang_thai_thi_truong_ml.ml_model import AI_Engine, DATA_DIR
from ml.trang_thai_thi_truong_ml.tao_feature import feature_dataset, features_vectorized

LOG_FILE = os.path.join(DATA_DIR, "trading_memory.csv")
engine = AI_Engine() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STATE_MAP = {
    0: "Đóng_Băng",           # Dead Market – không trade
    1: "Nén_Chặt",            # Squeeze – tích lũy nén, canh bứt phá
    2: "Đầu_Xu_Hướng",        # Early Trend – xu hướng chớm hình thành
    3: "Xu_Hướng_Mạnh",       # Strong Trend – H4/H1/M15 đồng thuận
    4: "Cao_Trào",            # Climax – giá chạy quá xa, sắp đảo chiều
    5: "Hồi_Quy",             # Mean Reversion – giật ngược về trung bình
    6: "Nhiễu_Động",          # Choppy – đi ngang biên độ hẹp
    7: "Quét_Thanh_Khoản"     # Liquidity Crisis – tin mạnh, risk-off
}

# Ánh xạ từ state_id sang tên chiến lược được dùng trong quan_ly_chien_luoc
STRATEGY_MAP = {
    0: None,               # Đóng_Băng  → không trade
    1: "Squeeze",          # Nén_Chặt   → đánh breakout khi squeeze nổ
    2: "Breakout",         # Đầu_XH     → vào sớm theo hướng breakout
    3: "Trend_following",  # XH_Mạnh    → follow trend đa khung
    4: "Mean_reversion",   # Cao_Trào   → đánh ngược khi kiệt sức
    5: "Mean_reversion",   # Hồi_Quy    → đánh ngược về trung bình
    6: "Scalping",         # Nhiễu_Động → scalp biên độ hẹp
    7: None,               # Quét_TK    → quá nguy hiểm, không trade
}

def du_doan_trang_thai_ml(df_5m, df_15m, df_1h, df_4h, last_state = None):

    feature_dict = feature_dataset(df_5m, df_15m, df_1h, df_4h, last_state=last_state)

    if feature_dict is None or (isinstance(feature_dict, pd.DataFrame) and feature_dict.empty):
        return None

    input_vector = {k: v.iloc[-1] for k, v in feature_dict.items()}

    state_id, conf, probs = engine.predict(input_vector)

    if state_id is None: return None

    strategy = STRATEGY_MAP.get(state_id)

    # Nếu state không có chiến lược tương ứng → không trade
    if strategy is None:
        return None

    packet = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'state_id': state_id,
        'state_name': STATE_MAP.get(state_id, "UNKNOWN"),
        'strategy_name': strategy,   # tên chiến lược cho quan_ly_chien_luoc routing
        'confidence': round(conf, 4),
        'probs': probs,
        'features_snapshot': input_vector
    }
    return packet

def du_doan_trang_thai_ml_vector(df_1m: pl.DataFrame) -> pl.DataFrame:
    """
    Nhận DataFrame 1m gốc. Xuất ra DataFrame có cột 'regime' và 'confidence' 
    do mô hình AI dự đoán (Xử lý hàng loạt - Vectorized Inference).
    """
    # 1. Trích xuất Features
    df_feat = features_vectorized(df_1m)
    
    # Trả về Mặc định nếu DataFrame rỗng
    if df_feat is None or df_feat.is_empty():
        return df_1m.with_columns([pl.lit(0).alias("regime"), pl.lit(0.0).alias("confidence")])

    engine_vector = AI_Engine()
    
    # 🔥 SAFEGUARD 1: KIỂM TRA ĐÃ CÓ MODEL CHƯA? (Thay vì raise ValueError gây crash bot)
    if engine_vector.model is None or engine_vector.scaler is None:
        print("⚠️ CẢNH BÁO: AI Model hoặc Scaler chưa được huấn luyện! Trả về Regime 0 mặc định.")
        return df_1m.with_columns([pl.lit(0).alias("regime"), pl.lit(0.0).alias("confidence")])

    feature_cols = engine_vector.feature_names
    
    # 🔥 SAFEGUARD 2: KIỂM TRA LỖI LỆCH TÊN CỘT (MISMATCH FEATURES)
    missing_cols = [c for c in feature_cols if c not in df_feat.columns]
    if len(missing_cols) > 0:
        print(f"❌ LỖI: Dataset bị thiếu {len(missing_cols)} cột Features (Vd: {missing_cols[:3]}). Đã Bypass về Regime 0.")
        return df_1m.with_columns([pl.lit(0).alias("regime"), pl.lit(0.0).alias("confidence")])

    # 2. Chuyển DataFrame thành Tensor
    # Đảm bảo thứ tự cột khớp với lúc train
    X_numpy = df_feat.select(feature_cols).to_numpy()
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

    # 3. Chạy qua Model Hàng Loạt (Batch Inference)
    engine_vector.model.eval()
    with torch.no_grad():
        # Scale data & đưa lên GPU/CPU
        X_scaled = engine_vector.scaler.transform(X_tensor).to(device)
        
        # Dự đoán
        logits = engine_vector.model(X_scaled)
        probs = torch.softmax(logits, dim=1)
        
        # Lấy Class có xác suất cao nhất và độ tự tin
        confidences, predictions = torch.max(probs, dim=1)

    # 4. Đưa kết quả về CPU và ghép vào DataFrame ban đầu
    preds_np = predictions.cpu().numpy()
    confs_np = confidences.cpu().numpy()
    
    # Tạo DataFrame kết quả trung gian
    df_results = pl.DataFrame({
        "timestamp": df_feat["timestamp"],
        "regime": preds_np.astype(np.int32),
        "confidence": confs_np.astype(np.float64)
    })

    # Ghép lại với df_1m gốc (Những dòng đầu bị NaN feature sẽ được fill regime=0)
    df_final = df_1m.join(df_results, on="timestamp", how="left") \
                    .with_columns([
                        pl.col("regime").fill_null(0).cast(pl.Int32),
                        pl.col("confidence").fill_null(0.0).cast(pl.Float64)
                    ])

    return df_final.select(["timestamp", "open", "high", "low", "close", "volume", "regime", "confidence"])

def danh_gia_ml(packet, pnl, dd, correct=None):

    if not packet: return

    if correct is None:
        correct = 'NaN'

    state_name = packet['state_name']

    if pnl > 0:
        reward = pnl * 1.0 
    else:
        reward = pnl * 2.0 

    # --- 2. ĐIỀU CHỈNH THEO CHIẾN THUẬT (khớp với STATE_MAP) ---
    if state_name == 'Nhiễu_Động':       # Regime 6 – scalping range hẹp
        if pnl < 0: reward -= 2.0
        elif 0 < pnl < 0.5: reward += 0.5
    elif state_name in ('Nén_Chặt', 'Đầu_Xu_Hướng'):   # Regime 1,2 – breakout
        if pnl < 0: reward -= 2.0
        elif pnl > 2.0: reward += 2.0
    elif state_name == 'Xu_Hướng_Mạnh': # Regime 3 – follow trend
        if pnl < 0: reward *= 1.5
        elif pnl > 3.0: reward += 3.0
    elif state_name in ('Cao_Trào', 'Hồi_Quy'):  # Regime 4,5 – mean reversion
        if pnl < -1.0: reward -= 3.0
        elif pnl > 1.5: reward += 2.0
    else:
        if dd < -1.0: reward -= 1.0

    reward = max(min(reward, 10), -10)

    # --- 3. LƯU LOG ---
    log_row = {
        'timestamp': packet['timestamp'],
        'state': packet['state_id'],      
        'correct': correct,
        'state_name': packet['state_name'],
        'confidence': packet['confidence'],
        'pnl': round(pnl, 4),
        'reward': round(reward, 4),
        'features_json': json.dumps(packet['features_snapshot'])
    }
    
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(log_row)