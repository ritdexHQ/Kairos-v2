import polars as pl
import numpy as np

class TradingTeacher:
    def __init__(self):
        # Bộ nhớ riêng biệt cho mỗi thực thể (Instance)
        self.last_state = 0
        self.change_count = 0

    def detect_regime(self, df_5m=None, df_15m=None, df_1h=None, df_4h=None):
        """
        Hedge Fund Level Regime Detection.
        Logic: Swing Liquidity, MTF, Express Lane cho Fast States.
        """
        
        def _calc_features(df_in):
            if df_in is None or df_in.height < 120:
                return None
            df = df_in 
            try:
                # ================= 1. Cấu trúc Swing & Biến động =================
                df = df.with_columns([
                    pl.col("close").diff().alias("diff"),
                    pl.col("high").rolling_max(20).shift(1).alias("prev_high"),
                    pl.col("low").rolling_min(20).shift(1).alias("prev_low"),
                    pl.max_horizontal([
                        (pl.col("high") - pl.col("low")),
                        (pl.col("high") - pl.col("close").shift(1)).abs(),
                        (pl.col("low") - pl.col("close").shift(1)).abs()
                    ]).alias("tr")
                ])
                
                # ================= 2. Volatility & Trend =================
                df = df.with_columns([
                    pl.col("tr").ewm_mean(span=14, adjust=False).alias("atr"),
                    pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50")
                ]).with_columns([
                    (pl.col("atr") / (pl.col("close") + 1e-9)).alias("ATRn"),
                    ((pl.col("high") - pl.col("low")) / (pl.col("atr") + 1e-9)).alias("SpreadATR"),
                    ((pl.col("ema_50") - pl.col("ema_50").shift(5)) / (pl.col("ema_50").shift(5) + 1e-9)).alias("S")
                ])

                # RSI & ADX (Wilder's Smoothing)
                df = df.with_columns([
                    pl.when(pl.col("diff") > 0).then(pl.col("diff")).otherwise(0).ewm_mean(span=14, adjust=False).alias("gain"),
                    pl.when(pl.col("diff") < 0).then(pl.col("diff").abs()).otherwise(0).ewm_mean(span=14, adjust=False).alias("loss")
                ]).with_columns([
                    (100 - (100 / (1 + (pl.col("gain") / (pl.col("loss") + 1e-9))))).alias("RSI")
                ]).with_columns([
                    (pl.col("RSI") - pl.col("RSI").shift(3)).alias("RSIslope")
                ])

                df = df.with_columns([
                    (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
                    (pl.col("low").shift(1) - pl.col("low")).alias("down_move")
                ]).with_columns([
                    pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0)).then(pl.col("up_move")).otherwise(0).alias("plus_dm"),
                    pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0)).then(pl.col("down_move")).otherwise(0).alias("minus_dm")
                ]).with_columns([
                    (pl.col("plus_dm").ewm_mean(span=27, adjust=False) / (pl.col("tr").ewm_mean(span=27, adjust=False) + 1e-9)).alias("di_plus"),
                    (pl.col("minus_dm").ewm_mean(span=27, adjust=False) / (pl.col("tr").ewm_mean(span=27, adjust=False) + 1e-9)).alias("di_minus")
                ]).with_columns([
                    (100 * ((pl.col("di_plus") - pl.col("di_minus")).abs() / (pl.col("di_plus") + pl.col("di_minus") + 1e-9))).ewm_mean(span=27, adjust=False).alias("ADX")
                ])

                # ================= 3. Nén & Squeeze =================
                df = df.with_columns([
                    pl.col("close").rolling_mean(20).alias("bb_mid"),
                    pl.col("close").rolling_std(20).alias("bb_std"),
                    pl.col("close").ewm_mean(span=20, adjust=False).alias("ema_20")
                ]).with_columns([
                    (pl.col("bb_mid") + 2*pl.col("bb_std")).alias("bb_upper"),
                    (pl.col("bb_mid") - 2*pl.col("bb_std")).alias("bb_lower")
                ]).with_columns([
                    pl.when((pl.col("bb_upper") < (pl.col("ema_20") + 1.5*pl.col("atr"))) & 
                            (pl.col("bb_lower") > (pl.col("ema_20") - 1.5*pl.col("atr")))).then(1.0).otherwise(0.0).alias("SQZ"),
                    ((pl.col("close") - pl.col("close").shift(10)).abs() / (pl.col("diff").abs().rolling_sum(10) + 1e-9)).alias("ER")
                ])
                
                # Choppiness
                df = df.with_columns([
                    pl.col("tr").rolling_sum(14).alias("tr_sum"),
                    pl.col("high").rolling_max(14).alias("high_max"),
                    pl.col("low").rolling_min(14).alias("low_min")
                ]).with_columns([
                    (100 * (pl.col("tr_sum") / (pl.col("high_max") - pl.col("low_min") + 1e-9)).log10() / np.log10(14)).alias("CHOP")
                ])
                
                # ================= 4. Volume & Wick Structure =================
                df = df.with_columns([
                    pl.col("volume").rolling_mean(20).alias("vol_sma"),
                    pl.col("volume").rolling_std(20).alias("vol_std"),
                    ((pl.col("close") * pl.col("volume")).rolling_sum(100) / (pl.col("volume").rolling_sum(100) + 1e-9)).alias("vwap"),
                    pl.max_horizontal("open", "close").alias("body_top"),
                    pl.min_horizontal("open", "close").alias("body_bottom"),
                    (pl.col("high") - pl.col("low") + 1e-9).alias("hl_range")
                ]).with_columns([
                    ((pl.col("volume") - pl.col("vol_sma")) / (pl.col("vol_std") + 1e-9)).alias("VOLz"),
                    ((pl.col("close") - pl.col("vwap")) / (pl.col("atr") + 1e-9)).alias("VWAPd"),
                    ((pl.col("high") - pl.col("body_top")) / pl.col("hl_range")).alias("WickUpProp"),
                    ((pl.col("body_bottom") - pl.col("low")) / pl.col("hl_range")).alias("WickDnProp"),
                    pl.col("ATRn").rolling_mean(100).alias("ATRn_avg100")
                ])
                return df.tail(1).to_dicts()[0]
            except Exception as e:
                return None

        # MTF Calculations
        c_h4 = _calc_features(df_4h) if df_4h is not None else _calc_features(df_1h)
        c_h1 = _calc_features(df_1h)
        c_m15 = _calc_features(df_15m)
        c_m5 = _calc_features(df_5m)

        if not all([c_h1, c_m15, c_m5]): return self.last_state, 0.0

        scores = {k: 0.0 for k in range(8)}

        # ================= KHỐI LUẬT GẮN NHÃN =================
        sweep_up = (c_m5['high'] > c_m5['prev_high']) and (c_m5['close'] < c_m5['prev_high']) and (c_m5['WickUpProp'] > 0.40)
        sweep_down = (c_m5['low'] < c_m5['prev_low']) and (c_m5['close'] > c_m5['prev_low']) and (c_m5['WickDnProp'] > 0.40)

        if (sweep_up or sweep_down) and c_m5['VOLz'] > 2.0:
            scores[7] = 100.0
        else:
            is_dead = (c_h1['VOLz'] < -1.2) and (c_h1['ATRn'] < (c_h1['ATRn_avg100'] * 0.65))
            if is_dead:
                scores[0] = 100.0
            else:
                if ((c_h1['SQZ'] == 1.0 or c_m15['SQZ'] == 1.0) and (c_m15['CHOP'] > 50 or c_h1['ER'] < 0.35)):
                    scores[1] += 60
                if (c_h1['ADX'] >= 28 and np.sign(c_h4['S']) == np.sign(c_h1['S']) == np.sign(c_m15['S']) and c_m15['ER'] > 0.38 and c_h1['CHOP'] < 45 and abs(c_h1['VWAPd']) < 2.2):
                    scores[3] += 65
                if abs(c_m15['S']) > 0.0015 and (c_m15['ER'] > 0.32) and (10 <= c_h1['ADX'] < 25): scores[2] += 55
                if abs(c_h1['VWAPd']) > 2.0 and (c_h1['RSI'] > 72 or c_h1['RSI'] < 28): scores[4] += 60
                if (
                    abs(c_h1['VWAPd']) > 1.5
                    and abs(c_m15['RSIslope']) > 4
                    and c_h1['ADX'] < 28
                    and c_m15['ER'] < 0.45
                ):
                    scores[5] += 62
                if (
                    c_h1['ER'] < 0.35
                    and c_h1['ADX'] < 22
                    and c_m15['CHOP'] > 48
                ):
                    scores[6] += 50
                    scores[2] *= 0.5
                    scores[3] *= 0.5

        # ================= QUÁN TÍNH & SOFTMAX =================
        scores[self.last_state] += max(4.0, 10.0 * confidence if 'confidence' in locals() else 6.0) 
        max_s = max(scores.values())
        exps = {k: np.exp(v - max_s) for k, v in scores.items()}
        total_exp = sum(exps.values())
        probs = {k: v / total_exp for k, v in exps.items()}

        raw_state = max(probs, key=probs.get)
        confidence = probs[raw_state]

        top2 = sorted(probs.values(), reverse=True)[:2]
        if confidence < 0.60 or (top2[0] - top2[1]) < 0.08:
            raw_state = self.last_state

        # ================= EXPRESS LANE PATCH =================
        fast_states = [4, 7]
        if raw_state in fast_states:
            self.last_state = raw_state
            self.change_count = 0
        else:
            if raw_state != self.last_state:
                self.change_count += 1
            else:
                self.change_count = 0

            if self.change_count >= 3:
                self.last_state = raw_state
                self.change_count = 0

        return self.last_state, float(confidence) 


import polars as pl
import numpy as np

def detect_regime_vectorized(
    df_1m: pl.DataFrame, 
    # =================================================================
    # ⚙️ KHU VỰC CÀI ĐẶT THÔNG SỐ (CHỈNH SỬA Ở ĐÂY)
    # =================================================================
    change_limit: int = 2,       # [Gốc: 3] Số nến liên tiếp cần để xác nhận đổi Regime (Lọc nhiễu)
    bonus_mult: float = 10.0,    # [Gốc: 10.0] Hệ số nhân quán tính. Tăng để nến "lì" hơn.
    min_bonus: float = 5.0,      # [Gốc: 4.0] Điểm cộng tối thiểu cho Regime hiện tại.
    conf_threshold: float = 0.60,# [Gốc: 0.60] Độ tự tin tối thiểu bắt buộc phải đạt để đổi màu.
    diff_threshold: float = 0.04 # [Gốc: 0.08] Độ chênh lệch điểm tối thiểu giữa Top 1 và Top 2.
):
    # =================================================================
    # BƯỚC 1: DÙNG POLARS ĐỂ TÍNH INDICATOR & ĐIỂM CƠ SỞ
    # =================================================================
    def _add_features(df):
        if df is None: return None
        # Cấu trúc & Biến động
        df = df.with_columns([
            pl.col("close").diff().alias("diff"),
            pl.col("high").rolling_max(20).shift(1).alias("prev_high"),
            pl.col("low").rolling_min(20).shift(1).alias("prev_low"),
            pl.max_horizontal([
                (pl.col("high") - pl.col("low")),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ]).alias("tr")
        ])
        
        # Volatility & Trend
        df = df.with_columns([
            pl.col("tr").ewm_mean(span=14, adjust=False).alias("atr"),
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
            pl.col("close").ewm_mean(span=20, adjust=False).alias("ema_20"),
        ]).with_columns([
            (pl.col("atr") / (pl.col("close") + 1e-9)).alias("ATRn"),
            ((pl.col("ema_50") - pl.col("ema_50").shift(5)) / (pl.col("ema_50").shift(5) + 1e-9)).alias("S")
        ])

        # RSI & ADX
        df = df.with_columns([
            pl.when(pl.col("diff") > 0).then(pl.col("diff")).otherwise(0).ewm_mean(span=14, adjust=False).alias("gain"),
            pl.when(pl.col("diff") < 0).then(pl.col("diff").abs()).otherwise(0).ewm_mean(span=14, adjust=False).alias("loss"),
            (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
            (pl.col("low").shift(1) - pl.col("low")).alias("down_move")
        ]).with_columns([
            (100 - (100 / (1 + (pl.col("gain") / (pl.col("loss") + 1e-9))))).alias("RSI"),
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0)).then(pl.col("up_move")).otherwise(0).alias("plus_dm"),
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0)).then(pl.col("down_move")).otherwise(0).alias("minus_dm")
        ]).with_columns([
            (pl.col("RSI") - pl.col("RSI").shift(3)).alias("RSIslope"),
            (pl.col("plus_dm").ewm_mean(span=27, adjust=False) / (pl.col("tr").ewm_mean(span=27, adjust=False) + 1e-9)).alias("di_plus"),
            (pl.col("minus_dm").ewm_mean(span=27, adjust=False) / (pl.col("tr").ewm_mean(span=27, adjust=False) + 1e-9)).alias("di_minus")
        ]).with_columns([
            (100 * ((pl.col("di_plus") - pl.col("di_minus")).abs() / (pl.col("di_plus") + pl.col("di_minus") + 1e-9))).ewm_mean(span=27, adjust=False).alias("ADX")
        ])

        # Squeeze & Choppiness
        df = df.with_columns([
            pl.col("close").rolling_mean(20).alias("bb_mid"),
            pl.col("close").rolling_std(20).alias("bb_std"),
            pl.col("tr").rolling_sum(14).alias("tr_sum"),
            pl.col("high").rolling_max(14).alias("high_max"),
            pl.col("low").rolling_min(14).alias("low_min")
        ]).with_columns([
            ((pl.col("bb_mid") + 2*pl.col("bb_std")) < (pl.col("ema_20") + 1.5*pl.col("atr"))).cast(pl.Float64).alias("SQZ"),
            ((pl.col("close") - pl.col("close").shift(10)).abs() / (pl.col("diff").abs().rolling_sum(10) + 1e-9)).alias("ER"),
            (100 * (pl.col("tr_sum") / (pl.col("high_max") - pl.col("low_min") + 1e-9)).log10() / np.log10(14)).alias("CHOP")
        ])

        # Volume & Wick
        df = df.with_columns([
            pl.col("volume").rolling_mean(20).alias("vol_sma"),
            pl.col("volume").rolling_std(20).alias("vol_std"),
            ((pl.col("close") * pl.col("volume")).rolling_sum(100) / (pl.col("volume").rolling_sum(100) + 1e-9)).alias("vwap"),
            pl.max_horizontal("open", "close").alias("body_top"),
            pl.min_horizontal("open", "close").alias("body_bottom"),
            (pl.col("high") - pl.col("low") + 1e-9).alias("hl_range")
        ]).with_columns([
            ((pl.col("volume") - pl.col("vol_sma")) / (pl.col("vol_std") + 1e-9)).alias("VOLz"),
            ((pl.col("close") - pl.col("vwap")) / (pl.col("atr") + 1e-9)).alias("VWAPd"),
            ((pl.col("high") - pl.col("body_top")) / pl.col("hl_range")).alias("WickUpProp"),
            ((pl.col("body_bottom") - pl.col("low")) / pl.col("hl_range")).alias("WickDnProp"),
            pl.col("ATRn").rolling_mean(100).alias("ATRn_avg100")
        ])
        return df

    # --- 2. Xử lý MTF an toàn (Chống rò rỉ tương lai - CỰC KỲ QUAN TRỌNG) ---
    def resample_and_shift(df, interval_str, shift_minutes):
        resampled = df.group_by_dynamic("timestamp", every=interval_str).agg([
            pl.col("open").first(), pl.col("high").max(), 
            pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum()
        ])
        features = _add_features(resampled)
        # Dịch chuyển timestamp để lúc ghép vào 1m không bị biết trước tương lai
        features = features.with_columns((pl.col("timestamp") + pl.duration(minutes=shift_minutes)).alias("timestamp"))
        return features

    # Tiền xử lý khung 1m để lấy thông số quét thanh khoản tức thì
    df_1m = _add_features(df_1m)

    f5m = resample_and_shift(df_1m, "5m", 5).select(pl.all().name.suffix("_5m"))
    f15m = resample_and_shift(df_1m, "15m", 15).select(pl.all().name.suffix("_15m"))
    f1h = resample_and_shift(df_1m, "1h", 60).select(pl.all().name.suffix("_1h"))
    f4h = resample_and_shift(df_1m, "4h", 240).select(pl.all().name.suffix("_4h"))

    df = df_1m.join(f5m, left_on="timestamp", right_on="timestamp_5m", how="left") \
              .join(f15m, left_on="timestamp", right_on="timestamp_15m", how="left") \
              .join(f1h, left_on="timestamp", right_on="timestamp_1h", how="left") \
              .join(f4h, left_on="timestamp", right_on="timestamp_4h", how="left") \
              .fill_null(strategy="forward")

    # --- 3. TÍNH TOÁN CÁC ĐIỀU KIỆN (CONDITIONS) ---
    # 🔥 ĐÃ FIX: cond_sweep sử dụng thông số trực tiếp từ khung 1m để bắt tín hiệu TỨC THỜI
    cond_sweep = (
        (((pl.col("high") > pl.col("prev_high")) & (pl.col("close") < pl.col("prev_high")) & (pl.col("WickUpProp") > 0.35)) |
         ((pl.col("low") < pl.col("prev_low")) & (pl.col("close") > pl.col("prev_low")) & (pl.col("WickDnProp") > 0.35))) &
        (pl.col("VOLz") > 1.5) # Yêu cầu Volume M1 đột biến mạnh hơn
    )
    
    cond_dead = (pl.col("VOLz_1h") < -1.2) & (pl.col("ATRn_1h") < (pl.col("ATRn_avg100_1h") * 0.65))
    cond_block = cond_sweep | cond_dead 

    df = df.with_columns([
        pl.when(cond_sweep).then(100.0).otherwise(0.0).alias("r7"),
        pl.when(cond_sweep).then(0.0).when(cond_dead).then(100.0).otherwise(0.0).alias("r0"),
        
        pl.when(cond_block).then(0.0).when(((pl.col("SQZ_1h") == 1.0) | (pl.col("SQZ_15m") == 1.0)) & ((pl.col("CHOP_15m") > 50) | (pl.col("ER_1h") < 0.35))).then(60.0).otherwise(0.0).alias("r1"),
        pl.when(cond_block).then(0.0).when((pl.col("VWAPd_1h").abs() > 2.0) & ((pl.col("RSI_1h") > 72) | (pl.col("RSI_1h") < 28))).then(60.0).otherwise(0.0).alias("r4"),
        pl.when(cond_block).then(0.0).when((pl.col("VWAPd_1h").abs() > 1.5) & (pl.col("RSIslope_15m").abs() > 4) & (pl.col("ADX_1h") < 28) & (pl.col("ER_15m") < 0.45)).then(62.0).otherwise(0.0).alias("r5"),
        pl.when(cond_block).then(0.0).when((pl.col("ER_1h") < 0.35) & (pl.col("ADX_1h") < 22) & (pl.col("CHOP_15m") > 48)).then(50.0).otherwise(0.0).alias("r6"),
        
        pl.when(cond_block).then(0.0).when((pl.col("S_15m").abs() > 0.0015) & (pl.col("ER_15m") > 0.32) & (pl.col("ADX_1h") >= 10) & (pl.col("ADX_1h") < 25)).then(55.0).otherwise(0.0).alias("raw_r2"),
        pl.when(cond_block).then(0.0).when((pl.col("ADX_1h") >= 28) & (pl.col("S_4h").sign() == pl.col("S_1h").sign()) & (pl.col("S_1h").sign() == pl.col("S_15m").sign()) & (pl.col("ER_15m") > 0.38) & (pl.col("CHOP_1h") < 45) & (pl.col("VWAPd_1h").abs() < 2.2)).then(65.0).otherwise(0.0).alias("raw_r3"),
    ])

    # Phạt điểm r2 và r3 nếu r6 kích hoạt
    df = df.with_columns([
        pl.when(pl.col("r6") > 0).then(pl.col("raw_r2") * 0.5).otherwise(pl.col("raw_r2")).alias("r2"),
        pl.when(pl.col("r6") > 0).then(pl.col("raw_r3") * 0.5).otherwise(pl.col("raw_r3")).alias("r3"),
    ])

    # =================================================================
    # BƯỚC 4: CỖ MÁY TRẠNG THÁI NUMPY (ĐÃ XÓA LOOKAHEAD BIAS)
    # =================================================================
    score_matrix = df.select(["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]).fill_null(0.0).to_numpy()
    
    n_rows = len(score_matrix)
    regimes = np.zeros(n_rows, dtype=np.int32)
    confidences = np.zeros(n_rows, dtype=np.float64)

    last_state = 0
    change_count = 0
    confidence = 0.0

    for i in range(n_rows):
        scores = score_matrix[i].copy()
        
        # 1. Quán tính (Hysteresis)
        bonus = max(min_bonus, bonus_mult * confidence) if i > 0 else 6.0
        scores[last_state] += bonus
        
        # 2. Softmax
        max_s = np.max(scores)
        exps = np.exp(scores - max_s)
        probs = exps / np.sum(exps)
        
        raw_state = int(np.argmax(probs))
        curr_confidence = probs[raw_state]
        
        # 3. Điều kiện Top 2 (Confidence)
        sorted_probs = np.sort(probs) 
        if curr_confidence < conf_threshold or (sorted_probs[-1] - sorted_probs[-2]) < diff_threshold:
            raw_state = last_state

        # 4. State Transition (KHÔNG TRUY HỒI ĐỂ TRÁNH LOOKAHEAD BIAS)
        if raw_state in (4, 7):
            # Các tín hiệu cực đoan như Climax (4) và Sweep (7) được "Express Lane" kích hoạt ngay lập tức
            last_state = raw_state
            change_count = 0
        else:
            if raw_state != last_state:
                change_count += 1
            else:
                change_count = 0

            # Cần tích lũy đủ số nến (change_limit) mới chịu đổi trạng thái, 
            # Đổi từ hiện tại trở đi, KHÔNG sửa lại quá khứ.
            if change_count >= change_limit:
                last_state = raw_state
                change_count = 0

        # Gán trạng thái hiện tại
        regimes[i] = last_state
        confidence = curr_confidence
        confidences[i] = curr_confidence

    # =================================================================
    # BƯỚC 5: GẮN KẾT QUẢ VÀO DATAFRAME
    # =================================================================
    df = df.with_columns([
        pl.Series("regime", regimes),
        pl.Series("confidence", confidences)
    ])

    return df.select(["timestamp", "open", "high", "low", "close", "volume", "regime", "confidence"])