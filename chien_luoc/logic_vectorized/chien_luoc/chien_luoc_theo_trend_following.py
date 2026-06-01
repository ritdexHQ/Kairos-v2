from chien_luoc.logic_vectorized.phan_tich_ky_thuat.dong_luong_dao_chieu import pt_rsi
from utils.ham_tien_ich import gop_va_dong_bo_data


def chien_luoc_theo_trend_following(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d ):

    du_lieu_cac_khung = {
            '1m': pt_rsi(df_1m, '1m'), '3m': pt_rsi(df_3m, '3m'), 
            '5m': pt_rsi(df_5m, '5m'), '15m': pt_rsi(df_15m, '15m'),
            '30m': pt_rsi(df_30m, '30m'), '1h': pt_rsi(df_1h, '1h'), 
            '4h': pt_rsi(df_4h, '4h'), '1d': pt_rsi(df_1d, '1d')
        }
    
    df = gop_va_dong_bo_data(du_lieu_cac_khung)
    
    # 1. Định nghĩa các ngưỡng tối ưu cho từng khung (đã phân tích từ vector)
    nguong = {
        'rsi_1d':  {'buy': 35, 'sell': 45},
        'rsi_4h':  {'buy': 45, 'sell': 50},
        'rsi_1h':  {'buy': 48, 'sell': 52},
        'rsi_30m': {'buy': 50, 'sell': 50},
        'rsi_15m': {'buy': 50, 'sell': 50},
        'rsi_5m':  {'buy': 55, 'sell': 45},
        'rsi_3m':  {'buy': 55, 'sell': 45},
        'rsi_1m':  {'buy': 60, 'sell': 40}
    }

    # 2. Khởi tạo các cột tính điểm đồng thuận
    df['buy_score'] = 0
    df['sell_score'] = 0

    # Danh sách các cột RSI có trong dữ liệu của bạn
    rsi_columns = ['rsi_1m', 'rsi_3m', 'rsi_5m', 'rsi_15m', 'rsi_30m', 'rsi_1h', 'rsi_4h', 'rsi_1d']

    # 3. Tính điểm (Score) dựa trên sự đồng thuận của các khung
    for col in rsi_columns:
        if col in df.columns:
            df['buy_score'] += (df[col] > nguong[col]['buy']).astype(int)
            
            df['sell_score'] += (df[col] < nguong[col]['sell']).astype(int)

    df['signal'] = 0 
    
    # ĐIỀU KIỆN MUA (LONG):
    # - Ít nhất 6 trên 8 khung đồng thuận Tăng (buy_score >= 6)
    # - Quan trọng: Khung 1 giờ (rsi_1h) phải giữ được trên mức 48 để tránh sập hầm
    if 'rsi_1h' in df.columns:
        dk_mua = (df['buy_score'] >= 6) & (df['rsi_1h'] > 48)
        dk_ban = (df['sell_score'] >= 6) & (df['rsi_1h'] < 52)
    else:
        # Dự phòng nếu thiếu cột rsi_1h
        dk_mua = (df['buy_score'] >= 6)
        dk_ban = (df['sell_score'] >= 6)

    df.loc[dk_mua, 'signal'] = 1
    df.loc[dk_ban, 'signal'] = -1

    # 5. Xác định điểm kích hoạt lệnh (Entry Trigger)
    # entry_signal = 1 hoặc -1 tại chính xác thời điểm tín hiệu mới bắt đầu xuất hiện
    # Thêm fillna(0) để bảo vệ dòng đầu tiên không bị lỗi NaN
    df['entry_signal'] = df['signal'].diff().fillna(0)
    
    return df