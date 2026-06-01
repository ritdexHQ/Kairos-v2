"""
chien_luoc/logic_bar_to_bar/chien_luoc/chien_luoc_theo_trend_following.py – Theo xu hướng
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tín hiệu: EMA hướng + ATR xác nhận biến động + RSI hỗ trợ.
Khung dài (1D/4H) có trọng số cao hơn để tránh noise ngắn hạn.
Dùng khi ML phát hiện regime Xu_Hướng_Mạnh (state 3) hoặc Đầu_Xu_Hướng (state 2).
"""
from joblib import Parallel, delayed
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.bien_dong import pt_atr
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.xu_huong import pt_ema_trend
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.dong_luong_dao_chieu import pt_rsi


#df_1m có 300 nến, df_3m có 100 nến, df_5m có 300 nến, df_15m có 100 nến, df_30m có 300 nến, df_1h có 150 nến, df_4h có 300 nến, df_1d có 50 nến
def tinh_toan_diem_xu_huong_don_vi(df, name, weight):
    """Xử lý tính toán cho từng khung thời gian trên một nhân CPU riêng biệt"""
    # 1. Gọi các hàm chỉ báo (Đảm bảo các hàm này đã được import)
    ema = pt_ema_trend(df)
    atr = pt_atr(df)
    rsi = pt_rsi(df)
    
    diem_khung = 0
    
    # Tính toán điểm dựa trên EMA
    if ema['trang_thai'] == 'TĂNG':
        diem_khung += weight
    else:
        diem_khung -= weight
            
    # 2. Hệ số nhân từ ATR (Biến động cao thì tin tưởng xu hướng hơn)
    if atr['trang_thai'] == 'BIẾN_ĐỘNG_CAO':
        diem_khung += (1 if ema['trang_thai'] == 'TĂNG' else -1) * (weight * 0.5)
            
    # 3. Hỗ trợ từ RSI
    if rsi['trang_thai'] == 'MẠNH' and ema['trang_thai'] == 'TĂNG':
        diem_khung += weight * 0.2
    elif rsi['trang_thai'] == 'YẾU' and ema['trang_thai'] == 'GIẢM':
        diem_khung -= weight * 0.2

    # Tạo thông điệp lý do cho khung này
    msg = f"{name}: {ema['trang_thai']} (RSI {rsi['trang_thai']})"
    
    return diem_khung, msg

def phan_tich_theo_xu_huong(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    danh_sach_khung = [
        (df_1d, "1D", 10),
        (df_4h, "4H", 8),
        (df_1h, "1H", 5),
        (df_15m, "15M", 3),
        (df_5m, "5M", 2)
    ]

    ket_qua = Parallel(n_jobs=-1)(
        delayed(tinh_toan_diem_xu_huong_don_vi)(df, name, weight) 
        for df, name, weight in danh_sach_khung
    )

    diem_tong = sum(res[0] for res in ket_qua)
    ly_do_list = [res[1] for res in ket_qua if res[1] != ""]
    
    tin_hieu = 'buy' if diem_tong > 15 else ('sell' if diem_tong < -15 else None)

    # TRẢ VỀ: 3 giá trị (để hàm vao_lenh nhận đủ)
    return tin_hieu, round(diem_tong, 2), "; ".join(ly_do_list)
