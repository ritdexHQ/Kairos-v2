"""
chien_luoc/logic_bar_to_bar/chien_luoc/chien_luoc_mean_reversion.py – Đảo chiều về trung bình
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tín hiệu: giá kiệt sức – RSI quá mua/bán + xác nhận Bollinger Bands + Volume đột biến.
Dùng khi ML phát hiện regime Hồi_Quy (state 5) hoặc Nhiễu_Động (state 6).
"""
from joblib import Parallel, delayed
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.khoi_luong import pt_volume
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.xu_huong import pt_ema_trend
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.bien_dong import pt_bollinger_squeeze
from chien_luoc.logic_bar_to_bar.phan_tich_ky_thuat.dong_luong_dao_chieu import pt_rsi

#df_1m có 300 nến, df_3m có 100 nến, df_5m có 300 nến, df_15m có 100 nến, df_30m có 300 nến, df_1h có 150 nến, df_4h có 300 nến, df_1d có 50 nến
def tinh_toan_diem_khung(df, name, weight):
    """
    Hàm xử lý độc lập cho từng khung thời gian. 
    Sử dụng .values[index] thay vì .iloc[index] để truy cập tốc độ NumPy.
    """
    # 1. Tính toán các chỉ báo
    rsi = pt_rsi(df)
    bb = pt_bollinger_squeeze(df)
    vol = pt_volume(df)
    ema = pt_ema_trend(df)
    
    # Lấy giá trị đóng cửa cuối cùng nhanh hơn qua NumPy
    price_now = df['close'].values[-1]
    
    diem = 0
    ly_do = ""

    # --- LOGIC ĐẢO CHIỀU GIẢM ---
    if rsi['muc_do'] == 'QUÁ_MUA' and ema['trang_thai'] == 'TĂNG':
        diem_dao_chieu = 0
        # Nếu volume cực cao tại đỉnh -> Dấu hiệu xả hàng
        if vol['trang_thai'] == 'DOT_BIEN':
            diem_dao_chieu += 2
        # Nếu giá vượt ngoài biên trên Bollinger
        if price_now > bb['upper_band']: # Giả định hàm BB trả về upper_band
            diem_dao_chieu += 2
            
        diem -= (diem_dao_chieu * weight)
        if diem_dao_chieu >= 2:
            ly_do = f"{name}: Kiệt sức TĂNG (RSI:{rsi['rsi_val']:.1f})"

        # --- LOGIC ĐẢO CHIỀU TĂNG (Tìm đáy để BUY) ---
    elif rsi['muc_do'] == 'QUÁ_BÁN' and ema['trang_thai'] == 'GIẢM':
        diem_dao_chieu = 0
        if vol['trang_thai'] == 'DOT_BIEN':
            diem_dao_chieu += 2
        if price_now < bb['lower_band']:
            diem_dao_chieu += 2
                
        diem += (diem_dao_chieu * weight)
        if diem_dao_chieu >= 2:
            ly_do = f"{name}: Kiệt sức GIẢM (RSI:{rsi['rsi_val']:.1f})"

    return diem, ly_do

def phan_tich_dao_chieu(df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d):
    # Cấu hình khung thời gian
    cau_hinh_khung = [
        (df_1h, "1H", 5),
        (df_15m, "15M", 3),
        (df_5m, "5M", 2)
    ]

    # THỰC THI SONG SONG: Chạy các khung thời gian cùng lúc trên nhiều nhân CPU
    # n_jobs=-1: Dùng tất cả nhân CPU hiện có
    ket_qua = Parallel(n_jobs=-1)(
        delayed(tinh_toan_diem_khung)(df, name, weight) for df, name, weight in cau_hinh_khung
    )

    # Tổng hợp kết quả
    tong_diem = sum(res[0] for res in ket_qua)
    # Lọc và nối các lý do (tránh việc chỉ lưu lý do cuối cùng)
    ly_do_list = [res[1] for res in ket_qua if res[1] != ""]
    ly_do_tong = " | ".join(ly_do_list) if ly_do_list else ""

    # Quyết định tín hiệu
    nguong_dao_chieu = 12
    tin_hieu = "buy" if tong_diem >= nguong_dao_chieu else ("sell" if tong_diem <= -nguong_dao_chieu else None)

    return tin_hieu, round(tong_diem, 2), ly_do_tong