""" THỜI GIAN & CHU KỲ (Time / Cycle) ⭐
👉 Khi nào thị trường hay phản ứng?
- Phiên Á – Âu – Mỹ
- Thời điểm ra tin
- Chu kỳ nến
- Session Range
📌 Dùng để:
- Tránh giao dịch lúc nhiễu
- Giảm leverage giờ rủi ro
- Bot futures rất cần """

import numpy as np
import pandas as pd

def pt_kiem_tra_ngay(df):
    """
    Kiểm tra ngày trong tuần cho toàn bộ DataFrame (Vectorized).
    Đầu vào: df (có Index là Datetime), time_frame (tên khung thời gian).
    """
    # 1. Trích xuất thứ từ Index (0: Thứ 2, 6: Chủ Nhật)
    thu_so = df.index.dayofweek 
    
    # 3. Định nghĩa điều kiện (Ví dụ: Crypto chạy 24/7 nên thường là HỢP_LỆ)
    conditions = [
        (thu_so <= 4), # Thứ 2 đến Thứ 6
        (thu_so > 4)   # Thứ 7, Chủ Nhật
    ]
    choices = ['True', 'False']
    
    # 4. Gán trạng thái Vectorized
    df['check_days'] = np.select(conditions, choices, default='None')
    
    return df

def pt_kiem_tra_gio(df):

    gio_xau = [5, 6]

    df['check_hours'] = ~df.index.hour.isin(gio_xau)
    
    return df