import sys
import os
from datetime import datetime

# --- IMPORT RICH ---
from rich.console import Console
from rich.live import Live


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from utils.log import logger
    from utils.doc_cau_hinh import lay_cau_hinh_giao_dich, lay_cau_hinh_ao
    from lay_du_lieu.lay_ohlcv import gop_nen, tai_du_lieu_lich_su, chuan_bi_du_lieu_da_khung
    from ml.tool.data_filter import tao_log_can_bang
    from ml.trang_thai_thi_truong_ml.ml_model import huan_luyen_model
    from ml.trang_thai_thi_truong_ml.ml_predict import du_doan_trang_thai_ml, danh_gia_ml, STATE_MAP
    from ml.tool.trading_teacher import TradingTeacher
    from ml.tool.dashboard import hien_thi_dashboard
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

def tao_model_lan_dau():
    config_backtest = lay_cau_hinh_ao()
    config_trading = lay_cau_hinh_giao_dich()
    
    START_DATE = config_backtest.get('ngay_bat_dau', '2025-01-01')
    END_DATE = config_backtest.get('ngay_ket_thuc', '2025-01-31')
    DS_SYMBOL = config_trading.get('cap_giao_dich', ['BTC/USDT'])

    symbol = DS_SYMBOL[0] 

    df_goc = tai_du_lieu_lich_su(symbol, START_DATE, END_DATE)

    if df_goc is None or df_goc.is_empty():
        logger.error("Không có dữ liệu để huấn luyện!")
        return
    
    current_time = df_goc.get_column("timestamp").last()

    dfs = chuan_bi_du_lieu_da_khung(df_goc, current_time)

    df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d = dfs

    huan_luyen_model(df_5m, df_15m, df_1h, df_4h)

console = Console()
def chay_training_cap_toc():
    config_backtest = lay_cau_hinh_ao()
    config_trading = lay_cau_hinh_giao_dich()
    
    START_DATE = config_backtest.get('ngay_bat_dau', '2025-01-01')
    END_DATE = config_backtest.get('ngay_ket_thuc', '2025-01-31')
    DS_SYMBOL = config_trading.get('cap_giao_dich', ['BTC/USDT'])

    with Live(console=console, refresh_per_second=10) as live:
        
        for symbol in DS_SYMBOL:
            
            df_goc = tai_du_lieu_lich_su(symbol, START_DATE, END_DATE)
            if df_goc is None or df_goc.is_empty(): continue

            idx_start = 43200 
            if df_goc.height < idx_start: continue
            timestamps = df_goc.get_column("timestamp").slice(idx_start).to_list()
            
            # Khởi tạo bộ đếm
            stats = {'correct': 0, 'wrong': 0}
            class_stats = {k: {'correct': 0, 'total': 0} for k in STATE_MAP.keys()}

            teacher = TradingTeacher()
           
            for current_time in timestamps:
                dfs = chuan_bi_du_lieu_da_khung(df_goc, current_time)
                if not dfs: continue

                df_1m, df_3m, df_5m, df_15m, df_30m, df_1h, df_4h, df_1d = dfs

                packet = du_doan_trang_thai_ml(df_5m, df_15m, df_1h, df_4h, last_state=teacher.last_state)

                if packet is None: continue
                ai_state = packet['state_id']
                conf = packet['confidence']

                teacher_state, teacher_conf = teacher.detect_regime(df_5m, df_15m, df_1h, df_4h)
                
                class_stats[teacher_state]['total'] += 1
                
                if ai_state == teacher_state:
                    stats['correct'] += 1
                    class_stats[teacher_state]['correct'] += 1
                    danh_gia_ml(packet, 1, 0, teacher_state) # Thưởng 1 điểm
                else:
                    stats['wrong'] += 1
                    danh_gia_ml(packet, -0.5, 0, teacher_state) # Phạt 0.5 điểm

                #time_str = current_time.strftime("%Y-%m-%d %H:%M") if isinstance(current_time, datetime) else str(current_time)
                layout = hien_thi_dashboard(symbol, current_time, stats, class_stats, ai_state, teacher_state, teacher_conf, conf)
                live.update(layout)

            console.print(f"[bold green]✅ HOÀN TẤT TRAINING {symbol}![/]")

def main():
    print("=== CHỌN CHỨC NĂNG ===")
    print("1. Tạo model lần đầu")
    print("2. Chạy training cấp tốc")
    print("3. Tự động học từ log")
    print("4. Lọc dữ liệu lệnh thắng từ log")
    print("0. Thoát")

    choice = input("Nhập lựa chọn: ").strip()

    if choice == "1":
        tao_model_lan_dau()

    elif choice == "2":
        chay_training_cap_toc()

    elif choice == "3":
        from ml.trang_thai_thi_truong_ml.ml_model import tu_dong_hoc_tu_log
        tu_dong_hoc_tu_log()

    elif choice == "4":
        tao_log_can_bang()


    elif choice == "0":
        print("Thoát chương trình.")

    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
