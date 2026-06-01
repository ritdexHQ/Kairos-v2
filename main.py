"""
Kairos Quant System v2.0 - Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Điểm khởi động ứng dụng PyQt6. Tạo cửa sổ chính với 4 tab:
  • Giao dịch Realtime  – kết nối sàn, thực thi lệnh thật
  • Tài khoản Demo      – paper trading không rủi ro
  • Chiến thuật & Backtest – kiểm thử chiến lược trên dữ liệu quá khứ
  • Phân tích Vector    – biểu đồ nến nâng cao, multi-timeframe
"""
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtGui import QIcon

# Tính toán đường dẫn gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# --- THÊM 2 DÒNG NÀY VÀO ---
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ---------------------------

# Import các Class giao diện từ 4 file của bạn
try:
    from hien_thi.dashboard_vectorized import CandlestickChartWidget
    from hien_thi.dashboard_backtest import DraggableDashboard
    from hien_thi.dashboard_demo import MainDashboard_demo
    from hien_thi.dashboard_realtime import MainDashboard_realtime
except ImportError as e:
    print(f"Lỗi import: {e}. Vui lòng kiểm tra lại đường dẫn và tên Class.")
    sys.exit(1)

# --- MÀU SẮC ĐỒNG BỘ ---
BG_COLOR = "#0B0E14"
CARD_BG = "#131722"
TEXT_MAIN = "#D1D4DC"
TEXT_SUB = "#787B86"
ACCENT_COLOR = "#2A3B6C"

class MetaTraderClone(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KAIROS QUANT SYSTEM v2.0")
        self.resize(1280, 800)
        
        # Khởi tạo Widget trung tâm là QTabWidget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Thiết lập Style cho Tab (Dark Mode)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab {
                background: #2b2b2b; color: #b1b1b1;
                padding: 10px 20px; border-top-left-radius: 4px;
            }
            QTabBar::tab:selected { background: #3c3c3c; color: white; }
        """)

        self.init_ui()

    def init_ui(self):
        # 1. Tab Realtime (Giao diện chính để trade)
        self.tab_realtime = MainDashboard_realtime()
        self.tabs.addTab(self.tab_realtime, QIcon(), "Giao dịch Realtime")

        # 2. Tab Demo
        self.tab_demo = MainDashboard_demo()
        self.tabs.addTab(self.tab_demo, "Tài khoản Demo")

        # 3. Tab Backtest (Phân tích dữ liệu quá khứ)
        self.tab_backtest = DraggableDashboard()
        self.tabs.addTab(self.tab_backtest, "Chiến thuật & Backtest")

        # 4. Tab Vectorized Chart (Biểu đồ phân tích nâng cao)
        self.tab_vector = CandlestickChartWidget()
        self.tabs.addTab(self.tab_vector, "Phân tích Vector")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MetaTraderClone()
    window.show()
    sys.exit(app.exec())