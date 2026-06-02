"""
Kairos Quant System v2 – Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chọn chức năng cần chạy từ menu.
Chạy từ thư mục gốc dự án:
    python main.py
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─── MENU ────────────────────────────────────────────────────────────────────

MENU = """
╔══════════════════════════════════════════════════════╗
║            KAIROS QUANT SYSTEM  v2                   ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   [1]  Giao dich Realtime      (live trading)        ║
║   [2]  Demo / Paper Trading    (khong rui ro)        ║
║                                                      ║
║   [3]  Backtest Don luong      (bar-to-bar)          ║
║   [4]  Backtest Da luong       (bar-to-bar parallel) ║
║   [5]  Vectorized Backtest     (toan bo dataset)     ║
║                                                      ║
║   [6]  ML Training             (huan luyen model)    ║
║                                                      ║
║   [7]  Dashboard Analytics     (GUI PyQt6)           ║
║                                                      ║
║   [0]  Thoat                                         ║
╚══════════════════════════════════════════════════════╝
"""


def chon_chuc_nang():
    print(MENU)
    lua_chon = input("Chon chuc nang [0-7]: ").strip()
    return lua_chon


# ─── RUNNERS ─────────────────────────────────────────────────────────────────

def chay_realtime():
    from chuc_nang.chay_realtime import chay_realtime as _run
    import threading, signal

    _run()

    stop = threading.Event()
    def _exit(sig, frame):
        print("\nDang tat bot...")
        stop.set()

    signal.signal(signal.SIGINT, _exit)
    stop.wait()


def chay_demo():
    from chuc_nang.chay_demo import chay_demo as _run
    import threading, signal

    _run()

    stop = threading.Event()
    def _exit(sig, frame):
        print("\nDang tat demo...")
        stop.set()

    signal.signal(signal.SIGINT, _exit)
    stop.wait()


def chay_backtest_don():
    from chuc_nang.backtest_donluong import chay_backtest
    chay_backtest()


def chay_backtest_da():
    from chuc_nang.backtest_daluong import chay_backtest
    chay_backtest()


def chay_vectorized():
    from chuc_nang.vectorized_backtest import vectorized_backtest
    lich_su, du_lieu = vectorized_backtest()
    if not lich_su:
        print("Khong co lenh nao duoc thuc hien.")


def chay_ml():
    import importlib.util, subprocess
    ml_main = os.path.join(PROJECT_ROOT, 'ml', 'main.py')
    if os.path.exists(ml_main):
        subprocess.run([sys.executable, ml_main], cwd=PROJECT_ROOT)
    else:
        print(f"Khong tim thay {ml_main}")


def chay_dashboard():
    from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
    from PyQt6.QtGui import QIcon

    try:
        from hien_thi.dashboard_vectorized import CandlestickChartWidget
        from hien_thi.dashboard_backtest import DraggableDashboard
        from hien_thi.dashboard_demo import MainDashboard_demo
        from hien_thi.dashboard_realtime import MainDashboard_realtime
    except ImportError as e:
        print(f"Loi import dashboard: {e}")
        return

    app = QApplication.instance() or QApplication(sys.argv)

    class KairosWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Kairos v2 – Analytics Dashboard")
            self.resize(1440, 900)
            self.setStyleSheet("""
                QMainWindow  { background-color: #0B0E14; }
                QTabWidget::pane { border: 1px solid #2A2E39; }
                QTabBar::tab {
                    background: #131722; color: #787B86;
                    padding: 10px 24px; font-size: 13px;
                }
                QTabBar::tab:selected { background: #1E2230; color: #D1D4DC; }
                QTabBar::tab:hover    { color: #D1D4DC; }
            """)
            tabs = QTabWidget()
            self.setCentralWidget(tabs)
            tabs.addTab(MainDashboard_realtime(), "Realtime")
            tabs.addTab(MainDashboard_demo(),     "Demo")
            tabs.addTab(DraggableDashboard(),     "Backtest")
            tabs.addTab(CandlestickChartWidget(), "Vectorized")

    window = KairosWindow()
    window.show()
    sys.exit(app.exec())


# ─── DISPATCH ────────────────────────────────────────────────────────────────

DISPATCH = {
    '1': chay_realtime,
    '2': chay_demo,
    '3': chay_backtest_don,
    '4': chay_backtest_da,
    '5': chay_vectorized,
    '6': chay_ml,
    '7': chay_dashboard,
}


if __name__ == '__main__':
    lua_chon = chon_chuc_nang()

    if lua_chon == '0':
        print("Tam biet.")
        sys.exit(0)

    handler = DISPATCH.get(lua_chon)
    if handler is None:
        print(f"Lua chon '{lua_chon}' khong hop le.")
        sys.exit(1)

    try:
        handler()
    except KeyboardInterrupt:
        print("\nDa dung.")
    except Exception as e:
        print(f"Loi: {e}")
        raise
