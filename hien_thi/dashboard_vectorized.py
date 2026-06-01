import sys
import os
import polars as pl
import pandas as pd
import bisect
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QComboBox, QApplication, QSplitter, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from chuc_nang.vectorized_backtest import vectorized_backtest
except ImportError:
    vectorized_backtest = None

# ==========================================
# 1. BẢNG MÀU CHUYÊN NGHIỆP (THEME)
# ==========================================
class Theme:
    BG = "#0e0e0e"
    CARD = "#0e0e0e"
    BORDER = "#2A2E39"
    GRID = "#1E222D"       
    TEXT_MAIN = "#D1D4DC"
    TEXT_SUB = "#787B86"
    WIN = "#089981"        
    LOSS = "#F23645"       
    ACCENT = "#C8AA6E"
    ENTRY = "#2962FF"      
    EXIT = "#FF9800"       
    TRADE_LINE = "#4c525e" 

def to_datetime(val):
    if val is None: return None
    if isinstance(val, datetime): return val.replace(tzinfo=None)
    try: return pd.to_datetime(val).replace(tzinfo=None)
    except: return None

# ==========================================
# 2. ĐA LUỒNG XỬ LÝ (CHỐNG LAG)
# ==========================================
class BacktestWorker(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    
    def run(self):
        try:
            if vectorized_backtest is None:
                self.error.emit("Không tìm thấy hàm vectorized_backtest")
                return
            trades, dict_dfs = vectorized_backtest()
            self.finished.emit(trades, dict_dfs)
        except Exception as e:
            self.error.emit(str(e))

class DataProcessorWorker(QThread):
    finished = pyqtSignal(object, list)
    
    def __init__(self, df_base, tf):
        super().__init__()
        self.df_base = df_base
        self.tf = tf
        
    def run(self):
        try:
            if self.tf == "1m" or self.df_base.is_empty():
                df_res = self.df_base
            else:
                agg_cols = [
                    pl.col("open").first(), pl.col("high").max(),
                    pl.col("low").min(), pl.col("close").last(),
                    pl.col("volume").sum() if "volume" in self.df_base.columns else pl.lit(0).alias("volume")
                ]
                
                # Giữ lại các cột tín hiệu quan trọng
                for col_name in ["signal", "entry_signal", "buy_score", "sell_score"]:
                    if col_name in self.df_base.columns:
                        agg_cols.append(pl.col(col_name).last())
                        
                df_res = self.df_base.group_by_dynamic("timestamp", every=self.tf).agg(agg_cols)
                
            timestamps = [to_datetime(ts) for ts in df_res["timestamp"].to_list()]
            self.finished.emit(df_res, timestamps)
        except Exception as e:
            print(f"Data Worker Error: {e}")
            timestamps = [to_datetime(ts) for ts in self.df_base["timestamp"].to_list()]
            self.finished.emit(self.df_base, timestamps)

# ==========================================
# 3. BẢNG QUẢN LÝ LỆNH (TABLE)
# ==========================================
class TradesTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cols = ["Id", "Type", "Price", "Change"]
        self.setColumnCount(len(self.cols))
        self.setHorizontalHeaderLabels(self.cols)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
        self.setStyleSheet(f"""
            QTableWidget {{ background-color: {Theme.CARD}; color: {Theme.TEXT_MAIN}; gridline-color: {Theme.GRID}; border: none; font-size: 12px; }}
            QHeaderView::section {{ background-color: {Theme.BG}; color: {Theme.TEXT_SUB}; font-weight: bold; padding: 8px 5px; border: 1px solid {Theme.BORDER}; border-top: none; }}
            QTableWidget::item:selected {{ background-color: {Theme.ACCENT}; color: white; }}
        """)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def load_trades(self, trades):
        self.setRowCount(0)
        for i, t in enumerate(trades):
            if 'exit_time' not in t: continue
            
            row_idx = self.rowCount()
            self.insertRow(row_idx)
            price_change_val = float(t.get('price_change', 0.0))
            
            row_data = [str(t['id']), str(t['direction']), str(t['hold_time']), f"{price_change_val:+.2f}"]
            for col, text in enumerate(row_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 3: 
                    color = Theme.WIN if price_change_val > 0 else Theme.LOSS if price_change_val < 0 else Theme.TEXT_MAIN
                    item.setForeground(QColor(color))
                self.setItem(row_idx, col, item)

# ==========================================
# 4. LÕI BIỂU ĐỒ VECTOR CHUYÊN NGHIỆP
# ==========================================
class CoreCandlestickChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.df_current = pl.DataFrame()
        self.current_timestamps = []
        self.trades = []
        
        self.candle_width = 8
        self.candle_gap = 2
        self.scroll_offset = 0 
        self.mouse_pos = None
        self.last_mouse_x = 0
        self.is_panning = False
        self.current_tf = "1m"

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.candle_width = min(40, self.candle_width + 2) if delta > 0 else max(3, self.candle_width - 2)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True; self.last_mouse_x = event.position().x()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.is_panning = False

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.position() 
        if self.is_panning:
            dx = event.position().x() - self.last_mouse_x
            candles_shifted = int(dx / (self.candle_width + self.candle_gap))
            if candles_shifted != 0:
                self.scroll_offset += candles_shifted
                max_offset = max(0, self.df_current.height - 5)
                self.scroll_offset = max(0, min(self.scroll_offset, max_offset))
                self.last_mouse_x = event.position().x()
        self.update() 

    def leaveEvent(self, event):
        self.mouse_pos = None; self.update()

    def get_x(self, timestamp, start_idx, end_idx, total_candles, chart_w, space_per_candle):
        if timestamp is None: return -1
        idx = bisect.bisect_left(self.current_timestamps, timestamp)
        if idx < len(self.current_timestamps):
            if idx < start_idx or idx > end_idx: return -1 
            return chart_w - (total_candles - idx - self.scroll_offset) * space_per_candle + self.candle_width / 2
        return -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(Theme.CARD))
        
        if self.df_current.is_empty():
            painter.setPen(QColor(Theme.TEXT_SUB))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Đang chờ dữ liệu...")
            return

        w, h = self.width(), self.height()
        margin_top, margin_bottom, margin_right = 25, 40, 70
        chart_w, chart_h = w - margin_right, h - margin_top - margin_bottom
        
        space_per_candle = self.candle_width + self.candle_gap
        max_visible_candles = int(chart_w // space_per_candle)
        total_candles = self.df_current.height
        end_idx = total_candles - self.scroll_offset
        start_idx = max(0, end_idx - max_visible_candles)
        
        if start_idx >= end_idx: return
        df_view = self.df_current[start_idx:end_idx]
        
        timestamps = [to_datetime(ts) for ts in df_view["timestamp"].to_list()]
        opens = df_view["open"].to_list()
        highs = df_view["high"].to_list()
        lows = df_view["low"].to_list()
        closes = df_view["close"].to_list()
        volumes = df_view["volume"].to_list() if "volume" in df_view.columns else [0] * len(opens)
        
        min_low, max_high = min(lows), max(highs)
        price_range = max_high - min_low if max_high != min_low else 1
        min_low -= price_range * 0.1
        max_high += price_range * 0.1
        price_range = max_high - min_low
        
        def get_y(price): return margin_top + chart_h - ((price - min_low) / price_range) * chart_h

        # LƯỚI TỌA ĐỘ
        grid_pen = QPen(QColor(Theme.GRID), 1, Qt.PenStyle.SolidLine)
        painter.setPen(grid_pen)
        painter.setFont(QFont("Segoe UI", 8))
        
        for i in range(9):
            py = margin_top + (chart_h / 8) * i
            painter.drawLine(0, int(py), int(chart_w), int(py))
            painter.setPen(QColor(Theme.TEXT_SUB))
            painter.drawText(int(chart_w + 8), int(py + 4), f"{max_high - (price_range / 8) * i:.4f}")
            painter.setPen(grid_pen)

        fm_time = painter.fontMetrics()
        time_step = max(1, int(fm_time.horizontalAdvance("00:00 00/00") * 1.5 // space_per_candle))

        for i in range(len(opens)):
            if (start_idx + i) % time_step == 0:
                x = chart_w - (len(opens) - i) * space_per_candle + self.candle_width / 2
                painter.drawLine(int(x), int(margin_top), int(x), int(margin_top + chart_h))

        hovered_candle_idx = -1 
        max_vol = max(volumes) if volumes and max(volumes) > 0 else 1
        vol_max_height = chart_h * 0.25 

        # VẼ NẾN
        for i in range(len(opens)):
            o, hi, lo, c, vol = opens[i], highs[i], lows[i], closes[i], volumes[i]
            x = chart_w - (len(opens) - i) * space_per_candle
            center_x = x + self.candle_width / 2
            yo, yc, yh, yl = get_y(o), get_y(c), get_y(hi), get_y(lo)
            candle_color = QColor(Theme.WIN) if c >= o else QColor(Theme.LOSS)

            if self.mouse_pos and x <= self.mouse_pos.x() <= x + space_per_candle:
                hovered_candle_idx = i
                painter.fillRect(QRectF(x - self.candle_gap/2, margin_top, space_per_candle, chart_h), QColor(255, 255, 255, 8))

            vol_h = (vol / max_vol) * vol_max_height
            vol_brush = QColor(candle_color); vol_brush.setAlpha(80) 
            painter.fillRect(QRectF(x, margin_top + chart_h - vol_h, self.candle_width, vol_h), vol_brush)

            painter.setPen(QPen(candle_color, 1.5))
            painter.drawLine(int(center_x), int(yh), int(center_x), int(yl))
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(candle_color)
            painter.drawRect(QRectF(x, min(yo, yc), self.candle_width, max(abs(yo - yc), 1)))

        # VẼ TRỤC X
        for i in range(len(opens)):
            if (start_idx + i) % time_step == 0:
                x = chart_w - (len(opens) - i) * space_per_candle + self.candle_width / 2
                painter.setPen(QPen(QColor(Theme.BORDER), 1.5))
                painter.drawLine(int(x), int(margin_top + chart_h), int(x), int(margin_top + chart_h + 6))
                
                painter.setPen(QColor(Theme.TEXT_SUB))
                ts = timestamps[i]
                if ts:
                    time_str = ts.strftime('%H:%M\n%d/%m') if self.current_tf in ['1m', '3m', '5m', '15m'] else ts.strftime('%d/%m\n%H:%M')
                    painter.drawText(QRectF(x - 30, margin_top + chart_h + 8, 60, 30), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, time_str)

        # VẼ LỆNH & TOOLTIP
        for t in self.trades:
            if 'exit_time' not in t: continue
            
            # 1. Tìm index của cây nến Entry để lấy giá High/Low thực tế
            en_time = t['entry_time']
            en_idx = bisect.bisect_left(self.current_timestamps, en_time)
            
            # Bỏ qua nếu không tìm thấy nến hoặc nến nằm ngoài vùng hiển thị
            if en_idx >= self.df_current.height: continue

            # Lấy dữ liệu nến thực tế tại điểm vào lệnh
            # entry_candle = self.df_current.row(en_idx, named=True) # Cách này chậm
            # Nên lấy trực tiếp từ List (opens, highs, lows, closes) đã có sẵn ở trên paintEvent
            # Nhưng ta cần tính toán lại start_idx/end_idx, việc này phức tạp.
            
            # Giải pháp tối ưu: Lấy giá trị trực tiếp từ DataFrame view
            # Ta cần chuyển đổi index tuyệt đối (en_idx) thành index tương đối trong df_view
            view_idx = en_idx - start_idx
            if not (0 <= view_idx < df_view.height): continue

            candle_high = highs[view_idx]
            candle_low = lows[view_idx]

            # Tính toán tọa độ X, Y
            x1 = self.get_x(en_time, start_idx, end_idx, total_candles, chart_w, space_per_candle)
            x2 = self.get_x(t['exit_time'], start_idx, end_idx, total_candles, chart_w, space_per_candle)
            y2 = get_y(t['exit_price'])

            # Không tính y1 từ entry_price nữa, ta sẽ tính trực tiếp khi vẽ dựa trên high/low

            direction = t['direction']
            val = float(t.get('price_change', 0))
            
            # --- Chỉ vẽ nếu điểm vào hoặc điểm ra nằm trong khung nhìn ---
            if x1 == -1 and x2 == -1: continue

            # 2. Xử lý tọa độ X nếu nằm ngoài khung nhìn (để vẽ đường nối)
            # (Đoạn này giữ nguyên logic cũ của bạn)
            if x1 == -1: 
                x1 = chart_w - (total_candles - en_idx - self.scroll_offset) * space_per_candle + self.candle_width/2
            if x2 == -1: 
                idx2 = bisect.bisect_left(self.current_timestamps, t['exit_time'])
                x2 = chart_w - (total_candles - idx2 - self.scroll_offset) * space_per_candle + self.candle_width/2

            # 3. Tính toán lại y1 từ entry_price ĐỂ VẼ ĐƯỜNG NỐI (cho chính xác mức giá vào)
            y1_price = get_y(t['entry_price'])
            
            # Nét đứt nối lệnh
            painter.setPen(QPen(QColor(Theme.TRADE_LINE), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(int(x1), int(y1_price), int(x2), int(y2))

            # 4. ĐÁNH DẤU ĐIỂM VÀO LỆNH (Mũi tên Xanh dương) - FIX LỖI RÂU NẾN
            if 0 <= x1 <= chart_w:
                painter.setPen(Qt.PenStyle.NoPen)
                poly = QPainterPath()
                padding = 6

                if direction == 'Long':
                    # --- LỆNH LONG: TAM GIÁC XANH DƯỚI NẾN ---
                    painter.setBrush(QColor(Theme.ENTRY)) # Màu mặc định là Xanh dương
                    y_base = get_y(candle_low)
                    top_y = y_base + padding
                    poly.moveTo(x1, top_y)   # Đỉnh nhọn hướng lên
                    poly.lineTo(x1 - 6, top_y + 12)
                    poly.lineTo(x1 + 6, top_y + 12)
                    poly.closeSubpath()
                else:
                    # --- LỆNH SHORT: TAM GIÁC TÍM TRÊN NẾN, ĐỈNH CHỈ XUỐNG ---
                    painter.setBrush(QColor("#C517FF")) 
                    y_base = get_y(candle_high)    # Lấy giá cao nhất của nến làm gốc
                    padding = 6                    # Khoảng cách từ râu nến đến đỉnh nhọn
                    top_y = y_base - padding       # Tọa độ Y của đỉnh nhọn (nằm phía trên nến)
                    poly = QPainterPath()
                    # 1. Đỉnh nhọn nằm dưới cùng của hình tam giác (sát nến nhất)
                    poly.moveTo(x1, top_y)          
                    # 2. Đáy trái nằm cao hơn đỉnh 12px (Y giảm đi)
                    poly.lineTo(x1 - 6, top_y - 12) 
                    # 3. Đáy phải nằm cao hơn đỉnh 12px (Y giảm đi)
                    poly.lineTo(x1 + 6, top_y - 12) 
                    poly.closeSubpath()
                    
                painter.drawPath(poly)

            # 5. ĐÁNH DẤU ĐIỂM THOÁT LỆNH (Dấu X màu Cam)
            if 0 <= x2 <= chart_w:
                painter.setPen(QPen(QColor(Theme.EXIT), 2))
                size = 5 
                painter.drawLine(int(x2 - size), int(y2 - size), int(x2 + size), int(y2 + size))
                painter.drawLine(int(x2 - size), int(y2 + size), int(x2 + size), int(y2 - size))

            # 6. HOVER TOOLTIP (Rê chuột vào dấu X)
            if self.mouse_pos and abs(self.mouse_pos.x() - x2) <= space_per_candle and 0 <= x2 <= chart_w:
                pnl_str = f"{val:+.2f}"
                tw = fm_time.horizontalAdvance(pnl_str) + 16
                pnl_color = QColor(Theme.WIN) if val > 0 else QColor(Theme.LOSS) if val < 0 else QColor(Theme.TEXT_MAIN)
                
                bg_col = QColor(pnl_color); bg_col.setAlpha(220) 
                painter.setBrush(bg_col); painter.setPen(QPen(pnl_color, 1))
                painter.drawRoundedRect(QRectF(x2 - tw/2, y2 - 35, tw, 22), 4, 4)
                painter.setPen(QColor("#FFFFFF")); painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(QRectF(x2 - tw/2, y2 - 35, tw, 22), Qt.AlignmentFlag.AlignCenter, pnl_str)

        painter.setPen(QPen(QColor(Theme.BORDER), 1))
        painter.drawLine(int(chart_w), 0, int(chart_w), h)
        painter.drawLine(0, int(margin_top + chart_h), int(chart_w), int(margin_top + chart_h))

        # ==========================================
        # CROSSHAIR & BẢNG THÔNG TIN NẾN CHUẨN (INFO)
        # ==========================================
        if self.mouse_pos and hovered_candle_idx != -1:
            mx, my = self.mouse_pos.x(), self.mouse_pos.y()
            mx = max(0, min(mx, chart_w))
            my = max(margin_top, min(my, margin_top + chart_h))
            hover_x = chart_w - (len(opens) - hovered_candle_idx) * space_per_candle + self.candle_width / 2

            painter.setPen(QPen(QColor(Theme.TEXT_SUB), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(hover_x), int(margin_top), int(hover_x), int(margin_top + chart_h))
            painter.drawLine(0, int(my), int(chart_w), int(my))

            ho, hh, hl, hc, hv = opens[hovered_candle_idx], highs[hovered_candle_idx], lows[hovered_candle_idx], closes[hovered_candle_idx], volumes[hovered_candle_idx]
            hover_time = timestamps[hovered_candle_idx]
            
            # Quét tìm Lệnh khớp với Tọa độ X (Chính xác hơn dùng Time)
            trades_here = []
            for t in self.trades:
                tx_en = self.get_x(t.get('entry_time'), start_idx, end_idx, total_candles, chart_w, space_per_candle)
                tx_ex = self.get_x(t.get('exit_time'), start_idx, end_idx, total_candles, chart_w, space_per_candle)
                if abs(hover_x - tx_en) < space_per_candle: trades_here.append((t, 'Entry', t['entry_price']))
                if abs(hover_x - tx_ex) < space_per_candle: trades_here.append((t, 'Exit', t['exit_price']))

            hud_lines = [
                "📌 INFO",
                f"T: {hover_time.strftime('%m-%d %H:%M') if hover_time else ''}",
                f"O: {ho:.2f} | C: {hc:.2f}",
                f"H: {hh:.2f} | L: {hl:.2f}"
            ]
            if trades_here:
                hud_lines.append("⚡ TRADE")
                for tr, action, val in trades_here:
                    hud_lines.append(f"ID {tr['id']} ({tr['direction']})")
                    hud_lines.append(f"{action}: {val:.2f}")
            
            # 1. Set font và khởi tạo bộ đo font (FontMetrics)
            font_hud = QFont("Consolas", 9)
            painter.setFont(font_hud)
            fm_hud = painter.fontMetrics() # Dùng fm_hud để đo chính xác cho font này

            # 2. Tính chiều ngang: Lấy dòng dài nhất + 15px padding (vừa đủ đẹp)
            max_w = max([fm_hud.horizontalAdvance(line) for line in hud_lines]) + 15
            
            # 3. Tính chiều cao: Mỗi dòng 16px + 10px padding trên dưới
            box_h = len(hud_lines) * 16 + 10

            # 4. Logic Né chuột (Giữ nguyên hoặc tinh chỉnh khoảng cách)
            hud_x = mx + 20
            hud_y = my + 20
            if mx > chart_w * 0.6: 
                hud_x = mx - max_w - 20
            if my > chart_h * 0.6: 
                hud_y = my - box_h - 20

            # 5. Vẽ khung
            bg_hud = QColor(Theme.CARD)
            bg_hud.setAlpha(240)
            painter.setPen(QPen(QColor(Theme.BORDER), 1))
            painter.setBrush(bg_hud)
            
            # Chuyển về int để tránh lỗi render mờ viền
            painter.drawRoundedRect(int(hud_x), int(hud_y), int(max_w), int(box_h), 4, 4)

            curr_y = hud_y + 16
            for line in hud_lines:
                painter.setPen(QColor(Theme.ACCENT) if "📌" in line or "⚡" in line else QColor(Theme.TEXT_MAIN))
                painter.drawText(int(hud_x + 10), int(curr_y), line)
                curr_y += 16

            cross_price = max_high - ((my - margin_top) / chart_h) * price_range
            painter.setBrush(QColor(Theme.TEXT_MAIN)); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(int(chart_w), int(my - 10), margin_right, 20)
            painter.setPen(QColor(Theme.BG)); painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(int(chart_w + 6), int(my + 4), f"{cross_price:.4f}")

# ==========================================
# 5. GIAO DIỆN CHÍNH (MAIN APP)
# ==========================================
class CandlestickChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(f"background-color: {Theme.BG};")
        
        self.dict_dfs = {} 
        self.raw_trades = []
        self.current_symbol = "UNKNOWN"
        self.df_base_1m = pl.DataFrame()

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)

        self.setup_toolbar()
        
        self.chart = CoreCandlestickChart()
        self.left_layout.addWidget(self.chart)

        self.table = TradesTableWidget()
        self.table.cellClicked.connect(self.on_table_click)

        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.table)
        self.splitter.setSizes([800, 200])

    def setup_toolbar(self):
        self.toolbar = QWidget()
        self.toolbar.setFixedHeight(45)
        self.toolbar.setStyleSheet(f"background: {Theme.CARD}; border-bottom: 1px solid {Theme.BORDER};")
        tb_layout = QHBoxLayout(self.toolbar)
        
        self.btn_load = QPushButton("▶ Chạy Vector Backtest")
        self.btn_load.setStyleSheet(f"background-color: {Theme.ACCENT}; color: #FFF; font-weight: bold; padding: 6px 15px; border-radius: 4px; border: none;")
        self.btn_load.clicked.connect(self.run_backtest)
        tb_layout.addWidget(self.btn_load)

        self.combo_symbol = QComboBox()
        self.combo_symbol.setStyleSheet(f"background: {Theme.BG}; color: {Theme.TEXT_MAIN}; border: 1px solid {Theme.BORDER}; padding: 4px; min-width: 100px;")
        self.combo_symbol.currentTextChanged.connect(self.on_symbol_changed)
        tb_layout.addWidget(self.combo_symbol)

        self.lbl_info = QLabel("Sẵn sàng.")
        self.lbl_info.setStyleSheet(f"color: {Theme.TEXT_SUB}; font-weight: bold; padding-left: 10px;")
        tb_layout.addWidget(self.lbl_info)
        tb_layout.addStretch()

        self.btn_group = []
        for tf in ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]:
            btn = QPushButton(tf)
            btn.setFixedSize(40, 26)
            btn.clicked.connect(lambda checked, t=tf: self.change_timeframe(t))
            btn.setStyleSheet(f"background: {Theme.BORDER}; color: {Theme.TEXT_MAIN};" if tf=="1m" else f"background: transparent; color: {Theme.TEXT_SUB};")
            tb_layout.addWidget(btn)
            self.btn_group.append(btn)
            
        self.left_layout.addWidget(self.toolbar)

    def run_backtest(self):
        self.lbl_info.setText("⏳ Đang xử lý Vector...")
        self.btn_load.setEnabled(False)
        self.worker = BacktestWorker()
        self.worker.finished.connect(self.on_backtest_finished)
        self.worker.error.connect(self.on_backtest_error)
        self.worker.start()

    def on_backtest_finished(self, trades, dict_dfs):
        self.raw_trades = trades.to_dicts() if hasattr(trades, 'to_dicts') else trades if isinstance(trades, list) else []
        if not dict_dfs:
            self.lbl_info.setText("⚠️ Dữ liệu trống.")
            self.btn_load.setEnabled(True)
            return
            
        self.dict_dfs = dict_dfs
        self.combo_symbol.blockSignals(True)
        self.combo_symbol.clear()
        self.combo_symbol.addItems(list(self.dict_dfs.keys()))
        self.combo_symbol.blockSignals(False)
        if self.combo_symbol.count() > 0: self.on_symbol_changed(self.combo_symbol.currentText())
        self.btn_load.setEnabled(True)

    def on_backtest_error(self, err):
        self.lbl_info.setText(f"❌ Lỗi: {err}")
        self.btn_load.setEnabled(True)

    def calculate_hold_time(self, en_t, ex_t):
        if not en_t or not ex_t: return "-"
        delta = ex_t - en_t
        hours, rem = divmod(delta.total_seconds(), 3600)
        return f"{int(hours)}h {int(rem // 60)}m" if hours > 0 else f"{int(rem // 60)}m"

    def process_and_extract_trades(self, raw_trades, df_base):
        clean_trades = []
        
        # 1. Đọc từ Trades List (Nếu có)
        if raw_trades:
            for i, t in enumerate(raw_trades):
                if not isinstance(t, dict): continue
                nk = {str(k).lower().replace(' ', '_'): v for k, v in t.items()}
                en_t = to_datetime(nk.get('entry_time', nk.get('time_in')))
                ex_t = to_datetime(nk.get('exit_time', nk.get('time_out')))
                if en_t is None: continue 

                direction = str(nk.get('direction', nk.get('type', 'Long'))).capitalize()
                en_p = float(nk.get('entry_price', 0))
                ex_p = float(nk.get('exit_price', 0))
                p_change = ex_p - en_p if direction == 'Long' else en_p - ex_p

                clean_trades.append({
                    'id': nk.get('id', i + 1),
                    'direction': direction,
                    'entry_time': en_t, 'exit_time': ex_t,
                    'entry_price': en_p, 'exit_price': ex_p,
                    'hold_time': self.calculate_hold_time(en_t, ex_t),
                    'price_change': p_change
                })
            if clean_trades: return clean_trades

        # 2. KHÔI PHỤC LỖI: Tự quét cột `signal` cực thông minh
        sig_col = next((col for col in ["signal", "entry_signal"] if col in df_base.columns), None)
        if sig_col:
            timestamps = [to_datetime(ts) for ts in df_base["timestamp"].to_list()]
            closes = df_base["close"].to_list()
            try: signals = df_base[sig_col].cast(pl.Float64).fill_null(0).to_list()
            except: signals = [0] * len(closes)
            
            curr_trade = None
            trade_id = 1
            
            for t, p, s in zip(timestamps, closes, signals):
                if s == 1:
                    if curr_trade is None:
                        curr_trade = {'id': trade_id, 'direction': 'Long', 'entry_time': t, 'entry_price': p}; trade_id += 1
                    elif curr_trade['direction'] == 'Short': # Đóng Short, Mở Long
                        curr_trade['exit_time'] = t; curr_trade['exit_price'] = p
                        curr_trade['price_change'] = curr_trade['entry_price'] - p
                        curr_trade['hold_time'] = self.calculate_hold_time(curr_trade['entry_time'], t)
                        clean_trades.append(curr_trade)
                        curr_trade = {'id': trade_id, 'direction': 'Long', 'entry_time': t, 'entry_price': p}; trade_id += 1
                elif s == -1:
                    if curr_trade is None:
                        curr_trade = {'id': trade_id, 'direction': 'Short', 'entry_time': t, 'entry_price': p}; trade_id += 1
                    elif curr_trade['direction'] == 'Long': # Đóng Long, Mở Short
                        curr_trade['exit_time'] = t; curr_trade['exit_price'] = p
                        curr_trade['price_change'] = p - curr_trade['entry_price']
                        curr_trade['hold_time'] = self.calculate_hold_time(curr_trade['entry_time'], t)
                        clean_trades.append(curr_trade)
                        curr_trade = {'id': trade_id, 'direction': 'Short', 'entry_time': t, 'entry_price': p}; trade_id += 1
                elif s == 0 and curr_trade is not None:
                    # Nếu signal về 0 tức là đóng toàn bộ lệnh
                    curr_trade['exit_time'] = t; curr_trade['exit_price'] = p
                    curr_trade['price_change'] = p - curr_trade['entry_price'] if curr_trade['direction'] == 'Long' else curr_trade['entry_price'] - p
                    curr_trade['hold_time'] = self.calculate_hold_time(curr_trade['entry_time'], t)
                    clean_trades.append(curr_trade)
                    curr_trade = None
        return clean_trades

    def on_symbol_changed(self, symbol):
        df = self.dict_dfs.get(symbol)
        if df is None: return
        
        if isinstance(df, pd.DataFrame):
            if 'timestamp' not in df.columns: df = df.reset_index()
            df = pl.from_pandas(df)
            
        if df.is_empty(): return
        if df["timestamp"].dtype == pl.Utf8: df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))
        
        self.df_base_1m = df.sort("timestamp")
        
        # Tiền xử lý dữ liệu Trade ở luồng chính trước khi chuyển cho Chart/Table vẽ
        self.chart.trades = self.process_and_extract_trades(self.raw_trades, self.df_base_1m)
        self.change_timeframe("1m")

    def change_timeframe(self, tf):
        for btn in self.btn_group:
            btn.setStyleSheet(f"background: {Theme.BORDER}; color: {Theme.TEXT_MAIN}; font-weight: bold;" if btn.text() == tf else f"background: transparent; color: {Theme.TEXT_SUB}; font-weight: bold;")
        
        self.lbl_info.setText(f"⏳ Đang tải khung {tf}...")
        self.chart.current_tf = tf
        
        self.resample_worker = DataProcessorWorker(self.df_base_1m, tf)
        self.resample_worker.finished.connect(self.on_resample_finished)
        self.resample_worker.start()

    def on_resample_finished(self, df_resampled, timestamps):
        self.chart.df_current = df_resampled
        self.chart.current_timestamps = timestamps
        self.chart.scroll_offset = 0
        
        closed_trades = [t for t in self.chart.trades if 'exit_time' in t]
        
        self.lbl_info.setText(f" {self.combo_symbol.currentText()} | {self.chart.current_tf} | {df_resampled.height} nến | Lệnh: {len(closed_trades)}")
        
        self.table.load_trades(self.chart.trades)
        self.chart.update()

    def on_table_click(self, row, col):
        if not self.chart.current_timestamps or row >= self.table.rowCount(): return
        t_id = self.table.item(row, 0).text()
        trade = next((t for t in self.chart.trades if str(t.get('id')) == t_id), None)
        if not trade or 'entry_time' not in trade: return
        
        idx = bisect.bisect_left(self.chart.current_timestamps, trade['entry_time'])
        if idx < len(self.chart.current_timestamps):
            visible = (self.chart.width() - 70) // (self.chart.candle_width + self.chart.candle_gap)
            self.chart.scroll_offset = max(0, len(self.chart.current_timestamps) - idx - int(visible/2))
            self.chart.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CandlestickChartWidget()
    window.setWindowTitle("KAIROS Professional Quant UI")
    window.show()
    sys.exit(app.exec())