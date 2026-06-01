"""
thuc_thi_lenh/bo_may_thuc_thi.py – Quản lý kết nối các sàn giao dịch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Singleton BoMayThucThi khởi tạo và giữ kết nối CCXT đến tất cả sàn
được cấu hình trong config/tai_khoan_api.json (Binance, OKX, Bybit).

Module-level instance `quan_ly_san` được import ở mọi nơi cần thực thi lệnh
thay vì tạo kết nối mới mỗi lần – tránh overhead và rate limit.
"""
import ccxt
from utils.doc_cau_hinh import lay_cau_hinh_api
from utils.log import logger

class BoMayThucThi:
    """Quản lý pool kết nối CCXT. Khởi tạo 1 lần, dùng lại xuyên suốt phiên giao dịch."""
    def __init__(self):
        self.configs = lay_cau_hinh_api()
        self.exchanges = {}
        self.khoi_tao_ket_noi()

    def khoi_tao_ket_noi(self):
        for san, config in self.configs.items():
            if san in self.exchanges:
                logger.info(f"ℹ️ Sàn {san} đã được kết nối trước đó. Bỏ qua.")
                break
            try:
                if san == 'binance':
                    self.exchanges[san] = ccxt.binance(config)
                elif san == 'okx':
                    self.exchanges[san] = ccxt.okx(config)
                elif san == 'bybit':
                    self.exchanges[san] = ccxt.bybit(config)
                self.exchanges[san].load_markets()
            except Exception as e:
                logger.error(f"❌ Lỗi kết nối sàn {san}: {e}")

    def lay_san(self, ten_san):
        return self.exchanges.get(ten_san)

    def mo_lenh_market(self, ten_san, symbol, side, amount, leverage):
        exchange = self.lay_san(ten_san)
        if not exchange: return None
        try:
            try:
                exchange.set_leverage(leverage, symbol)
            except:
                pass 
            # amount là số lượng coin
            order = exchange.create_order(symbol, 'market', side, amount)
            logger.info(f"🚀 KHỚP LỆNH {side.upper()} {symbol} trên {ten_san}. ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Lỗi mở lệnh: {e}")
            return None

    def dong_vi_the(self, ten_san, symbol, side_hien_tai, amount):
        side_dong = 'sell' if side_hien_tai == 'buy' else 'buy'
        return self.mo_lenh_market(ten_san, symbol, side_dong, amount, 1)

quan_ly_san = BoMayThucThi()