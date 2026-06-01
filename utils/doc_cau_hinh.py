"""
utils/doc_cau_hinh.py – Loader cấu hình hệ thống
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Đọc các file cấu hình từ thư mục config/ và trả về dict.
Mọi module khác đều lấy tham số qua file này thay vì hardcode.
  • tai_khoan_api.json       – API keys các sàn (KHÔNG commit lên git)
  • cau_hinh_giao_dich.yaml – danh sách coin, đòn bẩy, vốn mỗi lệnh
  • thong_tin_san.yaml       – phí giao dịch, giới hạn đòn bẩy từng sàn
  • cau_hinh_ao_config.json  – tham số paper trading (vốn ảo, phí, slippage)
"""
import json
import yaml
import os
from utils.log import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def lay_cau_hinh_api():
    path = os.path.join(BASE_DIR, 'config', 'tai_khoan_api.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}

def lay_cau_hinh_giao_dich():
    path = os.path.join(BASE_DIR, 'config', 'cau_hinh_giao_dich.yaml')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return {}

def lay_thong_tin_san():
    path = os.path.join(BASE_DIR, 'config', 'thong_tin_san.yaml')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return {}

def lay_cau_hinh_ao():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'cau_hinh_ao_config.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            return json.loads(content)
    except Exception as e:
        logger.info(f"⚠️ Không đọc được cau_hinh_ao_config.json ({e}). Sử dụng mặc định.")
        return {
            "so_du_ban_dau": 10000,
            "ngay_bat_dau": "2025-01-01",
            "ngay_ket_thuc": "2025-01-05",
            "phi_giao_dich": 0.0004,
            "do_truot_gia": 0.0001
        }