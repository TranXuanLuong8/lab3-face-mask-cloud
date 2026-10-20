# LAB 3-2 Cloud Face Mask

Ứng dụng này chạy toàn bộ website, backend và model trên Render. Browser chỉ
mở webcam, chụp JPEG và gửi ảnh đến endpoint `/predict`.

## Kiến trúc

```text
Webcam browser
  -> POST /predict
Render Flask + ONNX Runtime
  -> SSD face detector
  -> MobileNetV2 ONNX, 3 classes
  -> JSON label/probabilities/timings
Browser hiển thị kết quả
Render PostgreSQL lưu lịch sử
Telegram Bot API nhận ảnh cảnh báo + caption
```

`inference_ms` được đo bên trong Cloud từ lúc backend bắt đầu xử lý ảnh đến khi
model hoàn tất. `rtt_ms` được đo tại browser từ lúc gửi request đến lúc nhận
response. `total_ms` được đo từ lúc bắt đầu chụp frame đến lúc nhận kết quả.

## Deploy trên Render

1. Đẩy thư mục `lab3_2` lên một repository GitHub.
2. Trong Render chọn **New > Blueprint** và chọn repository đó.
3. Render đọc `render.yaml`, tạo web service và PostgreSQL.
4. Tạo hai Secret Environment Variables:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Deploy và mở URL của web service.

Không đặt token Telegram trong source code. PostgreSQL được chọn vì filesystem
của web service Cloud không bảo đảm giữ SQLite sau mỗi lần deploy/restart.

## Chạy local

```powershell
cd lab3_2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Nếu không đặt `DATABASE_URL`, ứng dụng dùng SQLite local để kiểm thử. Khi chạy
trên Render, `DATABASE_URL` do `render.yaml` nối tự động vào PostgreSQL.
