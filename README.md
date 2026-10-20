# LAB 3-2 Cloud Face Mask

Ứng dụng này chạy toàn bộ website, backend và model trên Render. Browser chỉ
mở webcam, chụp JPEG và gửi ảnh đến endpoint `/predict`.

## Kiến trúc

```text
Webcam browser
  -> POST /predict
Render Free Web Service (Flask + ONNX Runtime)
  -> SSD face detector
  -> MobileNetV2 ONNX, 3 classes
  -> JSON label/probabilities/timings
Browser hiển thị kết quả
SQLite tạm thời lưu lịch sử
Telegram Bot API nhận ảnh cảnh báo + caption
```

`inference_ms` được đo bên trong Cloud từ lúc backend bắt đầu xử lý ảnh đến khi
model hoàn tất. `rtt_ms` được đo tại browser từ lúc gửi request đến lúc nhận
response. `total_ms` được đo từ lúc bắt đầu chụp frame đến lúc nhận kết quả.

## Deploy Web Service thủ công trên Render (Free, không cần PostgreSQL)

1. Đẩy thư mục `lab3_2` lên một repository GitHub.
2. Trong Render chọn **New > Web Service** và chọn repository đó.
3. Nếu `lab3_2` không nằm ở thư mục gốc repository, đặt **Root Directory** là
   `lab3_2`.
4. Nhập cấu hình:

   ```text
   Runtime: Python
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app
   Instance Type: Free
   ```

5. Không tạo PostgreSQL, Persistent Disk hoặc tài nguyên trả phí.
6. Có thể tạo hai Secret Environment Variables sau:

   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
7. Bấm **Create Web Service**, chờ build hoàn tất rồi mở URL của service.

Không đặt token Telegram trong source code. PostgreSQL được chọn vì filesystem
của Web Service Free không bảo đảm giữ SQLite sau mỗi lần deploy/restart. Vì
vậy lịch sử detection có thể mất khi Render restart hoặc deploy lại. Nếu không
cấu hình hai biến Telegram, ứng dụng vẫn chạy nhưng không gửi cảnh báo.

## Chạy local

```powershell
cd lab3_2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Nếu không đặt `DATABASE_URL`, ứng dụng dùng SQLite. Cấu hình `render.yaml` hiện
chỉ khai báo Web Service, nên không yêu cầu tạo hoặc liên kết PostgreSQL.
