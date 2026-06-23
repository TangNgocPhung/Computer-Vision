# Hệ thống hỗ trợ người khiếm thị — Web App (Streamlit)

Nhận diện vật thể (YOLOv8) + Đọc văn bản (OCR) + Giọng nói tiếng Việt (gTTS).

## Cấu trúc thư mục (4 file)
```
streamlit_app/
├── app.py            # ứng dụng Streamlit
├── requirements.txt  # thư viện Python (torch CPU)
├── packages.txt      # gói hệ thống (tesseract tiếng Việt)
└── best.pt           # <-- BẠN TỰ THÊM: trọng số YOLOv8 đã train
```

## Bước 1: Lấy file trọng số `best.pt`
- Tải `best.pt` từ Drive: `MyDrive/yolo_runs/voc_yolov8s-2/weights/best.pt`
- Đổi tên thành `best.pt` (nếu chưa) và copy vào thư mục `streamlit_app/`.
- (best.pt ~22MB, dưới giới hạn 100MB của GitHub nên đẩy thẳng được.)

## Bước 2: Đưa lên GitHub
1. Tạo tài khoản GitHub (nếu chưa có) tại github.com.
2. Tạo repo mới (ví dụ `blind-assist-app`), để **Public**.
3. Upload cả 4 file (`app.py`, `requirements.txt`, `packages.txt`, `best.pt`)
   bằng nút **Add file → Upload files** → Commit.

## Bước 3: Deploy lên Streamlit Community Cloud
1. Vào **share.streamlit.io** → đăng nhập bằng GitHub.
2. Bấm **Create app → Deploy a public app from GitHub**.
3. Chọn:
   - Repository: `<tên-bạn>/blind-assist-app`
   - Branch: `main`
   - Main file path: `app.py`
4. Bấm **Deploy**. Lần đầu build ~5–10 phút (cài torch, easyocr...).
5. Xong → app chạy 24/7 tại link `https://<ten-app>.streamlit.app`

## Lưu ý quan trọng
- **RAM gói free ~1GB:** nạp cả YOLO + EasyOCR có thể hơi nặng. App đã
  **nạp EasyOCR kiểu lazy** (chỉ nạp khi dùng chế độ Văn bản). Nếu app crash
  vì hết RAM → ưu tiên dùng engine **Tesseract** (nhẹ hơn EasyOCR).
- **App "ngủ" khi không ai dùng:** gói free sẽ sleep sau một thời gian rảnh,
  lần truy cập sau tự thức dậy (~30 giây). Đây là bình thường.
- **Internet:** Streamlit Cloud có sẵn internet nên gTTS chạy được.
- Nếu lỗi `libGL` → đã có `libgl1` trong `packages.txt`; nếu vẫn lỗi, kiểm tra
  `opencv-python-headless` (không phải `opencv-python`) trong requirements.

## Chạy thử ở máy (tùy chọn, trước khi deploy)
```bash
pip install -r requirements.txt
streamlit run app.py
```
