# -*- coding: utf-8 -*-
"""
He thong ho tro nguoi khiem thi
Truong Dai hoc Su pham TP.HCM - Khoa Cong nghe thong tin
YOLOv8 (phat hien vat the) + OCR (Tesseract/EasyOCR) + gTTS (giong noi tieng Viet)
"""
import tempfile
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
import pytesseract

st.set_page_config(page_title="Hỗ trợ người khiếm thị | HCMUE",
                   page_icon="🦮", layout="wide")

# ============================================================
#  GIAO DIEN (CSS)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }

:root { --ink:#10323d; --teal:#1f7a8c; --teal-d:#155160; --line:#e4e9ec; --soft:#f5f8f9; }

.hero {
  background: linear-gradient(135deg, #155160 0%, #1f7a8c 100%);
  border-radius: 16px; padding: 30px 34px; color: #fff; margin-bottom: 22px;
}
.hero .uni { font-size: 13px; letter-spacing:.12em; text-transform:uppercase; opacity:.9; }
.hero .fac { font-size: 14px; opacity:.85; margin-top:2px; }
.hero h1 { font-size: 30px; font-weight:700; margin: 12px 0 6px; line-height:1.2; }
.hero .sub { font-size: 15px; opacity:.92; font-weight:400; }
.hero .chips { margin-top:16px; display:flex; gap:10px; flex-wrap:wrap; }
.hero .chip { background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
  padding:5px 13px; border-radius:999px; font-size:13px; }

.section { font-size:13px; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  color: var(--teal-d); margin: 6px 0 10px; border-left:3px solid var(--teal); padding-left:9px; }

.stButton>button {
  background: var(--teal); color:#fff; border:none; border-radius:10px;
  font-weight:600; padding:.55rem 1rem; width:100%; transition:.15s;
}
.stButton>button:hover { background: var(--teal-d); color:#fff; }

/* Radio dang the (pill) thay cho cham tron mac dinh */
div[role="radiogroup"] { gap:0; }
div[role="radiogroup"] > label {
  background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12);
  border-radius:9px; padding:8px 13px; margin-bottom:6px; width:100%;
  cursor:pointer; transition:.15s;
}
div[role="radiogroup"] > label:hover { border-color:#2e94a6; }
div[role="radiogroup"] > label > div:first-child { display:none; }
div[role="radiogroup"] > label:has(input:checked) {
  background:#1f7a8c; border-color:#1f7a8c;
}
div[role="radiogroup"] > label:has(input:checked) p { color:#fff; font-weight:600; }

[data-testid="stSidebar"] { background: #0f2e38; }
[data-testid="stSidebar"] * { color:#dfeaed; }
.sb-brand { font-size:16px; font-weight:700; color:#fff; line-height:1.3; }
.sb-sub { font-size:12.5px; color:#9fc0c8; margin-top:2px; }
.sb-h { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:#7fa9b3;
  margin:18px 0 8px; }
.std { background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);
  border-radius:10px; padding:10px 12px; margin-bottom:8px; }
.std .n { font-size:14px; font-weight:600; color:#fff; }
.std .c { font-size:12px; color:#9fc0c8; font-family:monospace; }

.foot { text-align:center; color:#7c949c; font-size:12.5px; margin-top:26px;
  padding-top:14px; border-top:1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

YOLO_WEIGHTS = "best.pt"
VN_NAMES = {
    "aeroplane": "máy bay", "bicycle": "xe đạp", "bird": "con chim",
    "boat": "con thuyền", "bottle": "cái chai", "bus": "xe buýt",
    "car": "ô tô", "cat": "con mèo", "chair": "cái ghế", "cow": "con bò",
    "diningtable": "bàn ăn", "dog": "con chó", "horse": "con ngựa",
    "motorbike": "xe máy", "person": "người", "pottedplant": "chậu cây",
    "sheep": "con cừu", "sofa": "ghế sô pha", "train": "tàu hỏa",
    "tvmonitor": "màn hình ti vi",
}

# ============================================================
#  MODEL & PIPELINE
# ============================================================
@st.cache_resource(show_spinner="Đang nạp mô hình phát hiện vật thể...")
def load_yolo():
    return YOLO(YOLO_WEIGHTS)

@st.cache_resource(show_spinner="Đang nạp EasyOCR (lần đầu hơi lâu)...")
def load_easyocr():
    import easyocr
    return easyocr.Reader(["vi"], gpu=False)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def perspective_correction(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    h, w = img_bgr.shape[:2]
    area = h * w
    doc = None
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.25 * area:
            doc = approx.reshape(4, 2).astype("float32")
            break
    if doc is None:
        return img_bgr
    rect = _order_points(doc)
    (tl, tr, br, bl) = rect
    mw = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    mh = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if mw < 10 or mh < 10:
        return img_bgr
    dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype="float32")
    return cv2.warpPerspective(img_bgr, cv2.getPerspectiveTransform(rect, dst), (mw, mh))

def detect_objects(img_bgr, conf=0.35):
    model = load_yolo()
    res = model.predict(img_bgr, conf=conf, verbose=False)[0]
    annotated = res.plot()
    labels = [VN_NAMES.get(model.names[int(b.cls)], model.names[int(b.cls)]) for b in res.boxes]
    counts = Counter(labels)
    if counts:
        desc = "Phía trước có " + ", ".join(f"{n} {name}" for name, n in counts.items()) + "."
    else:
        desc = "Không phát hiện được vật thể nào."
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), desc

def extract_text(img_bgr, engine="easyocr"):
    if engine == "tesseract":
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(rgb, lang="vie", config="--oem 1 --psm 6").strip()
    reader = load_easyocr()
    lines = reader.readtext(img_bgr, detail=0, paragraph=True)
    return "\n".join(lines).strip()

def text_to_speech(text):
    text = (text or "").strip() or "Không có nội dung để đọc."
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text=text, lang="vi", slow=False).save(tmp.name)
    return tmp.name

# ============================================================
#  SIDEBAR (thuong hieu + hoc vien + cau hinh)
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-brand">Hệ thống hỗ trợ<br>người khiếm thị</div>'
                '<div class="sb-sub">Computer Vision Application</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-h">Cấu hình</div>', unsafe_allow_html=True)
    mode = st.radio("Chế độ xử lý", ["Tự động", "Vật thể", "Văn bản"])
    engine = st.radio("Bộ nhận dạng chữ (OCR)", ["EasyOCR", "Tesseract"])

    st.markdown('<div class="sb-h">Giảng viên hướng dẫn</div>', unsafe_allow_html=True)
    st.markdown('<div class="std"><div class="n">PGS.TS. Hoàng Văn Dũng</div>'
                '<div style="font-size:11.5px;color:#9fc0c8;margin-top:1px;">'
                'Khoa Công nghệ Thông tin</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-h">Học viên cao học</div>', unsafe_allow_html=True)
    students = [
        ("Tăng Ngọc Phụng", "KHMT836027"),
        ("Hoàng Châu Ngọc Phương", "KHMT836028"),
        ("Lê Thị Mai Len", "KHMT836015"),
    ]
    for name, code in students:
        st.markdown(f'<div class="std"><div class="n">{name}</div>'
                    f'<div class="c">{code}</div></div>', unsafe_allow_html=True)

# ============================================================
#  HERO
# ============================================================
st.markdown("""
<div class="hero">
  <div class="uni">Trường Đại học Sư phạm Thành phố Hồ Chí Minh</div>
  <div class="fac">Khoa Công nghệ Thông tin</div>
  <h1>Hệ thống hỗ trợ người khiếm thị</h1>
  <div class="sub">Nhận diện vật thể, đọc văn bản và chuyển thành giọng nói tiếng Việt từ một bức ảnh.</div>
  <div class="chips">
    <span class="chip">Phát hiện vật thể · YOLOv8</span>
    <span class="chip">Nhận dạng chữ · OCR</span>
    <span class="chip">Giọng nói · gTTS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  NOI DUNG
# ============================================================
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="section">Ảnh đầu vào</div>', unsafe_allow_html=True)
    up = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")
    if up is not None:
        st.image(up, use_container_width=True)
    go = st.button("Bắt đầu xử lý")

with right:
    st.markdown('<div class="section">Kết quả</div>', unsafe_allow_html=True)
    if go and up is not None:
        data = np.frombuffer(up.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        spoken, text_out = [], ""
        if mode in ("Tự động", "Vật thể"):
            annotated, desc = detect_objects(apply_clahe(img_bgr))
            spoken.append(desc)
            st.image(annotated, caption="Vật thể nhận diện được", use_container_width=True)
        if mode in ("Tự động", "Văn bản"):
            eng = "tesseract" if engine == "Tesseract" else "easyocr"
            text_out = extract_text(apply_clahe(perspective_correction(img_bgr)), eng)
            if text_out:
                spoken.append("Nội dung văn bản: " + text_out)

        full_text = " ".join(spoken)
        st.text_area("Nội dung", full_text or "(Không có nội dung)", height=120)

        mp3 = text_to_speech(full_text)
        st.audio(mp3, autoplay=True)

        d1, d2 = st.columns(2)
        d1.download_button("Tải văn bản (.txt)", full_text or "", "ket_qua.txt",
                           use_container_width=True)
        with open(mp3, "rb") as f:
            d2.download_button("Tải giọng nói (.mp3)", f, "ket_qua.mp3",
                               use_container_width=True)
    else:
        st.info("Tải một bức ảnh ở bên trái rồi bấm **Bắt đầu xử lý**.")

st.markdown('<div class="foot">© 2026 · Đồ án môn Thị giác máy tính và Ứng dụng · '
            'Khoa Công nghệ Thông tin, Trường ĐH Sư phạm TP.HCM</div>',
            unsafe_allow_html=True)
