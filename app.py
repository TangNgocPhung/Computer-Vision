# -*- coding: utf-8 -*-
"""
Ung dung ho tro nguoi khiem thi (Streamlit)
Nhan dien vat the (YOLOv8) + OCR (Tesseract / EasyOCR) + giong noi tieng Viet (gTTS)
"""
import os
import tempfile
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
import pytesseract

st.set_page_config(page_title="Ho tro nguoi khiem thi", page_icon="eye", layout="wide")

YOLO_WEIGHTS = "best.pt"   # dat file trong so cung thu muc voi app.py

VN_NAMES = {
    "aeroplane": "máy bay", "bicycle": "xe đạp", "bird": "con chim",
    "boat": "con thuyền", "bottle": "cái chai", "bus": "xe buýt",
    "car": "ô tô", "cat": "con mèo", "chair": "cái ghế", "cow": "con bò",
    "diningtable": "bàn ăn", "dog": "con chó", "horse": "con ngựa",
    "motorbike": "xe máy", "person": "người", "pottedplant": "chậu cây",
    "sheep": "con cừu", "sofa": "ghế sô pha", "train": "tàu hỏa",
    "tvmonitor": "màn hình ti vi",
}

# ---------------- Nap model (cache, chi nap 1 lan) ----------------
@st.cache_resource(show_spinner="Đang nạp mô hình phát hiện vật thể...")
def load_yolo():
    return YOLO(YOLO_WEIGHTS)

@st.cache_resource(show_spinner="Đang nạp EasyOCR (lần đầu hơi lâu)...")
def load_easyocr():
    import easyocr
    return easyocr.Reader(["vi"], gpu=False)

# ---------------- Tien xu ly anh ----------------
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
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img_bgr, M, (mw, mh))

# ---------------- Phat hien vat the ----------------
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

# ---------------- OCR ----------------
def extract_text(img_bgr, engine="easyocr"):
    if engine == "tesseract":
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(rgb, lang="vie", config="--oem 1 --psm 6").strip()
    reader = load_easyocr()
    lines = reader.readtext(img_bgr, detail=0, paragraph=True)
    return "\n".join(lines).strip()

# ---------------- Text-to-Speech ----------------
def text_to_speech(text):
    text = (text or "").strip() or "Không có nội dung để đọc."
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text=text, lang="vi", slow=False).save(tmp.name)
    return tmp.name

# ---------------- Giao dien ----------------
st.title("👁️ Hệ thống hỗ trợ người khiếm thị")
st.caption("Nhận diện vật thể + Đọc văn bản (OCR) + Giọng nói tiếng Việt")

c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("Ảnh đầu vào", type=["jpg", "jpeg", "png"])
    mode = st.radio("Chế độ xử lý", ["Tự động", "Vật thể", "Văn bản"], horizontal=True)
    engine = st.radio("Bộ nhận dạng chữ (OCR)", ["EasyOCR", "Tesseract"], horizontal=True)
    go = st.button("Xử lý", type="primary", use_container_width=True)
    if up is not None:
        st.image(up, caption="Ảnh gốc", use_container_width=True)

with c2:
    if go and up is not None:
        data = np.frombuffer(up.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        spoken, text_out = [], ""
        if mode in ("Tự động", "Vật thể"):
            annotated, desc = detect_objects(apply_clahe(img_bgr))
            spoken.append(desc)
            st.image(annotated, caption="Kết quả phát hiện", use_container_width=True)
        if mode in ("Tự động", "Văn bản"):
            eng = "tesseract" if engine == "Tesseract" else "easyocr"
            text_out = extract_text(apply_clahe(perspective_correction(img_bgr)), eng)
            if text_out:
                spoken.append("Nội dung văn bản: " + text_out)

        st.text_area("Văn bản trích xuất", text_out or "(Không có văn bản)", height=120)
        mp3 = text_to_speech(" ".join(spoken))
        st.audio(mp3, autoplay=True)

        st.download_button("Tải văn bản (.txt)", text_out, "ket_qua.txt")
        with open(mp3, "rb") as f:
            st.download_button("Tải giọng nói (.mp3)", f, "ket_qua.mp3")
    elif go:
        st.warning("Hãy tải lên một ảnh trước.")
