import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import csv
import os
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="BISINDO Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# DESIGN SYSTEM
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-base: #0a0e13;
        --bg-surface: #12171f;
        --bg-elevated: #1a2029;

        --accent: #f0b429;
        --accent-soft: rgba(240, 180, 41, 0.12);
        --accent-border: rgba(240, 180, 41, 0.3);
        --success: #10b981;
        --success-soft: rgba(16, 185, 129, 0.1);

        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #475569;

        --border: #1e293b;
        --border-hover: #334155;

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    * { font-family: 'Plus Jakarta Sans', sans-serif !important; }

    .stApp {
        background: var(--bg-base) !important;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1280px !important;
    }

    /* ===== HEADER ===== */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
    }

    .brand-text h1 {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 2px 0;
    }

    .brand-text p {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin: 0;
        letter-spacing: 0.03em;
    }

    .metrics {
        display: flex;
        gap: 0.75rem;
    }

    .metric {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.65rem 1.1rem;
        text-align: center;
        min-width: 85px;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--accent);
    }

    .metric-label {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1px;
    }

    /* ===== MAIN CARD ===== */
    .main-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.75rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    /* ===== CONTROLS ===== */
    .control-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    .conf-display {
        background: var(--accent);
        color: var(--bg-base);
        padding: 0.4rem 0.7rem;
        border-radius: var(--radius-sm);
        font-size: 0.8rem;
        font-weight: 700;
        text-align: center;
        margin-top: 4px;
    }

    /* ===== IMAGE PANEL ===== */
    .panel-wrapper {
        background: var(--bg-base);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    .panel-header {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid var(--border);
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: var(--bg-elevated);
    }

    /* ===== RESULTS ===== */
    .results-wrap {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        padding: 0.9rem 1.1rem;
        background: var(--bg-elevated);
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        margin-top: 1rem;
    }

    .results-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--accent);
        color: var(--bg-base);
        padding: 0.4rem 0.85rem;
        border-radius: var(--radius-sm);
        font-weight: 700;
        font-size: 0.88rem;
    }

    .badge-conf {
        background: rgba(0,0,0,0.2);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.72rem;
    }

    .no-result {
        color: var(--text-muted);
        font-size: 0.82rem;
        font-style: italic;
    }

    /* ===== EMPTY STATE ===== */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 5rem 2rem;
        text-align: center;
    }

    .empty-icon {
        width: 72px;
        height: 72px;
        background: var(--bg-elevated);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--border);
    }

    .empty-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }

    .empty-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
        max-width: 280px;
    }

    /* ===== BOTTOM CARDS ===== */
    .info-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.1rem;
        height: 100%;
    }

    .info-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .info-title span { color: var(--accent); }

    .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }

    .tag {
        background: var(--bg-base);
        color: var(--text-secondary);
        padding: 4px 9px;
        border-radius: 5px;
        font-size: 0.65rem;
        font-weight: 600;
        border: 1px solid var(--border);
        letter-spacing: 0.02em;
        transition: all 0.15s;
    }

    /* ===== DIVIDER ===== */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.2rem 0;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 1.2rem;
        color: var(--text-muted);
        font-size: 0.7rem;
        margin-top: 1rem;
        border-top: 1px solid var(--border);
    }

    .app-footer strong { color: var(--accent); }

    /* ===== STREAMLIT OVERRIDES ===== */
    .stSlider label,
    .stFileUploader label,
    .stNumberInput label { display: none !important; }

    .stSlider > div { 
        padding-top: 0 !important; 
        background: transparent !important;
    }

    div[data-baseweb="slider"] > div:first-child {
        background: var(--accent) !important;
    }

    div[data-baseweb="slider"] > div > div[role="slider"] {
        background: var(--accent) !important;
        border: 3px solid var(--bg-base) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Number input styling */
    input[type="number"] {
        background: var(--bg-base) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-size: 0.85rem !important;
    }

    input[type="number"]:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-soft) !important;
    }

    /* File uploader */
    section[data-testid="stFileUploadDropzone"] {
        background: var(--bg-base) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: var(--radius-md) !important;
        transition: all 0.2s !important;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    section[data-testid="stFileUploadDropzone"] span { 
        font-size: 0.78rem !important; 
        color: var(--text-secondary) !important;
    }

    section[data-testid="stFileUploadDropzone"] small { 
        font-size: 0.65rem !important; 
        color: var(--text-muted) !important;
    }

    /* Remove Streamlit's default white background on widgets */
    .stSlider, .stFileUploader, .stNumberInput {
        background: transparent !important;
    }

    [data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: var(--success) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.5rem !important;
    }

    /* st.image captions */
    .stImage { border-radius: 0 0 var(--radius-md) var(--radius-md); overflow: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# INIT
# =====================================================
LOG_FILE = "detection_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "file", "label", "confidence"])

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================================================
# SESSION STATE — FIX UPLOAD BUG
# =====================================================
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="app-header">
    <div class="brand">
        <div class="brand-icon">🤟</div>
        <div class="brand-text">
            <h1>BISINDO Detection</h1>
            <p>Bahasa Isyarat Indonesia &nbsp;•&nbsp; YOLOv11 &nbsp;•&nbsp; UNNES</p>
        </div>
    </div>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">94.5%</div>
            <div class="metric-label">mAP@0.5</div>
        </div>
        <div class="metric">
            <div class="metric-value">~39</div>
            <div class="metric-label">FPS Local</div>
        </div>
        <div class="metric">
            <div class="metric-value">47</div>
            <div class="metric-label">Classes</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN CARD
# =====================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# --- Controls Row ---
ctrl1, ctrl2 = st.columns([1, 2])

with ctrl1:
    st.markdown('<div class="control-label">⚙️ Confidence Threshold</div>', unsafe_allow_html=True)
    inner1, inner2 = st.columns([4, 1])
    with inner1:
        conf = st.number_input(
            "conf",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            format="%.2f",
            label_visibility="collapsed"
        )
    with inner2:
        st.markdown(f'<div class="conf-display">{conf:.0%}</div>', unsafe_allow_html=True)

with ctrl2:
    st.markdown('<div class="control-label">📁 Upload Gambar BISINDO</div>', unsafe_allow_html=True)
    # FIX: gunakan key agar Streamlit track state dengan benar
    file = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_uploader"
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =====================================================
# DETECTION AREA
# =====================================================

# FIX: Simpan hasil ke session_state agar tidak hilang saat re-run
if file is not None:
    # Baca file → convert ke numpy
    img_pil = Image.open(file).convert("RGB")
    img_np  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Jalankan inferensi
    with st.spinner("🔍 Mendeteksi gesture..."):
        result   = model(img_np, conf=conf, imgsz=640)
        img_out  = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)

    # Kumpulkan label
    labels = []
    if result[0].boxes and len(result[0].boxes) > 0:
        for box in result[0].boxes:
            lbl = model.names[int(box.cls[0])]
            scr = float(box.conf[0])
            labels.append((lbl, scr))
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    file.name, lbl, f"{scr:.4f}"
                ])

    # Simpan ke session_state
    st.session_state.detection_result = {
        "img_pil":  img_pil,
        "img_out":  img_out,
        "labels":   labels,
        "filename": file.name,
    }

# Render hasil (dari session_state, tetap tampil walaupun ada re-run)
if st.session_state.detection_result:
    det = st.session_state.detection_result

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel-wrapper"><div class="panel-header">🖼️ &nbsp;Input Original</div>', unsafe_allow_html=True)
        st.image(det["img_pil"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel-wrapper"><div class="panel-header">✅ &nbsp;Hasil Deteksi</div>', unsafe_allow_html=True)
        st.image(det["img_out"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Results bar
    if det["labels"]:
        badges = "".join([
            f'<span class="badge">{l} <span class="badge-conf">{s:.0%}</span></span>'
            for l, s in det["labels"]
        ])
        st.markdown(
            f'<div class="results-wrap"><span class="results-label">🎯 Terdeteksi:</span> {badges}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="results-wrap"><span class="results-label">🎯 Hasil:</span> '
            '<span class="no-result">Tidak ada gesture terdeteksi — coba turunkan threshold</span></div>',
            unsafe_allow_html=True
        )

else:
    st.markdown('''
    <div class="empty-state">
        <div class="empty-icon">📷</div>
        <div class="empty-title">Upload Gambar untuk Memulai</div>
        <div class="empty-desc">Pilih file JPG / PNG gambar gesture BISINDO di atas</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # tutup .main-card

# =====================================================
# BOTTOM INFO
# =====================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="info-card"><div class="info-title"><span>🔤</span> Huruf (A–Z)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tags">' +
        "".join([f'<span class="tag">{c}</span>' for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]) +
        '</div></div>',
        unsafe_allow_html=True
    )

with c2:
    kata = [
        "AKU", "KAMU", "APA", "DIA", "AYAH", "KAKAK",
        "BAIK", "MAAF", "MARAH", "SABAR", "SEDIH", "SENANG",
        "SUKA", "MINUM", "RUMAH", "KERJA", "BERMAIN",
        "BANTU", "JANGAN", "KAPAN", "KEREN"
    ]
    st.markdown('<div class="info-card"><div class="info-title"><span>💬</span> Kata</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tags">' +
        "".join([f'<span class="tag">{k}</span>' for k in kata]) +
        '</div></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown('<div class="info-card"><div class="info-title"><span>📥</span> Export Log</div>', unsafe_allow_html=True)
    st.caption("Download riwayat hasil deteksi")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                "⬇️ Download Log CSV",
                f,
                "bisindo_log.csv",
                "text/csv",
                use_container_width=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    '<div class="app-footer">'
    '<strong>BISINDO Detection</strong> &nbsp;•&nbsp; YOLOv11 &nbsp;•&nbsp; '
    'Universitas Negeri Semarang &nbsp;•&nbsp; '
    f'{datetime.now().year}'
    '</div>',
    unsafe_allow_html=True
)