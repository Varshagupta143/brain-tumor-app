import streamlit as st
import numpy as np
import cv2
import pickle
import os
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0e1a; color: #e2e8f0; }

[data-testid="stSidebar"] { background: #0f1629 !important; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #38bdf8 !important; font-size: 1.6rem !important; letter-spacing: -0.5px; }
h2 { color: #7dd3fc !important; font-size: 1.1rem !important; }
h3 { color: #bae6fd !important; font-size: 0.95rem !important; }

.neuro-card {
    background: linear-gradient(135deg, #0f1e35 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: #0f1e35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.pred-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 8px 0;
}
.badge-glioma     { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
.badge-meningioma { background: #431407; color: #fdba74; border: 1px solid #f97316; }
.badge-notumor    { background: #052e16; color: #86efac; border: 1px solid #22c55e; }
.badge-pituitary  { background: #2e1065; color: #c4b5fd; border: 1px solid #8b5cf6; }

.conf-bar-bg {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    margin: 6px 0;
    overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 6px; transition: width 0.8s ease; }

.hist-item {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
[data-testid="stFileUploader"] {
    background: #0f1e35 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    transform: translateY(-1px) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0f1629; border-radius: 10px; gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 8px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] { background: #0369a1 !important; color: white !important; }
hr { border-color: #1e2d4a !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
.stProgress > div > div { background: #0369a1 !important; }
.stAlert { border-radius: 10px !important; background: #0f1e35 !important; }
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_data' not in st.session_state:
    st.session_state.model_data = None

# ── Class info ───────────────────────────────────────────────────────
CLASS_INFO = {
    'glioma': {
        'color': '#ef4444', 'badge': 'badge-glioma',
        'desc': 'A tumor originating in glial cells of the brain or spine. Can be low or high grade.',
        'severity': 'HIGH', 'icon': '🔴',
        'location': 'Cerebral hemisphere (frontal, temporal, parietal lobes)',
    },
    'meningioma': {
        'color': '#f97316', 'badge': 'badge-meningioma',
        'desc': 'Tumor arising from meninges. Usually benign and slow-growing.',
        'severity': 'MODERATE', 'icon': '🟠',
        'location': 'Brain surface / meninges layer',
    },
    'notumor': {
        'color': '#22c55e', 'badge': 'badge-notumor',
        'desc': 'No tumor detected. Brain tissue appears normal.',
        'severity': 'NONE', 'icon': '🟢',
        'location': 'N/A',
    },
    'pituitary': {
        'color': '#8b5cf6', 'badge': 'badge-pituitary',
        'desc': 'Tumor on the pituitary gland. Usually benign but affects hormones.',
        'severity': 'MODERATE', 'icon': '🟣',
        'location': 'Pituitary gland — center-bottom of brain',
    }
}

# ── Model loading ────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# ── Preprocessing ────────────────────────────────────────────────────
def preprocess_image(img_array, img_size=64):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    resized   = cv2.resize(gray, (img_size, img_size))
    equalized = cv2.equalizeHist(resized)
    return equalized

# ── Prediction ───────────────────────────────────────────────────────
def predict(img_array, model_data):
    img_size   = model_data.get('img_size', 64)
    model      = model_data['model']
    scaler     = model_data['scaler']
    pca        = model_data['pca']
    le         = model_data['encoder']
    processed  = preprocess_image(img_array, img_size)
    flat       = processed.flatten().reshape(1, -1)
    scaled     = scaler.transform(flat)
    pca_feat   = pca.transform(scaled)
    pred_enc   = model.predict(pca_feat)[0]
    proba      = model.predict_proba(pca_feat)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    classes    = le.classes_
    return pred_label, proba, classes, processed

# ── TUMOR LOCALIZATION ───────────────────────────────────────────────
def localize_tumor(processed_img, pred_label):
    """
    Localize tumor using Otsu thresholding + morphological operations.
    Tumors appear brighter than normal brain tissue in MRI.
    """
    h, w = processed_img.shape

    # Step 1 — Blur to reduce noise
    blurred = cv2.GaussianBlur(processed_img, (11, 11), 0)

    # Step 2 — Otsu: auto-find best threshold from pixel histogram
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3 — Morphological closing: fill holes inside tumor region
    kernel_close = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # Step 4 — Morphological opening: remove small noise dots
    kernel_open = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Step 5 — Remove skull border (outermost 15% = skull, not brain)
    border_mask = np.zeros_like(cleaned)
    margin = int(min(h, w) * 0.15)
    border_mask[margin:h-margin, margin:w-margin] = 255
    brain_only = cv2.bitwise_and(cleaned, border_mask)

    # Step 6 — Find contours of all bright regions
    contours, _ = cv2.findContours(
        brain_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 7 — Make smooth heatmap
    heatmap = brain_only.astype(float) / 255.0
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    # Step 8 — Pick largest contour as tumor region
    bbox         = None
    tumor_mask   = np.zeros_like(processed_img)
    best_contour = None

    if contours and pred_label != 'notumor':
        sorted_c = sorted(contours, key=cv2.contourArea, reverse=True)
        candidate = sorted_c[0]
        min_area  = (h * w) * 0.01   # must be > 1% of image
        if cv2.contourArea(candidate) > min_area:
            x, y, bw, bh = cv2.boundingRect(candidate)
            bbox         = (x, y, bw, bh)
            best_contour = candidate
            cv2.drawContours(tumor_mask, [candidate], -1, 255, -1)

    return heatmap, contours, tumor_mask, bbox, best_contour


def build_colored_overlay(processed_img, tumor_mask, pred_label):
    """Color the tumor region in the class-specific color."""
    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    if pred_label == 'notumor' or tumor_mask is None:
        return img_rgb
    hex_c = CLASS_INFO.get(pred_label, {}).get('color', '#ef4444').lstrip('#')
    r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
    colored = img_rgb.copy()
    colored[tumor_mask > 0] = [r, g, b]
    return cv2.addWeighted(img_rgb, 0.6, colored, 0.4, 0)


# ── Main 4-panel analysis plot ───────────────────────────────────────
def plot_analysis(img_array, pred_label, proba, classes, processed_img):
    heatmap, contours, tumor_mask, bbox, best_contour = localize_tumor(
        processed_img, pred_label
    )
    overlay   = build_colored_overlay(processed_img, tumor_mask, pred_label)
    color_hex = CLASS_INFO.get(pred_label, {}).get('color', '#38bdf8')

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor('#0a0e1a')
    for ax in axes:
        ax.set_facecolor('#0f1e35')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')

    # Panel 1 — Original
    axes[0].imshow(processed_img, cmap='gray')
    axes[0].set_title('Original MRI Scan', color='#94a3b8',
                      fontsize=9, pad=8, fontfamily='monospace')
    axes[0].axis('off')

    # Panel 2 — Otsu heatmap
    axes[1].imshow(processed_img, cmap='gray', alpha=0.45)
    im = axes[1].imshow(heatmap, cmap='hot', alpha=0.65)
    axes[1].set_title('Tumor Intensity Map\n(Otsu Thresholding)',
                      color='#94a3b8', fontsize=9, pad=8, fontfamily='monospace')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='#64748b')

    # Panel 3 — Colored overlay + bounding box
    axes[2].imshow(overlay)
    if bbox and pred_label != 'notumor':
        x, y, bw, bh = bbox
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=2.5, edgecolor=color_hex,
            facecolor='none', linestyle='--'
        )
        axes[2].add_patch(rect)
        # Tumor center crosshair
        cx, cy = x + bw // 2, y + bh // 2
        axes[2].plot([cx-5, cx+5], [cy, cy], color=color_hex, lw=1.5)
        axes[2].plot([cx, cx], [cy-5, cy+5], color=color_hex, lw=1.5)
        axes[2].text(
            x, max(0, y-2), 'Tumor Region',
            color=color_hex, fontsize=7, fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0e1a',
                      edgecolor=color_hex, alpha=0.85)
        )
    title3 = 'Tumor Location\n(Colored Overlay)' if pred_label != 'notumor' else 'No Tumor Detected'
    axes[2].set_title(title3, color='#94a3b8', fontsize=9, pad=8, fontfamily='monospace')
    axes[2].axis('off')

    # Panel 4 — Confidence bars
    bar_colors = [CLASS_INFO[c]['color'] if c in CLASS_INFO else '#38bdf8' for c in classes]
    bars = axes[3].barh(classes, proba * 100, color=bar_colors, alpha=0.85, height=0.5)
    axes[3].set_xlim(0, 110)
    axes[3].set_xlabel('Confidence (%)', color='#64748b', fontsize=9)
    axes[3].set_title('Class Probabilities', color='#94a3b8',
                      fontsize=9, pad=8, fontfamily='monospace')
    axes[3].tick_params(colors='#64748b', labelsize=9)
    for bar, val in zip(bars, proba * 100):
        axes[3].text(val + 1, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', color='#94a3b8', fontsize=8)

    plt.tight_layout(pad=1.5)
    return fig, bbox


# ── Zoomed localization detail ───────────────────────────────────────
def plot_localization_detail(processed_img, pred_label, bbox, best_contour):
    if pred_label == 'notumor' or bbox is None:
        return None

    x, y, bw, bh = bbox
    h, w          = processed_img.shape
    color_hex     = CLASS_INFO.get(pred_label, {}).get('color', '#ef4444')
    r, g, b       = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#0a0e1a')
    for ax in axes:
        ax.set_facecolor('#0f1e35')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')

    # Left — Full scan with filled highlight + crosshair
    axes[0].imshow(processed_img, cmap='gray')
    fill_rect   = patches.Rectangle((x, y), bw, bh,
                                     linewidth=0, facecolor=color_hex, alpha=0.25)
    border_rect = patches.Rectangle((x, y), bw, bh,
                                     linewidth=2.5, edgecolor=color_hex,
                                     facecolor='none', linestyle='--')
    axes[0].add_patch(fill_rect)
    axes[0].add_patch(border_rect)
    cx, cy = x + bw // 2, y + bh // 2
    axes[0].plot([cx-6, cx+6], [cy, cy], color=color_hex, lw=1.5)
    axes[0].plot([cx, cx], [cy-6, cy+6], color=color_hex, lw=1.5)
    axes[0].plot(cx, cy, 'o', color=color_hex, markersize=4)
    axes[0].set_title(f'Localization — {pred_label.upper()}',
                      color='#94a3b8', fontsize=10, pad=8, fontfamily='monospace')
    axes[0].text(2, h-4, f'Box: ({x},{y}) → ({x+bw},{y+bh})  |  Center: ({cx},{cy})',
                 color='#64748b', fontsize=7, fontfamily='monospace')
    axes[0].axis('off')

    # Right — Zoomed crop of tumor
    pad  = 8
    x1   = max(0, x-pad);  y1 = max(0, y-pad)
    x2   = min(w, x+bw+pad); y2 = min(h, y+bh+pad)
    crop = processed_img[y1:y2, x1:x2]

    if crop.size > 0:
        crop_rgb    = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        tint        = crop_rgb.copy()
        tint[:, :]  = [r, g, b]
        crop_tinted = cv2.addWeighted(crop_rgb, 0.65, tint, 0.35, 0)
        axes[1].imshow(crop_tinted)
        axes[1].set_title(f'Zoomed Tumor Region\n({bw}×{bh}px in 64×64 scan)',
                          color='#94a3b8', fontsize=9, pad=8, fontfamily='monospace')
    else:
        axes[1].text(0.5, 0.5, 'Region too small to zoom',
                     ha='center', va='center', color='#64748b',
                     transform=axes[1].transAxes)
    axes[1].axis('off')

    plt.tight_layout(pad=1.5)
    return fig


# ── Confusion matrix ─────────────────────────────────────────────────
def plot_confusion_matrix_chart(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0f1e35')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5, linecolor='#1e3a5f',
                annot_kws={'size': 11, 'color': 'white'})
    ax.set_xlabel('Predicted', color='#64748b', fontsize=10)
    ax.set_ylabel('Actual',    color='#64748b', fontsize=10)
    ax.set_title('Confusion Matrix', color='#94a3b8', fontsize=11, fontfamily='monospace')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    plt.tight_layout()
    return fig


# ── History timeline ─────────────────────────────────────────────────
def plot_history_timeline(history):
    if not history:
        return None
    labels       = [h['prediction'] for h in history]
    confs        = [h['confidence'] for h in history]
    fig, ax      = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0f1e35')
    colors_map   = {k: v['color'] for k, v in CLASS_INFO.items()}
    point_colors = [colors_map.get(l, '#38bdf8') for l in labels]
    ax.scatter(range(len(history)), confs, c=point_colors, s=80, zorder=5)
    ax.plot(range(len(history)), confs, color='#1e3a5f', lw=1, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(history)))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(history))], color='#64748b', fontsize=8)
    ax.set_ylabel('Confidence', color='#64748b', fontsize=9)
    ax.set_title('Prediction History', color='#94a3b8', fontsize=10, fontfamily='monospace')
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
    for i, (label, conf) in enumerate(zip(labels, confs)):
        ax.annotate(label, (i, conf), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7,
                    color=colors_map.get(label, '#38bdf8'))
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🧠 NeuroScan AI")
    st.markdown("---")
    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input("Model path (.pkl)", value="best_model_svm.pkl")

    if st.button("⚡ Load Model"):
        if os.path.exists(model_path):
            with st.spinner("Loading model..."):
                st.session_state.model_data   = load_model(model_path)
                st.session_state.model_loaded = True
            st.success("Model loaded!")
        else:
            st.error(f"File not found: {model_path}")

    if st.session_state.model_loaded:
        st.markdown('<span class="status-dot dot-green"></span> **Model ready**', unsafe_allow_html=True)
        md = st.session_state.model_data
        st.markdown(f"- Classes: `{', '.join(md['classes'])}`")
        st.markdown(f"- Image size: `{md.get('img_size', 64)}×{md.get('img_size', 64)}`")
    else:
        st.markdown('<span class="status-dot dot-red"></span> **No model loaded**', unsafe_allow_html=True)
        st.info("Load your model or use demo mode below.")

    st.markdown("---")
    st.markdown('<div class="section-title">Demo Mode</div>', unsafe_allow_html=True)
    demo_mode = st.toggle("Enable demo mode", value=not st.session_state.model_loaded)

    st.markdown("---")
    st.markdown('<div class="section-title">Localization Settings</div>', unsafe_allow_html=True)
    show_detail = st.toggle("Show zoomed tumor panel", value=True)

    st.markdown("---")
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#64748b; line-height:1.7'>
    Brain tumor MRI classifier using SVM + PCA.<br>
    Tumor localization via Otsu thresholding + morphological operations.<br><br>
    <b style='color:#475569'>Accuracy:</b> 96.43%<br>
    <b style='color:#475569'>CV Score:</b> 95.79% ± 0.25%<br>
    <b style='color:#475569'>Dataset:</b> Kaggle Brain Tumor MRI<br>
    <b style='color:#475569'>Images:</b> 5,600 across 4 classes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 NeuroScan AI")
st.markdown('<p style="color:#64748b; font-size:0.9rem; margin-top:-8px;">Brain Tumor MRI Classification & Localization System</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Diagnose", "📊 Model Metrics", "📈 History", "ℹ️ Classes"])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown("### Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Drop your MRI image here",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            label_visibility="collapsed"
        )
        if uploaded_file:
            img         = Image.open(uploaded_file).convert('RGB')
            img_array   = np.array(img)
            st.image(img, caption="Uploaded MRI", use_container_width=True)
            analyze_btn = st.button("🔬 Analyze & Localize", use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#475569; font-size:0.85rem;'>
                <div style='font-size:2.5rem; margin-bottom:12px'>🧠</div>
                Upload a brain MRI image to begin analysis
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with col_right:
        if uploaded_file and analyze_btn:
            with st.spinner("Analyzing & localizing tumor..."):
                time.sleep(0.4)

                if demo_mode or not st.session_state.model_loaded:
                    import random
                    demo_classes  = ['glioma', 'meningioma', 'notumor', 'pituitary']
                    pred_label    = random.choice(demo_classes)
                    proba         = np.random.dirichlet(np.ones(4) * 2)
                    proba[demo_classes.index(pred_label)] = max(proba) + 0.3
                    proba         = proba / proba.sum()
                    classes       = np.array(demo_classes)
                    processed_img = preprocess_image(img_array)
                    st.warning("⚠️ Demo mode — load your model for real predictions.")
                else:
                    pred_label, proba, classes, processed_img = predict(
                        img_array, st.session_state.model_data
                    )

                confidence = float(proba.max())
                info       = CLASS_INFO.get(pred_label, {})

                # Run localization to get bbox for result card
                _, _, _, bbox_card, _ = localize_tumor(processed_img, pred_label)

                # Save to history
                st.session_state.history.append({
                    'time':       datetime.now().strftime('%H:%M:%S'),
                    'filename':   uploaded_file.name,
                    'prediction': pred_label,
                    'confidence': confidence,
                    'severity':   info.get('severity', '-')
                })

            # ── Result card ──
            st.markdown("### Diagnosis Result")
            badge_class = info.get('badge', '')
            severity    = info.get('severity', '')
            sev_color   = '#ef4444' if severity == 'HIGH' else '#f97316' if severity == 'MODERATE' else '#22c55e'
            location    = info.get('location', 'N/A')
            tumor_found = pred_label != 'notumor'
            bbox_str    = f"({bbox_card[0]},{bbox_card[1]}) → ({bbox_card[0]+bbox_card[2]},{bbox_card[1]+bbox_card[3]})" if bbox_card else "Not detected"

            st.markdown(f"""
            <div class="neuro-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
                    <div>
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Diagnosis</div>
                        <span class="pred-badge {badge_class}">{pred_label.upper()}</span>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Severity</div>
                        <div style="font-family:monospace; font-size:1rem; font-weight:700; color:{sev_color}">{severity}</div>
                    </div>
                </div>
                <div style="font-size:0.85rem; color:#94a3b8; line-height:1.6; margin-bottom:10px">{info.get('desc','')}</div>
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:12px">
                    📍 <b style="color:#94a3b8">Typical location:</b> {location}
                </div>
                <div style="display:flex; gap:1.5rem; flex-wrap:wrap">
                    <div>
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Confidence</div>
                        <div style="font-family:monospace; font-size:1.4rem; font-weight:700; color:#38bdf8">{confidence*100:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Tumor Found</div>
                        <div style="font-family:monospace; font-size:1.4rem; font-weight:700; color:{'#ef4444' if tumor_found else '#22c55e'}">{'YES' if tumor_found else 'NO'}</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Region (64×64)</div>
                        <div style="font-family:monospace; font-size:0.8rem; font-weight:700; color:#94a3b8; margin-top:4px">{bbox_str}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Per-class probabilities ──
            st.markdown("**Class Probabilities**")
            for cls, prob in zip(classes, proba):
                cls_color = CLASS_INFO.get(cls, {}).get('color', '#38bdf8')
                st.markdown(f"""
                <div style="margin-bottom:6px">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-bottom:3px">
                        <span>{cls}</span>
                        <span style="font-family:monospace">{prob*100:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{prob*100:.1f}%; background:{cls_color}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif not uploaded_file:
            st.markdown("""
            <div style='text-align:center; padding:5rem 2rem; color:#334155'>
                <div style='font-size:3rem; margin-bottom:16px; opacity:0.4'>📋</div>
                <div style='font-size:0.9rem'>Results will appear here after analysis</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Full-width panels ──
    if uploaded_file and analyze_btn and 'processed_img' in dir():
        st.markdown("---")
        st.markdown("### Scan Analysis & Tumor Localization")

        fig, _ = plot_analysis(img_array, pred_label, proba, classes, processed_img)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Zoomed detail
        if show_detail and pred_label != 'notumor':
            st.markdown("### Zoomed Tumor Region")
            _, _, _, bbox_z, contour_z = localize_tumor(processed_img, pred_label)
            fig_zoom = plot_localization_detail(processed_img, pred_label, bbox_z, contour_z)
            if fig_zoom:
                st.pyplot(fig_zoom, use_container_width=True)
                plt.close()

        # How it works
        st.markdown(f"""
        <div class="neuro-card" style="margin-top:0.5rem">
            <div style="font-size:0.75rem; color:#64748b; line-height:2.0">
                <b style="color:#38bdf8; font-family:monospace">HOW LOCALIZATION WORKS</b><br>
                <b style="color:#94a3b8">Panel 1 — Original MRI:</b> Grayscale scan after 64×64 resize + histogram equalization.<br>
                <b style="color:#94a3b8">Panel 2 — Intensity Map:</b> Otsu thresholding finds bright regions automatically. Tumors appear brighter than normal tissue in MRI.<br>
                <b style="color:#94a3b8">Panel 3 — Colored Overlay:</b> Largest bright region highlighted in class color. Dashed box = bounding region. Crosshair = tumor center.<br>
                <b style="color:#94a3b8">Panel 4 — Probabilities:</b> SVM confidence scores for all 4 classes.<br>
                <b style="color:#ef4444">⚠️ Note:</b> Classical image processing (not deep learning GradCAM). Always confirm with a radiologist.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL METRICS
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (label, val, sub) in zip([m1,m2,m3,m4,m5], [
        ("Test Accuracy", "96.43%", "SVM"),
        ("CV Mean",       "95.79%", "5-fold"),
        ("CV Std",        "±0.25%", "Low variance"),
        ("Train Samples", "4,480",  "80% split"),
        ("Test Samples",  "1,120",  "20% split"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div style="font-size:0.7rem; color:#334155; margin-top:4px">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### All Models Comparison")
        model_names = ['Logistic\nRegression', 'KNN', 'SVM', 'Random\nForest', 'Gradient\nBoosting']
        accs = [84.91, 91.79, 96.43, 91.70, 91.25]
        cvs  = [85.36, 91.36, 95.79, 91.30, 89.64]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#0f1e35')
        x     = np.arange(len(model_names))
        bars1 = ax.bar(x - 0.18, accs, 0.34, label='Test Acc', color='#0369a1', alpha=0.9)
        bars2 = ax.bar(x + 0.18, cvs,  0.34, label='CV Mean',  color='#334155', alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, color='#64748b', fontsize=7)
        ax.set_ylim(80, 100)
        ax.set_ylabel('Accuracy (%)', color='#64748b', fontsize=9)
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
        ax.legend(fontsize=8, facecolor='#0a0e1a', edgecolor='#1e3a5f', labelcolor='#94a3b8')
        bars1[2].set_color('#38bdf8')
        for bar, v in zip(bars1, accs):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.1,
                    f'{v}', ha='center', va='bottom', fontsize=6.5, color='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("### Confusion Matrix (SVM)")
        cm = np.array([
            [268,  7,  3,  2],
            [  6, 258,  8,  8],
            [  1,  2, 275,  2],
            [  0,  0,  1, 279],
        ])
        fig_cm = plot_confusion_matrix_chart(cm, ['glioma','meningioma','notumor','pituitary'])
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("### Per-Class Performance (SVM)")
    pcd = {
        'Class':     ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'Precision': [96.75, 94.16, 95.82, 98.94],
        'Recall':    [95.71, 92.14, 98.21, 99.64],
        'F1-Score':  [96.23, 93.14, 97.00, 99.29],
    }
    fig_pc, ax_pc = plt.subplots(figsize=(10, 3.5))
    fig_pc.patch.set_facecolor('#0a0e1a')
    ax_pc.set_facecolor('#0f1e35')
    x_pc = np.arange(len(pcd['Class']))
    w    = 0.25
    for i, (metric, color) in enumerate(zip(
        ['Precision','Recall','F1-Score'], ['#38bdf8','#818cf8','#34d399']
    )):
        ax_pc.bar(x_pc + i*w - w, pcd[metric], w, label=metric, color=color, alpha=0.85)
    ax_pc.set_xticks(x_pc)
    ax_pc.set_xticklabels(pcd['Class'], color='#94a3b8', fontsize=9)
    ax_pc.set_ylim(85, 102)
    ax_pc.set_ylabel('Score (%)', color='#64748b', fontsize=9)
    ax_pc.tick_params(colors='#64748b')
    for spine in ax_pc.spines.values(): spine.set_edgecolor('#1e3a5f')
    ax_pc.legend(fontsize=9, facecolor='#0a0e1a', edgecolor='#1e3a5f', labelcolor='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig_pc, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Prediction History")
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center; padding:4rem; color:#334155'>
            <div style='font-size:2rem; margin-bottom:12px'>📭</div>
            No predictions yet. Upload an MRI scan to get started.
        </div>
        """, unsafe_allow_html=True)
    else:
        h           = st.session_state.history
        total       = len(h)
        avg_conf    = np.mean([x['confidence'] for x in h]) * 100
        tumor_count = sum(1 for x in h if x['prediction'] != 'notumor')

        c1, c2, c3 = st.columns(3)
        for col, (label, val) in zip([c1,c2,c3], [
            ("Total Scans",    str(total)),
            ("Avg Confidence", f"{avg_conf:.1f}%"),
            ("Tumor Detected", f"{tumor_count}/{total}"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        fig_hist = plot_history_timeline(h)
        if fig_hist:
            st.pyplot(fig_hist, use_container_width=True)
            plt.close()

        st.markdown("---")
        st.markdown("### Scan Log")
        for i, item in enumerate(reversed(h)):
            info  = CLASS_INFO.get(item['prediction'], {})
            badge = info.get('badge', '')
            icon  = info.get('icon', '•')
            st.markdown(f"""
            <div class="hist-item">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <div>
                        <span style="color:#475569; font-size:0.75rem">#{total-i} · {item['time']}</span>
                        <span style="margin-left:10px; color:#94a3b8">{item['filename']}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px">
                        <span style="font-family:monospace; font-size:0.8rem; color:#64748b">{item['confidence']*100:.1f}%</span>
                        <span class="pred-badge {badge}" style="font-size:0.75rem; padding:3px 10px">{icon} {item['prediction'].upper()}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — CLASSES
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Tumor Classification Guide")

    for cls_name, info in CLASS_INFO.items():
        badge     = info['badge']
        sev_color = '#ef4444' if info['severity'] == 'HIGH' else '#f97316' if info['severity'] == 'MODERATE' else '#22c55e'
        st.markdown(f"""
        <div class="neuro-card" style="border-left: 3px solid {info['color']}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start">
                <div style="flex:1">
                    <span class="pred-badge {badge}">{info['icon']} {cls_name.upper()}</span>
                    <p style="color:#94a3b8; font-size:0.875rem; margin-top:10px; margin-bottom:6px; line-height:1.6">{info['desc']}</p>
                    <p style="color:#64748b; font-size:0.8rem; margin:0">
                        📍 <b style="color:#475569">Typical location:</b> {info['location']}
                    </p>
                </div>
                <div style="text-align:right; margin-left:2rem; flex-shrink:0">
                    <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Severity</div>
                    <div style="font-family:monospace; font-weight:700; color:{sev_color}; font-size:0.9rem">{info['severity']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How Tumor Localization Works")
    st.markdown("""
    <div class="neuro-card">
        <div style="font-size:0.82rem; color:#94a3b8; line-height:2.0">
            <b style="color:#38bdf8">Step 1 — Gaussian Blur:</b> Smooths the image to remove noise.<br>
            <b style="color:#38bdf8">Step 2 — Otsu Thresholding:</b> Auto-finds the best intensity cutoff to separate bright tumor tissue from dark normal tissue.<br>
            <b style="color:#38bdf8">Step 3 — Morphological Closing:</b> Fills small holes inside detected bright regions.<br>
            <b style="color:#38bdf8">Step 4 — Morphological Opening:</b> Removes tiny noise dots.<br>
            <b style="color:#38bdf8">Step 5 — Skull Removal:</b> Masks out the outermost 15% border (skull) so only brain interior is analyzed.<br>
            <b style="color:#38bdf8">Step 6 — Contour Detection:</b> Finds all connected bright regions in brain interior.<br>
            <b style="color:#38bdf8">Step 7 — Largest Region:</b> Selects the biggest contour as the tumor region.<br>
            <b style="color:#38bdf8">Step 8 — Overlay:</b> Colors tumor region in the class color + draws bounding box + crosshair at center.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="neuro-card">
        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px">Disclaimer</div>
        <p style="color:#64748b; font-size:0.8rem; line-height:1.8; margin:0">
        This tool is for <b style="color:#94a3b8">research and educational purposes only</b>.
        Not a substitute for professional medical diagnosis.
        Always consult a qualified radiologist or neurologist.
        Model accuracy: 96.43% — errors possible especially for meningioma.
        Localization uses classical image processing, not deep learning segmentation.
        </p>
    </div>
    """, unsafe_allow_html=True)
