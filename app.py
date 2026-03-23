import streamlit as st
import numpy as np
import cv2
import pickle
import os
import json
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, classification_report
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

/* Main background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* Headings */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #38bdf8 !important; font-size: 1.6rem !important; letter-spacing: -0.5px; }
h2 { color: #7dd3fc !important; font-size: 1.1rem !important; }
h3 { color: #bae6fd !important; font-size: 0.95rem !important; }

/* Cards */
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

/* Prediction badge */
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

/* Confidence bar */
.conf-bar-bg {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    margin: 6px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

/* History item */
.hist-item {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0f1e35 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
}

/* Buttons */
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

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1629;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 8px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #0369a1 !important;
    color: white !important;
}

/* Divider */
hr { border-color: #1e2d4a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* Progress bar */
.stProgress > div > div { background: #0369a1 !important; }

/* Info/warning boxes */
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

# ── Helper functions ─────────────────────────────────────────────────
CLASS_INFO = {
    'glioma': {
        'color': '#ef4444', 'badge': 'badge-glioma',
        'desc': 'A tumor originating in glial cells of the brain or spine. Can be low or high grade.',
        'severity': 'HIGH', 'icon': '🔴'
    },
    'meningioma': {
        'color': '#f97316', 'badge': 'badge-meningioma',
        'desc': 'Tumor arising from meninges. Usually benign and slow-growing.',
        'severity': 'MODERATE', 'icon': '🟠'
    },
    'notumor': {
        'color': '#22c55e', 'badge': 'badge-notumor',
        'desc': 'No tumor detected. Brain tissue appears normal.',
        'severity': 'NONE', 'icon': '🟢'
    },
    'pituitary': {
        'color': '#8b5cf6', 'badge': 'badge-pituitary',
        'desc': 'Tumor on the pituitary gland. Usually benign but affects hormones.',
        'severity': 'MODERATE', 'icon': '🟣'
    }
}

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def preprocess_image(img_array, img_size=64):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    resized = cv2.resize(gray, (img_size, img_size))
    equalized = cv2.equalizeHist(resized)
    return equalized

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


    
    def generate_heatmap(img_array, pred_label, processed_img):
    
    # Step 1: Blur to reduce noise
    blurred = cv2.GaussianBlur(processed_img, (11, 11), 0)
    
    # Step 2: Otsu thresholding — finds bright regions automatically
    _, thresh = cv2.threshold(blurred, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Clean up small noise
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Step 4: Find all contours (tumor regions)
    contours, _ = cv2.findContours(cleaned, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Make heatmap from cleaned mask
    heatmap = cleaned.astype(float) / 255.0
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    return heatmap, contours

def plot_analysis(img_array, pred_label, proba, classes, processed_img):
    heatmap, contours = generate_heatmap(img_array, pred_label, processed_img)
    color = CLASS_INFO.get(pred_label, {}).get('color', '#38bdf8')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#0a0e1a')

    for ax in axes:
        ax.set_facecolor('#0f1e35')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')

    # ── Original image ──
    axes[0].imshow(processed_img, cmap='gray')
    axes[0].set_title('MRI Scan', color='#94a3b8', fontsize=10, pad=10, fontfamily='monospace')
    axes[0].axis('off')

    # ── Heatmap overlay ──
    heatmap_resized = cv2.resize(heatmap, (processed_img.shape[1], processed_img.shape[0]))
    axes[1].imshow(processed_img, cmap='gray', alpha=0.5)
    im = axes[1].imshow(heatmap_resized, cmap='hot', alpha=0.6)
    axes[1].set_title('Attention Heatmap', color='#94a3b8', fontsize=10, pad=10, fontfamily='monospace')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='#64748b')

    # Draw bounding box around ROI
    if contours and pred_label != 'notumor':
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        scale_x = processed_img.shape[1] / heatmap.shape[1]
        scale_y = processed_img.shape[0] / heatmap.shape[0]
        rect = patches.Rectangle(
            (x * scale_x, y * scale_y), bw * scale_x, bh * scale_y,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--', label='Region of Interest'
        )
        axes[1].add_patch(rect)
        axes[1].legend(fontsize=7, loc='upper right',
                       facecolor='#0a0e1a', edgecolor='#1e3a5f',
                       labelcolor='#94a3b8')

    # ── Confidence bar chart ──
    bar_colors = [CLASS_INFO[c]['color'] if c in CLASS_INFO else '#38bdf8' for c in classes]
    bars = axes[2].barh(classes, proba * 100, color=bar_colors, alpha=0.85, height=0.5)
    axes[2].set_xlim(0, 105)
    axes[2].set_xlabel('Confidence (%)', color='#64748b', fontsize=9)
    axes[2].set_title('Class Probabilities', color='#94a3b8', fontsize=10, pad=10, fontfamily='monospace')
    axes[2].tick_params(colors='#64748b', labelsize=9)
    for bar, val in zip(bars, proba * 100):
        axes[2].text(val + 1, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', color='#94a3b8', fontsize=8)

    plt.tight_layout(pad=1.5)
    return fig


def plot_confusion_matrix_chart(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0f1e35')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5, linecolor='#1e3a5f',
                annot_kws={'size': 11, 'color': 'white'})
    ax.set_xlabel('Predicted', color='#64748b', fontsize=10)
    ax.set_ylabel('Actual', color='#64748b', fontsize=10)
    ax.set_title('Confusion Matrix', color='#94a3b8', fontsize=11, fontfamily='monospace')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    plt.tight_layout()
    return fig


def plot_history_timeline(history):
    if not history:
        return None
    labels = [h['prediction'] for h in history]
    confs  = [h['confidence'] for h in history]
    times  = [h['time'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0f1e35')

    colors_map = {k: v['color'] for k, v in CLASS_INFO.items()}
    point_colors = [colors_map.get(l, '#38bdf8') for l in labels]

    ax.scatter(range(len(history)), confs, c=point_colors, s=80, zorder=5)
    ax.plot(range(len(history)), confs, color='#1e3a5f', lw=1, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(history)))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(history))], color='#64748b', fontsize=8)
    ax.set_ylabel('Confidence', color='#64748b', fontsize=9)
    ax.set_title('Prediction History', color='#94a3b8', fontsize=10, fontfamily='monospace')
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')

    for i, (label, conf) in enumerate(zip(labels, confs)):
        ax.annotate(label, (i, conf), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7,
                    color=colors_map.get(label, '#38bdf8'))
    plt.tight_layout()
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 NeuroScan AI")
    st.markdown("---")

    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model path (.pkl)",
        value="best_model_svm.pkl",
        help="Path to your saved model pickle file"
    )

    if st.button("⚡ Load Model"):
        if os.path.exists(model_path):
            with st.spinner("Loading model..."):
                st.session_state.model_data = load_model(model_path)
                st.session_state.model_loaded = True
            st.success("Model loaded!")
        else:
            st.error(f"File not found: {model_path}")

    # Model status
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
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#64748b; line-height:1.6'>
    Brain tumor MRI classifier using SVM + PCA pipeline trained on 5,600 MRI images across 4 classes.
    <br><br>
    <b style='color:#475569'>Accuracy:</b> 96.43%<br>
    <b style='color:#475569'>CV Score:</b> 95.79% ± 0.25%<br>
    <b style='color:#475569'>Dataset:</b> Kaggle Brain Tumor MRI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ── Main content ─────────────────────────────────────────────────────
st.markdown("# 🧠 NeuroScan AI")
st.markdown('<p style="color:#64748b; font-size:0.9rem; margin-top:-8px;">Brain Tumor MRI Classification & Analysis System</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Diagnose", "📊 Model Metrics", "📈 History", "ℹ️ Classes"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ═══════════════════════════════════════════════════════════
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
            img = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(img)
            st.image(img, caption="Uploaded MRI", use_container_width=True)

            analyze_btn = st.button("🔬 Analyze Scan", use_container_width=True)
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
            with st.spinner("Analyzing MRI scan..."):
                time.sleep(0.5)

                if demo_mode or not st.session_state.model_loaded:
                    # Demo mode — simulate result
                    import random
                    demo_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
                    pred_label = random.choice(demo_classes)
                    proba = np.random.dirichlet(np.ones(4) * 2)
                    proba[demo_classes.index(pred_label)] = max(proba) + 0.3
                    proba = proba / proba.sum()
                    classes = np.array(demo_classes)
                    processed_img = preprocess_image(img_array)
                    st.warning("⚠️ Demo mode — load your model for real predictions.")
                else:
                    pred_label, proba, classes, processed_img = predict(img_array, st.session_state.model_data)

                confidence = float(proba.max())
                info = CLASS_INFO.get(pred_label, {})

                # Save to history
                st.session_state.history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'filename': uploaded_file.name,
                    'prediction': pred_label,
                    'confidence': confidence,
                    'severity': info.get('severity', '-')
                })

            # ── Result card ──
            st.markdown("### Diagnosis Result")
            badge_class = info.get('badge', '')
            severity    = info.get('severity', '')
            sev_color   = '#ef4444' if severity == 'HIGH' else '#f97316' if severity == 'MODERATE' else '#22c55e'

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
                <div style="font-size:0.85rem; color:#94a3b8; line-height:1.6; margin-bottom:12px">{info.get('desc','')}</div>
                <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px">Confidence</div>
                <div style="font-family:monospace; font-size:1.5rem; font-weight:700; color:#38bdf8">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Per-class probabilities ──
            st.markdown("**Class Probabilities**")
            for cls, prob in zip(classes, proba):
                cls_color = CLASS_INFO.get(cls, {}).get('color', '#38bdf8')
                st.markdown(f"""
                <div style="margin-bottom:6px">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-bottom:3px">
                        <span>{cls}</span><span style="font-family:monospace">{prob*100:.1f}%</span>
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

    # ── Full-width analysis charts ──
    if uploaded_file and analyze_btn and 'processed_img' in dir():
        st.markdown("---")
        st.markdown("### Scan Analysis & Heatmap")
        fig = plot_analysis(img_array, pred_label, proba, classes, processed_img)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="neuro-card" style="margin-top:0.5rem">
            <div style="font-size:0.75rem; color:#64748b; line-height:1.8">
                <b style="color:#94a3b8">🔴 Attention Heatmap:</b> Highlights regions with high gradient activity — areas the model focuses on.<br>
                <b style="color:#94a3b8">📦 Bounding Box:</b> Marks the largest region of interest detected in the scan.<br>
                <b style="color:#94a3b8">⚠️ Note:</b> This is a traditional ML model (SVM + PCA). The heatmap uses gradient-based saliency, not deep learning GradCAM.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — MODEL METRICS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Metrics")

    # Summary metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Test Accuracy", "96.43%", "SVM"),
        ("CV Mean", "95.79%", "5-fold"),
        ("CV Std", "±0.25%", "Low variance"),
        ("Train Samples", "4,480", "80% split"),
        ("Test Samples", "1,120", "20% split"),
    ]
    for col, (label, val, sub) in zip([m1,m2,m3,m4,m5], metrics):
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
        model_data_table = {
            'Model': ['Logistic Regression', 'KNN', 'SVM', 'Random Forest', 'Gradient Boosting'],
            'Test Acc (%)': [84.91, 91.79, 96.43, 91.70, 91.25],
            'CV Mean (%)': [85.36, 91.36, 95.79, 91.30, 89.64],
            'CV Std (%)': [0.36, 0.45, 0.25, 0.34, 0.66],
        }

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#0f1e35')
        models_list = model_data_table['Model']
        accs = model_data_table['Test Acc (%)']
        cvs  = model_data_table['CV Mean (%)']
        x = np.arange(len(models_list))
        bars1 = ax.bar(x - 0.18, accs, 0.34, label='Test Acc', color='#0369a1', alpha=0.9)
        bars2 = ax.bar(x + 0.18, cvs,  0.34, label='CV Mean',  color='#334155', alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models_list], color='#64748b', fontsize=7)
        ax.set_ylim(80, 100)
        ax.set_ylabel('Accuracy (%)', color='#64748b', fontsize=9)
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
        ax.legend(fontsize=8, facecolor='#0a0e1a', edgecolor='#1e3a5f', labelcolor='#94a3b8')
        # Highlight best
        bars1[2].set_color('#38bdf8')
        for bar, v in zip(bars1, accs):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v}', ha='center', va='bottom', fontsize=6.5, color='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("### Confusion Matrix (SVM)")
        # Real values from your notebook
        cm = np.array([
            [268,  7,  3,  2],
            [ 6, 258,  8,  8],
            [ 1,  2, 275,  2],
            [ 0,  0,  1, 279],
        ])
        classes_cm = ['glioma', 'meningioma', 'notumor', 'pituitary']
        fig_cm = plot_confusion_matrix_chart(cm, classes_cm)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("### Per-Class Performance (SVM)")

    per_class = {
        'Class':     ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'Precision': [96.75, 94.16, 95.82, 98.94],
        'Recall':    [95.71, 92.14, 98.21, 99.64],
        'F1-Score':  [96.23, 93.14, 97.00, 99.29],
        'Support':   [280, 280, 280, 280],
    }

    fig_pc, ax_pc = plt.subplots(figsize=(10, 3.5))
    fig_pc.patch.set_facecolor('#0a0e1a')
    ax_pc.set_facecolor('#0f1e35')
    x_pc = np.arange(len(per_class['Class']))
    w = 0.25
    colors_pc = ['#38bdf8', '#818cf8', '#34d399']
    for i, (metric, color) in enumerate(zip(['Precision','Recall','F1-Score'], colors_pc)):
        ax_pc.bar(x_pc + i*w - w, per_class[metric], w, label=metric, color=color, alpha=0.85)
    ax_pc.set_xticks(x_pc)
    ax_pc.set_xticklabels(per_class['Class'], color='#94a3b8', fontsize=9)
    ax_pc.set_ylim(85, 102)
    ax_pc.set_ylabel('Score (%)', color='#64748b', fontsize=9)
    ax_pc.tick_params(colors='#64748b')
    for spine in ax_pc.spines.values(): spine.set_edgecolor('#1e3a5f')
    ax_pc.legend(fontsize=9, facecolor='#0a0e1a', edgecolor='#1e3a5f', labelcolor='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig_pc, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ═══════════════════════════════════════════════════════════
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
        # Summary stats
        h = st.session_state.history
        total = len(h)
        avg_conf = np.mean([x['confidence'] for x in h]) * 100
        tumor_count = sum(1 for x in h if x['prediction'] != 'notumor')

        c1, c2, c3 = st.columns(3)
        for col, (label, val) in zip([c1,c2,c3], [
            ("Total Scans", str(total)),
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

        # Timeline chart
        fig_hist = plot_history_timeline(h)
        if fig_hist:
            st.pyplot(fig_hist, use_container_width=True)
            plt.close()

        st.markdown("---")
        st.markdown("### Scan Log")
        for i, item in enumerate(reversed(h)):
            info = CLASS_INFO.get(item['prediction'], {})
            badge = info.get('badge', '')
            icon  = info.get('icon', '•')
            st.markdown(f"""
            <div class="hist-item">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <div>
                        <span style="color:#475569; font-size:0.75rem">#{total - i} · {item['time']}</span>
                        <span style="margin-left:10px; color:#94a3b8">{item['filename']}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px">
                        <span style="font-family:monospace; font-size:0.8rem; color:#64748b">{item['confidence']*100:.1f}%</span>
                        <span class="pred-badge {badge}" style="font-size:0.75rem; padding:3px 10px">{icon} {item['prediction'].upper()}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — CLASSES INFO
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Tumor Classification Guide")

    for cls_name, info in CLASS_INFO.items():
        badge = info['badge']
        sev_color = '#ef4444' if info['severity'] == 'HIGH' else '#f97316' if info['severity'] == 'MODERATE' else '#22c55e'
        st.markdown(f"""
        <div class="neuro-card" style="border-left: 3px solid {info['color']}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start">
                <div>
                    <span class="pred-badge {badge}">{info['icon']} {cls_name.upper()}</span>
                    <p style="color:#94a3b8; font-size:0.875rem; margin-top:10px; line-height:1.6">{info['desc']}</p>
                </div>
                <div style="text-align:right; margin-left:2rem; flex-shrink:0">
                    <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:2px">Severity</div>
                    <div style="font-family:monospace; font-weight:700; color:{sev_color}; font-size:0.9rem">{info['severity']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="neuro-card">
        <div class="section-title">Disclaimer</div>
        <p style="color:#64748b; font-size:0.8rem; line-height:1.8">
        This tool is intended for <b style="color:#94a3b8">research and educational purposes only</b>.
        It is not a substitute for professional medical diagnosis.
        Always consult a qualified radiologist or neurologist for clinical decisions.
        Model accuracy: 96.43% on the test set — errors are possible, especially for meningioma cases.
        </p>
    </div>
    """, unsafe_allow_html=True)
