"""
SIGAB Cilacap — Sistem Informasi Geospasial Antisipasi Banjir
Streamlit Web App + Peta Interaktif Folium
Skripsi: Klasifikasi Jalur Evakuasi Banjir Berbasis Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os, time
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# ── Import komponen peta ─────────────────────────────────────────
from map_component import (
    build_map, geocode_all_shelters, geocode_shelter,
    CILACAP_CENTER, CLASS_COLORS_HEX, RISK_LABEL
)

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SIGAB Cilacap — Jalur Evakuasi Banjir",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background:#0d2137; }
  [data-testid="stSidebar"] * { color:#e8f4fd !important; }
  [data-testid="stSidebar"] .stSlider > div > div > div { background:#1a6faf !important; }
  .badge-fastest  {background:#fff3cd;color:#856404;padding:4px 12px;border-radius:20px;font-weight:600}
  .badge-safest   {background:#d1f2eb;color:#0f6b50;padding:4px 12px;border-radius:20px;font-weight:600}
  .badge-balanced {background:#d6eaf8;color:#1a5276;padding:4px 12px;border-radius:20px;font-weight:600}
  .rec-card {
    border-radius:10px;padding:10px 14px;margin:6px 0;
    border-left:4px solid;cursor:pointer;
  }
  .rec-fastest  {background:#fffbf0;border-color:#f39c12}
  .rec-safest   {background:#f0faf5;border-color:#27ae60}
  .rec-balanced {background:#f0f6ff;border-color:#2980b9}
  .section-header {
    font-size:17px;font-weight:700;color:#0d2137;
    border-bottom:2px solid #1a6faf;padding-bottom:5px;
    margin:20px 0 12px 0;
  }
  div[data-testid="stMetricValue"] { font-size:22px !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
MEAN_RH    = 83.53
MEAN_RR    = 15.16
MAX_RR     = 135.0
CLASS_NAMES  = ['Fastest', 'Safest', 'Balanced']
CLASS_EMOJI  = {'Fastest':'⚡','Safest':'🛡️','Balanced':'⚖️'}
FEATURES     = ['Distance_km','Time_min','RR','RH','RiskWeight']

# ════════════════════════════════════════════════════════════════
# LOAD MODEL & DATA
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists("model_rf.pkl"):
        st.error("❌ `model_rf.pkl` tidak ditemukan. Jalankan training script dulu.")
        st.stop()
    with open("model_rf.pkl","rb") as f:
        model = pickle.load(f)
    info = {}
    if os.path.exists("model_info.json"):
        with open("model_info.json") as f:
            info = json.load(f)
    return model, info

rf, model_info = load_model()

@st.cache_data
def load_routes():
    """
    Load routes.csv. Kolom yang dibutuhkan:
      - Distance_km, Time_min
      - lat_asal, lon_asal     ← koordinat titik asal
      - lat_tujuan, lon_tujuan ← koordinat titik tujuan
      - Asal, Tujuan           ← nama titik (untuk shelter geocoding)

    Jika kolom koordinat belum ada tapi ada Asal/Tujuan,
    script akan geocode otomatis (satu kali, lalu di-cache).
    """
    if os.path.exists("routes.csv"):
        df = pd.read_csv("routes.csv")
    else:
        # ── Demo data jika file belum ada ────────────────────
        np.random.seed(42)
        shelters = [
            ("Desa Karangkandri",    -7.6850, 109.0200),
            ("Desa Padangjaya",      -7.7050, 109.0350),
            ("Desa Menganti",        -7.6750, 108.9800),
            ("Balai Desa Kuripan",   -7.7200, 109.0050),
            ("Masjid Al-Jihad",      -7.6950, 109.0150),
            ("Balai Desa Gombolharjo",-7.7350, 109.0250),
            ("Kantor Kelurahan Mertasinga",-7.6600,109.0100),
            ("RS Pertamina Cilacap", -7.7198, 109.0073),
            ("Alun-alun Cilacap",    -7.7276, 109.0128),
            ("BPBD Cilacap",         -7.7256, 109.0135),
        ]
        rows = []
        for _ in range(80):
            a  = shelters[np.random.randint(len(shelters))]
            b  = shelters[np.random.randint(len(shelters))]
            if a == b: continue
            dist = np.round(np.random.uniform(1, 15), 2)
            rows.append({
                'Route_ID'   : f"R{len(rows)+1:03d}",
                'Asal'       : a[0], 'lat_asal' : a[1], 'lon_asal' : a[2],
                'Tujuan'     : b[0], 'lat_tujuan': b[1], 'lon_tujuan': b[2],
                'Distance_km': dist,
                'Time_min'   : np.round(dist * 3.5, 1),
            })
        df = pd.DataFrame(rows)
    return df

df_routes = load_routes()

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def get_risk_weight(rh, rr):
    if rr > MAX_RR or rr == 8888: return 5, 5.0
    elif rh > MEAN_RH and rr >= MEAN_RR: return 4, 2.5
    elif rh > MEAN_RH or rr >= MEAN_RR:  return 3, 1.5
    elif 80 <= rh <= MEAN_RH:            return 2, 1.2
    else:                                return 1, 1.0

def risk_badge(level):
    icons = {1:"🟢",2:"🟡",3:"🟠",4:"🔴",5:"⛔"}
    return f"{icons.get(level,'⚪')} {RISK_LABEL.get(level,'?')}"

def predict_routes(df, rh, rr):
    level, weight = get_risk_weight(rh, rr)
    df = df.copy()
    df['RH'] = rh; df['RR'] = rr; df['RiskWeight'] = weight
    df['RiskLevel'] = level
    X = df[FEATURES].values
    df['Label']      = rf.predict(X)
    df['Label_Name'] = df['Label'].map(lambda x: CLASS_NAMES[x])
    proba = rf.predict_proba(X)
    for i,cn in enumerate(CLASS_NAMES):
        df[f'Prob_{cn}'] = np.round(proba[:,i]*100,1)
    df['Confidence'] = np.round(proba.max(axis=1)*100,1)
    return df

# ── Shelter geocoding (cached di session state) ──────────────────
def ensure_shelter_coords(df):
    """
    Pastikan koordinat shelter tersedia.
    Urutan prioritas:
      1. lat_asal/lon_asal sudah ada di CSV → langsung pakai
      2. Hanya punya nama → geocode via Nominatim (disimpan ke session)
    """
    if 'lat_asal' in df.columns and df['lat_asal'].notna().any():
        # Koordinat sudah ada di CSV ─ bangun dict dari data
        shelter_coords = {}
        for _, r in df.drop_duplicates('Tujuan').iterrows():
            if pd.notna(r.get('lat_tujuan')):
                shelter_coords[r['Tujuan']] = (r['lat_tujuan'], r['lon_tujuan'])
        return shelter_coords

    # Belum ada koordinat → geocode
    if 'shelter_coords' not in st.session_state:
        names = list(df['Tujuan'].dropna().unique()) if 'Tujuan' in df.columns else []
        if names:
            with st.spinner("Geocoding shelter locations (sekali saja, lalu di-cache)…"):
                st.session_state['shelter_coords'] = geocode_all_shelters(names)
        else:
            st.session_state['shelter_coords'] = {}

    # Tambahkan koordinat ke dataframe
    coords = st.session_state['shelter_coords']
    if 'Asal' in df.columns:
        df['lat_asal']   = df['Asal'].map(lambda n: coords.get(n,(None,None))[0])
        df['lon_asal']   = df['Asal'].map(lambda n: coords.get(n,(None,None))[1])
    if 'Tujuan' in df.columns:
        df['lat_tujuan'] = df['Tujuan'].map(lambda n: coords.get(n,(None,None))[0])
        df['lon_tujuan'] = df['Tujuan'].map(lambda n: coords.get(n,(None,None))[1])

    return coords

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SIGAB Cilacap")
    st.markdown("*Sistem Informasi Geospasial Antisipasi Banjir*")
    st.divider()

    st.markdown("### 📡 Data Cuaca BMKG")
    rh_input = st.slider("Kelembaban RH (%)", 50, 100, 85, 1,
                          help=f"Rata-rata historis: {MEAN_RH}%")
    rr_input = st.slider("Curah Hujan RR (mm)", 0, int(MAX_RR), 20, 1,
                          help=f"Rata-rata historis: {MEAN_RR} mm")

    level_now, weight_now = get_risk_weight(rh_input, rr_input)
    st.markdown(f"**Status:** {risk_badge(level_now)}")
    st.divider()

    st.markdown("### 🗺️ Pengaturan Peta")
    show_heatmap  = st.toggle("Flood Risk Heatmap",     value=True)
    show_flood_pt = st.toggle("Flood Points",           value=True)
    show_shelters = st.toggle("Evacuation Shelters",    value=True)

    st.markdown("**Tampilkan rute:**")
    show_fastest  = st.checkbox("⚡ Fastest",  value=True)
    show_safest   = st.checkbox("🛡️ Safest",   value=True)
    show_balanced = st.checkbox("⚖️ Balanced", value=True)

    st.divider()
    st.markdown("### 🔑 Google Earth Engine")
    gee_token = st.text_input(
        "Map Token (opsional)",
        type="password",
        placeholder="Paste EE map token…",
        help="Dari: ee.Image.getMapId()['token']\nKosongkan untuk pakai heatmap simulasi",
    )

    st.divider()
    selected_classes = [c for c, show in zip(
        CLASS_NAMES, [show_fastest, show_safest, show_balanced]) if show]

    max_dist = st.slider("Jarak maks (km)", 1, 50, 20)
    run_btn  = st.button("🚀 Analisis & Update Peta",
                          use_container_width=True, type="primary")

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("# 🌊 SIGAB Cilacap — Jalur Evakuasi Banjir")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Akurasi Model",   f"{model_info.get('accuracy',0)*100:.2f}%")
c2.metric("Cohen's Kappa",   f"{model_info.get('kappa',0):.4f}")
c3.metric("Total Rute",      f"{len(df_routes):,}")
c4.metric("Status Risiko",   risk_badge(level_now))
st.divider()

# ════════════════════════════════════════════════════════════════
# MAIN — RUN PREDICTION
# ════════════════════════════════════════════════════════════════
df_pred = predict_routes(df_routes.copy(), rh_input, rr_input)
shelter_coords = ensure_shelter_coords(df_pred)

# Filter jarak
df_filtered = df_pred[df_pred['Distance_km'] <= max_dist].copy()

# ════════════════════════════════════════════════════════════════
# LAYOUT: PETA (kiri lebar) + REKOMENDASI (kanan)
# ════════════════════════════════════════════════════════════════
col_map, col_rec = st.columns([3, 1], gap="medium")

# ────────────────────────────────────────────────────────────────
# KOLOM KIRI — Peta interaktif
# ────────────────────────────────────────────────────────────────
with col_map:
    st.markdown('<div class="section-header">🗺️ Peta Interaktif Jalur Evakuasi</div>',
                unsafe_allow_html=True)

    with st.spinner("Membangun peta…"):
        m = build_map(
            df_pred        = df_filtered,
            shelter_coords = shelter_coords,
            gee_token      = gee_token,
            show_heatmap   = show_heatmap,
            show_flood_pts = show_flood_pt,
            show_shelters  = show_shelters,
            show_routes    = True,
            selected_class = selected_classes,
            rh             = rh_input,
            rr             = rr_input,
            risk_level     = level_now,
        )

    map_data = st_folium(
        m,
        width         = "100%",
        height        = 560,
        returned_objects=["last_object_clicked"],
        key           = f"map_{rh_input}_{rr_input}_{show_heatmap}",
    )

    # ── Popup info klik peta ─────────────────────────────────────
    clicked = map_data.get("last_object_clicked")
    if clicked:
        lat_c = clicked.get("lat"); lon_c = clicked.get("lng")
        if lat_c and lon_c:
            # Cari rute terdekat dari titik klik
            if 'lat_asal' in df_filtered.columns:
                df_temp = df_filtered.dropna(subset=['lat_asal','lon_asal'])
                if not df_temp.empty:
                    df_temp = df_temp.copy()
                    df_temp['_dist_click'] = df_temp.apply(
                        lambda r: ((r['lat_asal']-lat_c)**2+(r['lon_asal']-lon_c)**2)**0.5,
                        axis=1
                    )
                    nearest = df_temp.nsmallest(1,'_dist_click').iloc[0]
                    label   = nearest['Label_Name']
                    color   = CLASS_COLORS_HEX.get(label,'#888')
                    st.markdown(f"""
                    <div style="background:{color}18;border-left:4px solid {color};
                                border-radius:8px;padding:10px 14px;margin-top:8px">
                      <b>{CLASS_EMOJI.get(label,'')} {nearest.get('Asal','?')}
                         → {nearest.get('Tujuan','?')}</b><br>
                      <small>{label} | {nearest.get('Distance_km','?')} km |
                      {nearest.get('Time_min','?')} mnt |
                      Confidence: {nearest.get('Confidence','?')}%</small>
                    </div>
                    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# KOLOM KANAN — Rekomendasi (mirip popup screenshot)
# ────────────────────────────────────────────────────────────────
with col_rec:
    st.markdown('<div class="section-header">📋 Rekomendasi</div>',
                unsafe_allow_html=True)

    # Kondisi
    st.markdown(f"**RH:** {rh_input}% | **RR:** {rr_input} mm")
    st.markdown(f"**Risiko:** {risk_badge(level_now)}")

    if level_now >= 4:
        st.error("⛔ Kondisi Ekstrem!\nGunakan rute Safest.")
    elif level_now == 3:
        st.warning("⚠️ Siaga — prioritas Safest")
    else:
        st.success("✅ Kondisi Aman")

    st.markdown("---")
    st.markdown("**RECOMMENDATIONS:**")

    # Top-1 per kelas
    for cls in CLASS_NAMES:
        if cls not in selected_classes:
            continue
        df_cls = df_filtered[df_filtered['Label_Name'] == cls] \
                    .sort_values('Confidence', ascending=False)
        if df_cls.empty:
            continue
        top = df_cls.iloc[0]
        color = CLASS_COLORS_HEX[cls]
        st.markdown(f"""
        <div class="rec-card rec-{cls.lower()}">
          <span style="font-weight:700;color:{color}">
            {CLASS_EMOJI[cls]} {cls}:</span>
          {top.get('Tujuan', '—')}<br>
          <small style="color:#666">
            {top.get('Distance_km','?')} km |
            {top.get('Time_min','?')} mnt
          </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**OTHER OPTIONS:**")

    # Top-5 lainnya
    df_others = df_filtered[df_filtered['Label_Name'].isin(selected_classes)] \
                    .sort_values('Confidence', ascending=False) \
                    .head(8)
    for _, r in df_others.iterrows():
        color = CLASS_COLORS_HEX.get(r['Label_Name'],'#888')
        st.markdown(f"""
        <div style="padding:7px 10px;border-bottom:1px solid #f0f0f0;font-size:13px">
          <b>{r.get('Tujuan','?')}</b><br>
          <small style="color:#888">
            {r.get('Distance_km','?')} km |
            {r.get('Time_min','?')} mnt
          </small>
          <span style="float:right;background:{color}22;color:{color};
                       padding:1px 7px;border-radius:10px;font-size:11px;
                       font-weight:600">{r['Label_Name']}</span>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# BAWAH — Tab evaluasi model & info
# ════════════════════════════════════════════════════════════════
st.divider()
tab1, tab2 = st.tabs(["📊 Distribusi Klasifikasi", "📈 Evaluasi Model"])

with tab1:
    dist = df_pred['Label_Name'].value_counts()
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4,3.5))
        colors_ = [CLASS_COLORS_HEX[n] for n in dist.index if n in CLASS_COLORS_HEX]
        ax.pie(dist.values, labels=dist.index, colors=colors_,
               autopct='%1.1f%%', startangle=90,
               wedgeprops=dict(edgecolor='white', linewidth=2))
        ax.set_title("Proporsi Kelas Rute", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    with c2:
        fig2, ax2 = plt.subplots(figsize=(4,3.5))
        colors2 = [CLASS_COLORS_HEX.get(n,'#888') for n in dist.index]
        bars = ax2.barh(dist.index, dist.values, color=colors2, height=0.5)
        for bar, val in zip(bars, dist.values):
            ax2.text(val + dist.max()*.02, bar.get_y()+bar.get_height()/2,
                     f'{val:,}', va='center', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Jumlah Rute'); ax2.grid(True, axis='x', linestyle='--', alpha=.4)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.set_title("Jumlah per Kelas", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

with tab2:
    st.metric("Accuracy",    f"{model_info.get('accuracy',0)*100:.2f}%")
    st.metric("Cohen's Kappa", f"{model_info.get('kappa',0):.4f}")
    eval_imgs = {
        "Confusion Matrix": "01_confusion_matrix.png",
        "ROC Curve":        "05_roc_curve.png",
        "Feature Importance":"04_feature_importance.png",
        "Learning Curve":   "03_learning_curve.png",
    }
    for title, fname in eval_imgs.items():
        if os.path.exists(fname):
            st.markdown(f"**{title}**"); st.image(fname, use_column_width=True)

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>SIGAB Cilacap | Random Forest + Folium + Google Earth Engine | "
    "Data BMKG Stasiun Cilacap</small></center>",
    unsafe_allow_html=True
)
