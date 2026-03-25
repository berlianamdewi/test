"""
map_component.py — Komponen peta interaktif untuk SIGAB Cilacap
Gabungkan file ini ke dalam app.py utama
"""

import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests, time, json
from functools import lru_cache

# ── Konstanta wilayah Cilacap ────────────────────────────────────
CILACAP_CENTER = [-7.7256, 109.0135]
CILACAP_BOUNDS = [[-7.95, 108.70], [-7.45, 109.35]]

CLASS_COLORS_HEX = {
    'Fastest' : '#f39c12',
    'Safest'  : '#27ae60',
    'Balanced': '#2980b9',
}
CLASS_EMOJI = {'Fastest': '⚡', 'Safest': '🛡️', 'Balanced': '⚖️'}

RISK_COLOR = {1: '#2ecc71', 2: '#f1c40f', 3: '#e67e22', 4: '#e74c3c', 5: '#8e44ad'}
RISK_LABEL = {1: 'Low', 2: 'Medium-Low', 3: 'Medium', 4: 'High', 5: 'Extreme'}

# ── GEE Tile URL builder ─────────────────────────────────────────
def get_gee_flood_tile_url(token: str) -> str:
    """
    Kembalikan URL tile WMS dari Google Earth Engine.
    token = EE map token dari ee.Image.getMapId()['token']
    Jika token kosong, gunakan fallback OpenStreetMap.
    """
    if token:
        return f"https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/maps/{token}/tiles/{{z}}/{{x}}/{{y}}"
    return ""


# ── Geocode shelter names → koordinat ───────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_shelter(name: str, region: str = "Cilacap, Jawa Tengah") -> tuple | None:
    """
    Geocode nama shelter via Nominatim (OSM).
    Return (lat, lon) atau None jika gagal.
    Rate-limit: 1 req/detik sesuai kebijakan Nominatim.
    """
    query = f"{name}, {region}, Indonesia"
    url   = "https://nominatim.openstreetmap.org/search"
    try:
        r = requests.get(url, params={
            "q": query, "format": "json", "limit": 1,
            "countrycodes": "id",
            "viewbox": "108.70,-7.45,109.35,-7.95",
            "bounded": 1,
        }, headers={"User-Agent": "SIGAB-Cilacap-Skripsi/1.0"}, timeout=5)
        data = r.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None


def geocode_all_shelters(shelter_names: list[str]) -> dict:
    """Geocode semua shelter dengan progress bar."""
    coords = {}
    bar    = st.progress(0, text="Geocoding shelter locations...")
    for i, name in enumerate(shelter_names):
        coords[name] = geocode_shelter(name)
        bar.progress((i + 1) / len(shelter_names),
                     text=f"Geocoding: {name}")
        time.sleep(1.1)   # Nominatim rate limit
    bar.empty()
    return coords


# ── Icon helpers ─────────────────────────────────────────────────
def shelter_icon(near_flood: bool) -> folium.Icon:
    return folium.Icon(
        icon="home", prefix="fa",
        color="orange" if near_flood else "green",
        icon_color="white",
    )

def flood_point_icon() -> folium.Icon:
    return folium.Icon(icon="tint", prefix="fa",
                       color="red", icon_color="white")


def make_route_popup(row: pd.Series) -> folium.Popup:
    label = row.get('Label_Name', 'Unknown')
    color = CLASS_COLORS_HEX.get(label, '#888')
    emoji = CLASS_EMOJI.get(label, '')
    html  = f"""
    <div style="font-family:sans-serif;min-width:200px">
      <b style="font-size:14px">{row.get('Asal','?')}</b>
      <div style="font-size:11px;color:#666;margin-bottom:6px">
        → {row.get('Tujuan','?')}
      </div>
      <span style="background:{color};color:white;padding:3px 10px;
                   border-radius:12px;font-size:12px;font-weight:600">
        {emoji} {label}
      </span>
      <table style="margin-top:8px;font-size:12px;width:100%">
        <tr><td><b>Jarak</b></td><td>{row.get('Distance_km','?')} km</td></tr>
        <tr><td><b>Waktu</b></td><td>{row.get('Time_min','?')} mnt</td></tr>
        <tr><td><b>Confidence</b></td>
            <td>{row.get('Confidence', '?')}%</td></tr>
        <tr><td><b>Risk Weight</b></td>
            <td>{row.get('RiskWeight', '?')}</td></tr>
      </table>
    </div>
    """
    return folium.Popup(html, max_width=280)


# ── Fungsi utama: build peta ─────────────────────────────────────
def build_map(
    df_pred       : pd.DataFrame,
    shelter_coords: dict,
    gee_token     : str  = "",
    show_heatmap  : bool = True,
    show_flood_pts: bool = True,
    show_shelters : bool = True,
    show_routes   : bool = True,
    selected_class: list = None,
    rh            : float = 85.0,
    rr            : float = 20.0,
    risk_level    : int   = 2,
) -> folium.Map:
    """
    Bangun peta Folium lengkap mirip screenshot:
    - Satellite basemap (Esri)
    - GEE flood risk layer (opsional, butuh token)
    - Shelter markers (hijau/oranye)
    - Flood point markers (merah)
    - Route polylines warna per kelas
    - Popup info per rute
    - Legend + Layer control
    """
    if selected_class is None:
        selected_class = ['Fastest', 'Safest', 'Balanced']

    # ── Base map ─────────────────────────────────────────────────
    m = folium.Map(
        location   = CILACAP_CENTER,
        zoom_start = 11,
        tiles      = None,
        max_bounds = True,
    )

    # Satellite basemap (Esri World Imagery — mirip screenshot)
    folium.TileLayer(
        tiles     = "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr      = "Esri World Imagery",
        name      = "🛰️ Satellite (Esri)",
        overlay   = False,
        control   = True,
    ).add_to(m)

    # OSM fallback
    folium.TileLayer(
        tiles   = "OpenStreetMap",
        name    = "🗺️ OpenStreetMap",
        overlay = False,
        control = True,
    ).add_to(m)

    # ── GEE Flood Risk layer ─────────────────────────────────────
    if gee_token:
        tile_url = get_gee_flood_tile_url(gee_token)
        folium.TileLayer(
            tiles   = tile_url,
            attr    = "Google Earth Engine — Flood Risk Cilacap",
            name    = "🌊 Flood Risk (GEE Sentinel-2)",
            overlay = True,
            control = True,
            opacity = 0.65,
        ).add_to(m)
    else:
        # Simulasi flood risk via heatmap dari shelter yang near-flood
        # (sebagai pengganti GEE jika token belum ada)
        pass

    # ── Layer groups ─────────────────────────────────────────────
    lg_shelter    = folium.FeatureGroup(name="🏠 Evacuation Shelters", show=show_shelters)
    lg_flood_pts  = folium.FeatureGroup(name="🔴 Flood Points",        show=show_flood_pts)
    lg_fastest    = folium.FeatureGroup(name="⚡ Route: Fastest",       show='Fastest'  in selected_class)
    lg_safest     = folium.FeatureGroup(name="🛡️ Route: Safest",        show='Safest'   in selected_class)
    lg_balanced   = folium.FeatureGroup(name="⚖️ Route: Balanced",      show='Balanced' in selected_class)

    layer_map = {
        'Fastest' : lg_fastest,
        'Safest'  : lg_safest,
        'Balanced': lg_balanced,
    }

    # ── Shelter markers ──────────────────────────────────────────
    if show_shelters:
        for name, coord in shelter_coords.items():
            if coord is None:
                continue
            lat, lon   = coord
            near_flood = _is_near_flood(lat, lon, df_pred)
            folium.Marker(
                location  = [lat, lon],
                tooltip   = name,
                popup     = folium.Popup(
                    f"<b>🏠 {name}</b><br>"
                    f"<small>{'⚠️ Dekat titik banjir' if near_flood else '✅ Lokasi aman'}</small>",
                    max_width=220,
                ),
                icon      = shelter_icon(near_flood),
            ).add_to(lg_shelter)

    # ── Flood point markers (titik asal rute yg near flood) ──────
    if show_flood_pts and 'lat_asal' in df_pred.columns:
        flood_pts = df_pred[df_pred['RiskLevel'] >= 3].drop_duplicates('Asal')
        for _, row in flood_pts.iterrows():
            if pd.notna(row.get('lat_asal')) and pd.notna(row.get('lon_asal')):
                folium.CircleMarker(
                    location     = [row['lat_asal'], row['lon_asal']],
                    radius       = 8,
                    color        = '#c0392b',
                    fill         = True,
                    fill_color   = '#e74c3c',
                    fill_opacity = 0.85,
                    tooltip      = f"⚠️ Titik banjir: {row['Asal']}",
                ).add_to(lg_flood_pts)

    # ── Route polylines ──────────────────────────────────────────
    if show_routes and 'lat_asal' in df_pred.columns:
        needed_cols = ['lat_asal','lon_asal','lat_tujuan','lon_tujuan']
        df_geo = df_pred.dropna(subset=needed_cols)

        for _, row in df_geo.iterrows():
            label = row.get('Label_Name', 'Balanced')
            if label not in selected_class:
                continue

            color  = CLASS_COLORS_HEX.get(label, '#888')
            weight = 5 if label == 'Safest' else 3

            # START marker
            folium.Marker(
                location = [row['lat_asal'], row['lon_asal']],
                tooltip  = f"START: {row.get('Asal','?')}",
                icon     = folium.DivIcon(html=f"""
                    <div style="background:#1a6faf;color:white;font-size:10px;
                                font-weight:700;padding:3px 7px;border-radius:4px;
                                white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.4)">
                      START
                    </div>""",
                    icon_size=(55, 22), icon_anchor=(0, 11)),
            ).add_to(layer_map[label])

            # DEST marker
            folium.Marker(
                location = [row['lat_tujuan'], row['lon_tujuan']],
                tooltip  = f"DEST: {row.get('Tujuan','?')}",
                icon     = folium.DivIcon(html=f"""
                    <div style="background:{color};color:white;font-size:10px;
                                font-weight:700;padding:3px 7px;border-radius:4px;
                                white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.4)">
                      DEST
                    </div>""",
                    icon_size=(55, 22), icon_anchor=(0, 11)),
            ).add_to(layer_map[label])

            # Polyline rute (L-shape atau garis lurus)
            pts = [
                [row['lat_asal'],   row['lon_asal']],
                [row['lat_asal'],   row['lon_tujuan']],   # elbow
                [row['lat_tujuan'], row['lon_tujuan']],
            ]
            folium.PolyLine(
                locations  = pts,
                color      = color,
                weight     = weight,
                opacity    = 0.85,
                dash_array = None if label != 'Safest' else None,
                tooltip    = f"{label}: {row.get('Distance_km','?')} km / {row.get('Time_min','?')} mnt",
                popup      = make_route_popup(row),
            ).add_to(layer_map[label])

    # ── Heatmap risiko banjir (simulasi jika tidak ada GEE) ──────
    if show_heatmap and not gee_token:
        heat_data = _generate_risk_heatmap(df_pred, risk_level)
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name      = "🌡️ Flood Risk Heatmap",
                min_opacity = 0.3,
                max_zoom  = 14,
                radius    = 18,
                blur      = 15,
                gradient  = {0.2: '#2ecc71', 0.5: '#f1c40f',
                             0.7: '#e67e22', 1.0: '#e74c3c'},
            ).add_to(m)

    # ── Tambahkan semua layer ke map ─────────────────────────────
    for lg in [lg_shelter, lg_flood_pts, lg_fastest, lg_safest, lg_balanced]:
        lg.add_to(m)

    # ── Legend HTML ──────────────────────────────────────────────
    legend_html = f"""
    <div style="position:fixed;bottom:30px;right:10px;z-index:9999;
                background:white;border-radius:8px;padding:12px 16px;
                box-shadow:0 2px 8px rgba(0,0,0,.3);font-family:sans-serif;
                font-size:12px;min-width:200px;max-width:240px">
      <b style="font-size:13px">Legend</b>
      <hr style="margin:6px 0;border-color:#eee">
      <b>Satellite Data (Sentinel-2):</b><br>
      <span style="color:#2196F3">NDWI: Dry → Water</span><br>
      <span style="color:#4CAF50">NDMI: Dry → Moist</span><br>
      Risk: <span style="color:#2ecc71">Low</span>
            <span style="color:#f1c40f"> Medium</span>
            <span style="color:#e74c3c"> High</span><br>
      <hr style="margin:6px 0;border-color:#eee">
      <b>Evacuation:</b><br>
      🔴 Flood Point (view route)<br>
      🟢 Safe Shelter<br>
      🟠 Shelter Near Flood<br>
      <hr style="margin:6px 0;border-color:#eee">
      <b>Route Types:</b><br>
      <span style="color:{CLASS_COLORS_HEX['Fastest']};font-weight:700">⚡ Fastest</span> |
      <span style="color:{CLASS_COLORS_HEX['Safest']};font-weight:700">🛡️ Safest</span> |
      <span style="color:{CLASS_COLORS_HEX['Balanced']};font-weight:700">⚖️ Balanced</span><br>
      <hr style="margin:6px 0;border-color:#eee">
      <small>RH: {rh}% | RR: {rr} mm<br>
      Risk Level: {RISK_LABEL.get(risk_level,'?')}</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ─────────────────────────────────────────────
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    # ── Fit bounds ke wilayah Cilacap ─────────────────────────────
    m.fit_bounds(CILACAP_BOUNDS)

    return m


# ── Helpers internal ─────────────────────────────────────────────
def _is_near_flood(lat: float, lon: float, df: pd.DataFrame,
                   threshold_km: float = 2.0) -> bool:
    """Cek apakah koordinat shelter dekat titik banjir (≤ threshold_km)."""
    if 'lat_asal' not in df.columns:
        return False
    flood_rows = df[df['RiskLevel'] >= 3].dropna(subset=['lat_asal','lon_asal'])
    for _, r in flood_rows.iterrows():
        dist = _haversine(lat, lon, r['lat_asal'], r['lon_asal'])
        if dist <= threshold_km:
            return True
    return False


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Jarak haversine dalam km."""
    R  = 6371
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a  = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def _generate_risk_heatmap(df: pd.DataFrame, risk_level: int) -> list:
    """
    Generate titik heatmap simulasi berdasarkan risk level.
    Dipakai jika GEE token belum tersedia.
    """
    if 'lat_asal' not in df.columns:
        return []
    heat = []
    for _, r in df.dropna(subset=['lat_asal','lon_asal']).iterrows():
        w = r.get('RiskWeight', 1.0) / 5.0
        heat.append([r['lat_asal'], r['lon_asal'], w])
    return heat
