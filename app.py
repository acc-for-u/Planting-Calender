"""
Agricultural Planting Calendar Advisor
Central Kalimantan, Indonesia — BMKG Blending 1991–2020 + ENSO (ONI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import datetime
import warnings
import geopandas as gpd
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Planting Calendar Advisor – Central Kalimantan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CONSTANTS
# =====================================================================
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

CROPS = {
    "Rice": {
        "icon": "🌾",
        "color": "#27ae60",
        "duration": 5,
        "water_need": [180, 200, 200, 180, 150],
        "optimal_start": [10, 11],
        "enso_sens": "very_high",
        "drought_tol": "low",
        "note_elnino": "High failure risk during El Niño. Delay planting to Oct–Nov when rainfall stabilises.",
        "note_lanina": "Favourable conditions. Monitor for waterlogging in Jan–Feb.",
        "note_normal": "Follow standard planting calendar. Oct–Nov are the best start months.",
    },
    "Oil Palm": {
        "icon": "🌴",
        "color": "#2980b9",
        "duration": 12,
        "water_need": [150] * 12,
        "optimal_start": list(range(1, 13)),
        "enso_sens": "high",
        "drought_tol": "medium",
        "note_elnino": "Prepare water reserves during Jun–Sep. Production may drop ~20%.",
        "note_lanina": "Good conditions. Maintain drainage to prevent root saturation.",
        "note_normal": "Stable year-round crop. Any month is suitable for new plantation establishment.",
    },
    "Cassava": {
        "icon": "🌿",
        "color": "#e74c3c",
        "duration": 8,
        "water_need": [80, 100, 100, 90, 80, 70, 60, 50],
        "optimal_start": [10, 11, 12],
        "enso_sens": "low",
        "drought_tol": "high",
        "note_elnino": "Best choice during El Niño — highly drought tolerant. Plant Oct–Dec.",
        "note_lanina": "Good conditions, but ensure good drainage to avoid root rot.",
        "note_normal": "Reliable choice. Oct–Dec are optimal start months.",
    },
}

ENSO_THRESH = {"elnino": 0.5, "lanina": -0.5}
SENS_SCORE  = {"low": 2, "medium": 5, "high": 8, "very_high": 10}
TOL_SCORE   = {"low": 2, "medium": 5, "high": 9}

# ENSO scoring delta — differentiated by magnitude and Ky (FAO-33)
# Strong La Niña bonus is smaller than Mild due to waterlogging/root rot risk
ENSO_SCORE_DELTA = {
    "Strong El Niño": {"low": -5,  "medium": -8,  "high": -12, "very_high": -20},
    "Mild El Niño":   {"low": -3,  "medium": -5,  "high":  -8, "very_high": -10},
    "Neutral":        {"low":  0,  "medium":  0,  "high":   0, "very_high":   0},
    "Mild La Niña":   {"low":  1,  "medium":  2,  "high":   3, "very_high":   5},
    "Strong La Niña": {"low":  0,  "medium":  1,  "high":   1, "very_high":   3},
}

ENSO_FACTORS = {
    1:  {"El Niño": 0.976, "La Niña": 0.907, "Neutral": 1.0},
    2:  {"El Niño": 1.174, "La Niña": 1.065, "Neutral": 1.0},
    3:  {"El Niño": 0.991, "La Niña": 0.991, "Neutral": 1.0},
    4:  {"El Niño": 0.941, "La Niña": 0.909, "Neutral": 1.0},
    5:  {"El Niño": 0.933, "La Niña": 0.840, "Neutral": 1.0},
    6:  {"El Niño": 0.859, "La Niña": 1.075, "Neutral": 1.0},
    7:  {"El Niño": 0.507, "La Niña": 1.310, "Neutral": 1.0},
    8:  {"El Niño": 0.410, "La Niña": 1.482, "Neutral": 1.0},
    9:  {"El Niño": 0.477, "La Niña": 1.331, "Neutral": 1.0},
    10: {"El Niño": 0.594, "La Niña": 1.093, "Neutral": 1.0},
    11: {"El Niño": 0.901, "La Niña": 1.009, "Neutral": 1.0},
    12: {"El Niño": 1.055, "La Niña": 0.930, "Neutral": 1.0},
}

CLIM_FALLBACK = pd.DataFrame({
    "mean": [275, 245, 265, 255, 220, 175, 155, 145, 158, 205, 265, 285],
    "std":  [ 85,  75,  80,  70,  65,  60,  55,  50,  55,  65,  75,  80],
}, index=range(1, 13))

from pathlib import Path
_BASE = Path(__file__).parent

DEFAULT_RAIN_PATH   = _BASE / "data" / "CH_BlendPos_CHIRP_clip.xlsx"
DEFAULT_ONI_PATH    = _BASE / "data" / "oni.ascii.txt"
DEFAULT_SHP_PATH    = _BASE / "shp"  / "gadm41_IDN_2.shp"
DEFAULT_IDN_JSON    = _BASE / "data" / "gadm41_IDN_1.json"
DEFAULT_KEC_PATH    = _BASE / "shp"  / "gadm41_IDN_3.shp"
DEFAULT_GAMBUT_PATH = _BASE / "shp"  / "gambut_kalteng.shp"
DEFAULT_LC_PATH     = _BASE / "shp"  / "landcover_kalteng.shp"
ONI_URL           = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Land cover category colours — KLHK 2021 classification
LC_STYLE = {
    "Sawah":               {"color": "#2980b9", "icon": "🌾"},
    "Perkebunan":          {"color": "#e67e22", "icon": "🌴"},
    "Pertanian LK":        {"color": "#f1c40f", "icon": "🌿"},
    "Hutan LK Primer":     {"color": "#1a6b3c", "icon": "🌲"},
    "Hutan LK Sekunder":   {"color": "#27ae60", "icon": "🌲"},
    "Hutan Gambut Primer": {"color": "#6e3b0f", "icon": "🟤"},
    "Hutan Gambut Sekunder":{"color": "#8b5a2b","icon": "🟤"},
    "Pertanian Gambut":    {"color": "#c0882b", "icon": "🟤"},
    "Hutan Rawa Primer":   {"color": "#16a085", "icon": "💧"},
    "Hutan Rawa Sekunder": {"color": "#1abc9c", "icon": "💧"},
    "Belukar Rawa":        {"color": "#76d7c4", "icon": "💧"},
    "Mangrove":            {"color": "#117a65", "icon": "🌊"},
    "Semak Belukar":       {"color": "#aab7b8", "icon": "🌿"},
    "Savana":              {"color": "#d4ac0d", "icon": "🌾"},
    "Rawa":                {"color": "#5dade2", "icon": "💧"},
    "Tubuh Air":           {"color": "#3498db", "icon": "💧"},
    "Permukiman":          {"color": "#e74c3c", "icon": "🏘️"},
    "Lainnya":             {"color": "#bdc3c7", "icon": "—"},
}

CROP_NOTES = {
    "Rice": {
        "en": {
            "El Niño": "Drought risk is high this season. Best to wait until Oct–Nov when rains return.",
            "La Niña": "Wet season will be stronger than usual — favourable for rice planting.",
            "Neutral": "Rainfall is near average. Oct–Nov give the best start for rice.",
        },
        "id": {
            "El Niño": "Risiko kekeringan tinggi musim ini. Sebaiknya tunggu Okt–Nov saat hujan kembali.",
            "La Niña": "Musim hujan lebih kuat dari biasanya — menguntungkan untuk tanam padi.",
            "Neutral": "Curah hujan mendekati rata-rata. Okt–Nov memberi awal tanam terbaik untuk padi.",
        },
    },
    "Oil Palm": {
        "en": {
            "El Niño": "Dry spell may reduce yield ~20%. Secure water reserves before planting.",
            "La Niña": "Rainfall above average. Maintain field drainage to protect roots.",
            "Neutral": "Stable conditions — oil palm can be established in any month of the year.",
        },
        "id": {
            "El Niño": "Kemarau dapat menurunkan produksi ~20%. Siapkan cadangan air sebelum tanam.",
            "La Niña": "Curah hujan di atas rata-rata. Jaga drainase lahan agar akar tidak busuk.",
            "Neutral": "Kondisi stabil — sawit dapat ditanam kapan saja sepanjang tahun.",
        },
    },
    "Cassava": {
        "en": {
            "El Niño": "Cassava is drought-tolerant — a safe choice even when rainfall is low.",
            "La Niña": "Higher rainfall suits cassava well. Ensure drainage to prevent tuber rot.",
            "Neutral": "Reliable in normal conditions. Oct–Dec give the best start.",
        },
        "id": {
            "El Niño": "Singkong tahan kekeringan — pilihan aman meski curah hujan rendah.",
            "La Niña": "Hujan lebih banyak cocok untuk singkong. Pastikan drainase baik agar umbi tidak busuk.",
            "Neutral": "Andalan di kondisi normal. Okt–Des memberi awal tanam terbaik.",
        },
    },
}

ENSO_OPTIONS = {
    "🟢  Normal":    ("Neutral",       0.0),
    "🔴  El Niño":   ("El Niño",       1.2),
    "🔵  La Niña":   ("La Niña",      -1.2),
    # kept for auto-detection by ONI magnitude
    "🟡  Mild El Niño":   ("Mild El Niño",   1.2),
    "🔴  Strong El Niño": ("Strong El Niño", 1.8),
    "🔵  Mild La Niña":   ("Mild La Niña",  -1.2),
    "🌊  Strong La Niña": ("Strong La Niña", -1.8),
}

ENSO_KEY_MAP = {
    "🟢  Normal":  "🟢  Normal",
    "🔴  El Niño": "🔴  El Niño",
    "🔵  La Niña": "🔵  La Niña",
}

T = {
    "en": {
        "brand": "Planting Advisor", "region": "CENTRAL KALIMANTAN",
        "live_climate": "📡 Live Climate", "simulate": "🔬 Simulate scenario",
        "override": "Override condition:",
        "sim_warning": "⚠️ Showing simulated scenario — not current real condition.",
        "lang_label": "🌐 Language",
        "hero_region": "🌏 Central Kalimantan · Indonesia",
        "hero_title": "Agricultural Planting Calendar", "hero_sub": "Advisor",
        "chip1": "📊 BMKG Blending 1991–2020", "chip2": "🌊 ENSO / ONI Integration",
        "chip3": "🗺️ Spatial Analysis", "chip4": "🌾 Rice · Oil Palm · Cassava",
        "banner_Neutral": "Normal Conditions", "banner_El Niño": "El Niño Active",
        "banner_La Niña": "La Niña Active",
        "enso_opts": ["🟢  Normal", "🔴  El Niño", "🔵  La Niña"],
        "sec_plant": "🗓️ Best Time to Plant — Next 3 Months",
        "sec_rec": "📋 General Recommendations",
        "sec_map": "🗺️ Best Crop per Area — Central Kalimantan",
        "best_start": "Best start month", "harvest": "Harvest",
        "perennial": "Year-round (perennial)", "start_in": "Start:",
        "s_rec": "Recommended", "s_cau": "Caution", "s_not": "Not Advised",
        "Rice": "Rice", "Oil Palm": "Oil Palm", "Cassava": "Cassava",
        "months": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "tip_elnino": ("💡 **Rice scores low in the next 3 months.** Consider switching to "
                       "**{icon} Cassava** — drought tolerant and best suited during El Niño."),
        "tip_lanina": "💡 **La Niña conditions favour Rice.** Best start month: **{month}**.",
        "tip_normal": "💡 **Best crop to plant in the next 3 months:** {icon} **{crop}** — start in **{month}**",
        "rec_header_El Niño": "🔴 **El Niño is active.** Expect drier conditions, especially Jun–Sep.",
        "rec_header_La Niña": "🔵 **La Niña is active.** Expect more rainfall than usual.",
        "rec_header_Neutral": "🟢 **Normal conditions.** Follow the standard planting calendar above.",
        "rec_body_El Niño": ("- **Cassava** is the safest crop choice this season\n"
                             "- Delay **Rice** planting until Oct–Nov when rains return\n"
                             "- Prepare water reserves for **Oil Palm** during the dry season\n"
                             "- Monitor ENSO forecasts from BMKG / NOAA"),
        "rec_body_La Niña": ("- Excellent conditions for **Rice** — wet season will be stronger\n"
                             "- **Oil Palm** production should improve\n"
                             "- Ensure proper drainage to avoid waterlogging\n"
                             "- Watch for increased pest and disease pressure"),
        "rec_body_Neutral": ("- All three commodities can be planted in their recommended months\n"
                             "- **Rice**: target Oct–Nov for best results\n"
                             "- **Oil Palm**: stable year-round, any month is suitable\n"
                             "- **Cassava**: Oct–Dec are optimal start months"),
        "map_caption": ("Each map shows the most suitable commodity per grid point for that month, "
                        "based on historical rainfall and current ENSO condition."),
        "kalteng_label": "📍 Central Kalimantan",
        "kalteng_desc": ("Central Kalimantan (highlighted in red) is the study area for this planting "
                         "calendar. Recommendations are based on gridded rainfall data covering the entire province."),
        "kalteng_area": "<li><b>Area:</b> ±153,564 km² · 14 regencies/municipalities</li>",
        "kalteng_src":  "<li><b>Data source:</b> Indonesian Agency for Meteorology, Climatology and Geophysics (BMKG)</li>",
        "kalteng_per":  "<li><b>Period:</b> 30-year climatology (1991–2020)</li>",
        "peat_spin": "Computing peatland mask (first load only)...",
        "peat_cap": ("⬜ **Grey points = Peatland (gambut)** — {n} of {total} grid points masked. "
                     "Water balance scoring is not applied here. Source: WRI Indonesia Peatland Map (2012). "
                     "Separate hydrological assessment required (BRG, 2019)."),
        "spatial_spin": "Running spatial analysis...",
        "sub_exp": "📋 Suitability by Sub-district",
        "sub_rec": "✅ Recommended — Suitable for planting this month",
        "sub_cau": "⚠️ Caution — Possible water deficit, monitor closely",
        "sub_not": "❌ Not Advised — High risk of crop failure",
        "sub_sel": "Select Regency / City:",
        "sub_col": "Sub-district",
        "lc_exp": "🗺️ Land Cover Context — Central Kalimantan (KLHK 2021)",
        "lc_cap": ("Land cover data: KLHK Peta Tutupan Lahan 2020/2021 revision (PL2020R_2021). "
                   "Showing actual land use across Kalimantan Tengah."),
        "lc_spin": "Loading land cover layer...",
        "lc_leg": "**Colour legend:**", "lc_notes_h": "**Notes:**",
        "lc_notes": ("- 🌾 **Sawah** = Active rice paddy location\n"
                     "- 🌴 **Perkebunan** = Palm oil & other plantations\n"
                     "- 🌿 **Pertanian LK** = Dry-land farming\n"
                     "- 🟤 **Gambut** = Not assessed (peatland)\n"
                     "- Data: KLHK 2021 · Scale 1:250,000"),
        "lc_nodata": "Land cover data not available.",
        "source_noaa": "Source: NOAA",
        "sec_calendar": "📅 Monthly Planting Calendar",
        "cal_note": "Based on 30-year climatology adjusted for current ENSO condition.",
        "col_month": "Month",
        "this_month": "← Now",
        "forecast_on":  "🛰️ Using real-time rainfall forecast from Open-Meteo (updated hourly). Next 3 months reflect actual predicted rainfall.",
        "forecast_off": "📊 Using 30-year historical climatology (Open-Meteo unavailable). Connect to internet for real-time forecast.",
        "forecast_rain": "Forecast rainfall",
        "prob_label": "Success probability",
        "uncertainty_note": ("⚠️ **Climate Uncertainty:** Historical data has natural variability. "
                             "Shaded area shows the range of possible rainfall based on 30-year records. "
                             "Actual conditions may differ, especially during ENSO events."),
        "sec_rainfall": "📈 Monthly Rainfall Outlook & Climate Uncertainty",
        "chart_clim": "30-yr Climatology (mean)",
        "chart_enso": "ENSO-adjusted",
        "chart_upper": "Upper bound (flood risk scenario)",
        "chart_lower": "Lower bound (drought risk scenario)",
        "chart_flood_thresh": "Flood risk threshold (300 mm)",
        "chart_drought_thresh": "Drought risk threshold (100 mm)",
        "warn_flood": ("🚨 **FLOOD RISK WARNING** — Rainfall in this period is projected above **300 mm/month**. "
                       "In peatland-dominated areas of Central Kalimantan, this may trigger compound flooding. "
                       "**Check micro-drainage channels before planting.** Avoid low-lying fields near river margins."),
        "warn_drought": ("🏜️ **DROUGHT RISK WARNING** — Rainfall projected below **100 mm/month**. "
                         "Water deficit may significantly reduce crop yields. "
                         "**Prepare supplemental irrigation** or consider delaying planting."),
        "warn_peat": ("🟤 **PEATLAND AREA NOTICE** — Agricultural activities on peatland carry additional risk of "
                      "subsidence and fire during dry spells. Consult BRG guidelines before proceeding."),
        "farmer_go":   "✅ PLANT NOW",
        "farmer_cau":  "⚠️ BE CAREFUL",
        "farmer_stop": "❌ DELAY PLANTING",
        "prob_high": "Chance of success: **High**",
        "prob_mid":  "Chance of success: **Moderate**",
        "prob_low":  "Chance of success: **Low**",
        "water_ok":   "💧 Water: Sufficient",
        "water_warn": "💧 Water: Monitor closely",
        "water_bad":  "💧 Water: Insufficient",
        "farmer_best_month": "Best planting month",
        "farmer_harvest_est": "Estimated harvest",
        "tech_exp": "📊 Technical Details — For Agricultural Extension Workers",
        "tech_exp_desc": ("The charts and analyses below are intended for agricultural extension workers "
                          "and users with a technical background."),
        "chart_exp": "📈 Rainfall Chart & Climate Uncertainty",
        "map_exp_label": "🗺️ Suitability Maps by Area",
    },
    "id": {
        "brand": "Kalender Tanam", "region": "KALIMANTAN TENGAH",
        "live_climate": "📡 Kondisi Iklim", "simulate": "🔬 Simulasi skenario",
        "override": "Ubah kondisi:",
        "sim_warning": "⚠️ Menampilkan skenario simulasi — bukan kondisi aktual saat ini.",
        "lang_label": "🌐 Bahasa",
        "hero_region": "🌏 Kalimantan Tengah · Indonesia",
        "hero_title": "Kalender Tanam Pertanian", "hero_sub": "Berbasis ENSO",
        "chip1": "📊 BMKG Blending 1991–2020", "chip2": "🌊 Integrasi ENSO / ONI",
        "chip3": "🗺️ Analisis Spasial", "chip4": "🌾 Padi · Sawit · Singkong",
        "banner_Neutral": "Kondisi Normal", "banner_El Niño": "El Niño Aktif",
        "banner_La Niña": "La Niña Aktif",
        "enso_opts": ["🟢  Normal", "🔴  El Niño", "🔵  La Niña"],
        "sec_plant": "🗓️ Waktu Tanam Terbaik — 3 Bulan ke Depan",
        "sec_rec": "📋 Rekomendasi Umum",
        "sec_map": "🗺️ Komoditas Terbaik per Wilayah — Kalimantan Tengah",
        "best_start": "Bulan tanam terbaik", "harvest": "Panen",
        "perennial": "Sepanjang tahun", "start_in": "Mulai:",
        "s_rec": "Direkomendasikan", "s_cau": "Hati-hati", "s_not": "Tidak Disarankan",
        "Rice": "Padi", "Oil Palm": "Sawit", "Cassava": "Singkong",
        "months": ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"],
        "tip_elnino": ("💡 **Skor Padi rendah 3 bulan ke depan.** Pertimbangkan beralih ke "
                       "**{icon} Singkong** — tahan kekeringan dan paling cocok saat El Niño."),
        "tip_lanina": "💡 **Kondisi La Niña menguntungkan Padi.** Bulan tanam terbaik: **{month}**.",
        "tip_normal": "💡 **Tanaman terbaik 3 bulan ke depan:** {icon} **{crop}** — mulai tanam **{month}**",
        "rec_header_El Niño": "🔴 **El Niño aktif.** Perkirakan kondisi lebih kering, terutama Jun–Sep.",
        "rec_header_La Niña": "🔵 **La Niña aktif.** Perkirakan curah hujan lebih tinggi dari biasanya.",
        "rec_header_Neutral": "🟢 **Kondisi normal.** Ikuti kalender tanam standar di atas.",
        "rec_body_El Niño": ("- **Singkong** adalah pilihan tanaman paling aman musim ini\n"
                             "- Tunda penanaman **Padi** hingga Okt–Nov saat hujan kembali\n"
                             "- Siapkan cadangan air untuk **Sawit** selama musim kemarau\n"
                             "- Pantau prakiraan ENSO dari BMKG / NOAA"),
        "rec_body_La Niña": ("- Kondisi sangat baik untuk **Padi** — musim hujan akan lebih kuat\n"
                             "- Produksi **Sawit** diperkirakan meningkat\n"
                             "- Pastikan drainase baik untuk mencegah genangan air\n"
                             "- Waspadai peningkatan hama dan penyakit tanaman"),
        "rec_body_Neutral": ("- Ketiga komoditas dapat ditanam pada bulan yang direkomendasikan\n"
                             "- **Padi**: targetkan Okt–Nov untuk hasil terbaik\n"
                             "- **Sawit**: stabil sepanjang tahun, bulan apa pun cocok\n"
                             "- **Singkong**: Okt–Des adalah bulan tanam optimal"),
        "map_caption": ("Setiap peta menampilkan komoditas paling sesuai per titik grid untuk bulan tersebut, "
                        "berdasarkan curah hujan historis dan kondisi ENSO saat ini."),
        "kalteng_label": "📍 Kalimantan Tengah",
        "kalteng_desc": ("Kalimantan Tengah (ditandai merah) adalah wilayah studi kalender tanam ini. "
                         "Rekomendasi didasarkan pada data curah hujan grid yang mencakup seluruh provinsi."),
        "kalteng_area": "<li><b>Luas:</b> ±153.564 km² · 14 kabupaten/kota</li>",
        "kalteng_src":  "<li><b>Sumber data:</b> Badan Meteorologi, Klimatologi dan Geofisika (BMKG)</li>",
        "kalteng_per":  "<li><b>Periode:</b> Klimatologi 30 tahun (1991–2020)</li>",
        "peat_spin": "Menghitung masker gambut (hanya saat pertama kali)...",
        "peat_cap": ("⬜ **Titik abu-abu = Lahan Gambut** — {n} dari {total} titik grid dimasker. "
                     "Penilaian neraca air tidak diterapkan di sini. Sumber: Peta Gambut WRI Indonesia (2012). "
                     "Diperlukan penilaian hidrologi terpisah (BRG, 2019)."),
        "spatial_spin": "Menjalankan analisis spasial...",
        "sub_exp": "📋 Kesesuaian per Kecamatan",
        "sub_rec": "✅ Direkomendasikan — Cocok untuk ditanam bulan ini",
        "sub_cau": "⚠️ Hati-hati — Kemungkinan defisit air, pantau terus",
        "sub_not": "❌ Tidak Disarankan — Risiko gagal panen tinggi",
        "sub_sel": "Pilih Kabupaten / Kota:",
        "sub_col": "Kecamatan",
        "lc_exp": "🗺️ Tutupan Lahan — Kalimantan Tengah (KLHK 2021)",
        "lc_cap": ("Data tutupan lahan: KLHK Peta Tutupan Lahan revisi 2020/2021 (PL2020R_2021). "
                   "Penggunaan lahan aktual di Kalimantan Tengah."),
        "lc_spin": "Memuat layer tutupan lahan...",
        "lc_leg": "**Keterangan warna:**", "lc_notes_h": "**Catatan:**",
        "lc_notes": ("- 🌾 **Sawah** = Lokasi aktual sawah padi\n"
                     "- 🌴 **Perkebunan** = Kebun sawit & tanaman lain\n"
                     "- 🌿 **Pertanian LK** = Ladang/tegalan\n"
                     "- 🟤 **Gambut** = Tidak dinilai water balance\n"
                     "- Data: KLHK 2021 · Skala 1:250.000"),
        "lc_nodata": "Data tutupan lahan tidak tersedia.",
        "source_noaa": "Sumber: NOAA",
        "sec_calendar": "📅 Kalender Tanam Bulanan",
        "cal_note": "Berdasarkan klimatologi 30 tahun yang disesuaikan dengan kondisi ENSO saat ini.",
        "col_month": "Bulan",
        "this_month": "← Sekarang",
        "forecast_on":  "🛰️ Menggunakan prakiraan curah hujan real-time dari Open-Meteo (diperbarui setiap jam). 3 bulan ke depan mencerminkan curah hujan prediksi aktual.",
        "forecast_off": "📊 Menggunakan klimatologi historis 30 tahun (Open-Meteo tidak tersedia). Hubungkan ke internet untuk prakiraan real-time.",
        "forecast_rain": "Prakiraan curah hujan",
        "prob_label": "Peluang keberhasilan",
        "uncertainty_note": ("⚠️ **Ketidakpastian Iklim:** Data historis memiliki variabilitas alami. "
                             "Area bayangan menunjukkan rentang kemungkinan curah hujan berdasarkan data 30 tahun. "
                             "Kondisi aktual bisa berbeda, terutama saat kejadian ENSO."),
        "sec_rainfall": "📈 Prakiraan Curah Hujan Bulanan & Ketidakpastian Iklim",
        "chart_clim": "Klimatologi 30 tahun (rerata)",
        "chart_enso": "Disesuaikan ENSO",
        "chart_upper": "Batas atas (skenario risiko banjir)",
        "chart_lower": "Batas bawah (skenario risiko kekeringan)",
        "chart_flood_thresh": "Batas risiko banjir (300 mm)",
        "chart_drought_thresh": "Batas risiko kekeringan (100 mm)",
        "warn_flood": ("🚨 **PERINGATAN RISIKO BANJIR** — Curah hujan pada periode ini diperkirakan di atas **300 mm/bulan**. "
                       "Di wilayah gambut Kalimantan Tengah, ini dapat memicu genangan majemuk (compound flooding). "
                       "**Periksa saluran drainase mikro sebelum menanam.** Hindari lahan rendah dekat tepian sungai."),
        "warn_drought": ("🏜️ **PERINGATAN RISIKO KEKERINGAN** — Curah hujan diperkirakan di bawah **100 mm/bulan**. "
                         "Defisit air dapat menurunkan hasil panen secara signifikan. "
                         "**Siapkan irigasi tambahan** atau pertimbangkan menunda waktu tanam."),
        "warn_peat": ("🟤 **PERHATIAN LAHAN GAMBUT** — Aktivitas pertanian di lahan gambut berisiko penurunan muka tanah "
                      "dan kebakaran saat musim kering. Konsultasikan dengan pedoman BRG sebelum memulai."),
        "farmer_go":   "✅ TANAM SEKARANG",
        "farmer_cau":  "⚠️ HATI-HATI",
        "farmer_stop": "❌ TUNDA DULU",
        "prob_high": "Kemungkinan berhasil: **Tinggi**",
        "prob_mid":  "Kemungkinan berhasil: **Sedang**",
        "prob_low":  "Kemungkinan berhasil: **Rendah**",
        "water_ok":   "💧 Air hujan: Cukup",
        "water_warn": "💧 Air hujan: Perlu dipantau",
        "water_bad":  "💧 Air hujan: Kurang",
        "farmer_best_month": "Waktu tanam terbaik",
        "farmer_harvest_est": "Perkiraan panen",
        "tech_exp": "📊 Detail Teknis — untuk Penyuluh Pertanian",
        "tech_exp_desc": ("Grafik dan analisis di bawah ini ditujukan untuk penyuluh pertanian "
                          "dan pengguna dengan latar belakang teknis."),
        "chart_exp": "📈 Grafik Curah Hujan & Ketidakpastian Iklim",
        "map_exp_label": "🗺️ Peta Kesesuaian per Wilayah",
    },
}

def t(key):
    return T[st.session_state.get("lang", "en")][key]

def mname(m):
    return t("months")[m - 1]

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_openmeteo_forecast():
    """Fetch 3-month rainfall forecast from Open-Meteo for Central Kalimantan (-1.5, 113.5)."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": -1.5,
                "longitude": 113.5,
                "monthly": "precipitation_sum",
                "timezone": "Asia/Jakarta",
                "forecast_months": 3,
            },
            timeout=8,
        )
        resp.raise_for_status()
        monthly = resp.json().get("monthly", {})
        times   = monthly.get("time", [])
        precip  = monthly.get("precipitation_sum", [])
        result  = {}
        for t_str, p in zip(times, precip):
            if p is not None:
                result[pd.Timestamp(t_str).month] = float(p)
        return result if result else None
    except Exception:
        return None

# =====================================================================
# HELPERS
# =====================================================================
def classify_enso(v):
    if v >= ENSO_THRESH["elnino"]: return "El Niño"
    if v <= ENSO_THRESH["lanina"]: return "La Niña"
    return "Neutral"

def classify_enso_detailed(v):
    if v >= 1.5:   return "Strong El Niño"
    if v >= 0.5:   return "Mild El Niño"
    if v <= -1.5:  return "Strong La Niña"
    if v <= -0.5:  return "Mild La Niña"
    return "Neutral"

def wrap_month(m):
    return ((m - 1) % 12) + 1

# =====================================================================
# DATA LOADERS
# =====================================================================
@st.cache_data(show_spinner=False, ttl=86400)
def load_oni(local_path):
    season_to_month = {
        "DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
        "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12,
    }
    for source in [ONI_URL, local_path]:
        try:
            raw = pd.read_csv(source, sep=r"\s+")
            raw["month"] = raw["SEAS"].map(season_to_month)
            raw["date"]  = pd.to_datetime(dict(year=raw["YR"], month=raw["month"], day=1))
            return raw.set_index("date")["ANOM"].sort_index()
        except Exception:
            continue
    raise RuntimeError("Cannot load ONI data from URL or local file.")

@st.cache_data(show_spinner=False)
def load_rain_grid(path_rain):
    """Load gridded rainfall; return spatial mean, climatology, grid climatology."""
    df_grid = pd.read_excel(path_rain, index_col=0, engine="openpyxl")
    df_grid.index = pd.to_datetime(df_grid.index)
    df_grid = df_grid.sort_index()
    rain      = df_grid.mean(axis=1)
    clim      = monthly_climatology(rain)
    df_m      = df_grid.copy()
    df_m["month"] = df_m.index.month
    grid_clim = df_m.groupby("month").mean()
    return rain, clim, grid_clim

def try_autoload():
    try:
        rain, clim, grid_clim = load_rain_grid(DEFAULT_RAIN_PATH)
        oni                   = load_oni(DEFAULT_ONI_PATH)
        comp, merged          = enso_composite(rain, oni)
        return rain, oni, clim, comp, merged, grid_clim
    except Exception:
        return None, None, CLIM_FALLBACK, None, None, None

# =====================================================================
# ANALYSIS
# =====================================================================
def monthly_climatology(rain):
    df = rain.to_frame("rain")
    df["month"] = df.index.month
    return df.groupby("month")["rain"].agg(["mean", "std"])

def enso_composite(rain, oni):
    df = pd.DataFrame({"rain": rain}).join(oni.rename("ONI"), how="inner")
    df["month"] = df.index.month
    df["ENSO"]  = df["ONI"].apply(classify_enso)
    comp = df.groupby(["month","ENSO"])["rain"].mean().unstack().reindex(index=range(1, 13))
    for phase in ["El Niño", "Neutral", "La Niña"]:
        if phase not in comp.columns:
            comp[phase] = np.nan
    return comp, df

def water_balance(clim, crop_name, start_month, oni_val):
    crop = CROPS[crop_name]
    enso = classify_enso(oni_val)
    rows = []
    for i, need in enumerate(crop["water_need"]):
        m         = wrap_month(start_month + i)
        base_rain = clim.loc[m, "mean"]
        adj_rain  = base_rain * ENSO_FACTORS[m][enso]
        rows.append({
            "month":         m,
            "month_name":    MONTH_NAMES[m - 1],
            "growth_stage":  i + 1,
            "rainfall_base": round(base_rain, 1),
            "rainfall_adj":  round(adj_rain, 1),
            "water_need":    need,
            "balance":       round(adj_rain - need, 1),
        })
    return pd.DataFrame(rows)

def planting_score(clim, crop_name, start_month, oni_val):
    crop = CROPS[crop_name]
    wb   = water_balance(clim, crop_name, start_month, oni_val)
    enso = classify_enso(oni_val)

    n_deficit     = (wb["balance"] < 0).sum()
    total_deficit = wb[wb["balance"] < 0]["balance"].sum()

    # Base score = midpoint of each suitability class — Sys et al. (1993)
    # S1(>75): midpoint≈85 | S2(50-75): midpoint≈63 | S3(25-50): midpoint≈38 | N(<25): midpoint≈18
    if n_deficit == 0:   score = 85   # S1 Highly Suitable
    elif n_deficit == 1: score = 63   # S2 Moderately Suitable
    elif n_deficit == 2: score = 38   # S3 Marginally Suitable
    elif n_deficit == 3: score = 18   # N Not Suitable
    else:                score = 10   # N Not Suitable (severe)

    # Water stress depth penalty — Doorenbos & Kassam (1979) FAO-33
    if total_deficit < -200:   score -= 15
    elif total_deficit < -100: score -= 8

    # ENSO penalty/bonus — differentiated by magnitude, proportional to Ky (FAO-33)
    enso_detailed = classify_enso_detailed(oni_val)
    score += ENSO_SCORE_DELTA[enso_detailed][crop["enso_sens"]]

    score = max(0, min(100, score))

    # Suitability thresholds — Sys et al. (1993)
    if score >= 75:   status, color = "✅ Recommended", "#2ecc71"
    elif score >= 50: status, color = "⚠️ Caution",     "#f39c12"
    else:             status, color = "❌ Not Advised",  "#e74c3c"

    return {
        "score": score, "status": status, "color": color,
        "n_deficit": int(n_deficit), "total_deficit": float(total_deficit),
        "enso": enso_detailed, "wb": wb,
    }

# =====================================================================
# MAP
# =====================================================================
@st.cache_data(show_spinner=False)
def make_indonesia_context_map():
    gdf = gpd.read_file(DEFAULT_IDN_JSON)

    other_lats, other_lons = [], []
    kalteng_lats, kalteng_lons = [], []

    for _, row in gdf.iterrows():
        is_kalteng = row["NAME_1"] == "KalimantanTengah"
        geom  = row.geometry
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            coords = list(poly.exterior.coords)
            lats   = [c[1] for c in coords] + [None]
            lons   = [c[0] for c in coords] + [None]
            if is_kalteng:
                kalteng_lats.extend(lats)
                kalteng_lons.extend(lons)
            else:
                other_lats.extend(lats)
                other_lons.extend(lons)

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=other_lats, lon=other_lons,
        mode="lines", fill="toself",
        fillcolor="rgba(189,195,199,0.5)",
        line=dict(color="rgba(120,120,120,0.6)", width=0.5),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattermapbox(
        lat=kalteng_lats, lon=kalteng_lons,
        mode="lines", fill="toself",
        fillcolor="rgba(231,76,60,0.75)",
        line=dict(color="#c0392b", width=1.5),
        hovertemplate="Central Kalimantan<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=-2.5, lon=118.0), zoom=3.2),
        height=260,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig

@st.cache_data(show_spinner=False)
def load_kecamatan_mapping(grid_cols):
    """Spatial join: map each CHIRPS grid column to kabupaten + kecamatan."""
    from shapely.geometry import Point
    gdf_kec  = gpd.read_file(DEFAULT_KEC_PATH)
    kalteng  = gdf_kec[gdf_kec["NAME_1"] == "Kalimantan Tengah"][
        ["NAME_2", "NAME_3", "geometry"]].copy()
    lats = [float(c.split("_")[0]) for c in grid_cols]
    lons = [float(c.split("_")[1]) for c in grid_cols]
    gdf_pts  = gpd.GeoDataFrame(
        {"col": list(grid_cols)},
        geometry=[Point(lon, lat) for lat, lon in zip(lats, lons)],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf_pts, kalteng, how="left", predicate="within")
    import pandas as _pd
    mapping = {}
    for _, row in joined.iterrows():
        kab = row.get("NAME_2")
        kec = row.get("NAME_3")
        if _pd.isna(kab) or _pd.isna(kec):
            kab, kec = None, None
        mapping[row["col"]] = (kab, kec)
    return mapping

def kecamatan_summary(grid_clim, crop_name, start_month, oni_val, mapping):
    """Mean suitability score per kecamatan for one crop."""
    _, _, scores = score_grid_points_vec(grid_clim, crop_name, start_month, oni_val)
    cols = grid_clim.columns.tolist()
    agg = {}
    for j, col in enumerate(cols):
        kab, kec = mapping.get(col, (None, None))
        if kab is None or kec is None:
            continue
        agg.setdefault(kab, {}).setdefault(kec, []).append(scores[j])
    rows = []
    for kab in sorted(agg.keys()):
        for kec in sorted(agg[kab].keys()):
            mean_sc = float(np.mean(agg[kab][kec]))
            rows.append({"Kabupaten": kab, "Kecamatan": kec, "score": mean_sc})
    return rows

@st.cache_data(show_spinner=False)
def load_kabupaten_boundaries(shp_path):
    """Load Kalteng kabupaten boundary coordinates for map overlay."""
    gdf     = gpd.read_file(shp_path)
    kalteng = gdf[gdf["NAME_1"] == "Kalimantan Tengah"].copy()

    all_lats, all_lons, all_names = [], [], []
    for _, row in kalteng.iterrows():
        geom = row.geometry
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            coords = list(poly.exterior.coords)
            all_lats.extend([c[1] for c in coords] + [None])
            all_lons.extend([c[0] for c in coords] + [None])
            all_names.append(row["NAME_2"])
    return all_lats, all_lons, all_names

@st.cache_data(show_spinner=False)
def load_peatland_overlay(shp_path):
    """Load peatland (gambut) polygon coordinates for Plotly overlay.
    Source: WRI Indonesia peatland shapefile (2012), clipped to Kalteng.
    """
    try:
        gdf = gpd.read_file(shp_path)
        lats, lons = [], []
        for _, row in gdf.iterrows():
            geom  = row.geometry
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                coords = list(poly.exterior.coords)
                lats.extend([c[1] for c in coords] + [None])
                lons.extend([c[0] for c in coords] + [None])
        return lats, lons
    except Exception:
        return [], []

@st.cache_data(show_spinner=False)
def load_landcover(shp_path):
    """Load KLHK 2021 land cover polygons for Kalteng, return per-category coords."""
    try:
        gdf = gpd.read_file(shp_path)
        traces = {}
        for cat, style in LC_STYLE.items():
            subset = gdf[gdf["category"] == cat]
            if subset.empty:
                continue
            lats, lons = [], []
            for _, row in subset.iterrows():
                geom  = row.geometry
                polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
                for poly in polys:
                    if poly.is_empty:
                        continue
                    coords = list(poly.exterior.coords)
                    lats.extend([c[1] for c in coords] + [None])
                    lons.extend([c[0] for c in coords] + [None])
            if lats:
                luas = subset["luas_ha"].sum() / 1e3  # ribu ha
                traces[cat] = {"lats": lats, "lons": lons,
                               "color": style["color"],
                               "icon":  style["icon"],
                               "luas_rb_ha": round(luas, 1)}
        return traces
    except Exception as e:
        return {}


def make_landcover_map(lc_traces, b_lats, b_lons):
    """Plotly map showing KLHK 2021 land cover categories."""
    fig = go.Figure()
    for cat, d in lc_traces.items():
        icon = LC_STYLE.get(cat, {}).get("icon", "")
        luas = d["luas_rb_ha"]
        fig.add_trace(go.Scattermapbox(
            lat=d["lats"], lon=d["lons"],
            mode="lines", fill="toself",
            fillcolor=d["color"] + "55",      # 33% opacity
            line=dict(color=d["color"], width=0.6),
            name=f"{icon} {cat} ({luas:,.0f} rb ha)",
            hovertemplate=f"{icon} {cat}<extra></extra>",
        ))
    if b_lats:
        fig.add_trace(go.Scattermapbox(
            lat=b_lats, lon=b_lons, mode="lines",
            line=dict(color="#1a1a1a", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=-1.5, lon=113.5), zoom=4.5),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=10)),
    )
    return fig


@st.cache_data(show_spinner=False)
def compute_peatland_mask(grid_cols, gambut_shp_path):
    """
    Spatial join: returns boolean array (True = grid point inside peatland).
    Uses geopandas STRtree spatial index for fast point-in-polygon.
    Cached after first run — ~10-30 s depending on hardware.
    """
    try:
        from shapely.geometry import Point
        gdf_gambut = gpd.read_file(gambut_shp_path)
        lats = [float(c.split("_")[0]) for c in grid_cols]
        lons = [float(c.split("_")[1]) for c in grid_cols]
        gdf_pts = gpd.GeoDataFrame(
            {"col": list(grid_cols)},
            geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(gdf_pts, gdf_gambut[["geometry"]],
                           how="left", predicate="within")
        # Build mask on original point indices (handle possible duplicate rows)
        peat_idx = set(joined[joined["index_right"].notna()].index.tolist())
        mask = np.array([i in peat_idx for i in range(len(grid_cols))])
        n_peat = int(mask.sum())
        return mask, n_peat
    except Exception:
        return None, 0

def score_grid_points_vec(grid_clim, crop_name, start_month, oni_val):
    """Vectorized suitability score for all grid points."""
    crop  = CROPS[crop_name]
    enso  = classify_enso(oni_val)
    n     = len(grid_clim.columns)
    def_c = np.zeros(n)
    def_t = np.zeros(n)

    for i, need in enumerate(crop["water_need"]):
        m        = wrap_month(start_month + i)
        adj_rain = grid_clim.loc[m].values * ENSO_FACTORS[m][enso]
        balance  = adj_rain - need
        def_c   += (balance < 0).astype(float)
        def_t   += np.where(balance < 0, balance, 0)

    scores = np.where(def_c == 0, 85,
             np.where(def_c == 1, 63,
             np.where(def_c == 2, 38,
             np.where(def_c == 3, 18, 10))))

    scores = np.where(def_t < -200, scores - 15,
             np.where(def_t < -100, scores - 8, scores))

    if start_month in crop["optimal_start"]:
        scores += 5

    enso_detailed = classify_enso_detailed(oni_val)
    scores += ENSO_SCORE_DELTA[enso_detailed][crop["enso_sens"]]

    scores = np.clip(scores, 0, 100)

    cols = grid_clim.columns.tolist()
    lats = [float(c.split("_")[0]) for c in cols]
    lons = [float(c.split("_")[1]) for c in cols]

    return lats, lons, scores

def make_crop_map(grid_clim, crop_name, start_month, oni_val, b_lats, b_lons,
                  g_lats=None, g_lons=None, peat_mask=None,
                  center_lat=-1.5, center_lon=113.5, zoom=4.5):
    """Map showing suitability score for one crop across all grid points.
    peat_mask: boolean array (True = peatland) — those points shown as 'Not Assessed'.
    """
    crop = CROPS[crop_name]
    lats, lons, scores = score_grid_points_vec(grid_clim, crop_name, start_month, oni_val)

    # Apply peatland mask — exclude peat points from scored tiers
    non_peat = ~peat_mask if peat_mask is not None else np.ones(len(scores), dtype=bool)

    # Bucket into 3 tiers (non-peatland only)
    tiers = [
        ("✅ Recommended", "#27ae60", (scores >= 75) & non_peat),
        ("⚠️ Caution",    "#f39c12", (scores >= 50) & (scores < 75) & non_peat),
        ("❌ Not Advised", "#e74c3c", (scores < 50) & non_peat),
    ]

    fig = go.Figure()
    for label, color, mask in tiers:
        mask = mask.astype(bool)
        if not mask.any():
            continue
        fig.add_trace(go.Scattermapbox(
            lat=[lats[j] for j in range(len(lats)) if mask[j]],
            lon=[lons[j] for j in range(len(lons)) if mask[j]],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.75),
            name=label,
            text=[f"Score: {scores[j]:.0f}" for j in range(len(lats)) if mask[j]],
            hovertemplate="%{text}<extra></extra>",
        ))

    # Peatland grid points — shown as grey "Not Assessed"
    if peat_mask is not None and peat_mask.any():
        pm = peat_mask.astype(bool)
        fig.add_trace(go.Scattermapbox(
            lat=[lats[j] for j in range(len(lats)) if pm[j]],
            lon=[lons[j] for j in range(len(lons)) if pm[j]],
            mode="markers",
            marker=dict(size=6, color="#95a5a6", opacity=0.6),
            name="⬜ Not Assessed (Peatland)",
            hovertemplate="Peatland (gambut)<br>Water balance not applicable here<extra></extra>",
        ))

    # Peatland boundary outline
    if g_lats:
        fig.add_trace(go.Scattermapbox(
            lat=g_lats, lon=g_lons,
            mode="lines",
            line=dict(color="#8b5a2b", width=1.5),
            name="🟤 Peatland boundary",
            hoverinfo="skip", showlegend=False,
        ))

    if b_lats:
        fig.add_trace(go.Scattermapbox(
            lat=b_lats, lon=b_lons,
            mode="lines",
            line=dict(color="#1a1a1a", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        height=370,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01, y=0.06,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            font=dict(size=10),
        ),
        showlegend=True,
    )
    return fig

# =====================================================================
# RAINFALL UNCERTAINTY CHART
# =====================================================================
def make_rainfall_envelope_chart(clim, enso_phase):
    months       = list(range(1, 13))
    month_labels = [mname(m) for m in months]
    mean_rain    = [float(clim.loc[m, "mean"]) for m in months]
    std_rain     = [float(clim.loc[m, "std"])  for m in months]
    adj_rain     = [clim.loc[m, "mean"] * ENSO_FACTORS[m][enso_phase] for m in months]
    upper = [mean_rain[i] + 1.5 * std_rain[i] for i in range(12)]
    lower = [max(0, mean_rain[i] - 1.5 * std_rain[i]) for i in range(12)]

    enso_color = {"El Niño": "#e74c3c", "La Niña": "#2980b9", "Neutral": "#27ae60"}[enso_phase]

    fig = go.Figure()

    # Uncertainty envelope (shaded)
    fig.add_trace(go.Scatter(
        x=month_labels + month_labels[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(39,174,96,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=t("chart_upper"),
        hoverinfo="skip",
        showlegend=True,
    ))

    # Climatological mean
    fig.add_trace(go.Scatter(
        x=month_labels, y=mean_rain,
        mode="lines+markers",
        line=dict(color="#27ae60", width=2, dash="dash"),
        marker=dict(size=6, color="#27ae60"),
        name=t("chart_clim"),
    ))

    # ENSO-adjusted line
    fig.add_trace(go.Scatter(
        x=month_labels, y=adj_rain,
        mode="lines+markers",
        line=dict(color=enso_color, width=3),
        marker=dict(size=8, color=enso_color),
        name=f"{t('chart_enso')} ({enso_phase})",
    ))

    # Flood risk threshold
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#eee", borderwidth=1, font=dict(size=11)),
        yaxis_title="Rainfall (mm/month)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f0f0f0", zeroline=False, rangemode="tozero"),
    )
    return fig

def make_kabupaten_rainfall_chart(grid_clim, mapping, kab_name, enso_phase):
    cols = [col for col, (kab, _) in mapping.items() if kab == kab_name]
    if not cols:
        return None
    monthly_mean = grid_clim[cols].mean(axis=1)
    months       = list(range(1, 13))
    month_labels = [mname(m) for m in months]
    means        = [float(monthly_mean.loc[m]) for m in months]
    adj          = [means[m - 1] * ENSO_FACTORS[m][enso_phase] for m in months]
    enso_color   = {"El Niño": "#e74c3c", "La Niña": "#2980b9", "Neutral": "#27ae60"}[enso_phase]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_labels, y=means,
        name=t("chart_clim"),
        marker_color="#27ae60", opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=month_labels, y=adj,
        mode="lines+markers",
        line=dict(color=enso_color, width=2.5),
        marker=dict(size=7, color=enso_color),
        name=f"{t('chart_enso')} ({enso_phase})",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#eee", borderwidth=1, font=dict(size=11)),
        yaxis_title="Rainfall (mm/month)",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f0f0f0", zeroline=False, rangemode="tozero"),
    )
    return fig


def kabupaten_map_bounds(mapping, kab_name):
    """Return (center_lat, center_lon, zoom) for a kabupaten's grid points."""
    pts = [(float(c.split("_")[0]), float(c.split("_")[1]))
           for c, (kab, _) in mapping.items() if kab == kab_name]
    if not pts:
        return -1.5, 113.5, 4.5
    lats, lons = zip(*pts)
    clat = (min(lats) + max(lats)) / 2
    clon = (min(lons) + max(lons)) / 2
    extent = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = 8.0 if extent < 1 else 7.0 if extent < 2 else 6.5 if extent < 3 else 5.5
    return clat, clon, zoom


def get_kabupaten_clim(grid_clim, mapping, kab_name):
    """Monthly climatology (mean + std) for a specific kabupaten, from grid_clim."""
    cols = [col for col, (kab, _) in mapping.items() if kab == kab_name]
    if not cols:
        return None
    return pd.DataFrame({
        "mean": grid_clim[cols].mean(axis=1),
        "std":  grid_clim[cols].std(axis=1),
    })


# =====================================================================
# FARMER MODE
# =====================================================================
def page_farmer(clim, enso_phase, oni_val, grid_clim=None):
    current_month = datetime.datetime.now().month
    lang = st.session_state.get("lang", "en")

    # ── Open-Meteo forecast — patch clim for next 3 months ───────
    forecast_rain = fetch_openmeteo_forecast()
    if forecast_rain:
        clim = clim.copy()
        for m, rain in forecast_rain.items():
            clim.loc[m, "mean"] = rain
    clim_kalteng = clim  # save all-Kalteng clim before possible kabupaten override

    # ── Climate banner ────────────────────────────────────────────
    banner_cfg = {
        "Neutral": ("🟢", "#d5f5e3", "#1a6b3c", "#27ae60"),
        "El Niño": ("🔴", "#fff5f5", "#922b21", "#e74c3c"),
        "La Niña": ("🔵", "#f0f7ff", "#1a5276", "#2980b9"),
    }
    icon, bg, fg, border = banner_cfg[enso_phase]
    label = t(f"banner_{enso_phase}")
    st.markdown(
        f"<div style='background:{bg};border-radius:14px;padding:18px 28px;"
        f"margin-bottom:24px;border-left:5px solid {border};"
        f"box-shadow:0 4px 16px rgba(0,0,0,0.06);display:flex;align-items:center;gap:18px'>"
        f"<div style='font-size:2.8rem;line-height:1'>{icon}</div>"
        f"<div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:{fg};margin-bottom:2px'>{label}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.caption("📊 Indonesian Agency for Meteorology, Climatology and Geophysics · NOAA ONI")

    # ── KABUPATEN SELECTOR ────────────────────────────────────────
    selected_kab = None
    mapping_kab  = None
    if grid_clim is not None:
        mapping_kab  = load_kecamatan_mapping(tuple(grid_clim.columns.tolist()))
        kab_list     = sorted({kab for kab, _ in mapping_kab.values() if kab is not None})
        all_lbl      = "🌏 All of Central Kalimantan" if lang == "en" else "🌏 Seluruh Kalimantan Tengah"
        sel_label    = "📍 Select your area:" if lang == "en" else "📍 Pilih wilayah Anda:"
        selected_kab = st.selectbox(sel_label, [all_lbl] + kab_list, key="kab_main_sel")

        if selected_kab != all_lbl:
            kab_clim = get_kabupaten_clim(grid_clim, mapping_kab, selected_kab)
            if kab_clim is not None:
                if forecast_rain:
                    for m, rain in forecast_rain.items():
                        kab_clim.loc[m, "mean"] = rain
                clim = kab_clim
        else:
            selected_kab = None  # treat None as "all Kalteng"

    # ── Compute scores ───────────────────────────────────────────
    next_3 = [wrap_month(current_month + i) for i in range(3)]
    next_3_names = [mname(m) for m in next_3]

    upcoming = {}
    for crop_name in CROPS:
        month_scores = {m: planting_score(clim, crop_name, m, oni_val) for m in next_3}
        best_m  = max(month_scores, key=lambda m: month_scores[m]["score"])
        upcoming[crop_name] = {
            "best_month":      best_m,
            "best_month_name": mname(best_m),
            "score":           month_scores[best_m]["score"],
            "status":          month_scores[best_m]["status"],
            "all":             month_scores,
        }

    # ── MAPS + KECAMATAN (shown first) ───────────────────────────
    if grid_clim is not None:
        try:
            b_lats, b_lons, _ = load_kabupaten_boundaries(DEFAULT_SHP_PATH)
        except Exception:
            b_lats, b_lons = [], []
        g_lats, g_lons = load_peatland_overlay(DEFAULT_GAMBUT_PATH)
        with st.spinner(t("peat_spin")):
            peat_mask, _ = compute_peatland_mask(
                tuple(grid_clim.columns.tolist()), DEFAULT_GAMBUT_PATH
            )

        _peat_cap = ("⬜ Grey = Peatland — suitability not assessed."
                     if lang == "en" else
                     "⬜ Abu-abu = Lahan gambut — tidak dinilai.")

        def _render_kec_table(df_filtered):
            st.markdown(
                f"<div style='display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap'>"
                f"<span style='background:#d5f5e3;color:#1a6b3c;padding:4px 10px;"
                f"border-radius:6px;font-size:0.82rem;font-weight:600'>{t('sub_rec')}</span>"
                f"<span style='background:#fef9e7;color:#7d6608;padding:4px 10px;"
                f"border-radius:6px;font-size:0.82rem;font-weight:600'>{t('sub_cau')}</span>"
                f"<span style='background:#fadbd8;color:#922b21;padding:4px 10px;"
                f"border-radius:6px;font-size:0.82rem;font-weight:600'>{t('sub_not')}</span>"
                f"</div>", unsafe_allow_html=True,
            )
            def color_status(v):
                if "✅" in str(v): return "background-color:#d5f5e3;color:#1a6b3c"
                if "⚠️" in str(v): return "background-color:#fef9e7;color:#7d6608"
                if "❌" in str(v): return "background-color:#fadbd8;color:#922b21"
                return ""
            crop_cols = [c for c in df_filtered.columns if c != t("sub_col")]
            st.dataframe(df_filtered.style.map(color_status, subset=crop_cols),
                         use_container_width=True, hide_index=True)

        def _build_kec_pivot():
            all_rows = []
            for cn in CROPS:
                bm = upcoming[cn]["best_month"]
                rows = kecamatan_summary(grid_clim, cn, bm, oni_val, mapping_kab)
                for r in rows:
                    r["Commodity"] = f"{CROPS[cn]['icon']} {t(cn)}"
                    r["Status"]    = (f"✅ {t('s_rec')}" if r["score"] >= 75 else
                                      f"⚠️ {t('s_cau')}" if r["score"] >= 50 else
                                      f"❌ {t('s_not')}")
                all_rows.extend(rows)
            df = pd.DataFrame(all_rows)
            pivot = df.pivot_table(index=["Kabupaten","Kecamatan"],
                                   columns="Commodity", values="Status",
                                   aggfunc="first").reset_index()
            pivot.columns.name = None
            return pivot.rename(columns={"Kecamatan": t("sub_col")})

        if selected_kab:
            st.markdown(
                f"<div class='section-header'>📋 {t('sub_exp')} — {selected_kab}</div>",
                unsafe_allow_html=True,
            )
            with st.spinner(t("spatial_spin")):
                pivot = _build_kec_pivot()
            df_filtered = pivot[pivot["Kabupaten"] == selected_kab].drop(
                columns=["Kabupaten"]).reset_index(drop=True)
            _render_kec_table(df_filtered)

            st.markdown(
                f"<div class='section-header'>{t('sec_map')} — {selected_kab}</div>",
                unsafe_allow_html=True,
            )
            st.caption(t("map_caption"))
            clat, clon, zoom_kab = kabupaten_map_bounds(mapping_kab, selected_kab)
            map_cols = st.columns(3)
            for col, crop_name in zip(map_cols, CROPS.keys()):
                crop   = CROPS[crop_name]
                best_m = upcoming[crop_name]["best_month"]
                with col:
                    st.markdown(
                        f"<div style='text-align:center;font-weight:700;font-size:1.05rem;"
                        f"color:{crop['color']};margin-bottom:4px'>"
                        f"{crop['icon']} {t(crop_name)}</div>"
                        f"<div style='text-align:center;font-size:0.82rem;color:#555;margin-bottom:6px'>"
                        f"{t('start_in')} {mname(best_m)}</div>",
                        unsafe_allow_html=True,
                    )
                    with st.spinner(f"{t(crop_name)}..."):
                        fig = make_crop_map(
                            grid_clim, crop_name, best_m, oni_val,
                            b_lats, b_lons, g_lats, g_lons, peat_mask,
                            center_lat=clat, center_lon=clon, zoom=zoom_kab,
                        )
                    st.plotly_chart(fig, use_container_width=True)
            st.caption(_peat_cap)

        else:
            st.markdown(
                f"<div class='section-header'>{t('sec_map')}</div>",
                unsafe_allow_html=True,
            )
            st.caption(t("map_caption"))
            try:
                st.plotly_chart(make_indonesia_context_map(), use_container_width=True)
            except Exception:
                pass
            map_cols = st.columns(3)
            for col, crop_name in zip(map_cols, CROPS.keys()):
                crop   = CROPS[crop_name]
                best_m = upcoming[crop_name]["best_month"]
                with col:
                    st.markdown(
                        f"<div style='text-align:center;font-weight:700;font-size:1.05rem;"
                        f"color:{crop['color']};margin-bottom:4px'>"
                        f"{crop['icon']} {t(crop_name)}</div>"
                        f"<div style='text-align:center;font-size:0.82rem;color:#555;margin-bottom:6px'>"
                        f"{t('start_in')} {mname(best_m)}</div>",
                        unsafe_allow_html=True,
                    )
                    with st.spinner(f"{t(crop_name)}..."):
                        fig = make_crop_map(
                            grid_clim, crop_name, best_m, oni_val,
                            b_lats, b_lons, g_lats, g_lons, peat_mask,
                        )
                    st.plotly_chart(fig, use_container_width=True)
            st.caption(_peat_cap)

            with st.expander(t("sub_exp")):
                with st.spinner(t("spatial_spin")):
                    pivot = _build_kec_pivot()
                kab_list_exp    = sorted(pivot["Kabupaten"].unique())
                sel_kab_exp     = st.selectbox(t("sub_sel"), kab_list_exp)
                df_filtered_exp = pivot[pivot["Kabupaten"] == sel_kab_exp].drop(
                    columns=["Kabupaten"]).reset_index(drop=True)
                _render_kec_table(df_filtered_exp)

    # ── WHAT TO DO IN THE NEXT 3 MONTHS ──────────────────────────
    kab_label = selected_kab if selected_kab else "Kalimantan Tengah"
    st.markdown(
        f"<div class='section-header'>{t('sec_plant')} "
        f"<span style='color:#27ae60;font-weight:600;font-size:0.9rem'>📍 {kab_label}</span>"
        f"<span style='color:#888;font-weight:400;font-size:0.9rem'> · ({', '.join(next_3_names)})</span></div>",
        unsafe_allow_html=True,
    )

    # ── SIMPLE FARMER BOXES (tampilan utama) ─────────────────────
    cols_now = st.columns(3)
    for col, crop_name in zip(cols_now, CROPS.keys()):
        info = upcoming[crop_name]
        crop = CROPS[crop_name]
        s    = info["score"]

        if crop["duration"] == 12:
            harvest_str = t("perennial")
        else:
            harvest_m   = wrap_month(info["best_month"] + crop["duration"])
            harvest_str = f"~{mname(harvest_m)}"

        # Water label based on actual water balance for this crop
        wb        = water_balance(clim, crop_name, info["best_month"], oni_val)
        n_deficit = int((wb["balance"] < 0).sum())
        water_lbl = t("water_ok") if n_deficit == 0 else (t("water_warn") if n_deficit <= 2 else t("water_bad"))
        prob_lbl  = t("prob_high") if s >= 75 else (t("prob_mid") if s >= 50 else t("prob_low"))

        if s >= 75:
            if crop["duration"] == 12:
                harvest_line = f"\n🗓️ {t('harvest')}: **{t('perennial')}**"
            else:
                harvest_m    = wrap_month(current_month + crop["duration"])
                harvest_line = f"\n🗓️ {t('harvest')}: **{mname(harvest_m)}**"
        else:
            harvest_line = ""

        score_lbl = f"📊 Score: **{s}/100**"
        msg = (f"### {crop['icon']} {t(crop_name)}\n\n"
               f"{prob_lbl}  \n"
               f"{water_lbl}  \n"
               f"{score_lbl}"
               f"{harvest_line}")
        with col:
            if s >= 75:
                st.success(f"**{t('farmer_go')}**\n\n" + msg)
            elif s >= 50:
                st.warning(f"**{t('farmer_cau')}**\n\n" + msg)
            else:
                st.error(f"**{t('farmer_stop')}**\n\n" + msg)

    # ── SUBSTITUTION TIP ─────────────────────────────────────────
    best_crop = max(upcoming, key=lambda c: upcoming[c]["score"])
    if enso_phase == "El Niño" and upcoming["Rice"]["score"] < 50:
        st.info(t("tip_elnino").format(icon=CROPS["Cassava"]["icon"]))
    elif enso_phase == "La Niña" and upcoming["Rice"]["score"] >= 70:
        st.info(t("tip_lanina").format(month=upcoming["Rice"]["best_month_name"]))
    else:
        st.success(t("tip_normal").format(
            icon=CROPS[best_crop]["icon"],
            crop=t(best_crop),
            month=upcoming[best_crop]["best_month_name"],
        ))

    # ── MONTHLY CALENDAR TABLE ────────────────────────────────────
    st.markdown(
        f"<div class='section-header'>{t('sec_calendar')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(t("cal_note"))

    crop_names   = list(CROPS.keys())
    crop_headers = [f"{CROPS[c]['icon']} {t(c)}" for c in crop_names]

    rows_html = ""
    for m in next_3:
        is_now    = (m == current_month)
        month_lbl = mname(m) + (f" <b style='color:#e74c3c'>{t('this_month')}</b>" if is_now else "")
        row_bg    = "background:#fffbe6;" if is_now else ""
        rain_mm = forecast_rain.get(m, clim.loc[m, "mean"]) if forecast_rain else clim.loc[m, "mean"]
        cells = ""
        for crop_name in crop_names:
            s  = planting_score(clim, crop_name, m, oni_val)["score"]
            if s >= 75:
                cell_bg, emoji, label = "#d5f5e3", "✅", t("farmer_go")
            elif s >= 50:
                cell_bg, emoji, label = "#fef9e7", "⚠️", t("farmer_cau")
            else:
                cell_bg, emoji, label = "#fadbd8", "❌", t("farmer_stop")
            cells += (
                f"<td style='background:{cell_bg};text-align:center;padding:8px 6px;"
                f"font-size:0.82rem;font-weight:600;border-radius:6px'>"
                f"{emoji} {label}</td>"
            )
        rain_src = "🛰️" if (forecast_rain and m in forecast_rain) else "📊"
        rows_html += (
            f"<tr style='{row_bg}'>"
            f"<td style='padding:8px 12px;font-weight:{'700' if is_now else '400'};"
            f"font-size:0.85rem;white-space:nowrap'>{month_lbl}"
            f"<br><span style='font-size:0.72rem;color:#999'>{rain_src} {rain_mm:.0f} mm</span></td>"
            f"{cells}</tr>"
        )

    header_cells = "".join(
        f"<th style='text-align:center;padding:8px;font-size:0.85rem;color:#555'>{h}</th>"
        for h in crop_headers
    )
    table_html = (
        f"<table style='width:100%;border-collapse:separate;border-spacing:4px'>"
        f"<thead><tr><th style='padding:8px 12px;text-align:left;color:#555;font-size:0.85rem'>"
        f"{t('col_month')}</th>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # ── RAINFALL CHART ────────────────────────────────────────────
    st.markdown(
        f"<div class='section-header'>{t('sec_rainfall')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(t("cal_note"))

    if grid_clim is not None and selected_kab and mapping_kab:
        fig_rain = make_kabupaten_rainfall_chart(grid_clim, mapping_kab, selected_kab, enso_phase)
    else:
        fig_rain = make_rainfall_envelope_chart(clim_kalteng, enso_phase)

    if fig_rain:
        st.plotly_chart(fig_rain, use_container_width=True)




# =====================================================================
# MAIN
# =====================================================================
def main():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background-color: #f0f4f8; }
    .block-container { padding-top: 1.5rem !important; max-width: 1150px !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0a3d20 0%, #1a6b3c 60%, #1e7a42 100%) !important;
    }
    section[data-testid="stSidebar"] > div:first-child { background: transparent !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.75) !important; }
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: white !important;
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(0,0,0,0.15) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    .hero-card {
        background: linear-gradient(135deg, #0a3d20 0%, #1a6b3c 45%, #27ae60 100%);
        border-radius: 20px;
        padding: 36px 48px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(26,107,60,0.3);
    }
    .hero-card::before {
        content: "🌾🌴🌿";
        position: absolute;
        top: -10px;
        right: 30px;
        font-size: 7rem;
        opacity: 0.08;
        filter: blur(1px);
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 28px 0 14px;
        padding-left: 14px;
        border-left: 4px solid #27ae60;
    }
    .crop-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.07);
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .crop-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }
    </style>
    """, unsafe_allow_html=True)

    # ── Language init ─────────────────────────────────────────────
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    # ── Hero banner ───────────────────────────────────────────────
    st.markdown(
        f"<div class='hero-card'>"
        f"<div style='font-size:0.8rem;color:rgba(255,255,255,0.6);letter-spacing:2px;"
        f"text-transform:uppercase;margin-bottom:10px'>{t('hero_region')}</div>"
        f"<div style='font-size:2rem;font-weight:700;color:white;line-height:1.25'>"
        f"{t('hero_title')}<br>"
        f"<span style='color:#90ee90;font-size:2.2rem'>{t('hero_sub')}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    rain, oni, clim, comp, merged, grid_clim = try_autoload()

    if oni is not None and len(oni) > 0:
        latest_oni   = float(oni.iloc[-1])
        latest_date  = oni.index[-1].strftime("%b %Y")
        latest_phase = classify_enso(latest_oni)
    else:
        latest_oni, latest_date, latest_phase = 0.0, None, "Neutral"

    if latest_phase == "El Niño":
        enso_key = "🔴  Strong El Niño" if latest_oni >= 1.5 else "🟡  Mild El Niño"
    elif latest_phase == "La Niña":
        enso_key = "🌊  Strong La Niña" if latest_oni <= -1.5 else "🔵  Mild La Niña"
    else:
        enso_key = "🟢  Normal"

    _, oni_val = ENSO_OPTIONS[enso_key]

    badge_map = {"El Niño": "🔴", "La Niña": "🔵", "Neutral": "🟢"}
    badge     = badge_map[latest_phase]
    enso_accent = {"El Niño": "#ff7675", "La Niña": "#74b9ff", "Neutral": "#55efc4"}

    with st.sidebar:
        # ── Language toggle ───────────────────────────────────────
        lang_col1, lang_col2 = st.columns(2)
        with lang_col1:
            if st.button("🇬🇧 English", use_container_width=True,
                         type="primary" if st.session_state["lang"] == "en" else "secondary"):
                st.session_state["lang"] = "en"
                st.rerun()
        with lang_col2:
            if st.button("🇮🇩 Indonesia", use_container_width=True,
                         type="primary" if st.session_state["lang"] == "id" else "secondary"):
                st.session_state["lang"] = "id"
                st.rerun()

        st.markdown(
            "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.15);margin:14px 0'>",
            unsafe_allow_html=True,
        )

        # ── Brand ─────────────────────────────────────────────────
        st.markdown(
            f"<div style='text-align:center;padding-bottom:16px;margin-bottom:16px;"
            f"border-bottom:1px solid rgba(255,255,255,0.15)'>"
            f"<div style='font-size:2.6rem;margin-bottom:4px'>🌾</div>"
            f"<div style='font-weight:700;font-size:1rem;color:white'>{t('brand')}</div>"
            f"<div style='font-size:0.68rem;color:rgba(255,255,255,0.45);letter-spacing:2px;"
            f"text-transform:uppercase;margin-top:2px'>{t('region')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── ENSO card ─────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:0.68rem;color:rgba(255,255,255,0.45);letter-spacing:2px;"
            f"text-transform:uppercase;margin-bottom:10px'>{t('live_climate')}</div>",
            unsafe_allow_html=True,
        )
        accent = enso_accent[latest_phase]
        st.markdown(
            f"<div style='background:rgba(0,0,0,0.28);border-radius:14px;padding:22px 16px;"
            f"text-align:center;border:1px solid rgba(255,255,255,0.1);margin-bottom:6px'>"
            f"<div style='font-size:3rem;margin-bottom:8px'>{badge}</div>"
            f"<div style='font-weight:700;color:white;font-size:1.2rem;margin-bottom:6px'>"
            f"{t(f'banner_{latest_phase}')}</div>"
            f"<div style='background:{accent}22;border-radius:8px;padding:6px 10px;display:inline-block'>"
            f"<span style='color:{accent};font-weight:700;font-size:0.9rem'>ONI {latest_oni:+.2f}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if latest_date:
            st.markdown(
                f"<div style='font-size:0.7rem;color:rgba(255,255,255,0.38);text-align:center;"
                f"margin-bottom:16px'>{t('source_noaa')} · {latest_date}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.15);margin:12px 0 16px'>",
            unsafe_allow_html=True,
        )

        # ── Simulate scenario ─────────────────────────────────────
        with st.expander(t("simulate")):
            display_opts = t("enso_opts")
            phase_to_simple = {"El Niño": "🔴  El Niño", "La Niña": "🔵  La Niña", "Neutral": "🟢  Normal"}
            auto_idx = next(
                (i for i, opt in enumerate(display_opts)
                 if opt == phase_to_simple.get(latest_phase, "🟢  Normal")),
                0,
            )
            sim_display  = st.selectbox(t("override"), display_opts, index=auto_idx)
            sim_phase    = {"🔴  El Niño": "El Niño", "🔵  La Niña": "La Niña", "🟢  Normal": "Neutral"}[sim_display]
            if sim_phase != latest_phase:
                _, oni_val   = ENSO_OPTIONS[sim_display]
                latest_phase = sim_phase
                st.info(t("sim_warning"))

    page_farmer(clim, latest_phase, oni_val, grid_clim)


if __name__ == "__main__":
    main()
