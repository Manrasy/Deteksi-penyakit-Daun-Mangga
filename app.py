import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Pemodelan Proporsi Guru MI",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

[data-testid="stMetric"]{
    background-color:white;
    padding:20px;
    border-radius:15px;
    border-left:6px solid #2563eb;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

section[data-testid="stSidebar"]{
    background-color:#0f172a;
}

section[data-testid="stSidebar"] *{
    color:white;
}

h1,h2,h3{
    color:#1e293b;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_excel(
        "data/Hasil_Analisis_Guru_MI.xlsx"
    )

df = load_data()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "📚 Menu Analisis"
)

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Dashboard",
        "Visualisasi Data",
        "Pemodelan Regresi",
        "Simulasi Prediksi",
        "Monitoring Distribusi",
        "What-If Analysis",
        "Analisis Gap",
        "Klasifikasi",
        "Laporan"
    ]
)

# ==================================================
# FILTER GLOBAL
# ==================================================

tahun_list = sorted(
    df["Tahun"].unique()
)

kota_list = sorted(
    df["Kota"].unique()
)

st.sidebar.markdown("---")

tahun_filter = st.sidebar.selectbox(
    "Filter Tahun",
    options=["Semua"] + list(tahun_list)
)

kota_filter = st.sidebar.selectbox(
    "Filter Kota",
    options=["Semua"] + list(kota_list)
)

df_filter = df.copy()

if tahun_filter != "Semua":
    df_filter = df_filter[
        df_filter["Tahun"] == tahun_filter
    ]

if kota_filter != "Semua":
    df_filter = df_filter[
        df_filter["Kota"] == kota_filter
    ]

# ==================================================
# DASHBOARD
# ==================================================

if menu == "Dashboard":

    st.title(
        "📊 Dashboard Pendidikan MI Jawa Barat"
    )

    total_guru = int(
        df_filter["Jumlah_Guru"].sum()
    )

    total_siswa = int(
        df_filter["Jumlah_Siswa"].sum()
    )

    total_sekolah = int(
        df_filter["Jumlah_Sekolah"].sum()
    )

    rata_rasio = round(
        df_filter["Rasio_Siswa_Guru"].mean(),
        2
    )

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "👨‍🏫 Total Guru",
            f"{total_guru:,}"
        )

    with col2:
        st.metric(
            "🎓 Total Siswa",
            f"{total_siswa:,}"
        )

    with col3:
        st.metric(
            "🏫 Total Sekolah",
            f"{total_sekolah:,}"
        )

    with col4:
        st.metric(
            "📈 Rasio Siswa-Guru",
            rata_rasio
        )

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        kondisi = (
            df_filter["Kondisi"]
            .value_counts()
            .reset_index()
        )

        kondisi.columns = [
            "Kondisi",
            "Jumlah"
        ]

        fig = px.pie(
            kondisi,
            values="Jumlah",
            names="Kondisi",
            hole=0.4,
            title="Distribusi Kondisi Guru"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        guru_tahun = (
            df.groupby("Tahun")
            ["Jumlah_Guru"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            guru_tahun,
            x="Tahun",
            y="Jumlah_Guru",
            markers=True,
            title="Perkembangan Jumlah Guru"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "Data Pendidikan MI"
    )

    st.dataframe(
        df_filter,
        use_container_width=True
    )
