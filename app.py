"""Aplikasi Streamlit — Diagnosis Penyakit Daun Anggur.

File ini berfokus pada UI/alur halaman. Logic model, preprocessing,
inference, threshold, dan class names tetap berada di modul utils.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from utils.inference import run_diagnosis
from utils.model_loader import load_all_models
from utils.ui_helpers import (
    inject_page_styles,
    render_about_section_html,
    render_footer,
    render_result_dashboard,
)

# ─────────────────────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnosis Daun Anggur — Skripsi USU",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGE_LABELS = {
    "beranda": "Beranda",
    "diagnosis": "Diagnosis Citra",
    "tentang": "Tentang Sistem",
}
LABEL_TO_PAGE = {label: key for key, label in PAGE_LABELS.items()}

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "beranda"
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

inject_page_styles()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def navigate(page_key: str) -> None:
    """Pindah halaman — hanya ubah state halaman utama."""
    st.session_state.page = page_key


def reroute(page_key: str) -> None:
    navigate(page_key)
    st.rerun()


@st.cache_resource
def load_models():
    """Load semua model sekali saja selama session Streamlit."""
    return load_all_models()


def show_navbar() -> None:
    """Navbar stabil berbasis radio horizontal, bukan button berjauhan."""
    nav_labels = list(PAGE_LABELS.values())
    current_label = PAGE_LABELS.get(st.session_state.page, "Beranda")

    brand_col, menu_col = st.columns([1.15, 1.0], vertical_alignment="center")
    with brand_col:
        st.markdown(
            """
            <div class="brand-row">
                <span class="brand-logo">🍇</span>
                <span class="brand-title">Diagnosis Daun Anggur</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with menu_col:
        selected_label = st.radio(
            "Navigasi",
            nav_labels,
            index=nav_labels.index(current_label),
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_page = LABEL_TO_PAGE[selected_label]
        if selected_page != st.session_state.page:
            navigate(selected_page)
            st.rerun()


def _image_card(image, title: str, caption: str, number: int) -> None:
    """Card gambar kecil dan konsisten untuk hasil segmentasi."""
    st.markdown(
        f"""
        <div class="image-card-head">
            <span class="image-card-num">{number}</span>
            <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.image(image, use_container_width=True)
        st.caption(caption)


# ─────────────────────────────────────────────────────────────
# Halaman Beranda
# ─────────────────────────────────────────────────────────────
def page_beranda() -> None:
    hero_left, hero_right = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")

    with hero_left:
        st.markdown(
            """
            <section class="hero-copy">
                <div class="pill">🌿 Sistem Cerdas untuk Pertanian Presisi</div>
                <h1 class="hero-title-main">Klasifikasi Penyakit Daun Anggur dan Estimasi Tingkat Keparahan</h1>
                <p class="hero-subtitle-main">Menggunakan EfficientNet-B0 dan U-Net</p>
                <div class="gold-line"></div>
                <div class="identity-list">
                    <p><span>👤</span><b>Ceycylia Dear Amizafatel</b> - 221402059</p>
                    <p><span>🎓</span>Program Studi Teknologi Informasi</p>
                    <p><span>🏛️</span>Fakultas Ilmu Komputer dan Teknologi Informasi - USU</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Mulai Diagnosis →", key="cta_start", type="primary"):
            reroute("diagnosis")

    with hero_right:
        st.markdown(
            """
            <section class="ai-visual-card">
                <div class="leaf-orbit">🍃</div>
                <h2>Deep Learning untuk Pertanian Presisi</h2>
                <div class="chip-row">
                    <span>U-Net</span>
                    <span>EfficientNet-B0</span>
                    <span>Segmentasi</span>
                    <span>Klasifikasi</span>
                    <span>Severity</span>
                </div>
                <div class="severity-mini">
                    <div class="severity-mini-label">Severity</div>
                    <div class="severity-mini-bar"></div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <section class="feature-grid">
            <article class="feature-card">
                <span class="feature-number">1</span>
                <div class="feature-icon">🍃</div>
                <h3>Segmentasi Daun</h3>
                <p>Memisahkan area daun anggur dari latar belakang citra.</p>
            </article>
            <article class="feature-card">
                <span class="feature-number">2</span>
                <div class="feature-icon">🔬</div>
                <h3>Segmentasi Lesi</h3>
                <p>Mendeteksi bercak atau area lesi pada permukaan daun.</p>
            </article>
            <article class="feature-card">
                <span class="feature-number">3</span>
                <div class="feature-icon">📊</div>
                <h3>Klasifikasi & Severity</h3>
                <p>Menghasilkan jenis penyakit serta kategori tingkat keparahan.</p>
            </article>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Halaman Diagnosis
# ─────────────────────────────────────────────────────────────
def page_diagnosis() -> None:
    intro_col, upload_col = st.columns([0.9, 1.1], gap="large", vertical_alignment="top")

    with intro_col:
        st.markdown(
            """
            <section class="diagnosis-intro">
                <div class="pill">🌿 Sistem Cerdas untuk Pertanian Presisi</div>
                <h1>Diagnosis Citra Daun Anggur</h1>
                <p>
                    Unggah satu citra daun anggur untuk dilakukan segmentasi daun,
                    segmentasi lesi, estimasi tingkat keparahan, dan klasifikasi penyakit.
                </p>
                <div class="gold-line"></div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with upload_col:
        with st.container(border=True):
            st.markdown('<h3 class="upload-title">☁️ Unggah Citra</h3>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Pilih citra daun anggur",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )

            image = None
            process = False
            if uploaded_file is not None:
                if st.session_state.last_file != uploaded_file.name:
                    st.session_state.diagnosis_result = None
                    st.session_state.last_file = uploaded_file.name

                image = Image.open(uploaded_file).convert("RGB")
                preview_col, action_col = st.columns([0.95, 1.05], gap="medium", vertical_alignment="center")
                with preview_col:
                    st.image(image, width=340)
                    st.caption(uploaded_file.name)
                with action_col:
                    st.markdown(
                        '<p class="helper-text">Pastikan citra menampilkan daun anggur dengan jelas.</p>',
                        unsafe_allow_html=True,
                    )
                    process = st.button(
                        "🔍 Proses Diagnosis",
                        key="btn_process",
                        type="primary",
                        use_container_width=True,
                    )
            else:
                st.info("Format yang didukung: JPG, JPEG, PNG, BMP, WEBP.")

            if process and image is not None:
                with st.spinner("Memproses citra..."):
                    try:
                        models = load_models()
                        st.session_state.diagnosis_result = run_diagnosis(image, models)
                    except Exception as exc:
                        st.error(f"Terjadi kesalahan saat inference: {exc}")
                        st.session_state.diagnosis_result = None

    result = st.session_state.get("diagnosis_result")
    if result is not None:
        st.markdown(
            """
            <section class="section-heading">
                <h2>Hasil Visualisasi</h2>
                <p>Hasil segmentasi daun, segmentasi lesi, dan citra hasil masking.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            _image_card(Image.fromarray(result["image_256"]), "Citra Asli", "Letterbox 256×256", 1)
        with c2:
            _image_card(result["leaf_overlay"], "Segmentasi Daun", "Area daun terdeteksi", 2)
        with c3:
            _image_card(result["masked_image"], "Daun BG Hitam", "Background di luar daun dihitamkan", 3)
        with c4:
            _image_card(result["lesion_overlay"], "Segmentasi Lesi", "Bercak lesi terdeteksi", 4)

        render_result_dashboard(result)


# ─────────────────────────────────────────────────────────────
# Halaman Tentang
# ─────────────────────────────────────────────────────────────
def page_tentang() -> None:
    st.markdown(
        """
        <section class="section-heading section-heading--large">
            <h1>Tentang Sistem</h1>
            <p>Ringkasan arsitektur model dan alur kerja sistem diagnosis berbasis deep learning.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(render_about_section_html(), unsafe_allow_html=True)
    st.markdown(
        """
        <section class="severity-card severity-card--standalone">
            <h3>Kategori Keparahan</h3>
            <div class="severity-row"><span class="dot dot-green"></span><b>Sehat</b><span>0% – 0,5%</span></div>
            <div class="severity-row"><span class="dot dot-light"></span><b>Ringan</b><span>&gt;0,5% – 25%</span></div>
            <div class="severity-row"><span class="dot dot-yellow"></span><b>Sedang</b><span>&gt;25% – 50%</span></div>
            <div class="severity-row"><span class="dot dot-red"></span><b>Berat</b><span>&gt;50%</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
show_navbar()

page = st.session_state.page
if page == "beranda":
    page_beranda()
elif page == "diagnosis":
    page_diagnosis()
elif page == "tentang":
    page_tentang()
else:
    reroute("beranda")

render_footer()
