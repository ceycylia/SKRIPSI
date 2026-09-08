"""Antarmuka penelitian diagnosis daun anggur; pipeline berada di utils."""
from __future__ import annotations

import streamlit as st
from PIL import Image, UnidentifiedImageError
from utils.inference import run_diagnosis
from utils.model_loader import load_all_models
from utils.ui_helpers import (
    inject_page_styles, render_about_section_html, render_footer,
    render_result_dashboard, render_hero_html, render_feature_cards_html,
    render_diagnosis_intro_html, icon, render_severity_legend,
)

st.set_page_config(page_title="Diagnosis Daun Anggur — Skripsi USU",
                   page_icon=":material/eco:", layout="wide", initial_sidebar_state="collapsed")
PAGE_LABELS = {"beranda": "Beranda", "diagnosis": "Diagnosis Citra", "tentang": "Tentang Sistem"}
LABEL_TO_PAGE = {label: key for key, label in PAGE_LABELS.items()}
for key, value in (("page", "beranda"), ("diagnosis_result", None), ("last_file", None)):
    if key not in st.session_state:
        st.session_state[key] = value
inject_page_styles()

def navigate(page_key):
    st.session_state.page = page_key

def reroute(page_key):
    navigate(page_key)
    st.rerun()

@st.cache_resource
def load_models():
    return load_all_models()

def show_navbar():
    with st.container(key="navigation"):
        brand, menu = st.columns([1, 1], vertical_alignment="center")
        with brand:
            st.markdown(f'<div class="brand-row">{icon("leaf")}<span>Diagnosis Daun Anggur<small>PENELITIAN · USU</small></span></div>', unsafe_allow_html=True)
        with menu:
            labels = list(PAGE_LABELS.values())
            selected = st.radio("Navigasi", labels,
                index=labels.index(PAGE_LABELS.get(st.session_state.page, "Beranda")),
                horizontal=True, label_visibility="collapsed")
            if LABEL_TO_PAGE[selected] != st.session_state.page:
                reroute(LABEL_TO_PAGE[selected])

def page_beranda():
    with st.container(key="hero"):
        left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
        with left:
            st.markdown(render_hero_html(), unsafe_allow_html=True)
            if st.button("Mulai Diagnosis", key="cta_start", type="primary"):
                reroute("diagnosis")
            st.markdown('<div class="researcher"><strong>Cecylia Dear Amizafatel <span> / 221402059</span></strong><p>Program Studi Teknologi Informasi<br>Fakultas Ilmu Komputer dan Teknologi Informasi – USU</p></div>', unsafe_allow_html=True)
        with right:
            st.markdown('''<figure class="specimen">
<div class="specimen-heading"><span>OBJEK PENELITIAN</span><span>01 / Vitis vinifera</span></div>
<svg class="leaf-study" viewBox="0 0 440 320" role="img" aria-label="Diagram pengamatan daun anggur, dengan penanda area daun dan lesi. Bukan hasil diagnosis.">
<path d="M220 264 C169 258 113 228 77 188 L122 178 L74 133 L130 139 L127 82 L180 117 L217 47 L248 116 L306 80 L297 141 L365 128 L326 175 L368 197 C318 235 272 260 220 264Z" fill="#DCEBE3" stroke="#26735B" stroke-width="2"/>
<g fill="none" stroke="#26735B" stroke-width="1.5"><path d="M218 289 L218 92 M218 244 L120 183 M218 221 L145 120 M218 199 L302 120 M218 244 L330 192"/><path d="M174 218 L169 188 M267 220 L287 191 M188 165 L179 139"/></g>
<g fill="#C59B45" fill-opacity=".45" stroke="#94732E"><ellipse cx="266" cy="167" rx="13" ry="10"/><ellipse cx="279" cy="184" rx="7" ry="5"/><ellipse cx="164" cy="173" rx="8" ry="6"/></g>
<g fill="none" stroke="#66736D"><path d="M142 127 L85 57 L34 57"/><path d="M279 166 L343 65 L409 65"/></g>
<g fill="#173F35" font-family="sans-serif" font-size="11"><text x="28" y="44">AREA DAUN</text><text x="345" y="52">AREA LESI</text></g>
</svg><figcaption><strong>Dari citra, memahami kondisi daun.</strong><span>Ilustrasi area pengamatan · bukan hasil diagnosis</span></figcaption>
</figure>''', unsafe_allow_html=True)
    st.markdown(render_feature_cards_html(), unsafe_allow_html=True)

def _image_card(image, title, caption, number):
    st.markdown(f'<div class="image-heading"><span>0{number}</span><h3>{title}</h3></div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(caption)

def page_diagnosis():
    left, right = st.columns([.85, 1.35], gap="large")
    with left:
        st.markdown(render_diagnosis_intro_html(), unsafe_allow_html=True)
    with right:
        with st.container(key="upload"):
            st.markdown(f'<h2 class="upload-title">{icon("upload")} Unggah citra daun</h2>', unsafe_allow_html=True)
            st.caption("JPG, JPEG, PNG, BMP, atau WEBP · maksimal 10 MB")
            uploaded_file = st.file_uploader("Pilih citra daun anggur",
                type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
            image = None
            if uploaded_file is not None:
                if st.session_state.last_file != uploaded_file.name:
                    st.session_state.diagnosis_result = None
                    st.session_state.last_file = uploaded_file.name
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    preview, detail = st.columns([1, 1], gap="medium")
                    with preview:
                        st.image(image, use_container_width=True)
                    with detail:
                        st.markdown("**Citra siap diperiksa**")
                        st.caption(uploaded_file.name)
                        st.caption("Periksa kembali ketajaman gambar dan pastikan seluruh daun terlihat.")
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                    st.error("Gambar tidak dapat dibaca. Coba unggah ulang dalam format JPG atau PNG.")
            process = st.button("Proses Diagnosis", key="btn_process", type="primary",
                                disabled=image is None, use_container_width=True)
            if image is None:
                st.caption("Hasil analisis akan muncul di bawah setelah citra diproses.")
            if process and image is not None:
                with st.status("Menyiapkan model diagnosis…", expanded=True) as status:
                    try:
                        models = load_models()
                        st.write("Memeriksa area daun dan lesi, lalu menghitung hasil diagnosis…")
                        st.session_state.diagnosis_result = run_diagnosis(image, models)
                        status.update(label="Analisis selesai. Hasil tersedia di bawah.", state="complete", expanded=False)
                    except Exception as exc:
                        status.update(label="Analisis belum berhasil.", state="error")
                        st.error(f"Terjadi kesalahan saat inference: {exc}")
                        st.session_state.diagnosis_result = None
    result = st.session_state.get("diagnosis_result")
    if result is not None:
        st.markdown('<div class="section-heading"><span class="eyebrow">HASIL ANALISIS</span><h2>Melihat bagian yang terdeteksi</h2><p>Bandingkan citra dengan area daun dan lesi yang dikenali model.</p></div>', unsafe_allow_html=True)
        with st.container(key="results-images"):
            cols = st.columns(4, gap="medium")
            figures = [
                (result["image_256"], "Citra asli", "Citra masukan, disesuaikan ke 256 × 256."),
                (result["leaf_overlay"], "Segmentasi daun", "Warna hijau menunjukkan area daun."),
                (result["lesion_overlay"], "Segmentasi lesi", "Warna merah menunjukkan area lesi."),
                (result["masked_image"], "Daun tanpa latar", "Latar di luar daun dihitamkan."),
            ]
            for idx, (col, figure) in enumerate(zip(cols, figures), 1):
                with col:
                    _image_card(*figure, idx)
        render_result_dashboard(result)

def page_tentang():
    st.markdown('<div class="page-heading"><span class="eyebrow">CATATAN PENELITIAN</span><h1>Tentang sistem</h1><p>Pendekatan berbasis citra untuk mengenali penyakit dan mengukur keparahan pada daun anggur.</p></div>', unsafe_allow_html=True)
    st.markdown(render_about_section_html(), unsafe_allow_html=True)
    with st.expander("Panduan kategori keparahan", expanded=True):
        st.markdown(render_severity_legend(), unsafe_allow_html=True)

show_navbar()
{"beranda": page_beranda, "diagnosis": page_diagnosis, "tentang": page_tentang}.get(st.session_state.page, page_beranda)()
render_footer()
