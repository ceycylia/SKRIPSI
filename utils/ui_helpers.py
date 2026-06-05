"""Helper UI — HTML & CSS custom untuk Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

__all__ = [
    "load_css",
    "inject_page_styles",
    "render_result_dashboard",
    "render_about_section_html",
    "render_footer",
    "get_diagnosis_label",
    "get_recommendation_key",
    # Backward-compatible names from older app.py versions.
    "render_hero_html",
    "render_feature_cards_html",
    "render_diagnosis_intro_html",
    "render_section_header",
]

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700;800&family=Merriweather:wght@700;800&display=swap"
)

RECOMMENDATIONS = {
    "Sehat": [
        "Lakukan pemantauan rutin pada tanaman anggur.",
        "Jaga kebersihan area tanaman dan sanitasi kebun.",
        "Hindari kelembapan berlebih pada daun.",
        "Pantau kembali perkembangan daun secara berkala.",
    ],
    "Black Measles": [
        "Lakukan pemantauan intensif pada tanaman yang terinfeksi.",
        "Kurangi bagian tanaman yang menunjukkan gejala berat.",
        "Jaga sanitasi kebun dan sirkulasi udara.",
        "Konsultasikan pengendalian dengan ahli pertanian.",
    ],
    "Black Rot": [
        "Pangkas dan buang bagian daun yang terinfeksi.",
        "Hindari kelembapan berlebih pada permukaan daun.",
        "Tingkatkan sirkulasi udara di area tanaman.",
        "Pertimbangkan fungisida sesuai anjuran budidaya.",
    ],
    "Isariopsis Leaf Spot": [
        "Buang daun yang menunjukkan infeksi berat.",
        "Perbaiki sirkulasi udara di sekitar tanaman.",
        "Hindari percikan air berlebih pada daun.",
        "Lakukan tindakan pengendalian sesuai kebutuhan.",
    ],
}

DISCLAIMER = (
    "Hasil sistem ini bersifat alat bantu analisis berbasis citra dan bukan "
    "pengganti pemeriksaan langsung oleh ahli pertanian."
)

SEVERITY_LEGEND = [
    ("Sehat", "0% – 0,5%", "green"),
    ("Ringan", ">0,5% – 25%", "light"),
    ("Sedang", ">25% – 50%", "yellow"),
    ("Berat", ">50%", "red"),
]

SEVERITY_RANGE_HINT = {
    "Sehat": "0% – 0,5%",
    "Ringan": ">0,5% – 25%",
    "Sedang": ">25% – 50%",
    "Berat": ">50%",
}


def load_css() -> str:
    return (ASSETS_DIR / "style.css").read_text(encoding="utf-8")


def inject_page_styles() -> None:
    st.markdown(f'<link rel="stylesheet" href="{FONTS_URL}">', unsafe_allow_html=True)
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


def get_diagnosis_label(result: dict) -> str:
    if result.get("is_healthy"):
        return "Sehat"
    classification = result.get("classification") or {}
    return classification.get("pred_class_display") or classification.get("pred_class") or "Tidak diketahui"


def get_recommendation_key(result: dict) -> str:
    if result.get("is_healthy"):
        return "Sehat"
    return get_diagnosis_label(result)


def _confidence_text(result: dict) -> tuple[str, str]:
    if result.get("is_healthy") or not result.get("classification"):
        return "—", "Tidak berlaku untuk daun sehat"
    conf = float(result["classification"].get("confidence", 0.0)) * 100
    return f"{conf:.1f}%", "Tingkat keyakinan model"


def render_result_dashboard(result: dict) -> None:
    """Ringkasan hasil: stat cards, probabilitas, rekomendasi, legenda."""
    label = get_diagnosis_label(result)
    category = result.get("severity_category") or "Sehat"
    severity = float(result.get("severity", 0.0))
    range_hint = SEVERITY_RANGE_HINT.get(category, "")
    conf_text, conf_sub = _confidence_text(result)

    stat_row = f"""
<section class="stat-row">
  <article class="stat-card stat-card--wide">
    <div class="stat-card__icon">🛡️</div>
    <div>
      <div class="stat-card__label">Hasil Diagnosis</div>
      <div class="stat-card__value">{label}</div>
      <div class="stat-card__sub">Keluaran utama sistem</div>
    </div>
  </article>
  <article class="stat-card">
    <div class="stat-card__icon">◎</div>
    <div>
      <div class="stat-card__label">Severity</div>
      <div class="stat-card__value">{severity:.2f}%</div>
      <div class="stat-card__sub">Persentase keparahan</div>
    </div>
  </article>
  <article class="stat-card">
    <div class="stat-card__icon">🍃</div>
    <div>
      <div class="stat-card__label">Kategori Keparahan</div>
      <div class="stat-card__value">{category}</div>
      <div class="stat-card__sub">{range_hint}</div>
    </div>
  </article>
  <article class="stat-card">
    <div class="stat-card__icon">📈</div>
    <div>
      <div class="stat-card__label">Confidence</div>
      <div class="stat-card__value">{conf_text}</div>
      <div class="stat-card__sub">{conf_sub}</div>
    </div>
  </article>
</section>
"""

    st.markdown(stat_row, unsafe_allow_html=True)

    col_prob, col_rec, col_leg = st.columns([1.15, 1.05, 0.9], gap="medium")
    with col_prob:
        _render_probability_section(result)
    with col_rec:
        _render_recommendation_section(result)
    with col_leg:
        st.markdown(_render_severity_legend_card(), unsafe_allow_html=True)


def _render_probability_section(result: dict) -> None:
    """Probabilitas kelas — Streamlit native untuk cabang penyakit (hindari HTML mentah)."""
    with st.container(border=True):
        st.markdown(
            '<h3 class="native-card-title">Probabilitas Kelas</h3>',
            unsafe_allow_html=True,
        )

        if result.get("is_healthy") or not result.get("classification"):
            st.markdown(
                '<p class="muted-text">Tidak tersedia karena daun dikategorikan sehat berdasarkan severity.</p>',
                unsafe_allow_html=True,
            )
            return

        probs = result.get("classification", {}).get("probabilities", {})
        if not probs:
            st.markdown(
                '<p class="muted-text">Data probabilitas tidak tersedia.</p>',
                unsafe_allow_html=True,
            )
            return

        for class_name, prob in probs.items():
            pct = float(prob) * 100
            st.markdown(f"**{class_name}** — {pct:.1f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))


def _render_recommendation_section(result: dict) -> None:
    """Rekomendasi — Streamlit native agar konsisten dengan bagian probabilitas."""
    key = get_recommendation_key(result)
    bullets = RECOMMENDATIONS.get(key, RECOMMENDATIONS["Sehat"])

    with st.container(border=True):
        st.markdown(
            '<h3 class="native-card-title">Rekomendasi Penanganan</h3>',
            unsafe_allow_html=True,
        )
        for item in bullets:
            st.markdown(f"✓ {item}")
        st.markdown(
            f'<div class="note-box">{DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )


def _render_severity_legend_card() -> str:
    rows = "".join(
        f'<div class="severity-row"><span class="dot dot-{dot}"></span><b>{name}</b><span>{rng}</span></div>'
        for name, rng, dot in SEVERITY_LEGEND
    )
    return f'<div class="detail-card detail-card--legend"><h3>Kategori Keparahan</h3>{rows}</div>'


def render_about_section_html() -> str:
    return """
<section class="about-card">
  <div class="about-grid">
    <article>
      <h3>Model Segmentasi</h3>
      <p><strong>U-Net + EfficientNet-B0</strong> digunakan untuk segmentasi daun dan segmentasi lesi.</p>
    </article>
    <article>
      <h3>Estimasi Severity</h3>
      <p><strong>Severity</strong> dihitung dari rasio piksel lesi terhadap piksel daun × 100%.</p>
    </article>
    <article>
      <h3>Klasifikasi Penyakit</h3>
      <p><strong>EfficientNet-B0</strong> mengklasifikasikan Black Measles, Black Rot, dan Isariopsis Leaf Spot.</p>
    </article>
    <article>
      <h3>Keputusan Sehat</h3>
      <p>Kondisi <strong>Sehat</strong> ditentukan jika severity ≤ 0,5%.</p>
    </article>
  </div>
</section>
"""


def render_footer() -> None:
    st.markdown(
        """
<footer class="site-footer">
  <div class="site-footer__inner">
    <div>
      <div class="footer-brand">🍇 Diagnosis Daun Anggur</div>
      <p>Sistem diagnosis penyakit daun anggur berbasis deep learning.</p>
    </div>
    <div>
      <div class="footer-title">Institusi</div>
      <p>Program Studi Teknologi Informasi<br>Fakultas Ilmu Komputer dan Teknologi Informasi<br>Universitas Sumatera Utara</p>
    </div>
    <div>
      <div class="footer-title">Identitas</div>
      <p>Ceycylia Dear Amizafatel<br>221402059</p>
    </div>
  </div>
</footer>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Backward-compatible helper names for older app.py versions.
# ─────────────────────────────────────────────────────────────
def render_hero_html() -> str:
    chips = "".join(f"<span>{x}</span>" for x in ["U-Net", "EfficientNet-B0", "Segmentasi", "Klasifikasi", "Severity"])
    return f"""
<section class="legacy-hero-html">
  <div class="hero-copy">
    <div class="pill">🌿 Sistem Cerdas untuk Pertanian Presisi</div>
    <h1 class="hero-title-main">Klasifikasi Penyakit Daun Anggur dan Estimasi Tingkat Keparahan</h1>
    <p class="hero-subtitle-main">Menggunakan EfficientNet-B0 dan U-Net</p>
  </div>
  <div class="ai-visual-card"><div class="leaf-orbit">🍃</div><h2>Deep Learning untuk Pertanian Presisi</h2><div class="chip-row">{chips}</div></div>
</section>
"""


def render_feature_cards_html() -> str:
    return """
<section class="feature-grid">
  <article class="feature-card"><span class="feature-number">1</span><div class="feature-icon">🍃</div><h3>Segmentasi Daun</h3><p>Memisahkan area daun anggur dari latar belakang citra.</p></article>
  <article class="feature-card"><span class="feature-number">2</span><div class="feature-icon">🔬</div><h3>Segmentasi Lesi</h3><p>Mendeteksi bercak atau area lesi pada permukaan daun.</p></article>
  <article class="feature-card"><span class="feature-number">3</span><div class="feature-icon">📊</div><h3>Klasifikasi & Severity</h3><p>Menghasilkan jenis penyakit serta kategori tingkat keparahan.</p></article>
</section>
"""


def render_diagnosis_intro_html() -> str:
    return """
<section class="diagnosis-intro">
  <div class="pill">🌿 Sistem Cerdas untuk Pertanian Presisi</div>
  <h1>Diagnosis Citra Daun Anggur</h1>
  <p>Unggah satu citra daun anggur untuk dilakukan segmentasi daun, segmentasi lesi, estimasi tingkat keparahan, dan klasifikasi penyakit.</p>
</section>
"""


def render_section_header(title: str, description: str) -> None:
    st.markdown(
        f'<section class="section-heading"><h2>{title}</h2><p>{description}</p></section>',
        unsafe_allow_html=True,
    )
