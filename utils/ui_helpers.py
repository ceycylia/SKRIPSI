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
    "family=Inter:wght@400;500;600;700;800&display=swap"
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
    ("Sehat", "< 0,50%", "green"),
    ("Ringan", "0,5% – 25%", "light"),
    ("Sedang", ">25% – 50%", "yellow"),
    ("Berat", ">50%", "red"),
]

SEVERITY_RANGE_HINT = {
    "Sehat": "< 0,50%",
    "Ringan": "0,5% – 25%",
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



def icon(name: str) -> str:
    paths = {
        "leaf": '<path d="M20 4C10 3 4 7 4 13a7 7 0 0 0 7 7c6 0 10-6 9-16Z"/><path d="M4 21 15 10"/>',
        "upload": '<path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5"/>',
    }
    return f'<svg class="outline-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths.get(name, paths["leaf"])}</svg>'

def render_result_dashboard(result: dict) -> None:
    from html import escape
    label = escape(get_diagnosis_label(result))
    category = result.get("severity_category") or "Sehat"
    severity = float(result.get("severity", 0))
    tone = {"Sehat": "green", "Ringan": "green", "Sedang": "yellow", "Berat": "red"}.get(category, "green")
    conf_text, conf_sub = _confidence_text(result)
    st.markdown(f'''<section class="result-summary">
<div><span class="eyebrow">HASIL DIAGNOSIS</span><h2>{label}</h2><span class="status status-{tone}">{escape(category)}</span></div>
<div class="result-number"><span>Keparahan daun</span><strong>{severity:.2f}<small>%</small></strong><span>Luas lesi / luas daun</span></div>
<div class="result-confidence"><span>Keyakinan klasifikasi</span><strong>{conf_text}</strong><span>{conf_sub}</span></div>
</section><p class="interpretation">Area lesi terdeteksi mencakup <strong>{severity:.2f}%</strong> dari area daun. Berdasarkan ambang sistem, hasil ini termasuk kategori <strong>{escape(category.lower())}</strong>.</p>''', unsafe_allow_html=True)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("### Probabilitas kelas")
        classification = result.get("classification") or {}
        if result.get("is_healthy"):
            st.caption("Klasifikasi penyakit tidak dijalankan karena severity berada di bawah 0,50%.")
        elif not classification.get("probabilities"):
            st.caption("Data probabilitas tidak tersedia.")
        else:
            for name, prob in classification["probabilities"].items():
                st.progress(min(max(float(prob), 0), 1), text=f"{name} · {float(prob)*100:.1f}%")
    with right:
        st.markdown("### Rekomendasi penanganan")
        for item in RECOMMENDATIONS.get(get_recommendation_key(result), RECOMMENDATIONS["Sehat"]):
            st.markdown(f"- {item}")
    st.markdown(f'<p class="system-note">{DISCLAIMER}</p>', unsafe_allow_html=True)
    with st.expander("Lihat panduan kategori keparahan"):
        st.markdown(render_severity_legend(), unsafe_allow_html=True)

def render_severity_legend() -> str:
    from html import escape
    rows = "".join(f'<div class="severity-row"><span class="dot dot-{tone}"></span><strong>{name}</strong><span>{escape(rng)}</span></div>' for name, rng, tone in SEVERITY_LEGEND)
    return f'<div class="severity-legend">{rows}</div>'

def render_hero_html() -> str:
    return '''<section class="hero-copy"><span class="eyebrow">SISTEM DIAGNOSIS BERBASIS DEEP LEARNING</span>
<h1>Kenali penyakit daun.<br><span class="hero-emphasis">Pahami keparahannya.</span></h1>
<p>Analisis citra daun anggur untuk mengenali jenis penyakit dan memperkirakan luas lesi melalui segmentasi dan klasifikasi.</p>
<div class="model-meta">U-Net <span>/</span> EfficientNet-B0</div></section>'''

def render_feature_cards_html() -> str:
    steps = [("Segmentasi daun", "Memisahkan daun dari latar."),
             ("Segmentasi lesi", "Menandai area yang bergejala."),
             ("Klasifikasi", "Mengenali jenis penyakit."),
             ("Estimasi keparahan", "Membaca proporsi area lesi.")]
    items = "".join(f'<li><span class="step-number">0{i}</span><h3>{name}</h3><p>{desc}</p></li>' for i, (name, desc) in enumerate(steps, 1))
    return f'<section class="workflow"><div class="workflow-heading"><span class="eyebrow">DARI CITRA KE INFORMASI</span><span>Empat komponen analisis</span></div><ol>{items}</ol><p class="workflow-note">Dalam pemrosesan, severity dihitung sebelum klasifikasi; daun dengan severity &lt; 0,50% dikategorikan sehat.</p></section>'

def render_diagnosis_intro_html() -> str:
    return '''<section class="page-heading"><span class="eyebrow">DIAGNOSIS CITRA</span><h1>Mulai dari<br>satu daun.</h1>
<p>Unggah citra daun anggur. Sistem akan menampilkan area daun, lesi, serta hasil analisisnya.</p>
<div class="upload-guidance"><h3>Agar citra mudah dianalisis</h3><ol><li>Gunakan foto satu daun yang terlihat utuh.</li><li>Pilih pencahayaan merata, tanpa bayangan kuat.</li><li>Pastikan fokus tajam dan gejala terlihat jelas.</li></ol></div></section>'''

def render_about_section_html() -> str:
    return '''<div class="research-doc">
<section><div class="doc-label">01 / TUJUAN</div><div><h2>Membaca kondisi daun melalui citra</h2><p>Penelitian ini menggabungkan segmentasi dan klasifikasi untuk membantu mengenali penyakit daun anggur serta mengestimasi tingkat keparahannya. Area lesi dibandingkan dengan area daun agar hasil dapat dibaca secara kuantitatif.</p></div></section>
<section><div class="doc-label">02 / METODE</div><div><h2>Dua model segmentasi, satu model klasifikasi</h2><p><strong>U-Net dengan encoder EfficientNet-B0</strong> memisahkan area daun dan lesi. <strong>EfficientNet-B0</strong> mengklasifikasikan tiga penyakit: Black Measles, Black Rot, dan Isariopsis Leaf Spot.</p><ol><li>Citra disesuaikan dengan letterbox 256 × 256 dan dinormalisasi.</li><li>Model daun membentuk mask (threshold 0,45); latar dihitamkan.</li><li>Model lesi membentuk mask (threshold 0,40), dibatasi area daun.</li><li>Severity dihitung dari piksel lesi / piksel daun × 100%.</li><li>Jika severity &lt; 0,50%, hasilnya sehat. Selain itu, klasifikasi dijalankan pada citra 224 × 224.</li></ol></div></section>
<section><div class="doc-label">03 / BATASAN</div><div><h2>Memahami konteks hasil</h2><p>Hasil dipengaruhi kualitas citra, pencahayaan, latar, dan kemiripan citra dengan data pelatihan. Cakupan klasifikasi terbatas pada tiga penyakit tersebut. Persentase keyakinan model bukan jaminan kebenaran diagnosis.</p><p>Sistem merupakan alat bantu penelitian berbasis citra, bukan pengganti pemeriksaan langsung oleh ahli pertanian.</p></div></section>
<section><div class="doc-label">04 / PENELITI</div><div><h2>Cecylia Dear Amizafatel</h2><p>221402059 · Program Studi Teknologi Informasi<br>Fakultas Ilmu Komputer dan Teknologi Informasi – USU</p></div></section>
</div>'''

def render_footer() -> None:
    st.markdown('<footer class="site-footer"><div><strong>Diagnosis Daun Anggur</strong><span>Aplikasi penelitian · Universitas Sumatera Utara</span></div><span>Cecylia Dear Amizafatel / 221402059</span></footer>', unsafe_allow_html=True)

def render_section_header(title: str, description: str) -> None:
    from html import escape
    st.markdown(f'<div class="section-heading"><h2>{escape(title)}</h2><p>{escape(description)}</p></div>', unsafe_allow_html=True)
