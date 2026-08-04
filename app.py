"""Aplikasi Streamlit untuk klasifikasi gambar berbasis model Keras.

File ini generik: seluruh perilaku diatur lewat config.yaml.
Untuk project baru, ubah config.yaml dan ganti isi folder models/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))  # agar `core` bisa diimpor dari mana pun app dijalankan

from core.config import Config, ConfigError, load_config  # noqa: E402
from core.model import ModelLoadError, load_labels, load_model, validate_model_labels  # noqa: E402
from core.predict import Prediction, predict  # noqa: E402

CONFIG_PATH = APP_DIR / "config.yaml"
BYTES_PER_MB = 1024 * 1024


@st.cache_resource(show_spinner="Memuat model...")
def get_model_and_labels(model_path: str, labels_path: str):
    """Muat model dan label sekali saja, lalu simpan di cache antar-sesi."""
    model = load_model(model_path)
    labels = load_labels(labels_path)
    validate_model_labels(model, labels)
    return model, labels


def render_sidebar(config: Config, labels: list[str]) -> None:
    with st.sidebar:
        st.header("Tentang")
        if config.app.about:
            st.markdown(config.app.about)

        st.subheader("Kelas yang dikenali")
        for label in labels:
            st.markdown(f"- {config.label_display.get(label, label)}")

        st.subheader("Info model")
        height, width = config.model.input_size
        st.caption(f"Ukuran input: {width}x{height} piksel")
        st.caption(f"Normalisasi: {config.model.normalization}")
        st.caption(f"Jumlah kelas: {len(labels)}")


def render_results(predictions: list[Prediction], config: Config) -> None:
    top = predictions[0]
    threshold = config.inference.confidence_threshold

    st.subheader("Hasil prediksi")
    st.metric(label=top.display_label, value=f"{top.probability:.1%}")

    if top.probability < threshold:
        st.warning(
            f"Keyakinan model di bawah ambang batas ({threshold:.0%}). "
            "Gunakan gambar yang lebih jelas atau periksa apakah objek termasuk kelas yang dikenali."
        )
    else:
        st.success(f"Model yakin gambar ini termasuk **{top.display_label}**.")

    top_k = predictions[: config.inference.top_k]
    if len(top_k) > 1:
        st.markdown(f"**{len(top_k)} kemungkinan teratas**")
        for rank, prediction in enumerate(top_k, start=1):
            st.write(f"{rank}. {prediction.display_label} — {prediction.probability:.2%}")
            st.progress(min(max(prediction.probability, 0.0), 1.0))

    if config.ui.show_all_probabilities:
        with st.expander("Probabilitas seluruh kelas"):
            frame = pd.DataFrame(
                {
                    "Kelas": [p.display_label for p in predictions],
                    "Probabilitas": [p.probability for p in predictions],
                }
            ).set_index("Kelas")
            st.bar_chart(frame)
            st.dataframe(
                frame.style.format({"Probabilitas": "{:.2%}"}),
                use_container_width=True,
            )


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
    except ConfigError as exc:
        st.set_page_config(page_title="Konfigurasi bermasalah", page_icon="⚠️")
        st.error(f"Konfigurasi bermasalah:\n\n{exc}")
        st.stop()
        return

    st.set_page_config(page_title=config.app.title, page_icon=config.app.icon, layout="centered")

    try:
        model, labels = get_model_and_labels(str(config.model.path), str(config.model.labels_path))
    except ModelLoadError as exc:
        st.error(f"Model gagal dimuat:\n\n{exc}")
        st.stop()
        return

    st.title(f"{config.app.icon} {config.app.title}")
    if config.app.subtitle:
        st.caption(config.app.subtitle)

    render_sidebar(config, labels)

    uploaded = st.file_uploader(
        "Unggah gambar",
        type=list(config.ui.allowed_types),
        help=f"Format: {', '.join(config.ui.allowed_types)}. Maksimal {config.ui.max_upload_mb} MB.",
    )

    if uploaded is None:
        st.info("Silakan unggah gambar untuk memulai klasifikasi.")
        return

    size_mb = uploaded.size / BYTES_PER_MB
    if size_mb > config.ui.max_upload_mb:
        st.error(f"Ukuran file {size_mb:.1f} MB melebihi batas {config.ui.max_upload_mb} MB.")
        return

    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"File tidak bisa dibaca sebagai gambar: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)
    with right:
        with st.spinner("Menganalisis gambar..."):
            try:
                predictions = predict(model, image, labels, config.model, config.label_display)
            except Exception as exc:  # noqa: BLE001 - tampilkan kegagalan inferensi ke pengguna
                st.error(f"Inferensi gagal: {exc}")
                return
        render_results(predictions, config)


if __name__ == "__main__":
    main()
