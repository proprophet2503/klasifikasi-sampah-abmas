"""Pemuatan model Keras dan file label.

Model hasil ekspor Teachable Machine dibuat dengan Keras 2. Keras 3 menolak
memuatnya karena layer DepthwiseConv2D menyimpan argumen `groups` yang sudah
dihapus. Loader di bawah mencoba beberapa strategi secara berurutan sehingga
template ini jalan baik di lingkungan Keras 2 maupun Keras 3.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Kurangi log TensorFlow agar output terminal bersih. Harus di-set sebelum impor TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class ModelLoadError(Exception):
    """Model gagal dimuat."""


def _patched_custom_objects() -> dict[str, Any]:
    """Custom object yang membuang argumen tak dikenal dari model Keras 2."""
    from keras.layers import DepthwiseConv2D

    class CompatDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    return {"DepthwiseConv2D": CompatDepthwiseConv2D}


def load_model(model_path: str | Path):
    """Muat model Keras dari file .h5 atau .keras, atau folder SavedModel."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelLoadError(f"File model tidak ditemukan: {model_path}")

    errors: list[str] = []

    # Strategi 1: tf_keras (Keras 2). Paling cocok untuk ekspor Teachable Machine.
    try:
        import tf_keras

        return tf_keras.models.load_model(str(model_path), compile=False)
    except ImportError:
        errors.append("tf_keras tidak terpasang")
    except Exception as exc:  # noqa: BLE001 - kumpulkan sebab, coba strategi berikutnya
        errors.append(f"tf_keras gagal: {exc}")

    # Strategi 2: keras bawaan (Keras 3) dengan patch kompatibilitas.
    try:
        import keras

        return keras.models.load_model(
            str(model_path), compile=False, custom_objects=_patched_custom_objects()
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"keras gagal: {exc}")

    # Strategi 3: keras bawaan tanpa patch.
    try:
        import keras

        return keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"keras (tanpa patch) gagal: {exc}")

    raise ModelLoadError(
        "Model tidak bisa dimuat oleh semua strategi yang tersedia.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nUntuk model Teachable Machine, pasang tf-keras:  pip install tf-keras"
    )


def load_labels(labels_path: str | Path) -> list[str]:
    """Baca labels.txt menjadi daftar nama kelas.

    Menerima dua format:
      "0 Organic_Full"  -> nomor indeks di depan dibuang
      "Organic_Full"    -> dipakai apa adanya
    Baris kosong diabaikan.
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise ModelLoadError(f"File label tidak ditemukan: {labels_path}")

    labels: list[str] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        head, _, tail = text.partition(" ")
        labels.append(tail.strip() if head.isdigit() and tail.strip() else text)

    if not labels:
        raise ModelLoadError(f"File label kosong: {labels_path}")
    return labels


def validate_model_labels(model, labels: list[str]) -> None:
    """Pastikan jumlah label sama dengan jumlah output model."""
    try:
        output_units = int(model.output_shape[-1])
    except (AttributeError, TypeError, IndexError):
        return  # bentuk output tidak terbaca, lewati validasi

    if output_units != len(labels):
        raise ModelLoadError(
            f"Jumlah label ({len(labels)}) tidak cocok dengan jumlah output model ({output_units}). "
            "Periksa 'model.labels_path' di config.yaml."
        )
