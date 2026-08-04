"""Praproses gambar dan inferensi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from core.config import ModelConfig


@dataclass(frozen=True)
class Prediction:
    label: str
    display_label: str
    probability: float


def _resize(image: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    """Sesuaikan ukuran gambar. `size` dalam urutan (tinggi, lebar)."""
    height, width = size
    target = (width, height)  # PIL memakai urutan (lebar, tinggi)

    if mode == "stretch":
        return image.resize(target, Image.Resampling.LANCZOS)
    if mode == "pad":
        return ImageOps.pad(image, target, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    # center_crop: potong bagian tengah lalu skala, sama seperti Teachable Machine
    return ImageOps.fit(image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _normalize(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "teachable_machine":
        return (array / 127.5) - 1.0
    if mode == "rescale":
        return array / 255.0
    return array


def preprocess_image(image: Image.Image, config: ModelConfig) -> np.ndarray:
    """Ubah PIL Image menjadi batch numpy siap masuk model, bentuk (1, H, W, 3)."""
    rgb = image.convert("RGB")
    resized = _resize(rgb, config.input_size, config.resize_mode)
    array = np.asarray(resized, dtype=np.float32)
    normalized = _normalize(array, config.normalization)
    return np.expand_dims(normalized, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent)


def predict(
    model,
    image: Image.Image,
    labels: list[str],
    config: ModelConfig,
    label_display: dict[str, str] | None = None,
) -> list[Prediction]:
    """Jalankan inferensi, kembalikan daftar Prediction terurut dari yang tertinggi."""
    label_display = label_display or {}
    batch = preprocess_image(image, config)
    raw = np.asarray(model.predict(batch, verbose=0))[0].astype(np.float64)

    probabilities = _softmax(raw) if config.apply_softmax else raw
    total = float(np.sum(probabilities))
    if total > 0 and not np.isclose(total, 1.0, atol=1e-3):
        probabilities = probabilities / total  # jaga-jaga output belum ternormalisasi

    ranked = [
        Prediction(
            label=label,
            display_label=label_display.get(label, label),
            probability=float(probability),
        )
        for label, probability in zip(labels, probabilities)
    ]
    return sorted(ranked, key=lambda p: p.probability, reverse=True)
