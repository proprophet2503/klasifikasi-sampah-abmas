"""Pemuatan dan validasi config.yaml.

Semua nilai divalidasi di sini agar kesalahan konfigurasi gagal cepat dengan
pesan yang jelas, bukan meledak di tengah inferensi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NORMALIZATIONS = ("teachable_machine", "rescale", "none")
RESIZE_MODES = ("center_crop", "stretch", "pad")


class ConfigError(Exception):
    """Konfigurasi tidak valid."""


@dataclass(frozen=True)
class AppConfig:
    title: str
    subtitle: str
    icon: str
    about: str


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    labels_path: Path
    input_size: tuple[int, int]
    normalization: str
    resize_mode: str
    apply_softmax: bool


@dataclass(frozen=True)
class InferenceConfig:
    top_k: int
    confidence_threshold: float


@dataclass(frozen=True)
class UIConfig:
    show_all_probabilities: bool
    max_upload_mb: int
    allowed_types: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    app: AppConfig
    model: ModelConfig
    inference: InferenceConfig
    ui: UIConfig
    label_display: dict[str, str] = field(default_factory=dict)


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"config.yaml: kunci '{key}' wajib ada di bagian '{where}'.")
    return section[key]


def _as_path(base_dir: Path, raw: Any, where: str) -> Path:
    path = Path(str(raw))
    resolved = path if path.is_absolute() else (base_dir / path)
    if not resolved.exists():
        raise ConfigError(f"config.yaml: '{where}' menunjuk ke file yang tidak ada: {resolved}")
    return resolved


def _as_input_size(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError("config.yaml: 'model.input_size' harus berupa dua angka, contoh [224, 224].")
    height, width = raw
    if not (isinstance(height, int) and isinstance(width, int)) or height <= 0 or width <= 0:
        raise ConfigError("config.yaml: 'model.input_size' harus bilangan bulat positif.")
    return (height, width)


def _as_choice(raw: Any, options: tuple[str, ...], where: str) -> str:
    value = str(raw)
    if value not in options:
        raise ConfigError(f"config.yaml: '{where}' harus salah satu dari {list(options)}, bukan '{value}'.")
    return value


def load_config(config_path: str | Path) -> Config:
    """Baca config.yaml dan kembalikan objek Config yang sudah tervalidasi."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"File konfigurasi tidak ditemukan: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml tidak bisa dibaca (format YAML salah): {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml harus berupa mapping (key: value) di level teratas.")

    base_dir = config_path.parent
    app_raw = raw.get("app", {}) or {}
    model_raw = raw.get("model", {}) or {}
    inference_raw = raw.get("inference", {}) or {}
    ui_raw = raw.get("ui", {}) or {}
    label_display = raw.get("label_display") or {}

    if not isinstance(label_display, dict):
        raise ConfigError("config.yaml: 'label_display' harus berupa mapping 'nama_asli: nama_tampilan'.")

    threshold = float(inference_raw.get("confidence_threshold", 0.0))
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("config.yaml: 'inference.confidence_threshold' harus antara 0.0 dan 1.0.")

    top_k = int(inference_raw.get("top_k", 3))
    if top_k < 1:
        raise ConfigError("config.yaml: 'inference.top_k' minimal 1.")

    return Config(
        app=AppConfig(
            title=str(app_raw.get("title", "Klasifikasi Gambar")),
            subtitle=str(app_raw.get("subtitle", "")),
            icon=str(app_raw.get("icon", "🖼️")),
            about=str(app_raw.get("about", "")),
        ),
        model=ModelConfig(
            path=_as_path(base_dir, _require(model_raw, "path", "model"), "model.path"),
            labels_path=_as_path(base_dir, _require(model_raw, "labels_path", "model"), "model.labels_path"),
            input_size=_as_input_size(model_raw.get("input_size", [224, 224])),
            normalization=_as_choice(
                model_raw.get("normalization", "teachable_machine"), NORMALIZATIONS, "model.normalization"
            ),
            resize_mode=_as_choice(
                model_raw.get("resize_mode", "center_crop"), RESIZE_MODES, "model.resize_mode"
            ),
            apply_softmax=bool(model_raw.get("apply_softmax", False)),
        ),
        inference=InferenceConfig(top_k=top_k, confidence_threshold=threshold),
        ui=UIConfig(
            show_all_probabilities=bool(ui_raw.get("show_all_probabilities", True)),
            max_upload_mb=int(ui_raw.get("max_upload_mb", 10)),
            allowed_types=tuple(str(t).lower() for t in ui_raw.get("allowed_types", ["jpg", "jpeg", "png"])),
        ),
        label_display={str(k): str(v) for k, v in label_display.items()},
    )
