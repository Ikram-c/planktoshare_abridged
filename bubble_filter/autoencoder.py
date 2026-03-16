# pre_process/bubble_filter/autoencoder.py
import logging
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

add_safe_globals([StandardScaler, list, str])

logger = logging.getLogger(__name__)

_checkpoint_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _detect_device() -> str:
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        logger.info(
            "CUDA device detected: %s", torch.cuda.get_device_name(torch.cuda.current_device())
        )
        return device
    if torch.backends.mps.is_available():
        logger.info("Apple MPS device detected")
        return "mps"
    logger.info("No GPU detected, using CPU")
    return "cpu"


_DEVICE: str = _detect_device()


class SupervisedAutoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        hidden = (64, 32)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden[1], latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden[1], hidden[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden[0], input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def __repr__(self) -> str:
        counts = {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in (
                ("encoder", self.encoder),
                ("decoder", self.decoder),
                ("classifier", self.classifier),
            )
        }
        parts = ", ".join(f"{k}_params={v}" for k, v in counts.items())
        return f"SupervisedAutoencoder({parts})"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), self.classifier(z), z

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def load_checkpoint(
    checkpoint_path: str, device: str = "auto"
) -> tuple[SupervisedAutoencoder, StandardScaler, list[str]]:
    resolved_device = _DEVICE if device == "auto" else device
    cache_key = f"{checkpoint_path}:{resolved_device}"

    with _cache_lock:
        if cache_key in _checkpoint_cache:
            logger.debug("Checkpoint cache hit: %s", cache_key)
            return _checkpoint_cache[cache_key]

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Verify the checkpoint_path in your config, or retrain the model."
        )

    ckpt: dict = torch.load(path, map_location=resolved_device, weights_only=True)
    feature_names: list[str] = ckpt["features"]
    scaler: StandardScaler = ckpt["scaler"]
    latent_dim = ckpt.get("latent_dim", 8)

    model = SupervisedAutoencoder(input_dim=len(feature_names), latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(resolved_device).eval()

    logger.info(
        "Loaded autoencoder: %d features, latent_dim=%d, device=%s",
        len(feature_names), latent_dim, resolved_device,
    )
    result = (model, scaler, feature_names)
    with _cache_lock:
        _checkpoint_cache[cache_key] = result
    return result


def _safe_float(v: float) -> float:
    return 0.0 if (np.isnan(v) or np.isinf(v)) else float(v)


def predict(
    feature_dict: dict[str, float],
    model: SupervisedAutoencoder,
    scaler: StandardScaler,
    feature_names: list[str],
) -> float:
    vector = np.array(
        [_safe_float(feature_dict.get(name, 0.0)) for name in feature_names],
        dtype=np.float32,
    ).reshape(1, -1)

    x_tensor = torch.from_numpy(scaler.transform(vector))
    with torch.no_grad():
        return model.classify(x_tensor).item()