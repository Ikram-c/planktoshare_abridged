import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SupervisedAutoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        y_pred = self.classifier(z)
        return x_recon, y_pred, z

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def load_checkpoint(
        checkpoint_path: str, device: str = "cpu"
) -> tuple[SupervisedAutoencoder, StandardScaler, list[str]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    feature_names: list[str] = ckpt["features"]
    scaler: StandardScaler = ckpt["scaler"]
    input_dim = len(feature_names)
    latent_dim = ckpt.get("latent_dim", 8)

    model = SupervisedAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded autoencoder: %d features, latent_dim=%d, device=%s",
        input_dim, latent_dim, device,
    )
    return model, scaler, feature_names


def predict(
        feature_dict: dict[str, float],
        model: SupervisedAutoencoder,
        scaler: StandardScaler,
        feature_names: list[str],
) -> float:
    vector = []
    for name in feature_names:
        value = feature_dict.get(name, 0.0)
        if np.isnan(value) or np.isinf(value):
            value = 0.0
        vector.append(value)

    x = np.array([vector], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.from_numpy(x_scaled)

    with torch.no_grad():
        score = model.classify(x_tensor).item()
    return score
