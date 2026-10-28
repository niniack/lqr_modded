"""Composable models for latent-state prediction."""

from .aikae import AIKAE
from .koopman_ae import KoopmanAE
from .layerwise_operator import LayerwiseKoopmanOperator
from .koopman_operator import KoopmanOperator
from .residual_layerwise_operator import ResidualLayerwiseKoopmanOperator
from .shared_modes_operator import SharedModesKoopmanOperator
from .rollout import StudentRollout

__all__ = [
    "AIKAE",
    "KoopmanAE",
    "LayerwiseKoopmanOperator",
    "KoopmanOperator",
    "ResidualLayerwiseKoopmanOperator",
    "SharedModesKoopmanOperator",
    "StudentRollout",
]
