"""Fit a causal latent forecast to generated normal conveyor video on CPU."""

from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from beltwatch.data import CONTEXT

_FRAME_SHAPE = (32, 48)
_LATENT_DIM = 16
_BATCH_SIZE = 8
SIGREG_WEIGHT = 1.0


class BeltModel(nn.Module):
    """Encode each frame independently and predict from the previous four embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 4 * 6, _LATENT_DIM),
        )
        self.predictor = nn.Sequential(
            nn.Linear(CONTEXT * _LATENT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, _LATENT_DIM),
        )
        last_layer = self.predictor[-1]
        assert isinstance(last_layer, nn.Linear) and last_layer.bias is not None
        nn.init.zeros_(last_layer.bias)
        nn.init.normal_(last_layer.weight, std=0.005)

    def encode(self, clips: Tensor) -> Tensor:
        """Return one embedding per grayscale frame without mixing clips or times."""
        batch, length = clips.shape[:2]
        frames = clips.reshape(-1, 1, *_FRAME_SHAPE) - 0.5
        return self.encoder(frames).reshape(batch, length, _LATENT_DIM)

    def forward(self, clips: Tensor) -> tuple[Tensor, Tensor]:
        """Return forecasts aligned to frames from CONTEXT onward, and all embeddings."""
        embeddings = self.encode(clips)
        return self._predict(embeddings), embeddings

    def _predict(self, embeddings: Tensor) -> Tensor:
        windows = embeddings[:, :-1].unfold(1, CONTEXT, 1).transpose(-1, -2)
        return windows[:, :, -1] + self.predictor(windows.flatten(start_dim=2))


def _prediction_errors(forecasts: Tensor, embeddings: Tensor) -> Tensor:
    # The observed target remains in the prediction loss's gradient path.
    return (forecasts - embeddings[:, CONTEXT:]).square().mean(dim=-1)


def _sigreg(samples: Tensor, *, generator: torch.Generator) -> Tensor:
    """Estimate a projected Gaussian distance using windowed Epps–Pulley quadrature."""
    directions = F.normalize(torch.randn(samples.size(-1), 32, dtype=samples.dtype, generator=generator), dim=0)
    frequencies = torch.linspace(0.0, 3.0, 17, dtype=samples.dtype)
    gaussian = (-frequencies.square() / 2).exp()
    phase = (samples @ directions).unsqueeze(-1) * frequencies
    error = (phase.cos().mean(dim=0) - gaussian).square() + phase.sin().mean(dim=0).square()
    # The integrand is even. Average the distances to keep the weight independent of batch size.
    return 2 * torch.trapezoid(error * gaussian, frequencies).mean()


@torch.inference_mode()
def frame_scores(model: BeltModel, clips: Tensor) -> Tensor:
    """Score surprising observations using mean squared latent forecast error."""
    return _prediction_errors(*model(clips))


@torch.inference_mode()
def latent_spread(model: BeltModel, clips: Tensor) -> float:
    """Return the mean standard deviation of the latent channels across frames."""
    embeddings = torch.cat([model.encode(batch).flatten(0, 1) for batch in clips.split(_BATCH_SIZE)])
    return embeddings.std(dim=0, correction=0).mean().item()


def fit_model(
    clips: Tensor, *, epochs: int, seed: int, sigreg_weight: float = SIGREG_WEIGHT
) -> tuple[BeltModel, list[dict[str, float]]]:
    """Train the encoder and predictor jointly, returning sample-weighted epoch losses."""
    if clips.ndim != 4 or clips.shape[2:] != _FRAME_SHAPE or len(clips) == 0 or clips.shape[1] <= CONTEXT:
        raise ValueError("training requires clips shaped [B,T,32,48] with B > 0 and T > CONTEXT")
    if clips.device.type != "cpu" or clips.dtype != torch.float32:
        raise ValueError("training requires float32 CPU clips")
    if epochs < 1 or not isfinite(sigreg_weight) or sigreg_weight < 0:
        raise ValueError("epochs must be positive and sigreg_weight must be finite and nonnegative")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = BeltModel()
    shuffle = torch.Generator().manual_seed(seed)
    sketch = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history: list[dict[str, float]] = []

    for _ in range(epochs):
        prediction_sum, sigreg_sum, total_sum = 0.0, 0.0, 0.0
        for indices in torch.randperm(len(clips), generator=shuffle).split(_BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            forecasts, embeddings = model(clips[indices])
            prediction = _prediction_errors(forecasts, embeddings).mean()
            sigreg = _sigreg(embeddings.flatten(0, 1), generator=sketch) if sigreg_weight else prediction.new_zeros(())
            total = prediction + sigreg_weight * sigreg
            total.backward()
            optimizer.step()

            prediction_sum += prediction.item() * len(indices)
            sigreg_sum += sigreg.item() * len(indices)
            total_sum += total.item() * len(indices)
        history.append(
            {
                "prediction": prediction_sum / len(clips),
                "sigreg": sigreg_sum / len(clips),
                "total": total_sum / len(clips),
            }
        )

    return model.eval(), history
