import pytest
import torch
from torch import Tensor

from beltwatch.data import CONTEXT, make_split
from beltwatch.model import BeltModel, _prediction_errors, _sigreg, fit_model, frame_scores, latent_spread


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _clips() -> Tensor:
    return torch.rand((3, CONTEXT + 3, 32, 48), generator=torch.Generator().manual_seed(42))


def test_forecast_cannot_read_target_or_future_pixels() -> None:
    model = BeltModel().eval()
    original = _clips()
    changed = original.clone()
    changed[:, CONTEXT:] = 0

    with torch.no_grad():
        first, _ = model(original)
        second, _ = model(changed)
    torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)

    scores = frame_scores(model, original)
    assert scores.shape == (3, 3)
    assert not scores.requires_grad
    assert not torch.equal(scores[:, 0], frame_scores(model, changed)[:, 0])


def test_prediction_loss_reaches_both_context_and_observed_target() -> None:
    model = BeltModel()
    clips = _clips().requires_grad_()
    (gradient,) = torch.autograd.grad(_prediction_errors(*model(clips))[:, 0].sum(), clips)

    assert gradient[:, :CONTEXT].abs().sum() > 0
    assert gradient[:, CONTEXT].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, CONTEXT + 1 :]) == 0


def test_sigreg_penalizes_collapsed_and_anisotropic_embeddings() -> None:
    normal = torch.randn((2048, 8), generator=torch.Generator().manual_seed(21)).requires_grad_()
    anisotropic = normal.detach()[:, :1] * torch.tensor([8**0.5, 0, 0, 0, 0, 0, 0, 0])

    def penalty(samples: Tensor) -> Tensor:
        return _sigreg(samples, generator=torch.Generator().manual_seed(37))

    loss = penalty(normal)
    assert 0 < loss < 0.01
    assert penalty(anisotropic) > 0.03
    assert penalty(torch.zeros_like(normal)) > 0.1
    loss.backward()
    assert normal.grad is not None and normal.grad.abs().sum() > 0


def test_seeded_training_repeats_and_preserves_caller_rng() -> None:
    clips = _clips()
    original_rng = torch.random.get_rng_state()
    first, history = fit_model(clips, epochs=2, seed=199)
    second, repeated = fit_model(clips, epochs=2, seed=199)

    assert torch.equal(torch.random.get_rng_state(), original_rng)
    assert history == repeated
    assert len(history) == 2
    for row in history:
        assert set(row) == {"prediction", "sigreg", "total"}
        assert row["prediction"] > 0 and row["sigreg"] > 0
        assert row["total"] == pytest.approx(row["prediction"] + row["sigreg"])
    torch.testing.assert_close(frame_scores(first, clips), frame_scores(second, clips), rtol=0, atol=0)


def test_zero_weight_disables_regularization() -> None:
    clips = _clips()
    ablated, [history] = fit_model(clips, epochs=1, seed=199, sigreg_weight=0)
    full, _ = fit_model(clips, epochs=1, seed=199)

    assert history["prediction"] == history["total"]
    assert history["sigreg"] == 0
    assert latent_spread(ablated, clips) > 0
    assert latent_spread(ablated, torch.zeros_like(clips)) == pytest.approx(0, abs=1e-7)
    assert not torch.equal(frame_scores(ablated, clips), frame_scores(full, clips))


def test_regularized_embeddings_retain_variation_on_nominal_conveyor() -> None:
    train = make_split("train", 24, seed=2026)
    calibration = make_split("calibration", 8, seed=2026)
    training = torch.stack([torch.from_numpy(clip.frames) for clip in train]).float().div(255)
    held_out = torch.stack([torch.from_numpy(clip.frames) for clip in calibration]).float().div(255)
    model, _ = fit_model(training, epochs=8, seed=2026)
    assert latent_spread(model, held_out) > 0.1
