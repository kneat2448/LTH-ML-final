"""Mask enforcement in every training mode, and the LR / sharpness schedules."""
import pytest

from src.models import init_model
from src.prune import global_magnitude_prune, ones_mask
from src.train import Trainer, lr_at, sharpness_steps
from tests.conftest import tiny_config


@pytest.mark.parametrize("over", [{}, {"warmup_iters": 5}, {"sam_rho": 0.05}, {"anchor_lam": 1e-2}])
def test_masked_weights_and_momentum_are_zero_after_10_steps(tiny_data, over):
    cfg = tiny_config(**over)
    model = init_model("resnet20", 0)
    w = {n: p.detach() for n, p in model.named_parameters()}
    mask = global_magnitude_prune(w, ones_mask(model), 0.5, "fc.weight", 0.5)
    trainer = Trainer(cfg, tiny_data, model, mask)
    rec = trainer.train(seed=0)
    assert len(rec["early_losses"]) == 10  # 10 optimizer steps
    params = dict(model.named_parameters())
    for n, m in mask.items():
        assert (params[n][m == 0] == 0).all()
        assert (trainer.opt.state[params[n]]["momentum_buffer"][m == 0] == 0).all()
    assert [r["step"] for r in rec["sharpness"]] == [0, 5, 10]


def test_lr_schedule():
    cfg = tiny_config(lr=0.03, warmup_iters=10000, milestones=[10000, 12500])
    assert lr_at(cfg, 0) == pytest.approx(0.03 / 10000)
    assert lr_at(cfg, 9999) == pytest.approx(0.03)
    assert lr_at(cfg, 11000) == pytest.approx(0.003)
    assert lr_at(cfg, 13000) == pytest.approx(0.0003)


def test_sharpness_schedule():
    cfg = tiny_config(sharpness={"batch": 2048, "iters": 20, "micro_batch": 512, "every": 250,
                                 "dense_until": 3000, "then_every": 1500})
    steps = sharpness_steps(cfg, 15000)
    assert steps[:13] == list(range(0, 3001, 250))
    assert steps[13:] == [4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000]
