import torch

from src.baselines import reinit, shuffle
from src.models import init_model
from src.prune import global_magnitude_prune, ones_mask


def _theta0_and_mask():
    model = init_model("resnet20", 0)
    w = {n: p.detach() for n, p in model.named_parameters()}
    return model.state_dict(), global_magnitude_prune(w, ones_mask(model), 0.5, "fc.weight", 0.5)


def test_shuffle_preserves_layer_multiset():
    theta0, mask = _theta0_and_mask()
    s = shuffle(theta0, mask, seed=0)
    for n, m in mask.items():
        keep = m.bool()
        assert torch.equal(torch.sort(s[n][keep]).values, torch.sort(theta0[n][keep]).values)
        assert (s[n][~keep] == 0).all()
        assert not torch.equal(s[n], theta0[n] * m)  # actually permuted
    assert torch.equal(s["conv1.weight"], theta0["conv1.weight"])  # unpruned layers untouched


def test_reinit_differs_and_is_masked():
    theta0, mask = _theta0_and_mask()
    r = reinit("resnet20", 0, mask)
    for n, m in mask.items():
        assert (r[n][~m.bool()] == 0).all()
        assert not torch.equal(r[n], theta0[n] * m)
