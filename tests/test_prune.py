import torch

from src.models import init_model
from src.prune import global_magnitude_prune, ones_mask, output_name, prunable_names


def _weights(model):
    return {n: p.detach() for n, p in model.named_parameters()}


def test_excluded_layers():
    model = init_model("resnet20", 0)
    names = prunable_names(model)
    assert "conv1.weight" not in names  # first conv never pruned
    assert "fc.weight" in names and output_name(model) == "fc.weight"
    assert all(n.endswith("weight") and "bn" not in n for n in names)


def test_global_prune_fraction_exact():
    model = init_model("resnet20", 0)
    weights, mask = _weights(model), ones_mask(model)
    body = [n for n in mask if n != "fc.weight"]
    for _ in range(3):  # several rounds, pruning among survivors
        alive_body = sum(mask[n].sum().item() for n in body)
        alive_fc = mask["fc.weight"].sum().item()
        new = global_magnitude_prune(weights, mask, 0.3, "fc.weight", 0.5)
        assert abs(sum(new[n].sum().item() for n in body) - 0.7 * alive_body) <= 1
        assert abs(new["fc.weight"].sum().item() - 0.85 * alive_fc) <= 1  # half rate
        assert all(((new[n] == 1) <= (mask[n] == 1)).all() for n in mask)  # only removes
        mask = new


def test_prune_removes_smallest():
    model = init_model("resnet20", 0)
    weights = _weights(model)
    new = global_magnitude_prune(weights, ones_mask(model), 0.3, "fc.weight", 0.5)
    body = [n for n in new if n != "fc.weight"]
    kept = torch.cat([weights[n][new[n].bool()].abs() for n in body])
    gone = torch.cat([weights[n][~new[n].bool()].abs() for n in body])
    assert kept.min() >= gone.max()


def test_refuses_nan_weights():
    model = init_model("resnet20", 0)
    weights = _weights(model)
    weights["fc.weight"] = weights["fc.weight"] * float("nan")
    try:
        global_magnitude_prune(weights, ones_mask(model), 0.3, "fc.weight", 0.5)
    except ValueError:
        return
    raise AssertionError("a mask was built from non-finite weights")
