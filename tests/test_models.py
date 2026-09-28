import torch

from src.models import init_model


def test_resnet20_parameter_count():
    """He et al. 2016 CIFAR ResNet-20 with BN and option-A shortcuts has ~0.27M parameters."""
    model = init_model("resnet20", 0)
    n = sum(p.numel() for p in model.parameters())
    assert n == 269_722
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_theta0_reload_is_bit_exact(tmp_path):
    path = tmp_path / "theta_0.pt"
    model = init_model("resnet20", 3, save_path=path)
    again = init_model("resnet20", 3)
    loaded = torch.load(path)
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded[k])
        assert torch.equal(v, again.state_dict()[k])  # same seed -> same init
