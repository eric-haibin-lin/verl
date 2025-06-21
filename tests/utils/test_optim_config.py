import torch
import torch.optim as optim
from omegaconf import OmegaConf

from verl.trainer.optimizer import OptimConfig
from verl.utils import omega_conf_to_dataclass


def test_optim_config_creation():
    """Test creating OptimConfig from OmegaConf DictConfig"""
    config_dict = {"lr": 1e-4, "weight_decay": 0.01, "betas": (0.9, 0.999), "clip_grad": 1.0, "lr_scheduler": "cosine", "warmup_steps_ratio": 0.1, "total_training_steps": 1000, "lr_warmup_steps": 100, "min_lr_ratio": 0.1, "num_cycles": 0.5, "warmup_style": "linear"}

    omega_config = OmegaConf.create(config_dict)
    optim_config = omega_conf_to_dataclass(omega_config, OptimConfig)

    assert optim_config.lr == 1e-4
    assert optim_config.weight_decay == 0.01
    assert optim_config.betas == [0.9, 0.999]  # OmegaConf converts tuples to lists
    assert optim_config.clip_grad == 1.0
    assert optim_config.lr_scheduler == "cosine"
    assert optim_config.warmup_steps_ratio == 0.1
    assert optim_config.total_training_steps == 1000
    assert optim_config.lr_warmup_steps == 100
    assert optim_config.min_lr_ratio == 0.1
    assert optim_config.num_cycles == 0.5
    assert optim_config.warmup_style == "linear"


def test_optim_config_defaults():
    """Test OptimConfig with default values"""
    config_dict = {"lr": 2e-5}
    omega_config = OmegaConf.create(config_dict)
    optim_config = omega_conf_to_dataclass(omega_config, OptimConfig)

    assert optim_config.lr == 2e-5
    assert optim_config.weight_decay == 0.01  # default
    assert optim_config.betas == [0.9, 0.999]  # default, OmegaConf converts tuples to lists
    assert optim_config.clip_grad == 1.0  # default
    assert optim_config.lr_scheduler == "cosine"  # default


def test_pytorch_optimizer_creation():
    """Test creating PyTorch optimizers using OptimConfig"""
    config_dict = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.95)}

    omega_config = OmegaConf.create(config_dict)
    optim_config = omega_conf_to_dataclass(omega_config, OptimConfig)

    model = torch.nn.Linear(10, 1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=optim_config.lr,
        weight_decay=optim_config.weight_decay,
        betas=tuple(optim_config.betas),  # Convert list back to tuple for PyTorch
    )

    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)  # PyTorch converts list back to tuple


def test_pytorch_sgd_optimizer_creation():
    """Test creating SGD optimizer using OptimConfig"""
    config_dict = {"lr": 0.01, "weight_decay": 1e-4}

    omega_config = OmegaConf.create(config_dict)
    optim_config = omega_conf_to_dataclass(omega_config, OptimConfig)

    model = torch.nn.Linear(5, 2)

    optimizer = optim.SGD(model.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay)

    assert isinstance(optimizer, optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_gradient_clipping_with_optim_config():
    """Test gradient clipping using OptimConfig"""
    config_dict = {"lr": 1e-3, "clip_grad": 0.5}

    omega_config = OmegaConf.create(config_dict)
    optim_config = omega_conf_to_dataclass(omega_config, OptimConfig)

    model = torch.nn.Linear(3, 1)
    optimizer = optim.Adam(model.parameters(), lr=optim_config.lr)

    x = torch.randn(10, 3)
    y = torch.randn(10, 1)
    loss = torch.nn.functional.mse_loss(model(x), y)

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optim_config.clip_grad)

    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm.item() >= 0

    optimizer.zero_grad()


if __name__ == "__main__":
    test_optim_config_creation()
    test_optim_config_defaults()
    test_pytorch_optimizer_creation()
    test_pytorch_sgd_optimizer_creation()
    test_gradient_clipping_with_optim_config()
    print("All tests passed!")
