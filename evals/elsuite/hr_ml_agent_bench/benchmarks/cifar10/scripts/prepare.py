from pathlib import Path

from torchvision import datasets

env_dir = Path(__file__).parent / ".." / "env"

train_dataset = datasets.CIFAR10(root=env_dir / "data", train=True, download=True)
test_dataset = datasets.CIFAR10(root=env_dir / "data", train=False, download=True)
