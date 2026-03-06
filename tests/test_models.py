"""Tests for model.py — construction and forward passes."""

import pytest
import torch
from model import MLP, ResNet18, ViT


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_default_construction(self):
        model = MLP()
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_custom_input_size(self):
        model = MLP(input_size=3072, hidden_sizes=[256, 128], num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_mnist_forward(self):
        model = MLP(input_size=784, hidden_sizes=[64, 32], num_classes=10)
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_single_hidden_layer(self):
        model = MLP(input_size=784, hidden_sizes=[128], num_classes=10)
        out = model(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 10)

    def test_deep_architecture(self):
        model = MLP(input_size=784, hidden_sizes=[128, 64, 32, 16], num_classes=10)
        out = model(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 10)

    def test_num_classes(self):
        model = MLP(input_size=12288, hidden_sizes=[64], num_classes=200)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 200)

    def test_batch_size_one(self):
        model = MLP(input_size=784, hidden_sizes=[64], num_classes=10)
        out = model(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# ResNet18
# ---------------------------------------------------------------------------

class TestResNet18:
    def test_rgb_cifar(self):
        model = ResNet18(in_channels=3, num_classes=10)
        out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_grayscale_mnist(self):
        model = ResNet18(in_channels=1, num_classes=10)
        out = model(torch.randn(4, 1, 28, 28))
        assert out.shape == (4, 10)

    def test_200_classes(self):
        model = ResNet18(in_channels=3, num_classes=200)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 200)

    def test_conv1_adapted(self):
        import torch.nn as nn
        model = ResNet18(in_channels=3, num_classes=10)
        assert model.model.conv1.kernel_size == (3, 3)
        assert model.model.conv1.stride == (1, 1)
        assert isinstance(model.model.maxpool, nn.Identity)


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------

class TestViT:
    def test_mnist_28x28(self):
        model = ViT(image_size=28, in_channels=1, patch_size=4, num_classes=10,
                    dim=64, depth=2, heads=4, mlp_dim=128)
        out = model(torch.randn(4, 1, 28, 28))
        assert out.shape == (4, 10)

    def test_cifar_32x32(self):
        model = ViT(image_size=32, in_channels=3, patch_size=4, num_classes=10,
                    dim=64, depth=2, heads=4, mlp_dim=128)
        out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_tiny_imagenet_64x64(self):
        model = ViT(image_size=64, in_channels=3, patch_size=4, num_classes=200,
                    dim=64, depth=2, heads=4, mlp_dim=128)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 200)

    def test_patch_size_assertion(self):
        with pytest.raises(AssertionError):
            ViT(image_size=28, patch_size=6)  # 28 not divisible by 6

    def test_num_patches(self):
        # 32 / 4 = 8, so 8x8 = 64 patches + 1 CLS = 65 tokens
        model = ViT(image_size=32, patch_size=4, dim=64, depth=1, heads=4, mlp_dim=128)
        assert model.pos_embedding.shape == (1, 65, 64)

    def test_batch_size_one(self):
        model = ViT(image_size=28, in_channels=1, patch_size=4, num_classes=10,
                    dim=32, depth=1, heads=4, mlp_dim=64)
        out = model(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)
