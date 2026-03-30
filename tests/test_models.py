"""Tests for model.py — construction and forward passes."""

import pytest
import torch
from model import MLP, ResNet18, ViT, GPT


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


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class TestGPT:
    def _make_model(self, vocab_size=64, seq_len=32, dim=32, depth=2, heads=4):
        return GPT(vocab_size=vocab_size, seq_len=seq_len, dim=dim,
                   depth=depth, heads=heads, mlp_dim=64, dropout=0.0)

    def test_output_shape(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 64)

    def test_shorter_sequence(self):
        model = self._make_model(seq_len=32)
        x = torch.randint(0, 64, (3, 16))
        out = model(x)
        assert out.shape == (3, 16, 64)

    def test_batch_size_one(self):
        model = self._make_model()
        x = torch.randint(0, 64, (1, 32))
        out = model(x)
        assert out.shape == (1, 32, 64)

    def test_output_is_finite(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 32))
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_causal_mask_no_future_leakage(self):
        """Token at position 0 should not depend on tokens at positions 1+."""
        model = self._make_model()
        model.eval()
        x1 = torch.randint(0, 64, (1, 32))
        x2 = x1.clone()
        x2[0, 1:] = torch.randint(0, 64, (31,))  # change all tokens after pos 0

        out1 = model(x1)
        out2 = model(x2)
        # Position 0 output should be identical regardless of future tokens
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), \
            "Causal mask violation: position 0 output changed when future tokens changed"

    def test_sequence_too_long_raises(self):
        model = self._make_model(seq_len=16)
        x = torch.randint(0, 64, (1, 17))
        with pytest.raises(AssertionError):
            model(x)

    def test_weight_tying(self):
        model = self._make_model()
        # LM head weights should be the same object as token embeddings
        assert model.head.weight is model.token_emb.weight

    def test_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(AssertionError):
            GPT(vocab_size=64, seq_len=16, dim=32, heads=3)  # 32 % 3 != 0

    def test_parameter_count_reasonable(self):
        model = self._make_model(vocab_size=256, seq_len=128, dim=128,
                                  depth=4, heads=4)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be between 1M and 50M for these settings
        assert 100_000 < n_params < 50_000_000
