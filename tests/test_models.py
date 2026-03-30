"""Tests for model.py — construction and forward passes."""

import pytest
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# MLP — additional tests
# ---------------------------------------------------------------------------

class TestMLPAdditional:
    def test_output_is_finite(self):
        model = MLP(input_size=784, hidden_sizes=[64, 32], num_classes=10)
        out = model(torch.randn(4, 1, 28, 28))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_no_hidden_layers(self):
        # Empty hidden_sizes → direct input→output linear
        model = MLP(input_size=784, hidden_sizes=[], num_classes=10)
        out = model(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 10)

    def test_gradients_flow(self):
        model = MLP(input_size=784, hidden_sizes=[64], num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_train_eval_mode_toggle(self):
        model = MLP(input_size=784, hidden_sizes=[64], num_classes=10)
        model.train()
        assert model.training
        model.eval()
        assert not model.training

    def test_net_is_sequential(self):
        model = MLP(input_size=784, hidden_sizes=[64, 32], num_classes=10)
        assert isinstance(model.net, nn.Sequential)

    def test_correct_number_of_linear_layers(self):
        # hidden_sizes=[64, 32] → 3 Linear layers (input→64, 64→32, 32→10)
        model = MLP(input_size=784, hidden_sizes=[64, 32], num_classes=10)
        linears = [m for m in model.net if isinstance(m, nn.Linear)]
        assert len(linears) == 3

    def test_flat_input_accepted(self):
        # MLP should also work with a pre-flattened 2-D input
        model = MLP(input_size=784, hidden_sizes=[64], num_classes=10)
        out = model(torch.randn(4, 784))
        assert out.shape == (4, 10)


# ---------------------------------------------------------------------------
# ResNet18 — additional tests
# ---------------------------------------------------------------------------

class TestResNet18Additional:
    def test_output_is_finite(self):
        model = ResNet18(in_channels=3, num_classes=10)
        out = model(torch.randn(2, 3, 32, 32))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradients_flow(self):
        model = ResNet18(in_channels=3, num_classes=10)
        out = model(torch.randn(2, 3, 32, 32))
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_conv1_in_channels_respected(self):
        model = ResNet18(in_channels=1, num_classes=10)
        assert model.model.conv1.in_channels == 1

    def test_fc_out_features(self):
        model = ResNet18(in_channels=3, num_classes=42)
        assert model.model.fc.out_features == 42

    def test_train_eval_mode_toggle(self):
        model = ResNet18(in_channels=3, num_classes=10)
        model.eval()
        assert not model.training
        model.train()
        assert model.training

    def test_batch_size_one(self):
        model = ResNet18(in_channels=3, num_classes=10)
        out = model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# ViT — additional tests
# ---------------------------------------------------------------------------

class TestViTAdditional:
    def _make_model(self, **kwargs):
        defaults = dict(image_size=32, in_channels=3, patch_size=4,
                        num_classes=10, dim=64, depth=2, heads=4, mlp_dim=128)
        defaults.update(kwargs)
        return ViT(**defaults)

    def test_output_is_finite(self):
        model = self._make_model()
        out = model(torch.randn(2, 3, 32, 32))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradients_flow(self):
        model = self._make_model()
        out = model(torch.randn(2, 3, 32, 32))
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_cls_token_shape(self):
        model = self._make_model(dim=64)
        assert model.cls_token.shape == (1, 1, 64)

    def test_pos_embedding_dim(self):
        # 32/4=8 → 64 patches; +1 CLS → 65; dim=64
        model = self._make_model(image_size=32, patch_size=4, dim=64)
        assert model.pos_embedding.shape == (1, 65, 64)

    def test_different_patch_sizes(self):
        # patch_size=2: 32/2=16 → 256 patches
        model = self._make_model(patch_size=2)
        out = model(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)

    def test_head_out_features(self):
        model = self._make_model(num_classes=100)
        assert model.head.out_features == 100

    def test_train_eval_determinism(self):
        # In eval mode (dropout=0) two identical inputs give identical outputs
        model = self._make_model(dropout=0.0)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        assert torch.allclose(model(x), model(x))


# ---------------------------------------------------------------------------
# GPT — additional tests
# ---------------------------------------------------------------------------

class TestGPTAdditional:
    def _make_model(self, vocab_size=64, seq_len=32, dim=32, depth=2, heads=4):
        return GPT(vocab_size=vocab_size, seq_len=seq_len, dim=dim,
                   depth=depth, heads=heads, mlp_dim=64, dropout=0.0)

    def test_logits_sum_to_non_zero(self):
        # Sanity: logits shouldn't all be zero after random init
        model = self._make_model()
        out = model(torch.randint(0, 64, (2, 16)))
        assert out.abs().sum() > 0

    def test_gradients_flow(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        out.sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_weight_tying_grad_shared(self):
        # After backward, token_emb and head share the same .grad tensor
        model = self._make_model()
        x = torch.randint(0, 64, (2, 16))
        model(x).sum().backward()
        assert model.head.weight.grad is model.token_emb.weight.grad

    def test_causal_mask_mid_position(self):
        # Position k should not depend on tokens at positions > k
        model = self._make_model()
        model.eval()
        k = 8
        x1 = torch.randint(0, 64, (1, 32))
        x2 = x1.clone()
        x2[0, k + 1:] = torch.randint(0, 64, (32 - k - 1,))
        out1 = model(x1)
        out2 = model(x2)
        assert torch.allclose(out1[0, k], out2[0, k], atol=1e-5)

    def test_pos_emb_shape(self):
        model = self._make_model(seq_len=32, dim=32)
        assert model.pos_emb.weight.shape == (32, 32)

    def test_token_emb_shape(self):
        model = self._make_model(vocab_size=64, dim=32)
        assert model.token_emb.weight.shape == (64, 32)

    def test_norm_layer_present(self):
        model = self._make_model()
        assert isinstance(model.norm, nn.LayerNorm)

    def test_eval_deterministic(self):
        model = self._make_model()
        model.eval()
        x = torch.randint(0, 64, (2, 16))
        assert torch.allclose(model(x), model(x))

    def test_sequence_length_one(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 1))
        out = model(x)
        assert out.shape == (2, 1, 64)

    def test_depth_controls_layers(self):
        model = self._make_model(depth=3)
        assert model.transformer.num_layers == 3
