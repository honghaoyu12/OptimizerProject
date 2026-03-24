"""Neural network architectures for image classification benchmarks.

This module provides three model families scaled appropriately for the small
image datasets (MNIST 28×28, CIFAR 32×32) used in this project:

  MLP      — Configurable fully-connected network; depth and width controlled
              by hidden_sizes.  Simple baseline for tabular and image tasks.

  ResNet18 — Torchvision ResNet-18 with two surgical modifications to avoid
              over-downsampling on small inputs: 3×3 stride-1 conv1 (instead
              of the ImageNet 7×7 stride-2) and an Identity maxpool.

  ViT      — Vision Transformer with Pre-LayerNorm (norm_first=True).  Patch
              size 4 divides evenly into both 28 and 32, giving 49 or 64 tokens
              respectively.

All three return raw logits (no softmax); cross-entropy in the training loop
applies the softmax internally via log_softmax + NLLLoss.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Configurable multi-layer perceptron for image classification.

    Architecture (example with hidden_sizes=[256, 128]):
        Flatten → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU
                → Linear(128, 10)

    WHY Flatten first: images arrive as (B, C, H, W); Linear expects 2-D.
    WHY no final activation: raw logits go into nn.CrossEntropyLoss which
    applies log_softmax internally — adding softmax here would be redundant
    and numerically unstable.

    Parameters
    ----------
    input_size : int
        Flattened input dimension.  784 for MNIST, 3072 for CIFAR-10.
    hidden_sizes : list[int]
        Width of each hidden layer.  Controls both depth (len) and width.
        Default [256, 128] gives: input_size -> 256 -> 128 -> num_classes.
    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] | None = None,
        num_classes: int = 10,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        # Build layer list dynamically so depth/width are fully configurable
        layers: list[nn.Module] = [nn.Flatten()]  # (B, C, H, W) → (B, input_size)
        in_features = input_size
        for width in hidden_sizes:
            # Each hidden block: Linear + ReLU activation
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        # Output layer: no activation (raw logits for cross-entropy)
        layers.append(nn.Linear(in_features, num_classes))

        # Pack into Sequential so forward() is a single call — also makes
        # named_modules() traverse in layer order, which linear_layer_names()
        # in train.py relies on.
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, input_size) → returns (B, num_classes) logits
        return self.net(x)


# ---------------------------------------------------------------------------
# ResNet-18 (small-image adaptation)
# ---------------------------------------------------------------------------

class ResNet18(nn.Module):
    """ResNet-18 adapted for small images (MNIST / CIFAR-10).

    Standard torchvision ResNet-18 is designed for ImageNet (224×224 inputs).
    Applied naively to 28×32 pixel images, the original first two layers would
    reduce spatial dimensions to ~3×3 before any residual learning happens,
    destroying most spatial structure.

    Modifications vs. the standard ImageNet variant:
    - conv1: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution on
      28/32px inputs; receptive field is still sufficient for small textures)
    - maxpool: replaced with nn.Identity() (avoids 4× spatial reduction that
      would leave only 7×7 or 8×8 feature maps before layer1)

    These two changes together keep the spatial dimensions at 28×28 / 32×32
    through the stem, matching the design choices in "CIFAR-10 ResNets"
    (He et al., 2016 — Section 4.2).

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale MNIST, 3 for RGB CIFAR).
    num_classes : int
        Number of output classes.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        # Load standard ImageNet architecture, then patch the two problem layers.
        # WHY use pretrained=False (default): we train from scratch on small
        # datasets; ImageNet weights are for 3-channel 224×224 images and
        # would need fine-tuning anyway.
        self.model = resnet18(num_classes=num_classes)

        # Patch 1: Replace the 7×7 stride-2 stem convolution.
        # Original: Conv2d(3, 64, 7, stride=2, padding=3)  → halves spatial dims
        # Patched:  Conv2d(in_channels, 64, 3, stride=1, padding=1)  → preserves
        # WHY bias=False: BatchNorm after conv absorbs any constant offset.
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Patch 2: Remove the 3×3 stride-2 max-pool that follows conv1.
        # nn.Identity() is a no-op that passes tensors through unchanged.
        # This prevents a 4× spatial downsampling that would leave only
        # 7×7 feature maps on MNIST before the residual blocks begin.
        self.model.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W) → returns (B, num_classes) logits
        return self.model(x)


# ---------------------------------------------------------------------------
# Vision Transformer (small-image adaptation)
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """Vision Transformer (ViT) adapted for small images (MNIST / CIFAR-10).

    Implements the architecture from "An Image is Worth 16x16 Words"
    (Dosovitskiy et al., 2020; https://arxiv.org/abs/2010.11929) with a
    smaller patch size suited to 28×32 pixel inputs.

    Pipeline:
        1. Split image into non-overlapping patches of size patch_size×patch_size.
        2. Linearly embed each flattened patch into a vector of dimension `dim`.
        3. Prepend a learnable CLS token (aggregates global classification info).
        4. Add learnable positional embeddings so the Transformer can distinguish
           positions (it is otherwise permutation-equivariant).
        5. Pass through `depth` Transformer encoder layers (Pre-LN variant).
        6. Read out the CLS token at the final layer and classify with a linear head.

    WHY patch_size=4: divides evenly into both 28 (→ 49 patches) and 32
    (→ 64 patches).  The original ViT uses 16×16 on 224×224 images; scaling
    to 4×4 on small images keeps the token count similar.

    WHY Pre-LN (norm_first=True): Post-LN (the original "Attention Is All You
    Need" variant) is known to diverge without careful LR warmup; Pre-LN is
    more stable and easier to train from scratch on small datasets.

    Parameters
    ----------
    image_size  : int   — side length of the (square) input image (28 or 32)
    in_channels : int   — 1 for grayscale, 3 for RGB
    patch_size  : int   — side length of each patch (default 4; divides both 28 and 32)
    num_classes : int   — number of output classes
    dim         : int   — token embedding dimension (d_model in Transformer terms)
    depth       : int   — number of stacked Transformer encoder layers
    heads       : int   — number of parallel self-attention heads (dim % heads == 0)
    mlp_dim     : int   — hidden dim of the MLP inside each Transformer layer
    dropout     : float — dropout probability (applied to embeddings and attention)
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 4,
        num_classes: int = 10,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Validate that image dimensions are evenly divisible by patch size
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        # Derived dimensions:
        #   num_patches: total number of non-overlapping patches (sequence length)
        #   patch_dim:   flattened size of one patch before embedding
        num_patches = (image_size // patch_size) ** 2   # e.g. (32//4)^2 = 64
        patch_dim   = in_channels * patch_size * patch_size  # e.g. 3*4*4 = 48
        self.patch_size = patch_size

        # Patch embedding pipeline (with Pre- and Post-LN for stability).
        # WHY double LayerNorm: the pre-norm normalises raw pixel variance
        # (which varies wildly across datasets), and the post-norm keeps
        # projected embeddings unit-scale before the positional add.
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),          # normalise raw patch values
            nn.Linear(patch_dim, dim),        # project to embedding dimension
            nn.LayerNorm(dim),                # normalise after projection
        )

        # Learnable CLS token — one vector shared across all examples in a
        # batch; expanded to (B, 1, dim) before concatenation in forward().
        # Initialised with truncated normal (std=0.02) following DeiT practice.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Positional embedding has one entry per patch PLUS the CLS token.
        # Sequence length = num_patches + 1 (CLS is position 0).
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token,    std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder — Pre-LN (norm_first=True) applies LayerNorm
        # *before* attention and MLP sub-layers, making gradient flow more
        # stable than the original Post-LN design.
        # WHY enable_nested_tensor=False: suppresses a PyTorch warning when
        # norm_first=True and no padding mask is provided; has no effect on
        # results, just silences noisy output.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,   # input shape: (B, seq_len, dim) not (seq_len, B, dim)
            norm_first=True,    # Pre-LN — more stable than Post-LN for small data
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,  # suppresses Pre-LN + no-mask warning
        )

        # Final LayerNorm applied to the CLS token output before classification
        self.norm = nn.LayerNorm(dim)
        # Linear classification head maps the CLS token to class logits
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, C, H, W) — batch of images

        Returns
        -------
        (B, num_classes) — raw logits (no softmax)
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # ── Step 1: Rearrange image into a sequence of flattened patches ──
        # (B, C, H, W)
        #   → reshape to (B, C, H//p, p, W//p, p)   [split each spatial dim into patches]
        #   → permute to (B, H//p, W//p, C, p, p)   [bring patch coords to front]
        #   → reshape to (B, num_patches, C*p*p)     [flatten each patch]
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)  # (B, num_patches, patch_dim)

        # ── Step 2: Embed patches into the token space ────────────────────
        x = self.to_patch_embedding(x)      # (B, num_patches, dim)

        # ── Step 3: Prepend CLS token and add positional embedding ────────
        # expand() shares memory with cls_token — no copy; just adjusts strides
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, dim)
        x   = torch.cat([cls, x], dim=1)                 # (B, num_patches+1, dim)
        x   = self.dropout(x + self.pos_embedding)       # add position info, then drop

        # ── Step 4: Transformer encoder ───────────────────────────────────
        x = self.transformer(x)             # (B, num_patches+1, dim)

        # ── Step 5: Classify from the CLS token (position 0) ─────────────
        # x[:, 0] extracts the CLS token for each example in the batch
        cls_out = self.norm(x[:, 0])        # (B, dim)
        return self.head(cls_out)           # (B, num_classes)
