import torch
import torch.nn as nn
from torchvision.models import resnet18


class MLP(nn.Module):
    """Configurable multi-layer perceptron for image classification.

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

        layers: list[nn.Module] = [nn.Flatten()]
        in_features = input_size
        for width in hidden_sizes:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        layers.append(nn.Linear(in_features, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet18(nn.Module):
    """ResNet-18 adapted for small images (MNIST / CIFAR-10).

    Modifications vs. the standard ImageNet variant:
    - conv1: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution on 28/32px inputs)
    - maxpool: replaced with nn.Identity() (avoids over-downsampling)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    num_classes : int
        Number of output classes.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        # Adapt for small images
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ViT(nn.Module):
    """Vision Transformer (ViT) adapted for small images (MNIST / CIFAR-10).

    Implements the architecture from "An Image is Worth 16x16 Words" (Dosovitskiy
    et al., 2020) with a patch size suited to small inputs.

    The image is split into non-overlapping patches, linearly projected into a
    token embedding, prepended with a learnable CLS token, and processed by a
    stack of standard Transformer encoder layers.  The CLS token at the final
    layer is passed to a linear classification head.

    Parameters
    ----------
    image_size  : int   — side length of the (square) input image (28 or 32)
    in_channels : int   — 1 for grayscale, 3 for RGB
    patch_size  : int   — side length of each patch (default 4; divides both 28 and 32)
    num_classes : int   — number of output classes
    dim         : int   — token embedding dimension
    depth       : int   — number of Transformer encoder layers
    heads       : int   — number of self-attention heads (dim must be divisible by heads)
    mlp_dim     : int   — hidden dimension of the MLP inside each encoder layer
    dropout     : float — dropout probability applied to embeddings and attention
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
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size

        # Linear patch embedding with pre- and post-norm (improves small-data stability)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (Pre-LN via norm_first=True, more stable than Post-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        # Rearrange image into a sequence of flattened patches: (B, num_patches, patch_dim)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)

        # Embed patches
        x = self.to_patch_embedding(x)                   # (B, num_patches, dim)

        # Prepend CLS token and add positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B, num_patches+1, dim)
        x = self.dropout(x + self.pos_embedding)

        # Transformer
        x = self.transformer(x)                          # (B, num_patches+1, dim)

        # Classify from CLS token
        cls_out = self.norm(x[:, 0])                     # (B, dim)
        return self.head(cls_out)                        # (B, num_classes)
