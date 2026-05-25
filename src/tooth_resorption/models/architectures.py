"""Attention and Transformer based model architectures.

Vision Transformer (ViT), Swin Transformer and attention-augmented CNN
hybrid models — the full zoo evaluated during the MSc work.

Each architecture exposes a ``freeze_backbone(freeze: bool)`` method so the
two-stage fine-tuning strategy (head-only warm-up, then unfreeze) can be
applied uniformly. Build an instance through :func:`create_model` to keep
the calling code decoupled from the concrete class.
"""

from __future__ import annotations

from typing import Callable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionBlock(nn.Module):
    """Multi-head self-attention block over a flattened feature map."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out)
        return attn_out.transpose(1, 2).reshape(b, c, h, w)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial gating)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VisionTransformerModel(nn.Module):
    """Thin wrapper around any ``timm`` ViT backbone."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "head"):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze every parameter except the classification head."""
        for name, param in self.model.named_parameters():
            if "head" not in name and "fc" not in name:
                param.requires_grad = not freeze


class SwinTransformerModel(nn.Module):
    """Thin wrapper around any ``timm`` Swin Transformer backbone."""

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        if hasattr(self.model, "head"):
            if hasattr(self.model.head, "fc"):
                in_features = self.model.head.fc.in_features
                self.model.head.fc = nn.Linear(in_features, num_classes)
            elif hasattr(self.model.head, "in_features"):
                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for name, param in self.model.named_parameters():
            if "head" not in name and "fc" not in name:
                param.requires_grad = not freeze


class ResNetCBAM(nn.Module):
    """ResNet50 backbone with CBAM attention applied at every stage."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = "resnet50_cbam"

        resnet = models.resnet50(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.cbam1 = CBAM(256)

        self.layer2 = resnet.layer2
        self.cbam2 = CBAM(512)

        self.layer3 = resnet.layer3
        self.cbam3 = CBAM(1024)

        self.layer4 = resnet.layer4
        self.cbam4 = CBAM(2048)

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def freeze_backbone(self, freeze: bool = True) -> None:
        for name, param in self.named_parameters():
            if "fc" not in name and "cbam" not in name:
                param.requires_grad = not freeze


class ResNetSE(nn.Module):
    """ResNet50 backbone with Squeeze-and-Excitation at every stage."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = "resnet50_se"

        resnet = models.resnet50(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.se1 = SEBlock(256)

        self.layer2 = resnet.layer2
        self.se2 = SEBlock(512)

        self.layer3 = resnet.layer3
        self.se3 = SEBlock(1024)

        self.layer4 = resnet.layer4
        self.se4 = SEBlock(2048)

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def freeze_backbone(self, freeze: bool = True) -> None:
        for name, param in self.named_parameters():
            if "fc" not in name and "se" not in name:
                param.requires_grad = not freeze


class EfficientNetAttention(nn.Module):
    """EfficientNet-B0 backbone with a single multi-head attention block."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = "efficientnet_b0_attention"

        efficientnet = models.efficientnet_b0(pretrained=pretrained)

        self.features = efficientnet.features
        self.attention = AttentionBlock(1280, num_heads=8)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.features.parameters():
            param.requires_grad = not freeze


class DenseNetAttention(nn.Module):
    """DenseNet121 backbone with a single multi-head attention block."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = "densenet121_attention"

        densenet = models.densenet121(pretrained=pretrained)

        self.features = densenet.features
        self.attention = AttentionBlock(1024, num_heads=8)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.features.parameters():
            param.requires_grad = not freeze


class ViTBaseWithAttention(nn.Module):
    """ViT-Base with an additional attention module at the head or per stage.

    Supports CBAM, SE and Multi-head Attention at one of two insertion
    points: ``"head"`` (single block before the classification head) or
    ``"stages"`` (one block after every group of three transformer blocks).
    """

    def __init__(
        self,
        attention_type: str = "cbam",
        position: str = "head",
        num_classes: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.position = position
        self.model_name = f"vit_base_16_{attention_type}_{position}"

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()

        if position == "head":
            self.attention = self._create_attention_block(self.embed_dim, attention_type)
        else:
            self.stage_attentions = nn.ModuleList(
                [self._create_attention_block(self.embed_dim, attention_type) for _ in range(4)]
            )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    @staticmethod
    def _create_attention_block(dim: int, attention_type: str) -> nn.Module:
        if attention_type == "cbam":
            return CBAM(dim, reduction=16)
        if attention_type == "se":
            return SEBlock(dim, reduction=16)
        if attention_type == "mha":
            return AttentionBlock(dim, num_heads=8)
        raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        if self.position == "stages":
            num_blocks = len(self.vit.blocks)
            blocks_per_stage = num_blocks // 4

            for i, block in enumerate(self.vit.blocks):
                x = block(x)

                stage_idx = i // blocks_per_stage
                if (i + 1) % blocks_per_stage == 0 and stage_idx < 4:
                    b, n, c = x.shape
                    x_spatial = x[:, 1:, :]
                    h = w = int((n - 1) ** 0.5)
                    x_spatial = x_spatial.transpose(1, 2).reshape(b, c, h, w)

                    x_spatial = self.stage_attentions[stage_idx](x_spatial)

                    x_spatial = x_spatial.flatten(2).transpose(1, 2)
                    x = torch.cat([x[:, :1, :], x_spatial], dim=1)
        else:
            x = self.vit.blocks(x)

        x = self.vit.norm(x)
        x = x[:, 0]

        if self.position == "head":
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.attention(x)
            x = x.squeeze(-1).squeeze(-1)

        x = self.norm(x)
        x = self.fc(x)

        return x

    def freeze_backbone(self, freeze: bool = True) -> None:
        for _, param in self.vit.named_parameters():
            param.requires_grad = not freeze
        if hasattr(self, "attention"):
            for param in self.attention.parameters():
                param.requires_grad = True
        if hasattr(self, "stage_attentions"):
            for att in self.stage_attentions:
                for param in att.parameters():
                    param.requires_grad = True


class SwinSmallWithAttention(nn.Module):
    """Swin-Small with an additional attention module at the head or per stage."""

    def __init__(
        self,
        attention_type: str = "cbam",
        position: str = "head",
        num_classes: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.position = position
        self.model_name = f"swin_small_{attention_type}_{position}"

        self.swin = timm.create_model("swin_small_patch4_window7_224", pretrained=pretrained)

        self.stage_dims = [96, 192, 384, 768]
        self.final_dim = 768

        self.swin.head = nn.Identity()

        if position == "head":
            self.attention = self._create_attention_block(self.final_dim, attention_type)
        else:
            self.stage_attentions = nn.ModuleList(
                [self._create_attention_block(dim, attention_type) for dim in self.stage_dims]
            )

        self.norm = nn.LayerNorm(self.final_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.final_dim, num_classes)

    @staticmethod
    def _create_attention_block(dim: int, attention_type: str) -> nn.Module:
        if attention_type == "cbam":
            return CBAM(dim, reduction=max(1, dim // 16))
        if attention_type == "se":
            return SEBlock(dim, reduction=max(1, dim // 16))
        if attention_type == "mha":
            heads = min(8, dim // 32) if dim >= 64 else 4
            return AttentionBlock(dim, num_heads=heads)
        raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.position == "stages":
            x = self.swin.patch_embed(x)

            for i, layer in enumerate(self.swin.layers):
                x = layer(x)

                _, _, _, _ = x.shape
                x_2d = x.permute(0, 3, 1, 2)
                x_2d = self.stage_attentions[i](x_2d)
                x = x_2d.permute(0, 2, 3, 1)

            x = self.swin.norm(x)
        else:
            x = self.swin.forward_features(x)

        x = x.permute(0, 3, 1, 2)

        if self.position == "head":
            x = self.attention(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        x = self.norm(x)
        x = self.fc(x)

        return x

    def freeze_backbone(self, freeze: bool = True) -> None:
        for _, param in self.swin.named_parameters():
            param.requires_grad = not freeze
        if hasattr(self, "attention"):
            for param in self.attention.parameters():
                param.requires_grad = True
        if hasattr(self, "stage_attentions"):
            for att in self.stage_attentions:
                for param in att.parameters():
                    param.requires_grad = True


def create_model(model_type: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Factory: instantiate one of the registered architectures by name.

    Args:
        model_type: Model identifier (see :data:`MODEL_REGISTRY` for the
            full list).
        num_classes: Number of output logits.
        pretrained: Whether to load pretrained weights for the backbone.

    Returns:
        An instantiated :class:`torch.nn.Module`.

    Raises:
        ValueError: If ``model_type`` is unknown.
    """
    model_dict: dict[str, Callable[[], nn.Module]] = {
        "vit_base_16": lambda: VisionTransformerModel("vit_base_patch16_224", num_classes, pretrained),
        "vit_base_32": lambda: VisionTransformerModel("vit_base_patch32_224", num_classes, pretrained),
        "vit_large_16": lambda: VisionTransformerModel("vit_large_patch16_224", num_classes, pretrained),
        "swin_tiny": lambda: SwinTransformerModel("swin_tiny_patch4_window7_224", num_classes, pretrained),
        "swin_small": lambda: SwinTransformerModel("swin_small_patch4_window7_224", num_classes, pretrained),
        "swin_base": lambda: SwinTransformerModel("swin_base_patch4_window7_224", num_classes, pretrained),
        "resnet50_cbam": lambda: ResNetCBAM(num_classes, pretrained),
        "resnet50_se": lambda: ResNetSE(num_classes, pretrained),
        "efficientnet_attention": lambda: EfficientNetAttention(num_classes, pretrained),
        "densenet_attention": lambda: DenseNetAttention(num_classes, pretrained),
        "vit_base_16_cbam_head": lambda: ViTBaseWithAttention("cbam", "head", num_classes, pretrained),
        "vit_base_16_se_head": lambda: ViTBaseWithAttention("se", "head", num_classes, pretrained),
        "vit_base_16_mha_head": lambda: ViTBaseWithAttention("mha", "head", num_classes, pretrained),
        "vit_base_16_cbam_stages": lambda: ViTBaseWithAttention("cbam", "stages", num_classes, pretrained),
        "vit_base_16_se_stages": lambda: ViTBaseWithAttention("se", "stages", num_classes, pretrained),
        "vit_base_16_mha_stages": lambda: ViTBaseWithAttention("mha", "stages", num_classes, pretrained),
        "swin_small_cbam_head": lambda: SwinSmallWithAttention("cbam", "head", num_classes, pretrained),
        "swin_small_se_head": lambda: SwinSmallWithAttention("se", "head", num_classes, pretrained),
        "swin_small_mha_head": lambda: SwinSmallWithAttention("mha", "head", num_classes, pretrained),
        "swin_small_cbam_stages": lambda: SwinSmallWithAttention("cbam", "stages", num_classes, pretrained),
        "swin_small_se_stages": lambda: SwinSmallWithAttention("se", "stages", num_classes, pretrained),
        "swin_small_mha_stages": lambda: SwinSmallWithAttention("mha", "stages", num_classes, pretrained),
    }

    if model_type not in model_dict:
        raise ValueError(
            f"Model type {model_type!r} not supported. Available: {sorted(model_dict)}"
        )

    return model_dict[model_type]()


def get_model_info(model: nn.Module) -> dict[str, float | int]:
    """Return total / trainable parameter counts and an approximate fp32 size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "model_size_mb": float(total_params * 4 / (1024 * 1024)),
    }


__all__ = [
    "AttentionBlock",
    "CBAM",
    "DenseNetAttention",
    "EfficientNetAttention",
    "ResNetCBAM",
    "ResNetSE",
    "SEBlock",
    "SwinSmallWithAttention",
    "SwinTransformerModel",
    "ViTBaseWithAttention",
    "VisionTransformerModel",
    "create_model",
    "get_model_info",
]
