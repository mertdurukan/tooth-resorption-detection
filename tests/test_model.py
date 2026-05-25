"""Unit tests for :mod:`tooth_resorption.models.model`."""

from __future__ import annotations

import pytest
import torch
from tooth_resorption.config import NUM_CLASSES
from tooth_resorption.models.model import build_model, count_parameters


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_model("not-a-model", pretrained=False)  # type: ignore[arg-type]


def test_baseline_cnn_forward_shape(baseline_model: torch.nn.Module) -> None:
    x = torch.randn(2, 3, 224, 224)
    out = baseline_model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_baseline_cnn_parameter_count(baseline_model: torch.nn.Module) -> None:
    n = count_parameters(baseline_model)
    assert 50_000 < n < 2_000_000


def test_baseline_cnn_gradient_flow(baseline_model: torch.nn.Module) -> None:
    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    target = torch.tensor([0, 1])
    logits = baseline_model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in baseline_model.parameters())


@pytest.mark.slow
def test_vit_tiny_forward_shape(vit_tiny_model: torch.nn.Module) -> None:
    x = torch.randn(2, 3, 224, 224)
    out = vit_tiny_model(x)
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.slow
def test_vit_tiny_freeze_backbone_toggle(vit_tiny_model: torch.nn.Module) -> None:
    from tooth_resorption.models.model import AttentionViT

    assert isinstance(vit_tiny_model, AttentionViT)
    vit_tiny_model.freeze_backbone(True)
    head_trainable = [
        p.requires_grad
        for n, p in vit_tiny_model.backbone.named_parameters()
        if "head" in n or "fc" in n
    ]
    body_trainable = [
        p.requires_grad
        for n, p in vit_tiny_model.backbone.named_parameters()
        if "head" not in n and "fc" not in n
    ]
    assert any(head_trainable)
    assert not any(body_trainable)
    vit_tiny_model.freeze_backbone(False)
    assert all(p.requires_grad for p in vit_tiny_model.parameters())
