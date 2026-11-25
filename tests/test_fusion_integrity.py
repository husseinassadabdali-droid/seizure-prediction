"""Focused integrity tests for Fusion architecture (no pretrained artifacts required)."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"models", ROOT/"features", ROOT/"training"):
    sys.path.insert(0,str(p))
from fusion_model import FusionModel
from fusion_classifier import FusionClassifier

def test_branch_embedding_shapes_and_finiteness():
    torch.manual_seed(7)
    model=FusionModel().eval()
    x=torch.randn(2,6,1000)
    with torch.no_grad():
        e=model(x)
    assert e["tcn_embedding"].shape==(2,128)
    assert e["mlp_embedding"].shape==(2,64)
    assert torch.isfinite(e["tcn_embedding"]).all()
    assert torch.isfinite(e["mlp_embedding"]).all()

def test_f1_f2_shapes_and_gate_range():
    torch.manual_seed(11)
    model=FusionClassifier().eval()
    x=torch.randn(2,6,256)
    with torch.no_grad():
        f1=model(x,"f1",return_details=True)
        f2=model(x,"f2",return_details=True)
    assert f1["logits"].shape==(2,3)
    assert f2["logits"].shape==(2,3)
    assert f1["representation"].shape==(2,384)
    assert f2["representation"].shape==(2,384)
    assert f1["gate"] is None
    assert f2["gate"].shape==(2,128)
    assert torch.all(f2["gate"]>=0) and torch.all(f2["gate"]<=1)
    for out in (f1,f2):
        for key in ("logits","fused","representation"):
            assert torch.isfinite(out[key]).all()

def test_frozen_branches_stay_eval_and_buffers_unchanged():
    model=FusionClassifier()
    model.train()
    backbone=model.backbone
    for branch in (backbone.attention,backbone.tcn_branch,backbone.mlp_branch):
        assert not branch.training
        assert all(not p.requires_grad for p in branch.parameters())
    before={n:b.detach().clone() for n,b in backbone.named_buffers()}
    x=torch.randn(2,6,256)
    model(x,"f2").sum().backward()
    after=dict(backbone.named_buffers())
    assert all(torch.equal(before[n],after[n]) for n in before)
    assert all(p.grad is None for branch in (backbone.attention,backbone.tcn_branch,backbone.mlp_branch)
               for p in branch.parameters())
    assert any(p.grad is not None for p in model.gate.parameters())
    assert any(p.grad is not None for p in model.head.parameters())

def test_tcn_sanitation():
    x=torch.tensor([[[float("nan"),float("inf"),float("-inf"),99.0,-99.0]]])
    y=FusionModel.sanitize_tcn_input(x)
    assert torch.isfinite(y).all()
    assert y.max()<=20 and y.min()>=-20
