"""Focused integrity tests for Fusion architecture (no pretrained artifacts required)."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"models", ROOT/"features", ROOT/"training"):
    sys.path.insert(0,str(p))
from fusion_model import FusionModel
from fusion_classifier import FusionClassifier
from channel_attention import ChannelAttention
from tcn_branch import TCNBranch
from mlp_branch import MLPBranch
from handcrafted_features import extract_features_from_window, N_FEATURES_PER_CHANNEL
import numpy as np

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
    x=torch.randn(2,6,1000)
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
    x=torch.randn(2,6,1000)
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


def test_fusion_embeddings_match_standalone_branches():
    """Fusion must preserve each pretrained branch's standalone forward path."""
    torch.manual_seed(23)
    fusion = FusionModel().eval()
    attention = ChannelAttention(n_channels=6).eval()
    tcn = TCNBranch(n_input_channels=6, output_dim=128).eval()
    mlp = MLPBranch(n_input_features=6 * N_FEATURES_PER_CHANNEL, output_dim=64).eval()

    attention.load_state_dict(fusion.attention.state_dict())
    tcn.load_state_dict(fusion.tcn_branch.state_dict())
    mlp.load_state_dict(fusion.mlp_branch.state_dict())

    x = torch.randn(2, 6, 1000)
    with torch.no_grad():
        tcn_expected = tcn(attention(FusionModel.sanitize_tcn_input(x)))
        feature_rows = np.stack([
            extract_features_from_window(sample, fs=250)
            for sample in x.numpy()
        ]).astype(np.float32)
        mlp_expected = mlp(torch.from_numpy(feature_rows))
        fusion_embeddings = fusion(x)

    assert torch.allclose(fusion_embeddings["tcn_embedding"], tcn_expected,
                          atol=1e-6, rtol=1e-5)
    assert torch.allclose(fusion_embeddings["mlp_embedding"], mlp_expected,
                          atol=1e-6, rtol=1e-5)

def test_f1_does_not_train_gate():
    model = FusionClassifier()
    model.train()
    x = torch.randn(2, 6, 1000)
    model(x, "f1").sum().backward()
    assert all(p.grad is None for p in model.gate.parameters())
    assert any(p.grad is not None for p in model.tcn_projection.parameters())
    assert any(p.grad is not None for p in model.mlp_projection.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
