"""F1 non-adaptive and F2 gated Fusion classifiers."""
import torch
import torch.nn as nn
try:  # Package import
    from .fusion_model import FusionModel
except ImportError:  # Direct-script/test import
    from fusion_model import FusionModel

class FusionClassifier(nn.Module):
    def __init__(self, backbone=None, fusion_dropout=0.45, hidden_dim=128, n_classes=3):
        super().__init__()
        self.backbone = backbone if backbone is not None else FusionModel()
        tcn_dim = getattr(self.backbone, "tcn_output_dim", 128)
        mlp_dim = getattr(self.backbone, "mlp_output_dim", 64)
        self.tcn_projection = nn.Sequential(nn.Linear(tcn_dim,128), nn.LayerNorm(128))
        self.mlp_projection = nn.Sequential(nn.Linear(mlp_dim,128), nn.LayerNorm(128))
        self.gate = nn.Sequential(nn.Linear(256,128), nn.Sigmoid())
        self.head = nn.Sequential(nn.Dropout(fusion_dropout), nn.Linear(384,hidden_dim),
                                 nn.ReLU(inplace=True), nn.Dropout(fusion_dropout),
                                 nn.Linear(hidden_dim,n_classes))
    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode)
        return self
    def forward(self, x, fusion_type="f1", return_details=False):
        if fusion_type not in ("f1","f2"):
            raise ValueError("fusion_type must be 'f1' or 'f2'")
        e=self.backbone(x)
        pt=self.tcn_projection(e["tcn_embedding"])
        pm=self.mlp_projection(e["mlp_embedding"])
        gate=self.gate(torch.cat((pt,pm),dim=1)) if fusion_type=="f2" else None
        fused=pt+pm if gate is None else gate*pt+(1.0-gate)*pm
        rep=torch.cat((fused,pt,pm),dim=1)
        logits=self.head(rep)
        if return_details:
            return {"logits":logits,"tcn_projected":pt,"mlp_projected":pm,
                    "fused":fused,"gate":gate,"representation":rep}
        return logits
