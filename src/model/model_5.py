import torch
import torch.nn as nn
from src.model.model_3 import generate_model as _generate_model


class model_5(nn.Module):
    def __init__(self,
                 model_depth=50,
                 projector_dim=None,
                 zero_init_residual=False,
                 normalize=True,
                 nmb_prototypes=3000,
                 eval_mode=False,
                 shortcut_type="B",
                 name="SWAV",
                 task="todo",
                 description="todo"):
        super(model_5, self).__init__()
        self.backbone_and_projector = _generate_model(0,
                                                      model_depth,
                                                      projector_dim,
                                                      normalize=False)
        self.name = name
        self.task = task
        self.description = description
        self.feature_dim = projector_dim[-1]

        # normalize output features
        self.l2norm = normalize

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(projector_dim[-1],
                                              nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(projector_dim[-1],
                                        nmb_prototypes,
                                        bias=False)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)

    def forward_head(self, representations):
        embedding = self.backbone_and_projector._forward_projector(
            representations)

        if self.l2norm:
            embedding = nn.functional.normalize(embedding, dim=1, p=2)
        if self.prototypes is not None:
            return embedding, self.prototypes(embedding)
        return embedding


    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone_and_projector._forward_backbone(
                torch.cat(inputs[start_idx:end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                features = _out
            else:
                features = torch.cat((features, _out))
            start_idx = end_idx
        output = self.forward_head(features)
        return output


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def generate_model(idx=5, model_depth=50, projector_dim=None):
    if projector_dim is None:
        projector_dim = [128, 128, 128]

    return model_5(model_depth, projector_dim)
