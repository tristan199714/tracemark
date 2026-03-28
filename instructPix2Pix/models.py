import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        groups = min(32, c_out)
        while c_out % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(groups, c_out),
            nn.SiLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class UserConditionedWriter(nn.Module):
    """
    Residual writer:
      x_w = clamp(x + alpha * tanh(delta(x, user_embed)), -1, 1)
    """
    def __init__(self, num_user: int, user_dim: int = 64, hidden: int = 96, blocks: int = 4):
        super().__init__()
        self.user_embedding = nn.Embedding(num_user, user_dim)
        layers = [ConvBNAct(3 + user_dim, hidden)]
        for _ in range(max(1, blocks - 1)):
            layers.append(ConvBNAct(hidden, hidden))
        self.encoder = nn.Sequential(*layers)
        self.to_delta = nn.Conv2d(hidden, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, user_ids: torch.Tensor, alpha: float) -> torch.Tensor:
        b, _, h, w = x.shape
        ue = self.user_embedding(user_ids).view(b, -1, 1, 1).expand(b, -1, h, w)
        feat = torch.cat([x, ue], dim=1)
        delta = self.tanh(self.to_delta(self.encoder(feat)))
        return torch.clamp(x + alpha * delta, -1.0, 1.0)


class RetrievalDetector(nn.Module):
    """
    Detector outputs normalized embedding and user logits via cosine prototypes.
    """
    def __init__(self, num_user: int, embed_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNAct(3, 64, s=2),
            ConvBNAct(64, 96, s=2),
            ConvBNAct(96, 128, s=2),
            ConvBNAct(128, 192, s=2),
            ConvBNAct(192, 256, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to_embed = nn.Linear(256, embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_user, embed_dim))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(16.0)))

    def forward(self, x: torch.Tensor):
        f = self.backbone(x).flatten(1)
        emb = F.normalize(self.to_embed(f), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * emb @ proto.t()
        return emb, logits
