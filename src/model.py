import torch
import torch.nn as nn
from torchvision import models


class MultiModalNet(nn.Module):
    def __init__(self, num_classes=4, text_vocab_size=256, text_emb_dim=64):
        super().__init__()

        # Image branch (ResNet18)
        self.img_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.img_backbone.fc.in_features
        self.img_backbone.fc = nn.Identity()  # output: [B, in_feats]

        self.img_proj = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Text branch (char IDs)
        self.text_emb = nn.Embedding(text_vocab_size, text_emb_dim, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, text_ids):
        # Image features
        img_feat = self.img_backbone(image)          # [B, in_feats]
        img_feat = self.img_proj(img_feat)           # [B, 256]

        # Text features: mean pooling over sequence
        emb = self.text_emb(text_ids)                # [B, L, E]
        mask = (text_ids != 0).float().unsqueeze(-1) # [B, L, 1]
        summed = (emb * mask).sum(dim=1)             # [B, E]
        denom = mask.sum(dim=1).clamp(min=1.0)       # [B, 1]
        mean = summed / denom                        # [B, E]
        txt_feat = self.text_proj(mean)              # [B, 128]

        # Fuse
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)
