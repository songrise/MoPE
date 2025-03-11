import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel, AutoImageProcessor


class PoolViT(nn.Module):
    def __init__(self, n_frames):
        super(PoolViT, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.model = ViTModel(ViTConfig())
        self.pooler = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, return_features=False):
        # x: [B, N, C, H, W]
        # only process the first frame
        N_FRAME = x.shape[1]
        xs = []
        for i in range(0, N_FRAME):
            x_frame = x[:, i, ...]
            x_frame = self.model(x_frame).last_hidden_state[:, 0, :]  # CLS
            xs.append(x_frame)
        xs = torch.stack(xs, dim=1)
        xs = xs.permute(0, 2, 1)
        pooled = self.pooler(xs).squeeze(-1)
        return pooled, pooled


if __name__ == "__main__":
    in_ = torch.randn(2, 8, 3, 224, 224)
    model = PoolViT(8)
    out = model(in_)
