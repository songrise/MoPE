# -*- coding : utf-8 -*-
# @FileName  : segmentor.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 07, 2023
# @Github    : https://github.com/songrise
# @Description: segmentor models


import torch
import torch.nn as nn
import torch.nn.functional as F



def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    exit(-1)

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class VisionTextSegmentor(nn.Module):
    def __init__(
        self,
        vision_model,
        text_model,
        num_classes,
        fusion_method="ens",
        train_instructor=True,
        moe_n_experts=8,
        moe_top_k=1,
        dense_routing=False,
        attn_fuse=False,
        route_per_layer=False,
    ):
        super(VisionTextSegmentor, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.fusion_head = None
        self.instruct_proj = None
        self.moe_proj = None
        self.fuse_method = fusion_method
        self.train_instructor = train_instructor
        self.moe_top_k = None
        self.moe_n_experts = None
        self.dense_routing = dense_routing
        self.attn_fuse = attn_fuse
        self.route_per_layer = route_per_layer
        if fusion_method == "late_concat":
            self.fusion_head = nn.Linear(
                768, 1024
            )  #!HARDCODED Oct 13: project text feature to vis hidden size
        if fusion_method == "instruct_t2v":
            self.instruct_proj = nn.Sequential(
                nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            )
        if fusion_method == "instruct_v2t":
            self.instruct_proj = nn.Linear(1024, 768)
        if fusion_method == "instruct_moe_t2v":
            self.instruct_proj = nn.Sequential(
                nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            )
            if self.route_per_layer:
                self.n_route_layers = 2 + 2 + 18 + 2 - 1
                self.moe_proj = nn.ModuleList(
                    [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
                )  # 4 swin enc layers
            else:
                self.moe_proj = nn.Linear(768, moe_n_experts)

            self.moe_top_k = moe_top_k
            self.moe_n_experts = moe_n_experts
        if fusion_method == "instruct_multimodal_moe_t2v":
            self.instruct_proj = nn.Sequential(
                nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            )
            if self.route_per_layer:
                self.n_route_layers = 2 + 2 + 18 + 2 - 1
                self.moe_proj = nn.ModuleList(
                    [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
                )  # 4 swin enc layers
            else:
                self.moe_proj = nn.Linear(768, moe_n_experts)
            self.moe_top_k = moe_top_k
            self.moe_n_experts = moe_n_experts
        if (
            fusion_method == "promptfuse"
            or fusion_method == "sequentialfuse"
            or fusion_method == "p_sequential"
        ):
            self.instruct_proj = nn.Sequential(
                nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            )  # same design as the mapper
            if fusion_method == "promptfuse":
                self.instruct_proj.requires_grad_(False)

    def forward(self, vision_input, text_input):
        """
        return :
            logits: final prediction
            extra_out: a dict of extra return key and values
        """
        if self.fuse_method == "ens":
            return self.forward_ens(vision_input, text_input)
        elif self.fuse_method == "late_concat":
            return self.forward_late_concat(vision_input, text_input)
        elif self.fuse_method == "instruct_t2v":
            return self.forward_instruct_t2v(vision_input, text_input)
        elif self.fuse_method == "instruct_v2t":
            return self.forward_instruct_v2t(vision_input, text_input)
        elif self.fuse_method == "instruct_moe_t2v":
            return self.forward_instruct_moe_t2v(vision_input, text_input)
        elif self.fuse_method == "sequentialfuse" or self.fuse_method == "p_sequential":
            return self.forward_instruct_t2v(vision_input, text_input)

    def forward_ens(self, vision_input, text_input):
        pass

    def forward_late_concat(self, vision_input, text_input):
        if self.train_instructor:
            _, text_feature = self.text_model(text_input, return_features=True)
        else:
            with torch.no_grad():
                _, text_feature = self.text_model(text_input, return_features=True)

        text_feature = self.fusion_head(text_feature)

        logits = self.vision_model.forward_late_concat(vision_input, text_feature)
        return logits, None

    def forward_instruct_t2v(self, vision_input, text_input):
        if self.train_instructor:
            _, text_feature = self.text_model(text_input, return_features=True)
        else:
            with torch.no_grad():
                _, text_feature = self.text_model(text_input, return_features=True)

        text_feature = self.instruct_proj(text_feature)

        logits = self.vision_model.forward_instruct(vision_input, text_feature)
        return logits, None

    def forward_instruct_moe_t2v(self, vision_input, text_input):
        if self.train_instructor:
            _, text_feature = self.text_model(text_input, return_features=True)
        else:
            with torch.no_grad():
                _, text_feature = self.text_model(text_input, return_features=True)
        extra_out = dict()
        if self.route_per_layer:
            moe_logits = torch.stack(
                [self.moe_proj[i](text_feature) for i in range(self.n_route_layers)],
                dim=1,
            )  # B, N_layer, N_expert
        else:
            moe_logits = self.moe_proj(text_feature)
        temperature = 1.0
        moe_logits = moe_logits / temperature

        # add normal dis N(0, 1/n_experts^2)
        noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (
            self.moe_n_experts**2
        )
        moe_scores = F.softmax(moe_logits + noise, dim=-1)
        # [B, n_experts], where top_k experts are 1, others are 0
        moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
        if self.route_per_layer:
            moe_mask.scatter_(2, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
        else:
            moe_mask.scatter_(1, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
        moe_scores = moe_scores / self.moe_top_k
        selected_expert = torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1]
        extra_out["moe_scores"] = moe_scores

        # importance loss
        sum_scores = torch.sum(moe_scores, dim=0)  # N_experts
        std_scores = torch.std(sum_scores)  # N_experts
        mean_scores = torch.mean(sum_scores)  # N_experts
        importance_loss = (std_scores / mean_scores) ** 2
        extra_out["importance_loss"] = importance_loss

        # entropy loss·
        entropy_loss = -torch.sum(moe_scores * torch.log(moe_scores + 1e-8), dim=1)
        entropy_loss = torch.mean(entropy_loss)
        extra_out["entropy_loss"] = entropy_loss

        text_feature = self.instruct_proj(text_feature)
        route_score = moe_scores if self.dense_routing else moe_mask
        logits = self.vision_model.forward_instruct_moe(
            vision_input, text_feature, route_score, self.attn_fuse
        )
        return logits, extra_out

    def forward_instruct_multimodal_moe_t2v(self, vision_input, text_input):
        pass

    def forward_instruct_v2t(self, vision_input, text_input, train_instructor=False):
        pass

    def forward_text(self, text_input):
        return self.text_model(text_input)

    def forward_vision(self, vision_input):
        return self.vision_model(vision_input)
