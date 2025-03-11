# -*- coding : utf-8 -*-
# @FileName  : classifier.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 09, 2023
# @Github    : https://github.com/songrise
# @Description: classifiers

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTextClassifiers(nn.Module):
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
        route_per_layer=False,
    ):
        super(VisionTextClassifiers, self).__init__()
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
        self.route_per_layer = route_per_layer
        if fusion_method == "late_concat":
            self.fusion_head = nn.Linear(
                1024 + 768, num_classes
            )  #!HARDCODED Sep 12: swin [CLS] 1024 and bert [CLS] 768
        if fusion_method == "instruct_t2v":
            self.instruct_proj = nn.Sequential(
                nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            )
        if fusion_method == "instruct_v2t" or fusion_method == "p_sequential":
            bottle_neck = 1500
            self.instruct_proj = nn.Sequential(
                nn.Linear(1024, bottle_neck),
                nn.BatchNorm1d(bottle_neck),
                nn.GELU(),
                nn.Linear(bottle_neck, 768),
                nn.GELU(),
            )  # same design as the mapper
            # self.instruct_proj = nn.Sequential(
            #     nn.Linear(1024,768)
            # ) #linear
        if fusion_method == "instruct_moe_t2v":
            #!HARDCODED Jan 25: here changed 384  and n for vit
            # self.instruct_proj = nn.Sequential(
            #     nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            # )
            self.instruct_proj = nn.Sequential(nn.Linear(768, 384), nn.GELU())
            if self.route_per_layer:
                self.n_route_layers = 2 + 2 + 18 + 2 - 1
                if self.vision_model.__class__.__name__ == "CLIPViT":
                    self.n_route_layers = 12

                self.moe_proj = nn.ModuleList(
                    [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
                )
            else:
                self.moe_proj = nn.Linear(768, moe_n_experts)
            self.moe_top_k = moe_top_k
            self.moe_n_experts = moe_n_experts

        if fusion_method == "instruct_mm_moe_t2v" or fusion_method == "mope":
            # the mappers are defined in Swin Transformer
            self.moe_top_k = moe_top_k
            self.moe_n_experts = moe_n_experts
        if fusion_method == "instruct_moe_v2t":
            self.instruct_proj = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Linear(512, 768),
                nn.GELU(),
            )
            # if self.route_per_layer:
            self.n_route_layers = 13
            self.moe_proj = nn.ModuleList(
                [nn.Linear(1024, moe_n_experts) for _ in range(self.n_route_layers)]
            )  # 4 swin enc layers
            # else:
            #     self.moe_proj = nn.Linear(1024, moe_n_experts)
            self.moe_top_k = moe_top_k
            self.moe_n_experts = moe_n_experts
        if fusion_method == "sequentialfuse":
            self.instruct_proj = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Linear(512, 768),
                nn.GELU(),
            )  # same design as the mapper

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
        elif self.fuse_method == "instruct_mm_moe_t2v" or self.fuse_method == "mope":
            return self.forward_instruct_multimodal_moe_t2v(vision_input, text_input)
        elif self.fuse_method == "instruct_moe_v2t":
            return self.forward_instruct_moe_v2t(vision_input, text_input)
        elif self.fuse_method == "sequentialfuse" or self.fuse_method == "p_sequential":
            return self.forward_instruct_v2t(vision_input, text_input)
        if self.fuse_method == "img_only":
            return self.forward_vision(vision_input)
        if self.fuse_method == "text_only":
            return self.forward_text(text_input)

    def forward_ens(self, vision_input, text_input):
        vision_logits = self.vision_model(vision_input)
        text_logits = self.text_model(text_input)
        logits = vision_logits + text_logits
        return logits / 2.0, None

    def forward_late_concat(self, vision_input, text_input):
        _, vision_feature = self.vision_model(vision_input, return_features=True)
        _, text_feature = self.text_model(text_input, return_features=True)
        concat_feature = torch.cat([vision_feature, text_feature], dim=-1)
        logits = self.fusion_head(concat_feature)
        extra_out = dict()
        extra_out["cls_"] = concat_feature
        return logits, extra_out

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
        temperature = 0.1
        moe_logits = moe_logits / temperature
        # add normal dis N(0, 1/n_experts^2)
        noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (
            self.moe_n_experts**2
        )
        if False:
            mask_out = torch.randint_like(moe_logits, 0, 2) * -10000
        else:
            mask_out = torch.zeros_like(moe_logits)
        if moe_logits.shape[-1] == 1:
            moe_scores = torch.zeros_like(moe_logits)
        else:
            moe_scores = F.softmax(
                moe_logits + noise + mask_out, dim=-1
            )  # [B, N_layer, K_expert]
        moe_scores_log = F.softmax(moe_logits + noise, dim=-1)
        # moe_scores = F.sigmoid(moe_logits + noise )
        # [B, n_experts], where top_k experts are 1, others are 0
        moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
        if self.route_per_layer:
            moe_mask.scatter_(2, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
        else:
            moe_mask.scatter_(1, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)

        selected_expert = torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1]
        extra_out["moe_scores"] = moe_scores

        # importance loss
        sum_scores = torch.sum(moe_scores, dim=0)  # N_layer, N_expert
        std_scores = torch.std(sum_scores, dim=-1)  # N_layer
        mean_scores = torch.mean(sum_scores, dim=-1)  # N_layer
        threshold = 0.05
        importance_loss = (std_scores / mean_scores) ** 2
        importance_loss = torch.where(
            importance_loss > threshold,
            importance_loss,
            torch.zeros_like(importance_loss),
        )
        extra_out["importance_loss"] = torch.mean(importance_loss)


        entropy_loss = -torch.sum(moe_scores * torch.log(moe_scores + 1e-7), dim=-1)
        entropy_loss = torch.mean(entropy_loss)
        extra_out["entropy_loss"] = entropy_loss

        text_feature = self.instruct_proj(text_feature)
        route_score = moe_scores if self.dense_routing else moe_mask

        logits, cls_, extra_loss = self.vision_model.forward_instruct_moe(
            vision_input,
            text_feature,
            route_score,
            return_features=True,
        )
        if extra_loss is not None:
            extra_out["othor_loss"] = extra_loss
        extra_out["cls_"] = cls_
        return logits, extra_out

    def forward_instruct_multimodal_moe_t2v(self, vision_input, text_input):
        if self.train_instructor:
            _, text_feature = self.text_model(text_input, return_features=True)
        else:
            with torch.no_grad():
                _, text_feature = self.text_model(text_input, return_features=True)

        # the routing logic is implemented in the vision model
        logits, extra_out = self.vision_model.forward_instruct_multimodal_moe(
            vision_input, text_feature
        )
        return logits, extra_out

    def forward_instruct_v2t(self, vision_input, text_input, train_instructor=False):
        if True:
            _, vision_feature = self.vision_model(vision_input, return_features=True)
        else:
            with torch.no_grad():
                _, vision_feature = self.vision_model(
                    vision_input, return_features=True
                )
        vision_feature = self.instruct_proj(vision_feature)
        # vision_feature = vision_feature.unsqueeze(1)
        logits, cls_ = self.text_model.forward_instruct(
            text_input, vision_embedding=vision_feature
        )
        extra_out = dict()
        extra_out["cls_"] = cls_
        return logits, extra_out

    def forward_instruct_moe_v2t(
        self,
        vision_input,
        text_input,
    ):
        if self.train_instructor:
            _, vision_feature = self.vision_model(vision_input, return_features=True)
        else:
            with torch.no_grad():
                _, vision_feature = self.vision_model(
                    vision_input, return_features=True
                )
        extra_out = dict()
        # if self.route_per_layer:
        moe_logits = torch.stack(
            [self.moe_proj[i](vision_feature) for i in range(13)], dim=1
        )  # B, N_layer, N_expert
        # else:
        #     moe_logits = self.moe_proj(vision_feature)
        temperature = 0.1
        moe_logits = moe_logits / temperature
        # add normal dis N(0, 1/n_experts^2)
        noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (
            self.moe_n_experts**2
        )
        if False:
            mask_out = torch.randint_like(moe_logits, 0, 2) * -10000
            #  # 0.01 chance
            # if torch.rand(1) < 0.01:
            #     mask_out = torch.ones_like(moe_logits) * -10000
        else:
            mask_out = torch.zeros_like(moe_logits)
        if moe_logits.shape[-1] == 1:
            moe_scores = torch.zeros_like(moe_logits)
        else:
            moe_scores = F.softmax(moe_logits + noise + mask_out, dim=-1)
        moe_scores_log = F.softmax(moe_logits + noise, dim=-1)
        # moe_scores = F.sigmoid(moe_logits + noise )
        # [B, n_experts], where top_k experts are 1, others are 0
        moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
        # if self.route_per_layer:
        moe_mask.scatter_(2, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
        # else:
        #     moe_mask.scatter_(1, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)

        selected_expert = torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1]
        extra_out["moe_scores"] = moe_scores

        # importance loss
        sum_scores = torch.sum(moe_scores, dim=0)  # N_layer, N_expert
        std_scores = torch.std(sum_scores, dim=-1)  # N_layer
        mean_scores = torch.mean(sum_scores, dim=-1)  # N_layer
        threshold = 0.05
        importance_loss = (std_scores / mean_scores) ** 2
        importance_loss = torch.where(
            importance_loss > threshold,
            importance_loss,
            torch.zeros_like(importance_loss),
        )
        extra_out["importance_loss"] = torch.mean(importance_loss)

        # entropy loss·
        # todo check here
        moe_scores_prob = moe_scores_log / torch.sum(
            moe_scores_log, dim=1, keepdim=True
        )
        entropy_loss = -torch.sum(
            moe_scores_prob * torch.log(moe_scores_prob + 1e-7), dim=1
        )
        entropy_loss = torch.mean(entropy_loss)
        extra_out["entropy_loss"] = entropy_loss

        vision_feature = self.instruct_proj(vision_feature)
        route_score = moe_scores if True else moe_mask
        logits = self.text_model.forward_instruct_moe(
            text_input, vision_feature, route_score
        )
        return logits, extra_out

    def forward_text(self, text_input):
        return self.text_model(text_input), None

    def forward_vision(self, vision_input):
        return self.vision_model(vision_input), None


# class AudioTextClassifiers(nn.Module):
#     def __init__(
#         self,
#         audio_model,
#         text_model,
#         num_classes,
#         fusion_method="ens",
#         train_instructor=True,
#         moe_n_experts=8,
#         moe_top_k=1,
#         dense_routing=False,
#         attn_fuse=False,
#         route_per_layer=False,
#     ):
#         super(AudioTextClassifiers, self).__init__()
#         self.audio_model = audio_model
#         self.text_model = text_model
#         self.fusion_head = None
#         self.instruct_proj = None
#         self.moe_proj = None
#         self.fuse_method = fusion_method
#         self.train_instructor = train_instructor
#         self.moe_top_k = None
#         self.moe_n_experts = None
#         self.dense_routing = dense_routing
#         self.attn_fuse = attn_fuse
#         self.route_per_layer = route_per_layer
#         if fusion_method == "late_concat" or fusion_method == "p_late_concat":
#             self.fusion_head = nn.Linear(768 + 768, num_classes)
#         if fusion_method == "instruct_t2v":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
#             )
#         if fusion_method == "instruct_v2t" or fusion_method == "p_sequential":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384),
#                 nn.BatchNorm1d(384),
#                 nn.GELU(),
#                 nn.Linear(384, 768),
#                 nn.GELU(),
#             )  # same design as the mapper
#             # self.instruct_proj = nn.Sequential(
#             #     nn.Linear(1024,768)
#             # ) #linear
#         if fusion_method == "instruct_moe_t2v":
#             #!HARDCODED Jan 25: here changed 384  and n for vit
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
#             )
#             if self.route_per_layer:
#                 self.n_route_layers = 2 + 2 + 18 + 2 - 1
#                 if self.audio_model.__class__.__name__ == "CLIPViT":
#                     self.n_route_layers = 12

#                 self.moe_proj = nn.ModuleList(
#                     [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
#                 )
#             else:
#                 self.moe_proj = nn.Linear(768, moe_n_experts)
#             self.moe_top_k = moe_top_k
#             self.moe_n_experts = moe_n_experts

#         if fusion_method == "instruct_mm_moe_t2v":

#             self.moe_top_k = moe_top_k
#             self.moe_n_experts = moe_n_experts
#         if fusion_method == "instruct_moe_v2t":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384),
#                 nn.BatchNorm1d(384),
#                 nn.GELU(),
#             )

#             # if self.route_per_layer:
#             self.n_route_layers = 13
#             self.moe_proj = nn.ModuleList(
#                 [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
#             )  # 4 swin enc layers
#             # else:
#             #     self.moe_proj = nn.Linear(1024, moe_n_experts)
#             self.moe_top_k = moe_top_k
#             self.moe_n_experts = moe_n_experts

#         if fusion_method == "sequentialfuse" or fusion_method == "p_sequential":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384),
#                 nn.BatchNorm1d(384),
#                 nn.GELU(),
#             )  # same design as the mapper

#     def forward(self, audio_input, text_input):
#         """
#         return :
#             logits: final prediction
#             extra_out: a dict of extra return key and values
#         """
#         if self.fuse_method == "ens":
#             return self.forward_ens(audio_input, text_input)
#         elif self.fuse_method == "late_concat" or self.fuse_method == "p_late_concat":
#             return self.forward_late_concat(audio_input, text_input)
#         elif self.fuse_method == "instruct_t2v":
#             return self.forward_instruct_t2v(audio_input, text_input)
#         elif self.fuse_method == "instruct_v2t":
#             return self.forward_instruct_v2t(audio_input, text_input)
#         elif self.fuse_method == "instruct_moe_t2v":
#             return self.forward_instruct_moe_t2v(audio_input, text_input)
#         elif self.fuse_method == "instruct_mm_moe_t2v":
#             return self.forward_instruct_multimodal_moe_t2v(audio_input, text_input)
#         elif self.fuse_method == "instruct_moe_v2t":
#             return self.forward_instruct_moe_v2t(audio_input, text_input)
#         elif self.fuse_method == "sequentialfuse" or self.fuse_method == "p_sequential":
#             return self.forward_instruct_v2t(audio_input, text_input)
#         if self.fuse_method == "img_only":
#             return self.forward_vision(audio_input)
#         if self.fuse_method == "text_only":
#             return self.forward_text(text_input)

#     def forward_ens(self, vision_input, text_input):
#         vision_logits = self.audio_model(vision_input)
#         text_logits = self.text_model(text_input)
#         logits = vision_logits + text_logits
#         return logits / 2.0, None

#     def forward_late_concat(self, audio_input, text_input):
#         _, vision_feature = self.audio_model(audio_input, return_features=True)
#         _, audio_feature = self.text_model(text_input, return_features=True)
#         concat_feature = torch.cat([vision_feature, audio_feature], dim=-1)
#         logits = self.fusion_head(concat_feature)
#         extra_out = dict()
#         extra_out["cls_"] = concat_feature
#         return logits, extra_out

#     def forward_instruct_t2v(self, vision_input, text_input):
#         if self.train_instructor:
#             _, text_feature = self.text_model(text_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, text_feature = self.text_model(text_input, return_features=True)
#         text_feature = self.instruct_proj(text_feature)
#         logits = self.audio_model.forward_instruct(vision_input, text_feature)
#         return logits, None

#     def forward_instruct_multimodal_moe_t2v(self, vision_input, text_input):
#         if self.train_instructor:
#             _, text_feature = self.text_model(text_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, text_feature = self.text_model(text_input, return_features=True)

#         # the routing logic is implemented in the vision model
#         logits, extra_out = self.audio_model.forward_instruct_multimodal_moe(
#             vision_input, text_feature
#         )
#         return logits, extra_out

#     def forward_instruct_v2t(self, vision_input, text_input, train_instructor=False):
#         if True:
#             _, vision_feature = self.audio_model(vision_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, vision_feature = self.audio_model(vision_input, return_features=True)
#         vision_feature = self.instruct_proj(vision_feature)
#         # vision_feature = vision_feature.unsqueeze(1)
#         logits, cls_ = self.text_model.forward_seqfuse(
#             text_input, vision_input=vision_feature
#         )
#         extra_out = dict()
#         extra_out["cls_"] = cls_
#         return logits, extra_out

#     def forward_instruct_moe_v2t(
#         self,
#         vision_input,
#         text_input,
#     ):
#         if self.train_instructor:
#             _, vision_feature = self.audio_model(vision_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, vision_feature = self.audio_model(vision_input, return_features=True)
#         extra_out = dict()
#         # if self.route_per_layer:
#         moe_logits = torch.stack(
#             [self.moe_proj[i](vision_feature) for i in range(13)], dim=1
#         )  # B, N_layer, N_expert
#         # else:
#         #     moe_logits = self.moe_proj(vision_feature)
#         temperature = 0.1
#         moe_logits = moe_logits / temperature
#         # add normal dis N(0, 1/n_experts^2)
#         noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (
#             self.moe_n_experts**2
#         )
#         if False:
#             mask_out = torch.randint_like(moe_logits, 0, 2) * -10000
#             #  # 0.01 chance
#             # if torch.rand(1) < 0.01:
#             #     mask_out = torch.ones_like(moe_logits) * -10000
#         else:
#             mask_out = torch.zeros_like(moe_logits)
#         if moe_logits.shape[-1] == 1:
#             moe_scores = torch.zeros_like(moe_logits)
#         else:
#             moe_scores = F.softmax(moe_logits + noise + mask_out, dim=-1)
#         moe_scores_log = F.softmax(moe_logits + noise, dim=-1)
#         # moe_scores = F.sigmoid(moe_logits + noise )
#         # [B, n_experts], where top_k experts are 1, others are 0
#         moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
#         # if self.route_per_layer:
#         moe_mask.scatter_(2, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
#         # else:
#         #     moe_mask.scatter_(1, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)

#         selected_expert = torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1]
#         extra_out["moe_scores"] = moe_scores

#         # importance loss
#         sum_scores = torch.sum(moe_scores, dim=0)  # N_layer, N_expert
#         std_scores = torch.std(sum_scores, dim=-1)  # N_layer
#         mean_scores = torch.mean(sum_scores, dim=-1)  # N_layer
#         threshold = 0.05
#         importance_loss = (std_scores / mean_scores) ** 2
#         importance_loss = torch.where(
#             importance_loss > threshold,
#             importance_loss,
#             torch.zeros_like(importance_loss),
#         )
#         extra_out["importance_loss"] = torch.mean(importance_loss)

#         # entropy loss·
#         # todo check here
#         moe_scores_prob = moe_scores_log / torch.sum(
#             moe_scores_log, dim=1, keepdim=True
#         )
#         entropy_loss = -torch.sum(
#             moe_scores_prob * torch.log(moe_scores_prob + 1e-7), dim=1
#         )
#         entropy_loss = torch.mean(entropy_loss)
#         extra_out["entropy_loss"] = entropy_loss

#         vision_feature = self.instruct_proj(vision_feature)
#         route_score = moe_scores if True else moe_mask
#         logits = self.text_model.forward_instruct_moe(
#             text_input, vision_feature, route_score
#         )
#         # extra_out["othor_loss"] = extra_loss
#         return logits, extra_out

#     def forward_text(self, text_input):
#         return self.text_model(text_input), None

#     def forward_vision(self, vision_input):
#         return self.audio_model(vision_input), None



# class VideoAudioTextClassifiers(nn.Module):
#     def __init__(
#         self,
#         video_model,
#         audio_model,
#         text_model,
#         num_classes,
#         fusion_method="ens",
#         train_instructor=True,
#         moe_n_experts=8,
#         moe_top_k=1,
#         dense_routing=False,
#         attn_fuse=False,
#         route_per_layer=False,
#     ):
#         super(VideoAudioTextClassifiers, self).__init__()
#         self.video_model = video_model
#         self.audio_model = audio_model
#         self.text_model = text_model
#         self.fusion_head = None
#         self.instruct_proj = None
#         self.moe_proj = None
#         self.fuse_method = fusion_method
#         self.train_instructor = train_instructor
#         self.moe_top_k = None
#         self.moe_n_experts = None
#         self.dense_routing = dense_routing
#         self.attn_fuse = attn_fuse
#         self.route_per_layer = route_per_layer
#         if fusion_method == "late_concat":
#             self.fusion_head = nn.Sequential(
#                 nn.Linear(768 + 768 + 768, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, num_classes),
#             )
#         if fusion_method == "instruct_t2v":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
#             )
#         if fusion_method == "instruct_v2t" or fusion_method == "p_sequential":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(1024, 384),
#                 nn.BatchNorm1d(384),
#                 nn.GELU(),
#                 nn.Linear(384, 768),
#                 nn.GELU(),
#             )  # same design as the mapper
#             # self.instruct_proj = nn.Sequential(
#             #     nn.Linear(1024,768)
#             # ) #linear
#         if fusion_method == "instruct_moe_t2v":
#             #!HARDCODED Jan 25: here changed 384  and n for vit
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
#             )
#             if self.route_per_layer:
#                 self.n_route_layers = 2 + 2 + 18 + 2 - 1
#                 if self.audio_model.__class__.__name__ == "CLIPViT":
#                     self.n_route_layers = 12

#                 self.moe_proj = nn.ModuleList(
#                     [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
#                 )
#             else:
#                 self.moe_proj = nn.Linear(768, moe_n_experts)
#             self.moe_top_k = moe_top_k
#             self.moe_n_experts = moe_n_experts

#         if fusion_method == "instruct_mm_moe_t2v":
#             self.moe_top_k = moe_top_k
#             self.moe_n_experts = moe_n_experts
#         if fusion_method == "instruct_moe_v2t":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(768, 384),
#                 nn.BatchNorm1d(384),
#                 nn.GELU(),
#             )

#             # if self.route_per_layer:
#             self.n_route_layers = 13
#             self.moe_proj = nn.ModuleList(
#                 [nn.Linear(768, moe_n_experts) for _ in range(self.n_route_layers)]
#             )  # 4 swin enc layers
#             # else:
#             #     self.moe_proj = nn.Linear(1024, moe_n_experts)
#             self.moe_top_k = moe_top_k
#             self.moe_n_experts = moe_n_experts
#         if fusion_method == "sequentialfuse":
#             self.instruct_proj = nn.Sequential(
#                 nn.Linear(1024, 512),
#                 nn.BatchNorm1d(512),
#                 nn.GELU(),
#                 nn.Linear(512, 768),
#                 nn.GELU(),
#             )  # same design as the mapper


#     def forward(self, video_input, audio_input, text_input):
#         """
#         return :
#             logits: final prediction
#             extra_out: a dict of extra return key and values
#         """
#         if self.fuse_method == "ens":
#             return self.forward_ens(audio_input, text_input)
#         elif self.fuse_method == "late_concat":
#             return self.forward_late_concat(video_input, audio_input, text_input)
#         elif self.fuse_method == "instruct_t2v":
#             return self.forward_instruct_t2v(audio_input, text_input)
#         elif self.fuse_method == "instruct_v2t":
#             return self.forward_instruct_v2t(audio_input, text_input)
#         elif self.fuse_method == "instruct_moe_t2v":
#             return self.forward_instruct_moe_t2v(audio_input, text_input)
#         elif self.fuse_method == "instruct_mm_moe_t2v":
#             return self.forward_instruct_multimodal_moe_t2v(
#                 video_input, audio_input, text_input
#             )
#         elif self.fuse_method == "instruct_moe_v2t":
#             return self.forward_instruct_moe_v2t(video_input, audio_input, text_input)
#         elif self.fuse_method == "sequentialfuse" or self.fuse_method == "p_sequential":
#             return self.forward_instruct_v2t(video_input, audio_input, text_input)
#         if self.fuse_method == "img_only":
#             return self.forward_vision(audio_input)
#         if self.fuse_method == "text_only":
#             return self.forward_text(text_input)

#     def forward_ens(self, vision_input, text_input):
#         vision_logits = self.audio_model(vision_input)
#         text_logits = self.text_model(text_input)
#         logits = vision_logits + text_logits
#         return logits / 2.0, None

#     def forward_late_concat(self, video_input, audio_input, text_input):
#         _, audio_feature = self.audio_model(audio_input, return_features=True)
#         _, video_feature = self.video_model(video_input, return_features=True)
#         # fused_feature = video_feature
#         _, text_feature = self.text_model(text_input, return_features=True)
#         concat_feature = torch.cat([audio_feature, video_feature, text_feature], dim=-1)
#         logits = self.fusion_head(concat_feature)
#         extra_out = dict()
#         extra_out["cls_"] = concat_feature
#         return logits, extra_out

#     def forward_instruct_t2v(self, vision_input, text_input):
#         if self.train_instructor:
#             _, text_feature = self.text_model(text_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, text_feature = self.text_model(text_input, return_features=True)
#         text_feature = self.instruct_proj(text_feature)
#         logits = self.audio_model.forward_instruct(vision_input, text_feature)
#         return logits, None

#     def forward_instruct_moe_t2v(self, vision_input, text_input):
#         if self.train_instructor:
#             _, text_feature = self.text_model(text_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, text_feature = self.text_model(text_input, return_features=True)
#         extra_out = dict()
#         if self.route_per_layer:
#             moe_logits = torch.stack(
#                 [self.moe_proj[i](text_feature) for i in range(self.n_route_layers)],
#                 dim=1,
#             )  # B, N_layer, N_expert
#         else:
#             moe_logits = self.moe_proj(text_feature)
#         temperature = 0.1
#         moe_logits = moe_logits / temperature
#         # add normal dis N(0, 1/n_experts^2)
#         noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (
#             self.moe_n_experts**2
#         )
#         if False:
#             mask_out = torch.randint_like(moe_logits, 0, 2) * -10000
#             #  # 0.01 chance
#             # if torch.rand(1) < 0.01:
#             #     mask_out = torch.ones_like(moe_logits) * -10000
#         else:
#             mask_out = torch.zeros_like(moe_logits)
#         if moe_logits.shape[-1] == 1:
#             moe_scores = torch.zeros_like(moe_logits)
#         else:
#             moe_scores = F.softmax(
#                 moe_logits + noise + mask_out, dim=-1
#             )  # [B, N_layer, K_expert]
#         moe_scores_log = F.softmax(moe_logits + noise, dim=-1)
#         # moe_scores = F.sigmoid(moe_logits + noise )
#         # [B, n_experts], where top_k experts are 1, others are 0
#         moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
#         if self.route_per_layer:
#             moe_mask.scatter_(2, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
#         else:
#             moe_mask.scatter_(1, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)

#         selected_expert = torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1]
#         extra_out["moe_scores"] = moe_scores

#         # importance loss
#         sum_scores = torch.sum(moe_scores, dim=0)  # N_layer, N_expert
#         std_scores = torch.std(sum_scores, dim=-1)  # N_layer
#         mean_scores = torch.mean(sum_scores, dim=-1)  # N_layer
#         threshold = 0.05
#         importance_loss = (std_scores / mean_scores) ** 2
#         importance_loss = torch.where(
#             importance_loss > threshold,
#             importance_loss,
#             torch.zeros_like(importance_loss),
#         )
#         extra_out["importance_loss"] = torch.mean(importance_loss)

#         # entropy loss·
#         # todo check here
#         # moe_scores_prob = moe_scores_log / torch.sum(moe_scores_log, dim = -1, keepdim = True)
#         # watch_layer = -1
#         # moe_scores_watch = moe_scores_log[:, watch_layer, :]
#         entropy_loss = -torch.sum(moe_scores * torch.log(moe_scores + 1e-7), dim=-1)
#         entropy_loss = torch.mean(entropy_loss)
#         extra_out["entropy_loss"] = entropy_loss

#         text_feature = self.instruct_proj(text_feature)
#         route_score = moe_scores if self.dense_routing else moe_mask
#         # if True:
#         #     #!HARDCODED Feb 22: random route!!!
#         #     route_score = torch.rand_like(route_score).to(route_score.device)
#         logits, cls_, extra_loss = self.audio_model.forward_instruct_moe(
#             vision_input,
#             text_feature,
#             route_score,
#             attn_fuse=self.attn_fuse,
#             return_features=True,
#         )
#         if extra_loss is not None:
#             extra_out["othor_loss"] = extra_loss
#         extra_out["cls_"] = cls_
#         return logits, extra_out

#     def forward_instruct_multimodal_moe_t2v(self, vision_input, text_input):
#         if self.train_instructor:
#             _, text_feature = self.text_model(text_input, return_features=True)
#         else:
#             with torch.no_grad():
#                 _, text_feature = self.text_model(text_input, return_features=True)

#         # the routing logic is implemented in the vision model
#         logits, extra_out = self.audio_model.forward_instruct_multimodal_moe(
#             vision_input, text_feature
#         )
#         return logits, extra_out

#     def forward_instruct_v2t(
#         self, vision_input, audio_input, text_input, train_instructor=False
#     ):
#         if True:
#             _, video_feature = self.video_model(vision_input, return_features=True)
#             _, audio_feature = self.audio_model(audio_input, return_features=True)
#             fused_feature = torch.cat([video_feature, audio_feature], dim=-1)

#         fused_feature = self.instruct_proj(fused_feature)
#         # vision_feature = vision_feature.unsqueeze(1)
#         logits, cls_ = self.text_model.forward_instruct(
#             text_input, vision_embedding=fused_feature
#         )
#         extra_out = dict()
#         extra_out["cls_"] = cls_
#         return logits, extra_out

#     def forward_instruct_moe_v2t(
#         self,
#         video_input,
#         audio_input,
#         text_input,
#     ):
#         if self.train_instructor:
#             _, audio_feature = self.audio_model(audio_input, return_features=True)
#             _, video_feature = self.video_model(video_input, return_features=True)
#             fused_feature = audio_feature + video_feature
#         else:
#             with torch.no_grad():
#                 _, audio_feature = self.audio_model(audio_input, return_features=True)
#                 _, video_feature = self.video_model(video_input, return_features=True)
#                 fused_feature = audio_feature + video_feature

#         extra_out = dict()
#         # if self.route_per_layer:
#         moe_logits = torch.stack(
#             [self.moe_proj[i](fused_feature) for i in range(13)], dim=1
#         )  # B, N_layer, N_expert
#         # else:
#         #     moe_logits = self.moe_proj(vision_feature)
#         temperature = 0.1
#         moe_logits = moe_logits / temperature
#         # add normal dis N(0, 1/n_experts^2)
#         noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (
#             self.moe_n_experts**2
#         )
#         if False:
#             mask_out = torch.randint_like(moe_logits, 0, 2) * -10000
#             #  # 0.01 chance
#             # if torch.rand(1) < 0.01:
#             #     mask_out = torch.ones_like(moe_logits) * -10000
#         else:
#             mask_out = torch.zeros_like(moe_logits)
#         if moe_logits.shape[-1] == 1:
#             moe_scores = torch.zeros_like(moe_logits)
#         else:
#             moe_scores = F.softmax(moe_logits + noise + mask_out, dim=-1)
#         moe_scores_log = F.softmax(moe_logits + noise, dim=-1)
#         # moe_scores = F.sigmoid(moe_logits + noise )
#         # [B, n_experts], where top_k experts are 1, others are 0
#         moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
#         # if self.route_per_layer:
#         moe_mask.scatter_(2, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)
#         # else:
#         #     moe_mask.scatter_(1, torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1], 1)

#         selected_expert = torch.topk(moe_scores, k=self.moe_top_k, dim=-1)[1]
#         extra_out["moe_scores"] = moe_scores

#         # importance loss
#         sum_scores = torch.sum(moe_scores, dim=0)  # N_layer, N_expert
#         std_scores = torch.std(sum_scores, dim=-1)  # N_layer
#         mean_scores = torch.mean(sum_scores, dim=-1)  # N_layer
#         threshold = 0.05
#         importance_loss = (std_scores / mean_scores) ** 2
#         importance_loss = torch.where(
#             importance_loss > threshold,
#             importance_loss,
#             torch.zeros_like(importance_loss),
#         )
#         extra_out["importance_loss"] = torch.mean(importance_loss)

#         # entropy loss·
#         # todo check here
#         moe_scores_prob = moe_scores_log / torch.sum(
#             moe_scores_log, dim=1, keepdim=True
#         )
#         entropy_loss = -torch.sum(
#             moe_scores_prob * torch.log(moe_scores_prob + 1e-7), dim=1
#         )
#         entropy_loss = torch.mean(entropy_loss)
#         extra_out["entropy_loss"] = entropy_loss

#         fused_feature = self.instruct_proj(fused_feature)
#         route_score = moe_scores if True else moe_mask
#         logits = self.text_model.forward_instruct_moe(
#             text_input, fused_feature, route_score
#         )
#         # extra_out["othor_loss"] = extra_loss
#         return logits, extra_out

#     def forward_text(self, text_input):
#         return self.text_model(text_input), None

#     def forward_vision(self, vision_input):
#         return self.audio_model(vision_input), None
