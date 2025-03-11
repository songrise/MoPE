# -*- coding : utf-8 -*-
# @FileName  : bert.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 09, 2023
# @Github    : https://github.com/songrise
# @Description: Bert model for text classification

from transformers.models.bert.modeling_bert import (
    BertModel,
)
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from utils import debug_utils



class BertClassifier(nn.Module):
    """
    Bert for sequence classification, with prompt tuning support.
    """

    def __init__(
        self,
        num_classes: int,
        use_prompt: bool = False,
        prompt_length: int = 0,
        prompt_depth: str = "all",
        use_instruct_moe=False,
        is_main_modal=False,
    ) -> None:
        super(BertClassifier, self).__init__()
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.num_classes = num_classes
        self.heads = nn.Linear(self.bert_encoder.config.hidden_size, num_classes)

        self.use_prompt = use_prompt
        self.prompt_length = prompt_length if use_prompt else 0
        # if use_instruct_moe:
        #     self.prompt_length = self.prompt_length * 2 + 1
        self.prompt_vector = None

        if use_prompt:
            if prompt_depth == "all":
                self.prompt_depth = self.bert_encoder.config.num_hidden_layers
            elif prompt_depth == "input":
                self.prompt_depth = 1
            self.prompt_depth = self.bert_encoder.config.num_hidden_layers
            self.prompt_vector = nn.Parameter(
                torch.randn(
                    self.prompt_depth,
                    prompt_length,
                    self.bert_encoder.config.hidden_size,
                )
            )
            self.prompt_dropout = nn.Dropout(0.1)
            nn.init.uniform_(self.prompt_vector, -0.3, 0.3)
            self.freeze_backbone()
        if True:  # this indentation for t2v, the normal pbert

            # additional projection from vision
            self.prompt_proj_act = nn.GELU()
            self.prompt_vectors = []
            #!HARDCODED Oct 13: assume vis model dim 768
            if is_main_modal:
                self.prompt_proj_0 = nn.Linear(384, 768)
                self.prompt_proj_1 = nn.Linear(384, 768)
                self.prompt_proj_2 = nn.Linear(384, 768)

            for _ in range(self.bert_encoder.config.num_hidden_layers):
                moe_n_experts = 16  if is_main_modal else 1 #hardcoded, 1 for a static prompt
                _n_prompt_vec = moe_n_experts + 1
                _prompt_vec = nn.Parameter(
                    torch.randn(_n_prompt_vec, prompt_length, 768)
                )
                nn.init.uniform_(_prompt_vec, -0.3, 0.3)
                self.prompt_vectors.append(_prompt_vec)
            self.prompt_vectors = nn.ParameterList(self.prompt_vectors)
        # if True: # this indentation for moe v2t ,


    def forward(
        self,
        text_input,
        return_features=False,
        dump_attn=False,
        prompt_depth_override=None,
        prompt_embeddings=None,
        prompt_length_override=None,
        vision_embedding=None,
        blind_prompt=False,
    ):
        # text_input: List[str]
        # return: logits
        if prompt_depth_override is not None:
            self.prompt_depth = prompt_depth_override
        if prompt_length_override is not None:
            prompt_length = prompt_length_override
        else:
            prompt_length = self.prompt_length

        tokenizer_output = self.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokenizer_output["input_ids"].to(self.bert_encoder.device)
        orig_input = self.tokenizer.decode(input_ids[0])
        token_type_ids = tokenizer_output["token_type_ids"].to(self.bert_encoder.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.bert_encoder.device)
        # txt_attn_mask = self.get_extended_txt_attn_mask(attention_mask)
        # txt_tokens = self.bert_encoder.embeddings(input_ids, token_type_ids)
        # for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers):
        #     txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](txt_tokens, attention_mask)[0]
        prompt_start_idx = 1
        # use_vision_feature = True
        # if vision_feature is not None: # assume vision feature is of same dimension as bert hidden size
        #         vision_feature = vision_feature.unsqueeze(1)
        #         prompt_start_idx  = 2
        #         use_vision_feature = True

        if not self.use_prompt:
            bert_output = self.bert_encoder(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            cls_ = bert_output["pooler_output"]
            logits = self.heads(cls_)
        else:
            if self.prompt_depth > 0:
                attention_mask = torch.cat(
                    [
                        attention_mask[:, :prompt_start_idx],
                        torch.ones(attention_mask.size(0), prompt_length).to(
                            self.bert_encoder.device
                        ),
                        attention_mask[:, prompt_start_idx:],
                    ],
                    dim=1,
                )  # add one more attention mask for prompt vector
                if blind_prompt:
                    attention_mask[
                        prompt_start_idx + prompt_length :, :prompt_length
                    ] = 0
            txt_tokens = self.bert_encoder.embeddings(input_ids, token_type_ids)
            # if use_vision_feature:
            #     #when use vision feature: input sequence is [CLS], [IMG], learnable prompt, text tokens, [SEP]
            #     txt_tokens = torch.cat([
            #         txt_tokens[:,:1,:],
            #         vision_feature,
            #         txt_tokens[:,1:,:]
            #         ],dim = 1)
            #     attention_mask = torch.cat([
            #         attention_mask[:,:1],
            #         torch.ones(attention_mask.size(0), 1).to(self.bert_encoder.device),
            #         attention_mask[:,1:]
            #         ], dim=1) # add one more attention mask for vision feature
            txt_attn_mask = self.get_extended_txt_attn_mask(attention_mask)

            for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers):
                if (
                    vision_embedding is not None and prompt_embeddings is not None
                ):  # both are used, it means promptfuse
                    mapped_prompt = self.prompt_dropout(vision_embedding)
                    static_prompt = prompt_embeddings[bert_layer_id].squeeze(1)
                    crt_prompt_vector = torch.cat(
                        [mapped_prompt.unsqueeze(1), static_prompt], dim=1
                    )

                elif prompt_embeddings is not None:
                    crt_prompt_vector = self.prompt_dropout(
                        prompt_embeddings[bert_layer_id]
                    )
                else:
                    crt_prompt_vector = self.prompt_dropout(
                        self.prompt_vector[bert_layer_id]
                        .unsqueeze(0)
                        .repeat(input_ids.size(0), 1, 1)
                    )
                if bert_layer_id < self.prompt_depth and self.prompt_depth > 1:
                    # insert prompt vector between [CLS] and the first token
                    txt_tokens = torch.cat(
                        [
                            txt_tokens[:, :prompt_start_idx, :],
                            crt_prompt_vector,
                            txt_tokens[:, prompt_start_idx:, :],
                        ],
                        dim=1,
                    )

                layer_output = self.bert_encoder.encoder.layer[bert_layer_id](
                    txt_tokens, txt_attn_mask, output_attentions=True
                )
                if dump_attn:
                    debug_utils.dump_tensor(
                        layer_output[1][0],
                        f"bert_layer_{bert_layer_id}_attn",
                        "./debug/dump",
                    )
                txt_tokens = layer_output[0]

                # remove prompt vector
                if bert_layer_id < self.prompt_depth:
                    txt_tokens = torch.cat(
                        [
                            txt_tokens[:, :prompt_start_idx, :],
                            txt_tokens[:, prompt_length + prompt_start_idx :, :],
                        ],
                        dim=1,
                    )
            cls_ = txt_tokens[:, 0, :]
            logits = self.heads(cls_)
        if not return_features:
            return logits
        else:
            return logits, cls_

    def forward_instruct(
        self,
        text_input,
        return_features=False,
        prompt_embeddings=None,
        prompt_length_override=None,
        vision_embedding=None,
        blind_prompt=False,
        prompt_depth_override=None,
    ):
        if (
            vision_embedding is not None
        ):  # when use vision embedding, then it means that we ablate promptfuse baseline
            self.prompt_length = 1
            B = vision_embedding.size(0)
            # init the static prompt embedding
            static_prompt_experts = [
                p[0, ...].unsqueeze(0).expand(B, -1, -1, -1)
                for p in self.prompt_vectors
            ]
            prompt_embeddings = static_prompt_experts
            prompt_length_override = static_prompt_experts[0].shape[2] + 1
            logit, cls_ = self.forward(
                text_input,
                return_features=True,
                prompt_embeddings=prompt_embeddings,
                prompt_length_override=prompt_length_override,
                vision_embedding=vision_embedding,
                blind_prompt=blind_prompt,
                prompt_depth_override=prompt_depth_override,
            )
            return logit, cls_

        return self.forward(
            text_input,
            return_features,
            prompt_embeddings=prompt_embeddings,
            prompt_length_override=prompt_length_override,
            vision_embedding=vision_embedding,
        )

    def forward_instruct_moe(
        self, text_input, vision_input, route_score, return_features=False
    ):
        """
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        [B, n_layer, n_expert] for per-layer routed moe
        """

        B = vision_input.size(0)
        if len(route_score.shape) == 2:
            route_per_layer = False
        elif len(route_score.shape) == 3:
            route_per_layer = True
        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(vision_input)))
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(vision_input)))
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(vision_input)))
        all_prompt_experts = [
            p[1:, ...].expand(B, -1, -1, -1) for p in self.prompt_vectors
        ]
        static_prompt_experts = [
            p[0, ...].unsqueeze(0).expand(B, -1, -1, -1) for p in self.prompt_vectors
        ]

        moe_prompt_embds = []
        for i in range(len(all_prompt_experts)):
            if route_per_layer:
                # b batch, k expert, l seq len, h hidden dim
                crt_prompt = torch.einsum(
                    "bk,bklh->blh", route_score[:, i, :], all_prompt_experts[i]
                )
            else:
                crt_prompt = torch.einsum(
                    "bk,bklh->blh", route_score, all_prompt_experts[i]
                )

            # concate projected prompt
            if i < 4:
                projected_prompt = y0
            elif i < 8:
                projected_prompt = y1
            else:
                projected_prompt = y2
            projected_prompt = projected_prompt.unsqueeze(1)
            crt_prompt = torch.cat(
                [projected_prompt, static_prompt_experts[i].squeeze(1), crt_prompt],
                dim=1,
            )  # dim 1 is l
            moe_prompt_embds.append(crt_prompt)
        prompt_length = 2 * self.prompt_length + 1
        return self.forward_instruct(
            text_input,
            prompt_embeddings=moe_prompt_embds,
            prompt_length_override=prompt_length,
        )

    def forward_seqfuse(self, text_input, vision_input, return_features=False):
        """
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        [B, n_layer, n_expert] for per-layer routed moe
        """

        B = vision_input.size(0)
        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(vision_input)))
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(vision_input)))
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(vision_input)))

        static_prompt_experts = [
            p[0, ...].unsqueeze(0).expand(B, -1, -1, -1) for p in self.prompt_vectors
        ]

        moe_prompt_embds = []
        for i in range(len(static_prompt_experts)):

            if i < 4:
                projected_prompt = y0
            elif i < 8:
                projected_prompt = y1
            else:
                projected_prompt = y2
            projected_prompt = projected_prompt.unsqueeze(1)
            crt_prompt = torch.cat(
                [projected_prompt, static_prompt_experts[i].squeeze(1)],
                dim=1,
            )  # dim 1 is l
            moe_prompt_embds.append(crt_prompt)
        prompt_length = self.prompt_length + 1
        return self.forward_instruct(
            text_input,
            prompt_embeddings=moe_prompt_embds,
            prompt_length_override=prompt_length,
            return_features=True,
        )

    def freeze_backbone(self):
        """
        freeze the backbone of the model, except the prompt vector and head
        """
        for name, param in self.named_parameters():
            if "prompt" not in name and "head" not in name:
                param.requires_grad = False

        if not self.use_prompt:
            print("[Warning] Freezeing Bert backbone but without prompt vector")

    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


if __name__ == "__main__":
    text = ["Hello, my dog is cute", "Hello, my cat is cute, too"]
    model = BertClassifier(4)
    logits = model(text)
    print(logits)
    print(F.log_softmax(logits))
