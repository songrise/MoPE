# -*- coding : utf-8 -*-
# @FileName  : bart.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Oct 13, 2023
# @Github    : https://github.com/songrise
# @Description: warpper class for bart (for sequence generation)


import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import (
    BartTokenizerFast,
    BartForSequenceClassification,
    BartModel,
    BartForConditionalGeneration,
)
from transformers.models.bart.modeling_bart import _expand_mask, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class PromptedBartEncoder(nn.Module):
    """
    Bart encoder with prompt tuning.
    """

    def __init__(
        self,
        use_prompt: bool = True,
        moe_n_experts: int = 8,
        use_static_prompt=True,
        prompt_length: int = 10,
        use_instruct: bool = True,
    ):
        super(PromptedBartEncoder, self).__init__()
        self.bart_encoder = BartModel.from_pretrained("facebook/bart-base").encoder
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        


        self.use_prompt = use_prompt
        self.use_static_prompt = use_static_prompt
        self.prompt_length = prompt_length if use_prompt else 0
        self.prompt_vector = None
        self.embed_dim = self.bart_encoder.config.d_model
        self.use_instruct = use_instruct
        if use_prompt:
            #!HARDCODED Oct 13: similar with swin, default init with moe expert, but use the first if moe is not used
            self.prompt_vectors = []
            self.prompt_dropout = nn.Dropout(0.1)
            # prompt expert for each layer
            for _ in range(self.bart_encoder.config.num_hidden_layers):
                _n_prompt_vec = (
                    moe_n_experts + 1 if use_static_prompt else moe_n_experts
                )
                _prompt_vec = nn.Parameter(
                    torch.randn(
                        _n_prompt_vec, prompt_length, self.bart_encoder.config.d_model
                    )
                )
                nn.init.uniform_(_prompt_vec, -0.3, 0.3)
                self.prompt_vectors.append(_prompt_vec)
            self.prompt_vectors = nn.ParameterList(self.prompt_vectors)
            if self.use_static_prompt:
                self.prompt_length = self.prompt_length * 2
            if self.use_instruct:
                self.prompt_length = self.prompt_length + 1
                # additional projection from vision
                self.prompt_proj_act = nn.GELU()
                #!HARDCODED Oct 13: assume vis model dim 768
                self.prompt_proj_0 = nn.Linear(768, self.embed_dim)
                self.prompt_proj_1 = nn.Linear(768, self.embed_dim)
        self.global_pooler = nn.AdaptiveAvgPool1d(1)
        # freeze all parameter with out prompt name
        for name, param in self.named_parameters():
            if "prompt" not in name:
                param.requires_grad = False      

    def forward(self, text_input, prompt_embeddings=None, **kwargs):
        # text_input: List[str]
        # return: text representations
        tokenizer_output = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=1010)
        input_ids = tokenizer_output["input_ids"].to(self.bart_encoder.device)

        attention_mask = tokenizer_output["attention_mask"].to(self.bart_encoder.device)
        if not self.use_prompt:
            bart_output = self.bart_encoder(input_ids, attention_mask=attention_mask)
            return bart_output["last_hidden_state"]
        # else: use prompt
 
        prompt_start_idx = 1
        attention_mask = torch.cat(
            [
                attention_mask[:, :prompt_start_idx],
                torch.ones(attention_mask.size(0), self.prompt_length).to(
                    self.bart_encoder.device
                ),
                attention_mask[:, prompt_start_idx:],
            ],
            dim=1,
        )

        txt_tokens = (
            self.bart_encoder.embed_tokens(input_ids) * self.bart_encoder.embed_scale
        )
        # # add dummy token to input_ids for pos embd
        # input_ids = torch.cat(
        #      [
        #         input_ids[:, :prompt_start_idx],
        #         torch.zeros(input_ids.size(0), self.prompt_length).to(
        #             self.bart_encoder.device
        #         ).long(),
        #         input_ids[:, prompt_start_idx:],
        #      ], dim = 1
        # )
        embed_pos = self.bart_encoder.embed_positions(input_ids)
        txt_tokens = txt_tokens + embed_pos
        txt_tokens = self.bart_encoder.layernorm_embedding(txt_tokens)
        txt_tokens = F.dropout(
            txt_tokens, p=self.bart_encoder.dropout, training=self.training
        )

        txt_attn_mask = self.get_extended_txt_attn_mask(attention_mask)
        # txt_attn_mask = _expand_mask(txt_attn_mask, txt_tokens.dtype)
        for bart_layer_id in range(self.bart_encoder.config.num_hidden_layers):
            B = txt_tokens.size(0)
            static_prompt_experts = self.prompt_vectors[bart_layer_id][0, ...].expand(B, -1, -1) 
            crt_prompt_embeddings = static_prompt_experts
            crt_prompt_embeddings = self.prompt_dropout(crt_prompt_embeddings)
            txt_tokens = torch.cat(
                [
                    txt_tokens[:, :prompt_start_idx, :],
                    crt_prompt_embeddings,
                    txt_tokens[:, prompt_start_idx:, :],
                ],
                dim=1,
            )

            layer_output = self.bart_encoder.layers[bart_layer_id](
                txt_tokens, attention_mask=None, layer_head_mask=None
            )
            txt_tokens = layer_output[0]
            txt_tokens = torch.cat(
                [
                    txt_tokens[:, :prompt_start_idx, :],
                    txt_tokens[:, self.prompt_length + prompt_start_idx :, :],
                ],
                dim=1,
            )
        # txt_tokens = self.global_pooler(txt_tokens.transpose(1, 2)).squeeze(-1)
        txt_tokens = txt_tokens[:, 0, :]
        return txt_tokens, txt_tokens

    def forward_instruct(self, text_input, prompt_embeddings=None, **kwargs):
        # text_input: List[str]
        # return: text representations
        tokenizer_output = self.tokenizer(text_input, return_tensors="pt", padding=True)
        input_ids = tokenizer_output["input_ids"].to(self.bart_encoder.device)

        attention_mask = tokenizer_output["attention_mask"].to(self.bart_encoder.device)
        if not self.use_prompt:
            bart_output = self.bart_encoder(input_ids, attention_mask=attention_mask)
            return bart_output["last_hidden_state"]
        # else: use prompt
        assert (
            prompt_embeddings is not None
        ), "prompt embeddings should be provided if prompt is used"
        prompt_start_idx = 1
        attention_mask = torch.cat(
            [
                attention_mask[:, :prompt_start_idx],
                torch.ones(attention_mask.size(0), self.prompt_length).to(
                    self.bart_encoder.device
                ),
                attention_mask[:, prompt_start_idx:],
            ],
            dim=1,
        )

        txt_tokens = (
            self.bart_encoder.embed_tokens(input_ids) * self.bart_encoder.embed_scale
        )
        # # add dummy token to input_ids for pos embd
        # input_ids = torch.cat(
        #      [
        #         input_ids[:, :prompt_start_idx],
        #         torch.zeros(input_ids.size(0), self.prompt_length).to(
        #             self.bart_encoder.device
        #         ).long(),
        #         input_ids[:, prompt_start_idx:],
        #      ], dim = 1
        # )
        embed_pos = self.bart_encoder.embed_positions(input_ids)
        txt_tokens = txt_tokens + embed_pos
        txt_tokens = self.bart_encoder.layernorm_embedding(txt_tokens)
        txt_tokens = F.dropout(
            txt_tokens, p=self.bart_encoder.dropout, training=self.training
        )

        txt_attn_mask = self.get_extended_txt_attn_mask(attention_mask)
        # txt_attn_mask = _expand_mask(txt_attn_mask, txt_tokens.dtype)
        for bart_layer_id in range(self.bart_encoder.config.num_hidden_layers):
            crt_prompt_embeddings = prompt_embeddings[bart_layer_id]
            crt_prompt_embeddings = self.prompt_dropout(crt_prompt_embeddings)
            txt_tokens = torch.cat(
                [
                    txt_tokens[:, :prompt_start_idx, :],
                    crt_prompt_embeddings,
                    txt_tokens[:, prompt_start_idx:, :],
                ],
                dim=1,
            )

            layer_output = self.bart_encoder.layers[bart_layer_id](
                txt_tokens, attention_mask=None, layer_head_mask=None
            )
            txt_tokens = layer_output[0]
            txt_tokens = torch.cat(
                [
                    txt_tokens[:, :prompt_start_idx, :],
                    txt_tokens[:, self.prompt_length + prompt_start_idx :, :],
                ],
                dim=1,
            )

        # return (
        #     BaseModelOutput(
        #         last_hidden_state=txt_tokens,
        #         hidden_states=txt_tokens,
        #         attentions=None,
        #     ),
        #     input_ids,
        # )
        return txt_tokens, input_ids

    def forward_instruct_moe(self, text_input, vision_input, route_score):
        """
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        [B, n_layer, n_expert] for per-layer routed moe
        """
        B = vision_input.size(0)
        if len(route_score.shape) == 2:
            route_per_layer = False
        elif len(route_score.shape) == 3:
            route_per_layer = True
        y0 = self.prompt_proj_act(self.prompt_proj_0(vision_input))
        y1 = self.prompt_proj_act(self.prompt_proj_1(vision_input))
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
            # concate the static prompt
            crt_prompt = torch.cat(
                [static_prompt_experts[i].squeeze(1), crt_prompt], dim=1
            )  # dim 1 is l
            # concate projected prompt
            projected_prompt = y0 if i < 6 else y1
            crt_prompt = torch.cat([crt_prompt, projected_prompt.unsqueeze(1)], dim=1)
            moe_prompt_embds.append(crt_prompt)

        return self.forward_instruct(text_input, prompt_embeddings=moe_prompt_embds)

    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def freeze_backbone(self):
        """
        freeze the backbone of the model, except the prompt vector and head
        """
        for name, param in self.named_parameters():
            if "prompt" not in name and "head" not in name:
                param.requires_grad = False
                
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class PromptedBartForGeneration(nn.Module):
    """
    Wrapper class for bart generation
    """

    def __init__(self, bart_encoder: PromptedBartEncoder):
        super(PromptedBartForGeneration, self).__init__()
        self.bart_encoder = bart_encoder
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.bart_decoder = model.get_decoder()
        self.config = model.model.config
        self.lm_head = model.lm_head
        self.final_logits_bias = model.final_logits_bias
        del model

    def forward(self, text_input, prompt_embeddings=None, **kwargs):
        # text_input: List[str]
        # return: text representations
        encoder_outputs, input_ids = self.bart_encoder(
            text_input, prompt_embeddings=prompt_embeddings
        )
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )
        decoder_outputs = self.bart_decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            **kwargs
        )
        lm_logit = (
            self.lm_head(decoder_outputs[0]) + self.bart_decoder.final_logits_bias.to(decoder_outputs[0].device)
        )

        return Seq2SeqLMOutput(
            logits=lm_logit,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def forward_instruct_moe(self, text_input, vision_input, route_score, **kwargs):
        encoder_outputs, input_ids = self.bart_encoder.forward_instruct_moe(
            text_input, vision_input, route_score
        )
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )
        decoder_outputs = self.bart_decoder(
            input_ids=decoder_input_ids,
            # encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs,
            **kwargs
        )
        lm_logit = self.lm_head(decoder_outputs[0]) + self.final_logits_bias.to(decoder_outputs[0].device)

        return Seq2SeqLMOutput(
            logits=lm_logit,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            #TODO Jan 10: encoder_outputs shall not be useful
            encoder_last_hidden_state=encoder_outputs,
            encoder_hidden_states=encoder_outputs,
            encoder_attentions=encoder_outputs,
        )

    def freeze_backbone(self):
        """
        freeze the backbone of the model, except the prompt vector and head
        """
        self.bart_encoder.freeze_backbone()

    def tokenize(self, texts):
        return self.bart_encoder.tokenizer(texts, return_tensors="pt", padding=True)

    def decode(self, ids):
        return self.bart_encoder.tokenizer.decode(ids)

    def batch_decode(self, ids):
        return self.bart_encoder.tokenizer.batch_decode(ids)

    def pad_token(self, tensor, max_len):
        return self.bart_encoder._pad_tensors_to_max_len(tensor, max_len)



if __name__ == "__main__":
    enc = PromptedBartEncoder()
    gen = PromptedBartForGeneration(enc)
    txt_input = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "This is a test sentence",
    ]

    vision_input = torch.randn(3, 768)
    route_score = torch.randn(3, 8)
    output = gen.forward_instruct_moe(txt_input, vision_input, route_score)
    # print(output.shape)

    # print(output)
    # translate into natrual language
    # convert logits to tokens
    print(output.logits)
    rand_logit = torch.randn(output.logits.shape)
    pred_ids = F.softmax(output.logits, dim=2)
    pred_ids = torch.argmax(pred_ids, dim=2)
    print(enc.tokenizer.decode(pred_ids[1]))

    rand_ids = F.softmax(rand_logit, dim=2)
    rand_ids = torch.argmax(rand_ids, dim=2)
