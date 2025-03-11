# -*- coding : utf-8 -*-
# @FileName  : wav2vec.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Oct 12, 2023
# @Github    : https://github.com/songrise
# @Description: wav2vec2 model
import os

os.environ["HF_HOME"] = "/root/autodl-tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/.cache"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    Wav2Vec2Config,
)
import librosa


class Wav2Vec2(nn.Module):
    def __init__(self) -> None:
        super(Wav2Vec2, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.cfg = Wav2Vec2Config()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h", config=self.cfg
        )
        self.wav2vec2.to("cuda")
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, return_features=False):
        a_feat = self.processor(
            x, return_tensors="pt", sampling_rate=16000, padding=True
        ).input_values.to("cuda")
        # cast to model precision
        a_feat = a_feat.to(self.wav2vec2.dtype)
        a_embed = self.wav2vec2(a_feat[0])["last_hidden_state"]
        a_pooled = self.pooler(a_embed.transpose(1, 2)).squeeze(2)
        if return_features:
            return a_pooled, a_pooled
        return a_pooled

    def freeze_backbone(self):
        self.wav2vec2.requires_grad = False


if __name__ == "__main__":
    wav2vec2 = Wav2Vec2()
    wav_path = "/root/autodl-tmp/mmsd/mmsd_raw_data/utterances_final/1_60.wav"
    wav_file = librosa.load(wav_path, sr=16000)[0]
    wav_file = torch.tensor(wav_file).unsqueeze(0).to("cuda")
    print(wav_file)
    print(wav_file.shape)
    wav2vec2_out = wav2vec2(wav_file)
    print(wav2vec2_out)
    print(wav2vec2_out.shape)
