# -*- coding : utf-8 -*-
# @FileName  : bow.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Oct 02, 2023
# @Github    : https://github.com/songrise
# @Description: bag of words model for text classification
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import torch.nn.utils.rnn as rnn_utils


class BoWClassifier(nn.Module):
    def __init__(self, num_classes:int, embedding_dim:int,**kwargs) -> None:
        super(BoWClassifier,self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.get_vocab()
        self.embeddings = BertModel.from_pretrained('bert-base-uncased').embeddings.word_embeddings
        self.linear = nn.Linear(embedding_dim, num_classes)        
        self.max_seq_length = kwargs.get("max_seq_length", 512)

    
    def freeze_backbone(self):
        self.embeddings.weight.requires_grad = False
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

    def forward(self, text_input, **kwargs):
        # text_input: List[str]
        # return: logits
        # tokenize
        encoded_inputs = self.tokenizer(text_input, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(self.embeddings.weight.device)
        #embedding
        text_embedding = self.embeddings(input_ids)
        #average
        text_embedding = text_embedding.mean(dim=1)
        #linear
        logits = self.linear(text_embedding)
        return logits, logits # to be compatible with other models

if __name__ == "__main__":
    text = ["hello world", "hello world, too"]
    model = BoWClassifier(768, 768)
    model.freeze_backbone()
    logits = model(text)
    print(logits)