import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import clip
import functools
import operator
from warnings import N

class CLIPViT(nn.Module):
    """
    ViT encoder for CLIP
    """
    def __init__(self, 
                 vpt_width:int, 
                 vpt_depth:int = 11, 
                 use_vpt:bool=True, 
                 unfreeze:bool=False,
                 n_expert:int=16,
                 num_classes= 1) -> None:
        """
        Param:
            clip_model: pretrained OpenAI CLIP model
            use_vpt: whether to use visual prompt tuning
            vpt_width: number of vpt token per layer
            vpt_depth: number of vpt layers. 1: vpt at the first layer (shallow), >1: deep vpt
            unfreeze: If true, unfreeze the CLIP model
        """
        raise NotImplementedError