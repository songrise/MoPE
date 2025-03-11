import numpy as np
import torch
import torch.nn as nn
import os
import pickle

def dump_tensor(tensor, name, path = "./"):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    with open(os.path.join(path, f"{name}.pkl"), "wb") as f:
        pickle.dump(tensor, f)
