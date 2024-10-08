from dataclasses import dataclass
from torch.nn import functional as F
import torch
import torch.nn as nn


#-----------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 256 # Maximum length of the input text
    vocab_size: int = 65 # Total number of unique tokens in the vocabulary
    n_layer: int = 6  # The number of layers in the transformer model
    n_head: int = 6  # number of attention heads 
    n_embd: int = 384 # The embedding dimention for each token

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config