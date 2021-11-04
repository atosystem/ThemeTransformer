"""Theme Transformer Architecture

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""
import torch.nn as nn
import torch
import math
import numpy as np
import myTransformer 

class PositionalEncoding(nn.Module):
    """
    For positional encoding in transformer.

    """
    
    def __init__(self, d_model,pos_enc_start=0, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(pos_enc_start, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class WordEmbedding(nn.Embedding):
    """
    For Token Embedding
    
    """
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class myLM(nn.Module):
    """
    The overall structure of model
    
    """
    def __init__(self, ntoken,d_model=512,dropout = 0.1,num_encoder_layers=6,xorpattern=[0,1]*3):
        super(myLM, self).__init__()

        # model parameter
        self.num_encoder_layers = num_encoder_layers
        self.xorpattern = xorpattern

        # model dimension
        self.d_model = d_model
        self.dropout = dropout

        # vocab size
        self.ntoken = ntoken

        # transformer model
        self.transformer_model = myTransformer.Transformer(
            d_model = self.d_model, 
            nhead=8,
            dim_feedforward=self.d_model*4,
            num_encoder_layers=num_encoder_layers, 
            xor_pattern=xorpattern,
            activation = "gelu"
            )
        
        # positional encoding used for encoder
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # token embedding
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)    

        # output layer
        self.output_layer = nn.Linear(self.d_model,self.ntoken)


    def forward(self,src,tgt,tgt_label,tgt_mask=None,src_key_padding_mask=None,tgt_key_padding_mask=None,memory_mask=None):
        """forward pass of the model

        Args:
            src (tensor): src sequence
            tgt (tensor): tgt sequence
            tgt_label (tensor): information for theme-aligned positional encoding
            tgt_mask (tensor, optional): mask for cross attending encoder's output. Defaults to None.
            src_key_padding_mask (tensor, optional): src key padding mask. Defaults to None.
            tgt_key_padding_mask (tensor, optional): tgt key padding mask. Defaults to None.
            memory_mask (tensor, optional): for causal attention mask, Leave it None for bidirectional. Defaults to None.

        Returns:
            tensor: output logits
        """

        src = self.token_embedding(src)
        src = self.pos_encoding(src)
        tgt = self.token_embedding(tgt)

        out = self.transformer_model(src, tgt,
                                        tgt_mask=tgt_mask,
                                        src_key_padding_mask=src_key_padding_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=src_key_padding_mask,
                                        memory_mask=memory_mask,
                                        tgt_label=tgt_label)
        
        out = self.output_layer(out)

        return out
    
    
    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][-1]
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word
