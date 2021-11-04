"""Music Dataset for theme-based transformer

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""
import torch
from torch.utils.data.dataset import Dataset
import sys, pickle
import numpy as np
from glob import glob
from copy import deepcopy

import preprocess.vocab

# create vocab
myvocab = preprocess.vocab.Vocab()


import random

class MusicDataset(Dataset):
    def __init__(self,data,max_seq_len,vocab = None,aug=None):
        self.vocab = myvocab if vocab == None else vocab
        self.event2id = self.vocab.token2id
        self.id2event = self.vocab.id2token

        self.data = data

        self.max_seq_len = max_seq_len

        self.ep_start_pitchaug = 0       

        self.pitchaug_range = aug
        self.final_training_data = []

        self.constants = {
            "max_src_len" : max([ len(x["src"]) for x in self.data]),
            "max_tgt_len" : min(max_seq_len,max([ len(x["tgt"]) for x in self.data])) + 1
        }
        print(self.constants)
    
    def data_pitch_augment(self,src_seq,tgt_seq):
        """pitch shift for data augmentation

        Args:
            src_seq (list): src sequence
            tgt_seq (list): tgt sequence

        Returns:
            tuple: augmented src sequence, augmented target sequence
        """
        # pitch augement

        # get all pitch tokens
        all_pitches = [ self.vocab.getPitch(x) for x in src_seq] + [self.vocab.getPitch(x) for x in tgt_seq]
        all_pitches = [x for x in all_pitches if x > 0]
        if len(all_pitches) == 0 :
            return src_seq,tgt_seq

        pitch_offsets = np.random.randint(1-min(all_pitches),127 - max(all_pitches), size=1)
        pitch_offset = pitch_offsets[0]

        aug_src_phrase = deepcopy(src_seq)
        aug_tgt_phrase = deepcopy(tgt_seq)
        for t in range(len(aug_src_phrase)):
            if self.vocab.getPitch(aug_src_phrase[t]) > 0:
                aug_src_phrase[t] += pitch_offset
                assert self.vocab.getPitch(aug_src_phrase[t]) > 0
        
        for t in range(len(aug_tgt_phrase)):
            if self.vocab.getPitch(aug_tgt_phrase[t]) > 0:
                aug_tgt_phrase[t] += pitch_offset
                assert self.vocab.getPitch(aug_tgt_phrase[t]) > 0

        return aug_src_phrase, aug_tgt_phrase

    def __getitem__(self, index):
        """return data given index

        Args:
            index (int): the index fo data

        Returns:
            obj: {
                "src"           : <src sequence>
                "src_msk"       : <src sequence padding mask>,
                "src_theme_msk" : <src sequence theme mask>,
                "tgt"           : <target sequence>,
                "tgt_msk"       : <target sequence padding mask>,
                "tgt_theme_msk" : <target sequence theme mask>,
            }
        """

        # pitch augment
        src, tgt = self.data_pitch_augment(src_seq=self.data[index]["src"],tgt_seq=self.data[index]["tgt"])
        
        src_theme_msk = self.data[index]["src_theme_msk"]
        tgt_theme_msk = self.data[index]["tgt_theme_msk"]
        
        src_theme_msk = list(map(int,src_theme_msk))
        tgt_theme_msk = list(map(int,tgt_theme_msk))
        
        
        # padding
        src_msk = [0]* len(src) + [1] * (self.constants["max_src_len"] - len(src))
        src_theme_msk.extend([0]*(self.constants["max_src_len"] - len(src)))
        src.extend([0]*(self.constants["max_src_len"] - len(src)))
        tgt_msk = []
        if len(tgt) > self.constants["max_tgt_len"]:
            print("Should not be here")
            assert True
            tgt = tgt[:self.constants["max_tgt_len"]]
            tgt_theme_msk = tgt_theme_msk[:self.constants["max_tgt_len"]]
            tgt_msk = [0] * len(tgt)
        else:
            tgt_msk = [0]* len(tgt) + [1] * (self.constants["max_tgt_len"] - len(tgt))
            tgt_theme_msk.extend([0]*(self.constants["max_tgt_len"] - len(tgt)))
            tgt.extend([0]*(self.constants["max_tgt_len"] - len(tgt)))
        
        current_entry = {
            "src"           : src,
            "src_msk"       : src_msk,
            "src_theme_msk" : src_theme_msk,
            "tgt"           : tgt,
            "tgt_msk"       : tgt_msk,
            "tgt_theme_msk" : tgt_theme_msk,
        }
        assert(len(src) == self.constants["max_src_len"])
        assert(len(tgt) == self.constants["max_tgt_len"])
        assert(len(src_msk) == self.constants["max_src_len"])
        assert(len(tgt_msk) == self.constants["max_tgt_len"])
        assert(len(src_theme_msk) == self.constants["max_src_len"])
        assert(len(tgt_theme_msk) == self.constants["max_tgt_len"])

        return {key: torch.tensor(value) for key, value in current_entry.items()}

    def __len__(self):
        return len(self.data)

    # def set_epoch(self,ep):
        

def getMusicDataset(pkl_path,args,vocab):
    """load data from pkl file and return torch dataset

    Args:
        pkl_path (str): data pkl file path
        args (obj): all args from argparser
        vocab (Vocab): vocab instance

    Returns:
        torch.util.data.dataset: the dataset insrance
    """
    with open(pkl_path,"rb") as f:
        train_data = pickle.load(f)
    
    dataset = MusicDataset(data=train_data,max_seq_len=args.max_len,vocab=vocab)

    return dataset
if __name__ == '__main__':
    pass
    