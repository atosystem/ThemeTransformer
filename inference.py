"""Theme Transformer Inferencing Code

usage: inference.py [-h] [--model_path MODEL_PATH] --theme THEME
                    [--seq_length SEQ_LENGTH] [--seed SEED]
                    [--out_midi OUT_MIDI] [--cuda] [--max_len MAX_LEN]
                    [--temp TEMP] [--nbars NBARS]
  --model_path MODEL_PATH   model file
  --theme THEME             theme file
  --seq_length SEQ_LENGTH   generated seq length
  --seed SEED               random seed (set to -1 to use random seed) (change different if the model stucks)
  --out_midi OUT_MIDI       output midi file
  --cuda                    use CUDA
  --max_len MAX_LEN         number of tokens to predict
  --temp TEMP               temperature
  --nbars NBARS             number of bars to generate

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
"""

import argparse
import numpy as np
import torch
import torch.optim

from mymodel import myLM

from preprocess.music_data import MusicDataset

from preprocess.vocab import Vocab

import time
import os
import pickle
import json

from randomness import set_global_random_seed

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./trained_model/model_ep2311.pt',
                    help='model file')

parser.add_argument('--theme', type=str, required=True,default='./theme_files/874_theme.mid',
                    help='theme file')

parser.add_argument('--seq_length', type=str, default='',
                    help='generated seq length')

parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (set to -1 to use random seed)')

parser.add_argument('--out_midi', type=str, default='output.mid',
                    help='output midi file')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--max_len', type=int, default=512,
                    help='number of tokens to predict')

parser.add_argument('--temp', type=float, default=1.2,
                    help='temperature')

parser.add_argument('--nbars', type=float, default=32,
                    help='number of bars to generate')

args = parser.parse_args()

if not args.seed == -1:
    set_global_random_seed(args.seed)

# create vocab
myvocab = Vocab()

# devices
device = torch.device('cuda:0' if args.cuda else 'cpu')
device_cpu = torch.device('cpu')


# model definition
model = myLM(myvocab.n_tokens,d_model=256,num_encoder_layers=6,xorpattern=[0,0,0,1,1,1])

print("Loading model from {}".format(args.model_path))
model.load_state_dict(torch.load(args.model_path))
print("Using device {}".format(device))

def inference(n_bars, strategies, params,theme_seq,prompt=None):
    """inference function

    Args:
        n_bars (int): numbers of bar to generate
        strategies (dict): inferencing strategies
        params (dict): parameters for inferencing strategies
        theme_seq (list): given theme condition
        prompt (list, optional): initial tokens fed to the theme transformer. Defaults to None.

    Returns:
        list: token sequence of generated music
    """
    model.eval()
    words = [[]]

    word2event = myvocab.id2token

    initial_flag, initial_cnt = True, 0
    generated_bars = 0
    
    fail_cnt = 0
    subbeats_accumulate = 0

    input_theme = torch.tensor(theme_seq)
    input_theme = input_theme.reshape((-1,1))
    input_theme = input_theme.to(device)

    label_list = []

    previous_labeled = False

    last_theme_bar_idx = -1

    new_motif_tmp_array = []
    
    bar_count = 0
    position_anchor = -1

    with torch.no_grad():
        while bar_count < n_bars:
            print("events #{} Generating Bars #{}/{}".format(len(words[0]),bar_count ,bar_count),end='\r')
            if fail_cnt:
                print ('failed iterations:', fail_cnt)
            
            if fail_cnt >1024:
                print ('model stuck ...\nPlease change a seed sand inference again!')
                return words[0]

            # prepare input
            if initial_flag:
                if not prompt == None:
                    # prompt given
                    input_x = torch.tensor(prompt)
                    words[0].extend(prompt)
                    label_list = [0]*len(prompt)
                    for i,x in enumerate(prompt):
                        if myvocab.id2token[x] == "Theme_Start":
                            previous_labeled = True
                        elif myvocab.id2token[x] == "Theme_End":
                            previous_labeled = False
                        if previous_labeled:
                            if i == 0:
                                label_list[i] = 1
                            else:
                                label_list[i] = label_list[i-1] +  1
                        if myvocab.id2token[x].startswith("Position"):
                            position_anchor = int(myvocab.id2token[x].split("_")[1])
                        if myvocab.id2token[x] == "Bar":
                            position_anchor = - 1
                            n_bars += 1
                    
                    label_input = torch.tensor(label_list)
                else:
                    # no prompt given
                    input_x = torch.tensor([theme_seq[0]])
                    label_list = [0]
                    words[0].append(theme_seq[0])
                    if myvocab.id2token[theme_seq[0]] == "Theme_Start":
                        previous_labeled = True
                    label_input = torch.tensor(label_list)

                initial_flag = False
            else:
                input_x = torch.tensor(words[0][-args.max_len:])
                label_input = torch.tensor(label_list[-args.max_len:])



            input_x = input_x.reshape((-1,1))
            label_input = label_input.reshape((-1,1))
            
            input_x_att_msk = model.transformer_model.generate_square_subsequent_mask(input_x.shape[0])
            input_x = input_x.to(device)
            label_input = label_input.to(device)
            input_x_att_msk = input_x_att_msk.to(device)


            logits = model(
                src=input_theme,
                tgt=input_x,
                tgt_label=label_input,
                tgt_mask = input_x_att_msk
            )   
            logits = logits[-1:]
            logits = torch.squeeze(logits)
            logits = logits.cpu().numpy()
            



            # temperature or not
            if 'temperature' in strategies:
                probs = model.temperature(logits=logits, temperature=params['t'])
            else:
                probs = model.temperature(logits=logits, temperature=1.)

            # sampling
            # word : the generated remi event
            word = model.nucleus(probs=probs, p=params['p'])
            
            print("Generated new remi word {}".format(myvocab.id2token[word]))
            # skip padding
            if word in [0]:
                fail_cnt += 1
                continue
            
            # grammar checking ========================================================

            #  check Theme_Start -> Bar
            if 'Theme_Start' in  word2event[words[0][-1]] and 'Bar' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue

            #  check Theme_End -> Bar
            if 'Theme_End' in  word2event[words[0][-1]] and 'Bar' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue

            # check Note-On-[track] -> Note-Duration-[track]
            if 'Note-On' in  word2event[words[0][-1]] and 'Note-Duration' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-On' in  word2event[words[0][-1]] and 'Note-Duration' in  word2event[word]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-On,Duration Track Inconsistency")
                    continue

            if 'Note-Duration' in  word2event[word] and 'Note-On' not in  word2event[words[0][-1]]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-Duration' in  word2event[word] and 'Note-On' in  word2event[words[0][-1]]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-On,Duration Track Inconsistency")
                    continue

            # check Note-Duration-[track] -> Note-Velocity-[track]
            if 'Note-Duration' in  word2event[words[0][-1]] and 'Note-Velocity' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-Duration' in  word2event[words[0][-1]] and 'Note-Velocity' in  word2event[word]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-Duration,Velocity Track Inconsistency")
                    continue

            if 'Note-Velocity' in  word2event[word] and 'Note-Duration' not in  word2event[words[0][-1]]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-Velocity' in  word2event[word] and 'Note-Duration' in  word2event[words[0][-1]]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-Duration,Velocity Track Inconsistency")
                    continue
            

            if word2event[word].startswith("Tempo") or word2event[word].startswith("Note"):
                if position_anchor == -1:
                    print("Position not yet set")
                    fail_cnt += 1 
                    continue
            
            # check position number
            if word2event[word].startswith("Position"):
                pos = int(word2event[word].split("_")[1])
                if position_anchor == pos:
                    print("Position not increasing")
                    fail_cnt += 1 
                    continue
                else:
                    position_anchor = pos

            
            # check theme region
            if myvocab.id2token[word].startswith("Theme"):
                if myvocab.id2token[word] == "Theme_Start" and not previous_labeled:
                    previous_labeled = True
                    last_theme_bar_idx = bar_count

                elif myvocab.id2token[word] == "Theme_End" and previous_labeled:
                    previous_labeled = False
                else:
                    print("Theme region error")
                    fail_cnt += 1
                    continue
            
            # add new event to record sequence
            words[0].append(word)
            if previous_labeled:
                label_list.append(label_list[-1]+1)
            else:
                label_list.append(0)

            if word2event[word] == "Bar":
                bar_count += 1
                position_anchor = -1
                if bar_count > n_bars:
                    return words[0]
            
            fail_cnt = 0


    print ('generated {} events'.format(len(words[0])))
    return words[0]


# load tempo information
with open("./tempo_dict.json") as f:
    tempo_dict = json.load(f)

given_theme = myvocab.midi2REMI(args.theme,theme_annotations=False)
given_theme = [myvocab.token2id["Theme_Start"]] + given_theme + [myvocab.token2id["Theme_End"]]


midiID = os.path.basename(args.theme).split('.')[0].split('_')[0]
print("Tempo from original : {}".format(tempo_dict[midiID]))
tmp = myvocab._tempo_bins[np.argmin(abs(tempo_dict[midiID] - myvocab._tempo_bins))]
given_tempo = myvocab.token2id["Tempo_{}".format(tmp)]

tempo_in_theme = [x for x in given_theme if myvocab.id2token[x].startswith("Tempo")]
if not len(tempo_in_theme) == 0:
    # remove error tempo
    given_theme = [x for x in given_theme if not myvocab.id2token[x].startswith("Tempo")]

model.to(device)
word_seq = inference(
            n_bars = args.nbars,
            strategies=['temperature', 'nucleus'],
            params={'t': args.temp, 'p': 0.9},
            theme_seq=given_theme,
            prompt=[given_tempo,myvocab.token2id["Theme_Start"]],
)

# check if no tempo in front , add it
position_events = [i for i,x in enumerate(word_seq) if myvocab.id2token[x].startswith("Position")]
if not word_seq[position_events[0]+1] == given_tempo:
    word_seq.insert(position_events[0]+1,given_tempo)

# remove tempo
word_seq = word_seq[1:]

# save to disk
myvocab.REMIID2midi(word_seq,args.out_midi)
print("{} saved".format(args.out_midi))
