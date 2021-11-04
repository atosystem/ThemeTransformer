"""Theme Transformer Training Code

usage: inference.py [-h] [--model_path MODEL_PATH] [--theme THEME]
                    [--seq_length SEQ_LENGTH] [--seed SEED]
                    [--out_midi OUT_MIDI] [--cuda] [--max_len MAX_LEN]
                    [--temp TEMP]
  --model_path MODEL_PATH   model file
  --theme THEME             theme file
  --seq_length SEQ_LENGTH   generated seq length
  --seed SEED               random seed (set to -1 to use random seed) (change different if the model stucks)
  --out_midi OUT_MIDI       output midi file
  --cuda                    use CUDA
  --max_len MAX_LEN         number of tokens to predict
  --temp TEMP               temperature

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""
device_str = 'cuda:0'
import shutil
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim

from mymodel import myLM

from preprocess.music_data import getMusicDataset

from preprocess.vocab import Vocab

from parse_arg import *

import time
import os

import logger

from randomness import set_global_random_seed

# Set the random seed manually for reproducibility.
set_global_random_seed(args.seed)

# create vocab
myvocab = Vocab()

# create directory for training purpose
os.makedirs("./ckpts",exist_ok=True)
os.makedirs("./logs",exist_ok=True)

# create work directory
while(1):
    exp_name = input("Enter exp name : ")
    if os.path.exists(os.path.join("./ckpts", exp_name)):
        ans = input("work dir exists! overwrite? [Y/N]:")
        if ans.lower() == "y":
            break
    else:
        break

os.makedirs(os.path.join(
    "./ckpts/", exp_name), exist_ok=True)
os.makedirs(os.path.join("./ckpts/",
            exp_name, "script"), exist_ok=True)
os.makedirs(os.path.join("./ckpts/",
            exp_name, "script", "preprocess"), exist_ok=True)
os.makedirs(os.path.join("./ckpts/",
            exp_name, "log"), exist_ok=True)

checkpoint_folder = "./ckpts/{}".format(
    exp_name)
# copy scripts
file_to_save = ['train.py', 'inference.py', 'myTransformer.py','randomness.py',
                'parse_arg.py', 'mymodel.py', 'preprocess/vocab.py', 'preprocess/music_data.py']
for x in file_to_save:
    shutil.copyfile(x, os.path.join(checkpoint_folder, "script", x))

# create logger for log
mylogger = logger.logger(filepath=os.path.join(
    checkpoint_folder, "log/log_{}.txt".format(exp_name)),overrite=True)
if os.path.exists("logs/log_{}.txt".format(exp_name)):
    os.remove("logs/log_{}.txt".format(exp_name))
os.link(mylogger.filepath, "logs/log_{}.txt".format(exp_name))
mylogger.log("Exp_dir : {}".format(checkpoint_folder))
mylogger.log("Exp_Name : {}".format(exp_name))


# devices
device = torch.device( device_str if args.cuda else 'cpu')
device_cpu = torch.device('cpu')


# dataset
train_dataset = getMusicDataset(pkl_path="./data_pkl/train_seg2_512.pkl",
                                args=args,
                                vocab=myvocab)


val_dataset = getMusicDataset(pkl_path="./data_pkl/val_seg2_512.pkl",
                                args=args,
                                vocab=myvocab)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=2,
                        shuffle=False,
                        num_workers=4)

# define model
model = myLM(myvocab.n_tokens, d_model=256,num_encoder_layers=6,xorpattern=[0,0,0,1,1,1])

mylogger.log("Model hidden dim : {}".format(model.d_model))
mylogger.log("Encoder Layers #{}".format(model.num_encoder_layers))
mylogger.log("Decoder Layers #{}".format(len(model.xorpattern)))
mylogger.log("Decoder Pattern #{}".format(model.xorpattern))
mylogger.log("Batch size #{}".format(args.batch_size))
mylogger.log("lr : {}".format(args.lr))
# optimizer
# adam
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.max_step,eta_min=args.lr_min)

if not args.restart_point == '':
    # restart from checkpoint
    mylogger.log("Restart from {}".format(args.restart_point))
    model.load_state_dict(torch.load(args.restart_point,map_location=device_str))
    mylogger.log("model loaded")
    optimizer.load_state_dict(torch.load(args.restart_point.replace('model_','optimizer_'),map_location=device_str))
    mylogger.log("optimizer loaded")
    scheduler.load_state_dict(torch.load(args.restart_point.replace('model_','scheduler_'),map_location=device_str))
    mylogger.log("scheduler loaded")

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

# loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

mylogger.log("Using device : {}".format(logger.getCyan(device)))

train_step = 0


def train(epoch_num):
    """train the model

    Args:
        epoch_num (int): epoch number
    """
    global train_step

    model.train()
    start_time = time.time()
    total_loss = 0
    total_acc = 0
    steps = 0
    for batch_idx, data in enumerate(train_loader):
        print("Epoch {} Step [{}/{}] ".format(epoch_num,
              batch_idx, len(train_loader)), end='')

        data = {key: value.to(device) for key, value in data.items()}

        optimizer.zero_grad()

        data["src_msk"] = data["src_msk"].bool()
        data["tgt_msk"] = data["tgt_msk"].bool()

        tgt_input_msk = data["tgt_msk"][:, :-1]
        tgt_output_msk = data["tgt_msk"][:, 1:]

        data["src"] = data["src"].permute(1, 0)
        data["tgt"] = data["tgt"].permute(1, 0)
        data["tgt_theme_msk"] = data["tgt_theme_msk"].permute(1, 0)

        fullsong_input = data["tgt"][:-1, :]
        fullsong_output = data["tgt"][1:, :]

        att_msk = model.transformer_model.generate_square_subsequent_mask(
            fullsong_input.shape[0]).to(device)

        mem_msk = None


        output = model(
            src=data["src"],
            tgt=fullsong_input,
            tgt_mask=att_msk,
            tgt_label=data["tgt_theme_msk"][:-1, :],
            src_key_padding_mask=data["src_msk"],
            tgt_key_padding_mask=tgt_input_msk,
            memory_mask=mem_msk
        )

        loss = criterion(output.view(-1, myvocab.n_tokens),
                         fullsong_output.reshape(-1))

        predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)

        correct = predict.eq(fullsong_output.reshape(-1))

        correct = torch.sum(
            correct * (~tgt_output_msk).reshape(-1).float()).item()

        correct = correct / \
            torch.sum((~tgt_output_msk).reshape(-1).float()).item()

        total_acc += correct

        print("Acc : {:.2f} ".format(correct), end="")


        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if train_step < args.warmup_step:
            curr_lr = args.lr * train_step / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            scheduler.step()
        

        total_loss += loss.item()

        print("Loss : {:.2f} lr:{:.4f} ".format(
            loss.item(), optimizer.param_groups[0]['lr']), end='\r')

        steps += 1
        train_step += 1

    mylogger.log("Epoch {} lr:{:.4f} train_acc : {:.2f} train_loss : {:.2f}  time:{:.2f} ".format(epoch_num,
                                                                                                  optimizer.param_groups[0]['lr'], total_acc/steps, total_loss/steps, time.time()-start_time), end='')
    

def evalulate(epoch_num):
    """evaluate validation set

    Args:
        epoch_num (int): epoch number
    """
    model.eval()
    start_time = time.time()
    total_loss = 0
    total_acc = 0
    steps = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # print("Epoch {} Step {}/{} ".format( epoch_num,batch_idx,len(val_loader)),end='')
            optimizer.zero_grad()

            data = {key: value.to(device) for key, value in data.items()}

            data["src_msk"] = data["src_msk"].bool()
            data["tgt_msk"] = data["tgt_msk"].bool()

            tgt_input_msk = data["tgt_msk"][:, :-1]
            tgt_output_msk = data["tgt_msk"][:, 1:]

            data["src"] = data["src"].permute(1, 0)
            data["tgt"] = data["tgt"].permute(1, 0)
            data["tgt_theme_msk"] = data["tgt_theme_msk"].permute(1, 0)

            fullsong_input = data["tgt"][:-1, :]
            fullsong_output = data["tgt"][1:, :]

            att_msk = model.transformer_model.generate_square_subsequent_mask(
                fullsong_input.shape[0]).to(device)

            mem_msk = None

            output = model(
                src=data["src"],
                tgt=fullsong_input,
                tgt_mask=att_msk,
                tgt_label=data["tgt_theme_msk"][:-1, :],
                src_key_padding_mask=data["src_msk"],
                tgt_key_padding_mask=tgt_input_msk,
                memory_mask=mem_msk
            )

            loss = criterion(output.view(-1, myvocab.n_tokens),
                             fullsong_output.reshape(-1))

            predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)

            correct = predict.eq(fullsong_output.reshape(-1))

            correct = torch.sum(
                correct * (~tgt_output_msk).reshape(-1).float()).item()

            correct = correct / \
                torch.sum((~tgt_output_msk).reshape(-1).float()).item()

            total_acc += correct

            total_loss += loss.item()

            steps += 1

        mylogger.log("val_acc: {:.2f} val_loss : {:.2f}".format(
            total_acc/steps, total_loss/steps))

start_epoch = 0

if not args.restart_point =='':
    start_epoch = int(args.restart_point.split('_')[-1].split('.')[0][2:]) + 1
    mylogger.log("starting from epoch {}".format(start_epoch))

max_epoch = 15000
mylogger.log("max epoch :{}".format(max_epoch))
for i in range(start_epoch,max_epoch):
    model.to(device)
    train(i)
    evalulate(i)
    model.to(device_cpu)

    if i % 10 == 0:
        # save state dicts
        torch.save(model.state_dict(), os.path.join(
            checkpoint_folder, "model_ep{}.pt".format(i)))
        torch.save(optimizer.state_dict(), os.path.join(
            checkpoint_folder, "optimizer_ep{}.pt".format(i)))
        torch.save(scheduler.state_dict(), os.path.join(
            checkpoint_folder, "scheduler_ep{}.pt".format(i)))

    

    
