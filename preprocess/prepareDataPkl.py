import vocab
import numpy as np
import glob
import os
import pickle

myvocab = vocab.Vocab()


# Check the paths for your own case
# the theme annotated midi files
MIDI_FILES = "./pop909_midis/*.mid"

# the tokens converted from theme annotated midi files
MIDI_FILES_PKLs_DIR = "./pop909_midi_pkls"

# the output training data 
OUTPUT_DATA_PKL = "../data_pkl/train_seg2_512.pkl"

all_mids = sorted(glob.glob(MIDI_FILES))

for _midiFile in all_midis:
    # convert midi files to token representation and save as .pkl file
    output_pkl_fp = midifp.replace(".mid",".pkl")
    remi_seq = myvocab.midi2REMI(midifp,include_bridge=False,bar_first=False,verbose=False)
    ret = myvocab.preprocessREMI(remi_seq,always_include=True,max_seq_len=512,verbose=True)    
    pickle.dump(ret, open(output_pkl_fp, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# collect all .pkl files and generate .pkl file for training/testing
al_pkls = sorted(glob.glob(os.path.join(MIDI_FILES_PKLs_DIR,"*.pkl")))

for i,fn in enumerate(al_pkls):
    print(">>[{}/{}][train] Now processing {}".format(i+1,len(al_pkls),os.path.split(fn)[-1]))
    with open(fn,"rb") as f:
        data = pickle.load(f)
    src = data["src"]
    src_theme_binary_msk = data["src_theme_binary_msk"]
    
    for i_tgt, tgt in enumerate(data["tgt_segments"]):
        training_data.append({"src":src,"tgt":tgt,
        "tgt_theme_msk":data["tgt_segments_theme_binary_msk"][i_tgt],
        "src_theme_msk":src_theme_binary_msk})

# dump to pkl file for training
pickle.dump(training_data, open(OUTPUT_DATA_PKL, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)






