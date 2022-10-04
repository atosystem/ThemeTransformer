# Theme Transformer
<p>
    <a href=""><img alt="LICENSE" src="https://img.shields.io/github/license/atosystem/ThemeTransformer"/> </a>
    <a href=""><img alt="STAR" src="https://img.shields.io/github/stars/atosystem/ThemeTransformer"/> </a>
    <a href="https://github.com/atosystem/ThemeTransformer/issues"><img alt="ISSUE" src="https://img.shields.io/github/issues/atosystem/ThemeTransformer" /></a>
    <a href="https://github.com/atosystem/ThemeTransformer/pulls"><img alt="PR" src="https://img.shields.io/github/issues-pr/atosystem/ThemeTransformer" /></a>
</p>

This is the official implementation of Theme Transformer.

*Checkout our demo and paper* : [Demo](https://atosystem.github.io/ThemeTransformer/) | [arXiv](https://arxiv.org/abs/2111.04093)

## Environment: 
* Clone this Repo 

    ```bash
    git clone https://github.com/atosystem/ThemeTransformer.git -b main --single-branch
    ```

* using python version 3.6.8
* install python dependencies: 

    `pip install -r requirements.txt`

## To train the model with GPU:

`python train.py --cuda`

## To generate music from theme

`python inference.py --cuda --theme <theme midi file> --out_midi <output midi file>`


##  Details of the files in this repo
```
.
├── ckpts                   For saving checkpoints while training
├── data_pkl                Stores train and val data
│   ├── train_seg2_512.pkl
│   └── val_seg2_512.pkl
├── inference.py            For generating music. (Detailed usage are written in the file)
├── logger.py               For logging
├── mymodel.py              The overal Theme Transformer Architecture
├── myTransformer.py        Our transformer revision code 
├── parse_arg.py            Some arguments for training
├── preprocess              For data preprocessing  
│   ├── music_data.py       Theme Transformer pytorch dataset definition
│   └── vocab.py            Our vocabulary for transformer
├── randomness.py           For fixing random seed
├── readme.txt              Readme
├── tempo_dict.json         The original tempo information from POP909 (used in inference time)
├── theme_files/            The themes from our testing set.
├── trained_model           The model we trained.
│   └── model_ep2311.pt
└── train.py                Code for training Theme Transformer
```

## Citation
If you find this work helpful and use our code in your research, please kindly cite our paper:
```
@article{shih2022theme,
  title={Theme Transformer: Symbolic Music Generation with Theme-Conditioned Transformer},
  author={Yi-Jen Shih and Shih-Lun Wu and Frank Zalkow and Meinard Müller and Yi-Hsuan Yang},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
