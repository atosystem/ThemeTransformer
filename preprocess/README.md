# Data Preprocess

To prepare data for Theme Transformer, you must first annotated your midi file with the following format (including theme annotation information), then run the code to generate pkl file for training.

## For midi annotations:
![image](https://user-images.githubusercontent.com/32225867/147511432-6992f921-f44b-41f1-82b4-8a2c91463e67.png)
Each midi file must contain 5 tracks. (The first 3 tracks directly come from POP909 only preprocessed with 1/16 bar quantize)

others:
* `boundary_track` : indicates the boundary of each music segment. (We employ 2-bar segmentation in our paper. You are welcome to use other phrase segmentation methods). The onset and offset of the notes correspond to the start and the end of the music segment. The pitch has no physical meaning, but please use the same value for every segment.

* `melody_boundary_track` (Optional, you can leave it blank): indicates the cluster for each music segment. The segments with same pitch are in the same cluster (they sound similar). Notice that the lowest pitch for the notes in this track must be the pitch in `boundary_track`

*  `theme info track` : indicates the music segments that are selected as **theme**

> Because we do not provide the theme retrieval code currently, you must decide the theme region by yourself, as well as boundaries.


| `boundary_track` | `melody_boundary_track` | `theme info track` |
| -------- | -------- | -------- |
| ![image](https://user-images.githubusercontent.com/32225867/147512228-973c6284-298a-444e-889e-0ead13c52ee9.png)    | ![image](https://user-images.githubusercontent.com/32225867/147512235-d9d1b8cd-ed3c-43c1-a7ed-5b045494e46a.png)     | ![image](https://user-images.githubusercontent.com/32225867/147512256-f186d26f-f235-40b8-b4d4-3c5eca166187.png)     |


## Convert to REMI Representation

1. Put all the annotated midi files under a directory (ex: `./pop909_midis`)
2. Make sure the paths in `prepareDataPkl.py` is correct
3. Execute : `python prepareDataPkl.py`