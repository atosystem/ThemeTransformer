"""Vocabulary for theme-based transformer

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""
import pickle
import numpy as np
import miditoolkit
import os
import math
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct

class Vocab(object):
    def __init__(self):
        """initialize some vocabulary settings
        """        
        
        # split each beat into 4 subbeats
        self.q_beat = 4

        # dictionary for matching token ID to name and the other way around.
        self.token2id = {}
        self.id2token = {}

        # midi pitch number : 1 ~ 127 (highest pitch) 
        self._pitch_bins = np.arange(start=1,stop=128)

        # duration tokens 1~64 of self.q_beat
        self._duration_bins = np.arange(start=1,stop=self.q_beat*16+1)

        # velocity tokens 1~127 (corressponding to midi format)
        self._velocity_bins =  np.arange(start=1,stop=127)

        # tempo tokens 17~197 (determined from our dataset)
        self._tempo_bins = np.arange(start=17,stop=197,step=3)

        # position(subbeat) tokens 0~15, indicate the relative position with in a bar
        self._position_bins = np.arange(start=0,stop=16)

        self.n_tokens = 0

        self.token_type_base = {}

        self.tracks = ["MELODY","BRIDGE","PIANO"]

        self.build()

        # vocab
        # Note-On (129) : 0 (padding) ,1 ~ 127(highest pitch) , 128 (rest)
        # Note-Duration : 1 ~ 16 beat * 3
        # min resulution 1/12 notes        

    def build(self):
        """build our vocab
        """
        self.token2id = {}
        self.id2token = {}
        
        self.n_tokens = 0

        self.token2id['padding'] = 0
        self.n_tokens += 1

        # Note related tokens==================================================================
        # Create Note-On tokens for each track
        for track in self.tracks:
            # Note-On
            self.token_type_base = {'Note-On-{}'.format(track) : 1}
            for i in self._pitch_bins:
                self.token2id[ 'Note-On-{}_{}'.format(track,i) ] = self.n_tokens
                self.n_tokens += 1
        
        # Create Note-Duration tokens for each track
        for track in self.tracks:
            # Note-Duration
            self.token_type_base['Note-Duration-{}'.format(track)] = self.n_tokens
            for note_dur in self._duration_bins:
                self.token2id[ 'Note-Duration-{}_{}'.format(track,note_dur) ] = self.n_tokens
                self.n_tokens += 1

        # Create Note-Velocity tokens for each track
        for track in self.tracks:
            # Note-Velocity
            self.token_type_base['Note-Velocity-{}'.format(track)] = self.n_tokens
            for vel in self._velocity_bins:
                self.token2id[ 'Note-Velocity-{}_{}'.format(track,vel) ] = self.n_tokens
                self.n_tokens += 1
        
        # Metric related tokens==================================================================
        # Tempo
        self.token_type_base['Tempo'] = self.n_tokens
        for tmp in self._tempo_bins:
            self.token2id[ 'Tempo_{}'.format(tmp) ] = self.n_tokens
            self.n_tokens += 1

        # Positions
        self.token_type_base['Position'] = self.n_tokens
        for pos in self._position_bins:
            self.token2id[ 'Position_{}'.format(pos) ] = self.n_tokens
            self.n_tokens += 1

        # Bar
        self.token_type_base['Bar'] = self.n_tokens
        self.token2id[ 'Bar' ] = self.n_tokens
        self.n_tokens += 1
        
        # Theme related tokens==================================================================
        # Phrase annotation (not used in our final paper)
        self.token_type_base['Phrase'] = self.n_tokens
        self.token2id[ 'Phrase_Start' ] = self.n_tokens
        self.n_tokens += 1

        self.token2id[ 'Phrase_End' ] = self.n_tokens
        self.n_tokens += 1

        # Theme annotation
        self.token_type_base['Theme'] = self.n_tokens
        self.token2id[ 'Theme_Start' ] = self.n_tokens
        self.n_tokens += 1

        self.token2id[ 'Theme_End' ] = self.n_tokens
        self.n_tokens += 1


        for w , v in self.token2id.items():
          self.id2token[v] = w

        self.n_tokens = len(self.token2id)

    def getPitch(self,input_event):
        """Return corresponding note pitch
        if input_event is not a note, it returns -1

        Args:
            input_event (str or int): REMI Event Name or vocab ID
        """
        if isinstance(input_event,int):
            input_event = self.id2token[input_event]
        elif isinstance(input_event,str):
            pass
        else:
            try:
                input_event = int(input_event)
                input_event = self.id2token[input_event]
            except:
                raise TypeError("input_event should be int or str, input_event={}, type={}".format(input_event,type(input_event)))
        
        if not input_event.startswith("Note-On"):
            return -1

        assert int(input_event.split("_")[1]) >=1 and int(input_event.split("_")[1]) <=127
        return int(input_event.split("_")[1])
        

    def midi2REMI(self,midi_path,trim_intro = True,trim_outro=True,include_bridge=False,extend_theme=False,bar_first=False,theme_annotations=True,verbose = False):
        """convert midi file to token representation

        Args:
            midi_path (str): the path of input midi file
            trim_intro (bool, optional): ignore the intro part of the song. Defaults to True.
            trim_outro (bool, optional): ignore the outro part of the song. Defaults to True.
            include_bridge (bool, optional): ignore the intro part of the song. Defaults to False.
            extend_theme (bool, optional): extend the theme region to at least MIN_MEL_NOTES=8 notes. Defaults to False.
            bar_first (bool, optional): place Bar token in front of Theme-Start token. Defaults to False.
            theme_annotations (bool, optional): including theme-realted tokens. Defaults to True.
            verbose (bool, optional): print some message. Defaults to False.

        Returns:
            list: sequence of tokens
        """
        MIN_MEL_NOTES = 8
        midi_obj = mid_parser.MidiFile(midi_path)

        if theme_annotations:
            # select theme info track
            theme_boundary_track = list(filter(lambda x: x.name=="theme info track",midi_obj.instruments))
            assert len(theme_boundary_track) == 1
        
        # parsing notes in each tracks (ignore BRIDGE)
        notesAndtempos = []
        midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes,key=lambda x: x.start)
        # add notes
        melody_start = sorted(midi_obj.instruments[0].notes,key=lambda x: x.start)[0].start
        melody_end = sorted(midi_obj.instruments[0].notes,key=lambda x: x.start)[-1].end
        for i in range(3):
            if not include_bridge and midi_obj.instruments[i].name == "BRIDGE":
                continue
            if midi_obj.instruments[i].name == "Theme info track":
                continue

            notes = midi_obj.instruments[i].notes
            for n in notes:
                # assert (trim_intro and n.start>=melody_start or not trim_intro)
                if trim_intro and n.start>=melody_start or not trim_intro:
                    if trim_outro and n.start<=melody_end or not trim_outro:
                        notesAndtempos.append({
                                "priority" : i+1,
                                "priority_1" : n.pitch,
                                "start_tick" : n.start,
                                "obj_type" : "Note-{}".format(midi_obj.instruments[i].name),
                                "obj" : n
                            })
        # add tempos
        for t in midi_obj.tempo_changes:
            # assert (trim_intro and t.time>=melody_start or not trim_intro)
            if trim_intro and t.time>=melody_start or not trim_intro or trim_intro:
                if trim_outro and t.time<=melody_end or not trim_outro:
                    notesAndtempos.append({
                            "priority" : 0,
                            "priority_1" : 0,
                            "start_tick" : t.time,
                            "obj_type" : "Tempo",
                            "obj" : t
                        })

        if trim_intro and len([x for x in midi_obj.tempo_changes if x.time==melody_start]) == 0:
            t = [x for x in sorted(midi_obj.tempo_changes,key= lambda z: z.time) if x.time < melody_start]
            if not len(t) == 0:
                t = t[-1]
                notesAndtempos.append({
                        "priority" : 0,
                        "priority_1" : 0,
                        "start_tick" : melody_start,
                        "obj_type" : "Tempo",
                        "obj" : t
                })
        notesAndtempos = sorted(notesAndtempos,key=lambda x: (x["start_tick"],x["priority"],-x["priority_1"]))

        
        
        if theme_annotations:
            theme_boundary_track = theme_boundary_track[0]
            theme_boundary_pitch = min([x.pitch for x in theme_boundary_track.notes])
            theme_boundaries = [ [x.start,x.end] for x in theme_boundary_track.notes if x.pitch == theme_boundary_pitch]
            assert not len(theme_boundaries) == 0
            if verbose:
                print(theme_boundaries)
        
        if extend_theme:
            # extend theme region 8~9
            for b_i,b in enumerate(theme_boundaries[:-1]):
                
                melody_notes = [x for x in midi_obj.instruments[0].notes if x.start>= b[0] and x.start< theme_boundaries[b_i+1][0] ]
                cur_bound = 0
                for x in melody_notes:
                    if x.start < b[1]:
                        cur_bound += 1
                    else:
                        break
                if  cur_bound + 1 >= MIN_MEL_NOTES :
                    continue
                # try to extend
                extend_idx = min(MIN_MEL_NOTES,len(melody_notes)) - 1
                theme_boundaries[b_i][1] = melody_notes[extend_idx].end

            
        
        
        
        b_i = 0
        in_theme = False
        # group
        bar_segments = []
        bar_ticks = midi_obj.ticks_per_beat * 4
        if verbose:
            print("Bar tick length: {}".format(bar_ticks))
        
        for bar_start_tick in range(0,notesAndtempos[-1]["start_tick"],bar_ticks):
            if verbose:
                print("Bar {} at tick: {}".format(bar_start_tick // bar_ticks,bar_start_tick))
            bar_end_tick = bar_start_tick + bar_ticks
            current_bar = []
            bar_objs = list(filter(lambda x: x["start_tick"] >=bar_start_tick and x["start_tick"]< bar_end_tick,notesAndtempos))
            bar_objs.insert(0,{"start_tick":-1})

            if not bar_first:
                if theme_annotations and not in_theme and  theme_boundaries[b_i][0] ==  bar_start_tick:
                    current_bar.append("Theme_Start")
                    in_theme = True
                    if verbose:
                        print("Theme start")

                if not in_theme and trim_intro and bar_start_tick+bar_ticks < melody_start:
                    if verbose:
                        print("into trimmed")
                    continue
            
                current_bar.append("Bar")
            else:
                if not in_theme and trim_intro and bar_start_tick+bar_ticks < melody_start:
                    if verbose:
                        print("into trimmed")
                    continue
            
                current_bar.append("Bar")

                if theme_annotations and not in_theme and  theme_boundaries[b_i][0] ==  bar_start_tick:
                    current_bar.append("Theme_Start")
                    in_theme = True
                    if verbose:
                        print("Theme start")

            
            for i,obj in enumerate(bar_objs):
                if obj["start_tick"]==-1 : continue
                if not obj["start_tick"] == bar_objs[i-1]["start_tick"]:
                    # insert Position Event
                    pos = (obj["start_tick"] - bar_start_tick) / midi_obj.ticks_per_beat * self.q_beat
                    pos_index = np.argmin(abs(pos - self._position_bins)) # use the closest position
                    pos = self._position_bins[pos_index]
                    current_bar.append("Position_{}".format(pos))

                if obj["obj_type"].startswith("Note"):
                    track_name = obj["obj_type"].split('-')[1].upper()
                    # add pitch
                    current_bar.append("Note-On-{}_{}".format(track_name,obj["obj"].pitch))
                    # add duration
                    dur = (obj["obj"].end - obj["obj"].start) / midi_obj.ticks_per_beat * self.q_beat
                    dur_index = np.argmin(abs(dur - self._duration_bins)) # use the closest position
                    dur = self._duration_bins[dur_index]
                    current_bar.append("Note-Duration-{}_{}".format(track_name,dur))
                    # add velocity
                    vel_index = np.argmin(abs(obj["obj"].velocity - self._velocity_bins)) # use the closest position
                    vel = self._velocity_bins[vel_index]
                    current_bar.append("Note-Velocity-{}_{}".format(track_name,vel))
                elif obj["obj_type"].startswith("Tempo"):
                    # tempo
                    tmp_index = np.argmin(abs(obj["obj"].tempo - self._tempo_bins)) # use the closest position
                    tmp = self._tempo_bins[tmp_index]
                    current_bar.append(obj["obj_type"] + "_{}".format(tmp))
                else:
                    # theme start end
                    current_bar.append(obj["obj_type"])

            if theme_annotations and in_theme and  theme_boundaries[b_i][1] ==  bar_start_tick + bar_ticks:
                current_bar.append("Theme_End")
                in_theme = False
                if verbose:
                    print("Theme End")
                if not  b_i == len(theme_boundaries) - 1:
                    b_i += 1
            bar_segments.extend(current_bar)

        output_ids = [self.token2id[x] for x in bar_segments]

        return output_ids

    def preprocessREMI(self,remi_sequence,always_include=False,max_seq_len=512,strict=True,verbose=False):
        """Preprocess token sequence

        slicing the sequence for training our models

        Args:
            remi_sequence (List): the music token seqeunce
            always_include (Bool): selected the data including either Theme-Start or Theme-End
            max_seq_len (Int): maximum sequence length for each data
            strict (Bool): the returning sequence should always include Theme-Start

        Return:
            {
                "src" : <corressponding theme condition>,
                "src_theme_binary_msk" : <corressponding theme condition's theme msk>,
                "tgt_segments" : <list of target sequences>,
                "tgt_segments_theme_binary_msk" : <list of target sequences theme msk>
            }
        """
        theme_binary_msk = []
        in_theme = False
        src = []
        src_theme_binary_msk = []
        for r in remi_sequence:
            if self.id2token[r] == "Theme_Start":
                in_theme = True
            elif self.id2token[r] == "Theme_End":
                in_theme = False
            theme_binary_msk.append(int(in_theme))
        
        for i in range(1,len(theme_binary_msk)):
            theme_binary_msk[i] = theme_binary_msk[i-1]*theme_binary_msk[i] + theme_binary_msk[i]


        start_first_theme = remi_sequence.index(self.token2id["Theme_Start"])
        end_first_theme = remi_sequence.index(self.token2id["Theme_End"])
        src = remi_sequence[start_first_theme:end_first_theme+1]
        src_theme_binary_msk = theme_binary_msk[start_first_theme:end_first_theme+1]

        tgt_segments = []
        tgt_segments_theme_msk = []
        
        s = 0
        if strict:
            theme_start_pos = [i for i in range(len(remi_sequence)) if remi_sequence[i] == self.token2id["Theme_Start"]]
            for t in theme_start_pos:
                tgt_segments.append(remi_sequence[t:t+max_seq_len+1])
                tgt_segments_theme_msk.append(theme_binary_msk[t:t+max_seq_len+1])
        else:
            total_s =  math.ceil(len(remi_sequence) / max_seq_len)
            for x in range(0,len(remi_sequence),max_seq_len):
                if always_include:
                    # if self.token2id["Theme_Start"] in remi_sequence[x:x+max_seq_len+1]:
                    if self.token2id["Theme_Start"] in remi_sequence[x:x+max_seq_len+1] or self.token2id["Theme_End"] in remi_sequence[x:x+max_seq_len+1]:
                        s += 1 
                        tgt_segments.append(remi_sequence[x:x+max_seq_len+1])
                        tgt_segments_theme_msk.append(theme_binary_msk[x:x+max_seq_len+1])
                else:
                    tgt_segments.append(remi_sequence[x:x+max_seq_len+1])
                    tgt_segments_theme_msk.append(theme_binary_msk[x:x+max_seq_len+1])

            if verbose and always_include:
                print("Include Theme Start {}/{}".format(s,total_s))
        


        return {
            "src" : src,
            "src_theme_binary_msk" : src_theme_binary_msk,
            "tgt_segments" : tgt_segments,
            "tgt_segments_theme_binary_msk" : tgt_segments_theme_msk
        }

    def REMIID2midi(self,event_ids,midi_path,verbose = False):
        """convert tokens to midi file
        The output midi file will contains 3 tracks:
            MELODY : melodt notes
            PIANO : accompaniment notes
            Theme info track : notes indicating theme region (using note start tick and end tick as the boundary of theme region)
        Args:
            event_ids (list): sequence of tokens
            midi_path (str): the output midi file path
            verbose (bool, optional): print some message. Defaults to False.
        """

        # create midi file
        new_mido_obj = mid_parser.MidiFile()
        new_mido_obj.ticks_per_beat = 120

        # create tracks
        music_tracks = {}
        music_tracks["MELODY"] = ct.Instrument(program=0, is_drum=False, name='MELODY')
        music_tracks["PIANO"] = ct.Instrument(program=0, is_drum=False, name='PIANO')
        music_tracks["Theme info track"] = ct.Instrument(program=0, is_drum=False, name='Theme info track')

        # all our generated music are 4/4
        new_mido_obj.time_signature_changes.append(miditoolkit.TimeSignature(4,4,0))

        ticks_per_step = new_mido_obj.ticks_per_beat / self.q_beat



        # convert tokens from id to string
        events = []
        for x in event_ids:
            events.append(self.id2token[x])
        
        # parsing tokens
        last_tick = 0
        current_bar_anchor = 0
        current_theme_boundary = []
        motif_label_segs = []
        idx = 0
        first_bar = True
        while(idx < len(events)):
            if events[idx] == "Bar":
                if first_bar:
                    current_bar_anchor = 0
                    first_bar = False
                else:
                    current_bar_anchor += new_mido_obj.ticks_per_beat * 4
                idx+=1
            elif events[idx].startswith("Position"):
                pos = int(events[idx].split('_')[1])
                last_tick = pos * ticks_per_step + current_bar_anchor
                idx += 1
            elif events[idx].startswith("Tempo"):
                tmp = pos = int(events[idx].split('_')[1])
                new_mido_obj.tempo_changes.append(ct.TempoChange(
                    tempo=int(tmp),
                    time=int(last_tick)
                ))
                idx += 1
            elif events[idx].startswith("Note"):
                track_name = events[idx].split("_")[0].split("-")[2]
                assert track_name in music_tracks
                assert events[idx].startswith("Note-On")
                assert events[idx+1].startswith("Note-Duration")
                assert events[idx+2].startswith("Note-Velocity")

                new_note = miditoolkit.Note(
                    velocity=int(events[idx+2].split("_")[1]),
                    pitch=int(events[idx].split("_")[1]),
                    start=int(last_tick),
                    end=int(int(events[idx+1].split('_')[1]) * ticks_per_step) + int(last_tick)
                )
                music_tracks[track_name].notes.append(new_note)
                idx += 3
            elif events[idx] == "Theme_Start":
                assert len(current_theme_boundary) == 0
                current_theme_boundary.append(last_tick)
                idx += 1
            elif events[idx] == "Theme_End":
                assert len(current_theme_boundary) == 1
                current_theme_boundary.append(last_tick)
                motif_label_segs.append(current_theme_boundary)
                music_tracks["Theme info track"].notes.append(
                    miditoolkit.Note(
                        velocity=1,
                        pitch=1,
                        start=int(current_theme_boundary[0]),
                        end=int(current_theme_boundary[1])
                    )
                )
                current_theme_boundary = []
                idx += 1
        
        # add tracks to midi file
        new_mido_obj.instruments.extend([music_tracks[ins] for ins in music_tracks])

        if verbose:
            print("Saving midi to ({})".format(midi_path))
        
        # save to disk
        new_mido_obj.dump(midi_path)

    def __str__(self):
        """return all tokens

        Returns:
            str: string of all tokens
        """
        ret = ""
        for w,i in self.token2id.items():
            ret = ret + "{} : {}\n".format(w,i)

        for i,w in self.id2token.items():
            ret = ret + "{} : {}\n".format(i,w)

        ret += "\nTotal events #{}".format(len(self.id2token))

        return ret

    def __repr__(self):
        """return string all token

        Returns:
            str: string of sll tokens
        """
        return self.__str__() 


if __name__ == '__main__':
    # print all tokens
    myvocab = Vocab()
    
    print(myvocab)

    