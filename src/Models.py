from __future__ import annotations

from typing import TYPE_CHECKING
from SongInfo import TrainingDataProcessedInfo, OrnamentGrouping, OrnamentNote
from music21 import corpus, instrument
from Rhythm import RhythmHMM
from music21 import stream, note, tempo, instrument
from config import *
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from Note import TrainingNote
    from Timings import KeyTiming

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import pandas as pd

"""
    Model reponsible for generating ornament notes. 
    Contains a list of dictionaries maping from pitch offset to markov chains.
"""
class OrnamentNoteMCs:
    def __init__(self):
        self.mcs = {}

    def save_model(self, filepath='models/ornament_mcs.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath='models/ornament_mcs.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    """
        Splits a list of ornament groupings into a dictionary to be used for training
        data for each offset.
    """
    def split_song_ornaments(self, ornament_groupings: list[OrnamentGrouping]):
        offset_function_dict = defaultdict(list)

        for grouping in ornament_groupings:
            note_offset = grouping.get_group_note_interval()
            offset_function_dict[note_offset].append(grouping)

        return offset_function_dict

    def train_mcs(self, training_data: list[OrnamentGrouping]):
        offset_training_data = self.split_song_ornaments(training_data)
        for offset, training_data in offset_training_data.items():
            mc = OrnamentMC()
            succ_train = mc.train_mc(training_data)
            if succ_train:
                self.mcs[offset] = mc

"""
Individual first-order Markov Chain for a particular skeleton offset
"""
class OrnamentMC:
    def __init__(self):
        self.transition_matrix = {}
        self.initial_probabilities = {}
        self.recent_index_used = -1
        self.recent_offset = None

    """
        Train initial emissions
    """
    def calc_initial_probabilities(self, ornament_groupings: list[OrnamentGrouping]):
        initial_count = defaultdict(int)

        for grouping in ornament_groupings:
            if len(grouping.ornament_notes) == 0:
                continue
            initial_note_offset = grouping.ornament_notes[0].offset
            initial_count[initial_note_offset] += 1

        initial_probs = {}
        total_count = sum(initial_count.values())
        for initial_emission, count in initial_count.items():
            initial_probs[initial_emission] = count / total_count

        return initial_probs

    """
        Train state transistion probabilities
    """
    def calc_transition_matrix(self, ornament_groupings: list[OrnamentGrouping]):
        transition_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            # No ornament notes
            if len(grouping.ornament_notes) == 0:
                continue

            for i in range(len(grouping.ornament_notes)-1):
                curr_note_offset = grouping.ornament_notes[i].offset
                next_note_offset = grouping.ornament_notes[i+1].offset
                transition_count[curr_note_offset][next_note_offset] += 1

        if len(transition_count) == 0:
            print('no training data')

        transition_probs = {}
        for current_offset in transition_count.keys():
            total_count = sum(transition_count[current_offset].values())
            for next_offset, count in transition_count[current_offset].items():
                if current_offset not in transition_probs:
                    transition_probs[current_offset] = {}
                transition_probs[current_offset][next_offset] = count / total_count

        return transition_probs

    """
        Generate a new offset
    """
    def sample_next_offset(self, current_offset):
        if current_offset not in self.transition_matrix:
            raise ValueError("Invalid offset.")

        next_offsets = list(self.transition_matrix[current_offset].keys())
        offsets_probs = list(self.transition_matrix[current_offset].values())

        return np.random.choice(next_offsets, p=offsets_probs)

    def sample_initial_state(self):
        offsets = list(self.initial_probabilities.keys())
        probs = list(self.initial_probabilities.values())
        return np.random.choice(offsets, p=probs)

    def train_mc(self, ornament_groupings: list[OrnamentGrouping]):
        try:
            self.transition_matrix = self.calc_transition_matrix(ornament_groupings)
            self.initial_probabilities = self.calc_initial_probabilities(ornament_groupings)
            return True
        except ValueError as e:
            print(f'Error training model: {e}')
            return False

    def generate(self, skeleton_note_index, current_note_index):
        # Last time this MC was used was before current skeleotn note index. therefore need to sample new note
        if self.recent_index_used < skeleton_note_index:
            emission = self.sample_initial_state()
            print(f'sampling from initila')
        else:
            emission = self.sample_next_offset(self.recent_offset)
            print(f'sampling from old one')

        print(f'note emission: {emission}')
        self.recent_index_used = current_note_index
        self.recent_offset = emission

        return emission

"""
    User facing class that will be called to generate a piece of music.
"""
class MusicGen:
    def __init__(self, chord_progression_hmm: ChordHMM, ornament_mcs: OrnamentNoteMCs, rhythm_model: RhythmHMM, bass_generator: BassNoteGenerator):
        self.chord_progression_hmm = chord_progression_hmm
        self.ornament_note_mcs = ornament_mcs
        self.rhythm_model = rhythm_model
        self.bass_generator = bass_generator
        self.fixed_ornament_note_pitches = {}

    """
        Function that will be called by main to actually generate the piece.
    """
    def generate_midi_score(self, key, num_notes, file_output='output.mid'):
        melody, bass_notes, _ = self.generate_lines(key, num_notes)
        score = stream.Score()
        treble = stream.Part()
        treble.append(instrument.Piano())
        treble.append(tempo.MetronomeMark(number=METRONOME_80_BPM))
        bass = stream.Part()
        bass.append(instrument.Piano())

        # Add bass notes
        for i in range(len(bass_notes)-1):
            pitch, onset = bass_notes[i]
            n = note.Note(pitch)
            n.quarterLength = MINIM_DURATION_CROTCHET
            bass.append(n)

            _, next_note_onset = bass_notes[i+1]
            if next_note_onset != (onset + STEPS_PER_BEAT * 2):
                rest_duration = next_note_onset - onset - STEPS_PER_BEAT * 2
                r = note.Rest()
                r.quarterLength = rest_duration / STEPS_PER_BEAT
                bass.append(r)

        # Add last bass note
        pitch, _ = bass_notes[-1]
        n = note.Note(pitch)
        n.quarterLength = MINIM_DURATION_CROTCHET
        bass.append(n)

        # Add melody notes to treble clef
        for i in range(len(melody)-1):
            curr_note = melody[i]
            pitch = curr_note['pitch']
            n = note.Note(pitch)
            duration = curr_note['end'] - curr_note['start']
            n.quarterLength = duration / STEPS_PER_BEAT  # Duration in quarter notes
            treble.append(n)

            next_note = melody[i+1]
            if next_note['start'] != curr_note['end']:
                # Need a rest!
                rest_duration = next_note['start'] - curr_note['end']
                r = note.Rest()
                r.quarterLength = rest_duration / STEPS_PER_BEAT
                treble.append(r)

        # Add last treble clef note
        last_note = melody[-1]
        pitch = last_note['pitch']
        n = note.Note(pitch)
        duration = last_note['end'] - last_note['start']
        n.quarterLength = duration / STEPS_PER_BEAT  # Duration in quarter notes
        treble.append(n)
        
        score.append(treble)
        score.append(bass)
        score.write('midi', fp=file_output)

    def generate_lines(self, key, num_notes):
        melody = []
        bass = []

        sampled_beats = self.chord_progression_hmm.generate(num_notes)
        sampled_beats = sampled_beats[::2]
        sampled_rhythm = self.rhythm_model.generate_rhythm_sequence(num_notes)

        print(f'rhythm: {sampled_rhythm}')
        original_sampled_beats = sampled_beats.copy()
        print(sampled_beats[:10])

        for i in range(10):
            note = sampled_beats[i]
            print(f'note chord tone: {note.note_chord_tone}. chord function: {note.chord_function}')

        i = 0
        while i < len(sampled_rhythm):
            #current_time = i * seconds_per_step
            ## Check if on strong beat position
            if i % 8 == 0:
                ## Need to start playing a fresh note
                if sampled_rhythm[i] == NOTE_ONSET:
                    # Check if ornament note or skeelton note
                    end_index = i + 1
                    while (end_index < len(sampled_rhythm) and sampled_rhythm[end_index] == 1):
                        end_index += 1

                    # Skeleton note
                    # create MIDI note
                    skeleton_note = sampled_beats.pop(0)
                    midi_pitch = skeleton_note.calc_midi_pitch(key)
                    chord_function = skeleton_note.chord_function
                    note = {
                        'pitch': midi_pitch,
                        'start': i,
                        'end': end_index,
                        'chord_function': chord_function
                    }
                    melody.append(note)

                    # Generate cooresponding bass note
                    bass_note_chord_tone = self.bass_generator.get_bass_note(skeleton_note.note_chord_tone)
                    print(f' skeleton note: chord function: {skeleton_note.chord_function} trelle chord tone: {skeleton_note.note_chord_tone} bass note chord tone: {bass_note_chord_tone}')
                    bass_note_midi_pitch = SkeletonEmission(bass_note_chord_tone, skeleton_note.chord_function).calc_midi_pitch(key) - OCTAVE_SEMITONES
                    bass.append((bass_note_midi_pitch, i))
                elif sampled_rhythm[i] == NOTE_CONTINUE or sampled_rhythm[i] == NOTE_REST:
                    # Sustaining previous note or rest on this strong beat
                    if sampled_beats:
                        skeleton_note = sampled_beats.pop(0)
                        # Generate cooresponding bass note
                        bass_note_chord_tone = self.bass_generator.get_bass_note(skeleton_note.note_chord_tone)
                        bass_note_midi_pitch = SkeletonEmission(bass_note_chord_tone, skeleton_note.chord_function).calc_midi_pitch(key) - OCTAVE_SEMITONES
                        bass.append((bass_note_midi_pitch, i))
            else:
                # need an ornament note first check to see whether we need to generate a new note or note
                if sampled_rhythm[i] == NOTE_ONSET:
                    # check to see if ornament note pitch has already been decided
                    if i in self.fixed_ornament_note_pitches:
                        midi_pitch = self.fixed_ornament_note_pitches[i].calc_midi_pitch(key)
                        chord_function = self.fixed_ornament_note_pitches[i].chord_function
                    else:
                        try:
                            previous_skeleton_note, previous_skeleton_note_index = self.find_previous_skeleton_note(i, melody, sampled_rhythm)
                            previous_skeleton_note_pitch = previous_skeleton_note['pitch']
                            previous_skeleton_note_chord_function = previous_skeleton_note['chord_function']
                            next_skeleton_note = self.find_next_skeleton_note(i, sampled_beats, sampled_rhythm)
                            next_skeleton_note_pitch = next_skeleton_note.calc_midi_pitch(key)
                        except ValueError as e:
                            # Default to 0 offset and setting previous chord function to 'I'
                            print(e)
                            next_skeleton_note_pitch = 60
                            previous_skeleton_note_pitch = 60
                            previous_skeleton_note_index = 0

                        # Check if this note is the note that sutains onto the next strong beat note
                        if i in self.fixed_ornament_note_pitches:
                            midi_pitch = self.fixed_ornament_note_pitches[i].calc_midi_pitch(key)
                        else:
                            try:
                                skeleton_offset = previous_skeleton_note_pitch - next_skeleton_note_pitch 
                                ornament_mc = self.ornament_note_mcs.mcs[skeleton_offset]
                                ornament_note_offset = ornament_mc.generate(previous_skeleton_note_index, i)
                            except Exception:
                                ornament_note_offset = 0

                            if len(melody) > 0:
                                midi_pitch = melody[-1]['pitch'] + ornament_note_offset
                            else:
                                midi_pitch = previous_skeleton_note_pitch

                    end_index = i + 1
                    while (end_index < len(sampled_rhythm) and sampled_rhythm[end_index] == NOTE_CONTINUE):
                        end_index += 1

                    note = {
                        'pitch': midi_pitch,
                        'start': i,
                        'end': end_index,
                        'chord_function': chord_function
                    }
                    melody.append(note)
            i += 1

        self.postprocess_melody(melody, key)
        return melody, bass, original_sampled_beats

    """
        Returns the pitch and chord function of the next skeleton note
    """
    def find_next_skeleton_note(self, current_index, skeleton_notes, sampled_rhythm):
        # first find out index of next skeleton note
        next_strong_beat_index = current_index + 1
        found_strong_beat = False
        while (not found_strong_beat and next_strong_beat_index < len(sampled_rhythm)):
            if next_strong_beat_index % (STEPS_PER_BEAT * 2) == 0:
                found_strong_beat = True
                break
            else:
                next_strong_beat_index += 1

        if not found_strong_beat:
            raise ValueError("no next strong beat found")

        # Note being generated at the strong beat
        if sampled_rhythm[next_strong_beat_index] == NOTE_ONSET:
            return skeleton_notes[0]
        elif sampled_rhythm[next_strong_beat_index] == NOTE_ONSET:
            # note is being sustained, figure out index of note that will sustain it
            sustained_note_index = next_strong_beat_index-1
            while sustained_note_index >= 0:
                if sampled_rhythm[sustained_note_index] == NOTE_ONSET:
                    break
                else:
                    sustained_note_index -= 1
            self.fixed_ornament_note_pitches[sustained_note_index] = skeleton_notes[0]
        else:
            raise ValueError("rest being played at skeleton note.")
        
        return skeleton_notes[0]

    """ 
        issue right now is that sound playing at previous skeleton note might not be played 
        exactly on a skeleton note

        so to find the pitch, we should search backwards to find out which index is responsible for
        that not being played. then we can search for whichever note was generated at that index what the pitch is
        can find index by doing start / seconds_per_step
    """
    def find_previous_skeleton_note(self, current_index, current_melody, sampled_rhythm):
        # search backwards and find most recent strong beat index
        recent_strong_beat_index = current_index - 1
        found_strong_beat = False
        while (not found_strong_beat and recent_strong_beat_index >= 0):
            if recent_strong_beat_index % (STEPS_PER_BEAT * 2) == NOTE_ONSET:
                found_strong_beat = True
                break
            else:
                recent_strong_beat_index -= 1

        if not found_strong_beat:
            raise ValueError("no previous skeleton note found")

        # Check if this strong beat was a 1 (new note) or 2 (sustained pitch from previous note)
        if sampled_rhythm[recent_strong_beat_index] == NOTE_ONSET:
            for note in reversed(current_melody):
                if note['start'] == recent_strong_beat_index:
                    return note, recent_strong_beat_index
        elif sampled_rhythm[recent_strong_beat_index] == NOTE_ONSET:
            # note was sustained from somewhere else, need to find note responsible for it 
            # by searching to the left somemore for the most recent 1 note
            recent_new_note_index = recent_strong_beat_index - 1
            found_new_note = False
            while (not found_new_note and recent_new_note_index >= 0):
                if sampled_rhythm[recent_new_note_index] == NOTE_ONSET:
                    found_new_note = True
                    break
                else:
                    recent_new_note_index -= 1
            
            if not found_new_note:
                raise ValueError("no previous skeleton note found")
            
            for note in reversed(current_melody):
                if note['start'] == recent_new_note_index:
                    return note, recent_strong_beat_index

        raise ValueError("no previous skeleton note found")

    def postprocess_melody(self, notes, key):
        """
        Clean up generated melody by avoiding really bad notes.
        """

        key_name = key.name
        
        # Define tritones from tonic (notes to absolutely avoid)
        if key_name not in KEY_SCALES:
            return notes  # Skip processing if key not defined
        
        allowed_notes = set(KEY_SCALES[key_name])
        tritone = TRITONES.get(key_name)
        
        for note in notes:
            pitch_class = note['pitch'] % OCTAVE_SEMITONES
            # Check if its tritone
            if pitch_class == tritone:
                # Move tritone to nearest safe note
                if tritone + 1 in allowed_notes:
                    corrected_pitch = note['pitch'] + 1
                elif tritone - 1 in allowed_notes:
                    corrected_pitch = note['pitch'] - 1
                else:
                    corrected_pitch = note['pitch'] + 2  # Fallback
                
                print(f"Fixed tritone: {note['pitch']} -> {corrected_pitch}")
                note['pitch'] = corrected_pitch
                
            # Check if it's outside the key scale
            elif pitch_class not in allowed_notes:
                # Find nearest note in scale
                distances = [(abs(pitch_class - allowed), allowed) for allowed in allowed_notes]
                distances.sort(key=lambda x: x[0])

                nearest_note = distances[0][1]
                note['pitch'] = nearest_note

class BassGen:
    def __init__(self):
        self.emission_probs = {}

    def get_bass_note(self, soprano_chord_tone):
        if soprano_chord_tone not in self.emission_probs:
            return 'root'
        
        bass_notes = list(self.emission_probs[soprano_chord_tone].keys())
        bass_probs = list(self.emission_probs[soprano_chord_tone].values())

        return np.random.choice(bass_notes, p=bass_probs)

    def train_model(self):
        soprano_bass_pairs = self.create_soprano_bass_pairs()

        emission_count = defaultdict(lambda: defaultdict(int))
        for soprano_note, bass_note in soprano_bass_pairs:
            if soprano_note is None or bass_note is None:
                continue
            emission_count[soprano_note][bass_note] += 1

        emission_probs = {}
        for soprano_note in emission_count.keys():
            if soprano_note not in emission_probs:
                emission_probs[soprano_note] = {}
            total_count = sum(emission_count[soprano_note].values())
            for bass_note, count in emission_count[soprano_note].items():
                emission_probs[soprano_note][bass_note] = count / total_count

        print(emission_probs)

        self.emission_probs = emission_probs

    def save_model(self, filepath='models/bass_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath='models/bass_model.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def create_soprano_bass_pairs(self):
        bach_paths = corpus.getComposer('bach')
        chorale_paths = [p for p in bach_paths if 'bwv' in str(p).lower()]
        pairs = []

        for path in chorale_paths:
            print(f'currently processing {path}...')
            score = corpus.parse(path)
            key_mode: str = score.analyze('key').mode

            if 'minor' in key_mode.lower():
                continue

            soprano = None
            for p in score.parts:
                if 'Soprano' in p.id:
                    soprano = p
                    break
            if soprano is None:
                continue

            bass = None
            for p in score.parts:
                if 'Bass' in p.id:
                    bass = p
                    break
            if bass is None:
                continue

            soprano_notes = soprano.flatten().notes
            bass_notes = bass.flatten().notes
            chords = score.chordify()
            chord_offset_dict = {chord.offset: chord for chord in chords.flatten().getElementsByClass('Chord') }

            for soprano_note in soprano_notes:
                target_offset = soprano_note.offset
                matching_bass_note = None

                # Find matching bass note
                for bass_note in bass_notes:
                    if bass_note.offset == target_offset:
                        matching_bass_note = bass_note

                if matching_bass_note is not None:
                    chord_offset = max([o for o in chord_offset_dict if o <= soprano_note.offset])
                    chord = chord_offset_dict.get(chord_offset, None)
                    if chord is not None:
                        try:
                            soprano_chord_tone = self.get_chord_tone(soprano_note, chord)
                            bass_chord_tone = self.get_chord_tone(matching_bass_note, chord)
                            pairs.append((soprano_chord_tone, bass_chord_tone))
                        except Exception:
                            continue
                    else:
                        continue
            
        return pairs
            
    def get_chord_tone(self, note, chord):
        if chord is None:
            return 'root'

        note_pc = note.pitch.pitchClass
        root_pc = chord.root().pitchClass

        interval = (note_pc - root_pc) % 12

        return SCALE_DEGREE_NOTE_NAME_MAPPING.get(interval, 'root')

class SkeletonEmission:
    def __init__(self, note_chord_tone, chord_function):
        self.note_chord_tone = note_chord_tone
        self.octave_offset = 4
        self.chord_function = chord_function

    def calc_midi_pitch(self, key: KeyTiming):
        key_root_note = key.get_root_note()
        key_pitch_class = note_to_pitch_class.get(key_root_note, 0)
        chord_pitch_class = (key_pitch_class + roman_numeral_to_semitones.get(self.chord_function, 0)) % 12
        interval = NOTE_NAME_TO_SCALE_DEGREE_MAPPING.get(self.note_chord_tone, 0)
        note_midi_pitch = (self.octave_offset + 1) * 12 + chord_pitch_class + interval
        return note_midi_pitch

class ChordHMM:
    def __init__(self):
        self.transition_matrix = {}
        self.duration_matrix = {}
        self.emission_matrix = {}
        self.initial_probabilities = {}

    def save_model(self, filepath="models/chord_hmm.pkl"):
        transition_probs = {}
        emission_probs = {}
        duration_probs = {}

        if self.duration_matrix:
            duration_probs = {k: dict(v) for k, v in self.duration_matrix.items()}

        if self.transition_matrix:
            transition_probs = {k: dict(v) for k, v in self.transition_matrix.items()}

        if self.emission_matrix:
            emission_probs = {k: dict(v) for k, v in self.emission_matrix.items()}

        model_data = {
            'transition_probs': transition_probs,
            'duration_probs': duration_probs,
            'emission_probs': emission_probs,
            'initial_probs': self.initial_probabilities
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}.")

    @classmethod
    def load(cls, filepath='models/chord_hmm.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.transition_matrix = model_data['transition_probs']
        model.duration_matrix = model_data['duration_probs']
        model.emission_matrix = model_data['emission_probs']
        model.initial_probabilities = model_data['initial_probs']

        return model

    def calc_hidden_state_duration_matrix(self, beat_chords_list: list[list[str]]):
        """
        song_notes_dict is the dict of 001
        want it to look like 'I': {4: 0.05, 3: 0.95}
        """
        duration_count = defaultdict(lambda: defaultdict(int))
        """
        loop through items, while the next one is same as previous one
        increment counter. when we see next one is different, save current count to probablity. reset counter to 0
        """

        # [I, I, I, iv, v, v, N]
        for curr_chord_sequence in beat_chords_list:
            duration_counter = 0
            for i in range(len(curr_chord_sequence)):

                # Last chord in list
                if i == len(curr_chord_sequence) - 1:
                    # If last chord is not the same as previous one, then should increment count for '1' only if not 'N' chord
                    if i > 0 and curr_chord_sequence[i] != 'N' and curr_chord_sequence[i] != curr_chord_sequence[i-1]:
                        duration_count[curr_chord_sequence[i]][1] += 1
                    # If last chord is same as previous one, then should increment duration counter only for if non 'N'
                    elif i > 0 and curr_chord_sequence[i] != 'N' and curr_chord_sequence[i] == curr_chord_sequence[i-1]:
                        duration_counter += 1
                        duration_count[curr_chord_sequence[i]][duration_counter] += 1
                    else:
                        continue
                else:
                    curr_chord = curr_chord_sequence[i]
                    next_chord = curr_chord_sequence[i+1]

                    if curr_chord == 'N':
                        continue

                    # End of current chords duration
                    if next_chord == 'N':
                        duration_counter += 1
                        duration_count[curr_chord][duration_counter] += 1
                        duration_counter = 0
                        continue

                    if next_chord == curr_chord:
                        duration_counter += 1
                    else:
                        # End of current chord's duration
                        duration_counter += 1
                        duration_count[curr_chord][duration_counter] += 1
                        duration_counter = 0
        
        duration_probs = defaultdict(lambda: defaultdict(float))
        for chord_function in duration_count.keys():
            total_count = sum(duration_count[chord_function].values())
            for beat_duration, count in duration_count[chord_function].items():
                duration_probs[chord_function][beat_duration] = count / total_count

        return duration_probs

    # Want a matrix that contains chord function transition probabilities
    def calc_hidden_state_transition_matrix(self, beat_chords_list: list[list[str]]):
        """
            looks like 'I': {'II': 0.05, 'IV': 0.03}, 'II': {'I':0.01}
        """
        transition_count = defaultdict(lambda: defaultdict(int))

        for curr_chord_sequence in beat_chords_list:

            collapsed_chord_sequence = self.collapse_chord_sequence(curr_chord_sequence)
            for i in range(len(collapsed_chord_sequence)-2):
                first_chord = collapsed_chord_sequence[i]
                second_chord = collapsed_chord_sequence[i+1]
                third_chord = collapsed_chord_sequence[i+2]

                if first_chord == 'N' or second_chord == 'N' or third_chord == 'N': 
                    continue

                transition_count[(first_chord, second_chord)][third_chord] += 1

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_chord in transition_count.keys():
            total_count = sum(transition_count[curr_chord].values())
            for next_chord, count in transition_count[curr_chord].items():
                transition_probs[curr_chord][next_chord] = count / total_count

        return transition_probs

    def calc_emission_state_transition_matrix(self, notes: list[TrainingNote]):
        """
            want it to be like {'IV': {root: 0.03}}
        """
        emission_count = defaultdict(lambda: defaultdict(int))

        for i in range(len(notes)):
            chord_function = notes[i].get_chord_function()
            note_chord_tone = notes[i].get_chord_tone()

            emission_count[chord_function][note_chord_tone] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for chord_function in emission_count.keys():
            total_count = sum(emission_count[chord_function].values())
            for note_chord_tone, count in emission_count[chord_function].items():
                emission_probs[chord_function][note_chord_tone] = count / total_count

        return emission_probs

    def get_note_beat_sequence(self, chord_beat_notes_list):
        return [(i[0]) for i in chord_beat_notes_list]

    def sample_emission(self, hidden_state):
        if hidden_state not in self.emission_matrix:
            raise ValueError("Invalid hidden state")

        emission_notes = list(self.emission_matrix[hidden_state].keys())
        emission_probs = list(self.emission_matrix[hidden_state].values())

        return emission_notes[np.random.choice(len(emission_notes), p=emission_probs)]

    def sample_hidden_state_duration(self, current_hidden_state):
        if current_hidden_state not in self.duration_matrix:
            raise ValueError("Invalid hidden state")

        duration_beats = list(self.duration_matrix[current_hidden_state].keys())
        duration_probs = list(self.duration_matrix[current_hidden_state].values())

        return duration_beats[np.random.choice(len(duration_beats), p=duration_probs)]

    def sample_next_hidden_state(self, prev_chord, curr_chord):
        if (prev_chord, curr_chord) not in self.transition_matrix:
            raise ValueError("Invalid hidden state")

        next_hidden_states = list(self.transition_matrix[(prev_chord, curr_chord)].keys())
        next_hidden_probs = list(self.transition_matrix[(prev_chord, curr_chord)].values())

        return next_hidden_states[np.random.choice(len(next_hidden_states), p=next_hidden_probs)]

    def train_model(self, song_notes_dict, beat_chords_dict):
        self.transition_matrix = self.calc_hidden_state_transition_matrix(beat_chords_dict)
        self.duration_matrix = self.calc_hidden_state_duration_matrix(beat_chords_dict)
        self.emission_matrix = self.calc_emission_state_transition_matrix(song_notes_dict)

    def generate(self, num_beats):
        def sample_beats(hidden_state, num_beats):
            print(f'Sampling {num_beats} for chord {hidden_state}')
            beats = []
            for _ in range(num_beats):
                chord_tone = self.sample_emission(hidden_state)
                beats.append(SkeletonEmission(chord_tone, hidden_state))
            return beats

        sampled_chord_function_beat_duration = []
        sampled_beats = []

        # Sample initial hidden state
        #hidden_states = list(self.emission_matrix.keys())
        #hidden_states_probs = list(self.initial_probabilities[state] for state in hidden_states)
        #current_state = hidden_states[np.random.choice(len(hidden_states), p=hidden_states_probs)]
        #current_emission = self.sample_emission(current_state)

        # Force first state to be 'I'
        prev_prev_state = 'I'
        print('I')
        print(f'yeetics: {self.duration_matrix}')
        current_state_duration = self.sample_hidden_state_duration(prev_prev_state)
        sampled_chord_function_beat_duration.append((prev_prev_state, current_state_duration))
        sampled_beats.extend(sample_beats(prev_prev_state, current_state_duration))

        prev_state = 'V'
        print('V')
        current_state_duration = self.sample_hidden_state_duration(prev_state)
        sampled_chord_function_beat_duration.append((prev_state, current_state_duration))
        sampled_beats.extend(sample_beats(prev_state, current_state_duration))

        while len(sampled_beats) < num_beats:
            # Sample next hidden state and emission.
            current_state = self.sample_next_hidden_state(prev_prev_state, prev_state)

            current_state_duration = self.sample_hidden_state_duration(current_state)

            sampled_chord_function_beat_duration.append((current_state, current_state_duration))
            sampled_beats.extend(sample_beats(current_state, current_state_duration))

            prev_prev_state = prev_state
            prev_state = current_state

        return sampled_beats[:num_beats]
    
    def generate_chord_transitions(self, num_chords):
        sampled_chords = ['I', 'V']
        prev_prev_state = 'I'
        prev_state = 'V'

        while len(sampled_chords) < num_chords:
            current_state = self.sample_next_hidden_state(prev_prev_state, prev_state)
            sampled_chords.append(current_state)
            prev_prev_state = prev_state
            prev_state = current_state

        return sampled_chords

    def get_hidden_states(self):
        return list(self.transition_matrix.keys())

    def collapse_chord_sequence(self, chord_sequence):
        if len(chord_sequence) <= 1:
            return chord_sequence

        collapsed_sequence = [chord_sequence[0]]
        for i in range(1, len(chord_sequence)):
            if chord_sequence[i] != collapsed_sequence[-1]:
                collapsed_sequence.append(chord_sequence[i])

        return collapsed_sequence

if __name__ == "__main__":
    #bass_gen = BassNoteGenerator()
    #bass_gen.train_model()
    #bass_gen.save_model()

    #bass_model = BassNoteGenerator.load_model()
    #plot_bass_heatmap(bass_model)

    #training_directory = "./POP909_training"
    #training_data = TrainingDataProcessedInfo()
    #training_data.load_training_data(training_directory)
    ###print(training_data.beat_chords)

    testing_directory = "./POP909_testing"
    testing_data = TrainingDataProcessedInfo()
    testing_data.load_training_data(testing_directory)

    second_order = ChordHMM.load()

    #chord_hmm_second_order = ChordHMM()
    #chord_hmm_second_order.train_model(training_data.notes, training_data.beat_chords)
    #chord_hmm_second_order.save_model('models/chord_second_order.pkl')

    #chord_hmm_first_order = ChordHMMFirstOrder()
    #chord_hmm_first_order.train_model(training_data.notes, training_data.beat_chords)
    #chord_hmm_first_order.save_model('models/chord_first_order.pkl')

    #chord_hmm_third_order = ChordHMMThirdOrder()
    #chord_hmm_third_order.train_model(training_data.notes, training_data.beat_chords)
    #chord_hmm_third_order.save_model('models/chord_third_order.pkl')

    #chord_probs_occurence = count_num_occurences(training_data.beat_chords)
    #print(f'baseline{calculate_unigram_baseline(testing_data.beat_chords, chord_probs_occurence)}')

    #print(f'unigram base line training: {calculate_unigram_baseline(training_data.beat_chords, chord_probs_occurence)}')
    #print(f'unigram base line testing: {calculate_unigram_baseline(testing_data.beat_chords, chord_probs_occurence)}')

    #first_order = ChordHMMFirstOrder.load('models/chord_first_order.pkl')
    #second_order = ChordHMM.load('models/chord_second_order.pkl')

    #print(second_order.emission_matrix)
    #plot_emission_heat_map()
    #print(emission_log_likelihood(testing_data.notes, second_order))

    #emission_probs = count_num_occurences_emission_notes(training_data.notes)
    #print(f' unigram testing nll: {calculate_unigram_baseline_emission(testing_data.notes, emission_probs)}')
    #print(f' unigram training nll: {calculate_unigram_baseline_emission(training_data.notes, emission_probs)}')

    #print(f'emission_log_likelihood training: {emission_log_likelihood(training_data.notes, second_order)}')
    #print(f'emission_log_likelihood testing: {emission_log_likelihood(testing_data.notes, second_order)}')
    chord_transitions = second_order.generate_chord_transitions(100000)
    flattened_real_chords = [chord for sequence in testing_data.beat_chords for chord in sequence]
    flattened_real_chords = collapse_chord_sequence(flattened_real_chords)
    chords = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii']

    gen_matrix = get_bigram_matrix(chord_transitions, chords)
    real_matrix = get_bigram_matrix(flattened_real_chords, chords)


    #fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    #sns.heatmap(real_matrix, annot=True, fmt='.2f', xticklabels=chords, 
                #yticklabels=chords, cmap='Blues', vmin=0, vmax=1)
    #plt.title('Real Chords')
    #plt.xlabel('Next Chord')
    #plt.ylabel('Current Chord')

    #plt.tight_layout()
    #plt.savefig('heatmap_real.png', dpi=300, bbox_inches='tight')

    #sns.heatmap(gen_matrix, annot=True, fmt='.2f', xticklabels=chords, 
                #yticklabels=chords, cmap='Blues', vmin=0, vmax=1)
    #plt.title('Generated Chords')
    #plt.xlabel('Next Chord')
    #plt.ylabel('Current Chord')

    #plt.tight_layout()
    #plt.savefig('heatmap_gen.png', dpi=300, bbox_inches='tight')

    #sns.heatmap(gen_matrix, annot=True, fmt='.2f', xticklabels=chords, 
                #yticklabels=chords, ax=axes[1], cmap='Blues', vmin=0, vmax=1)
    #axes[1].set_title('Generated Sequence')
    #axes[1].set_xlabel('Next Chord')
    #axes[1].set_ylabel('Current Chord')

    #print(f'chord hmm log likelihod for 2nd order hmm training {hsmm_chord_ll(training_data.beat_chords, second_order)}')
    #print(f'chord hmm log likelihod for 2nd order hmm testing {hsmm_chord_ll(testing_data.beat_chords, second_order)}')

    #print(f'chord hmm log likelihod for first order hmm training {hsmm_chord_ll_first_order(training_data.beat_chords, first_order)}')
    #print(f'chord hmm log likelihod for first order hmm testing {hsmm_chord_ll_first_order(testing_data.beat_chords, first_order)}')

    #print(f'chord hmm log likelihod for third order hmm training {hsmm_chord_ll_third_order(training_data.beat_chords, chord_hmm_third_order)}')
    #print(f'chord hmm log likelihod for third order hmm training {hsmm_chord_ll_third_order(testing_data.beat_chords, chord_hmm_third_order)}')

    #first_order_model = ChordHMMFirstOrder()
    #first_order_model.train_model(training_data.notes, training_data.beat_chords)
    #print(f'chord hmm log likelihod for first order hmm {hsmm_chord_ll_first_order(testing_data.beat_chords, first_order_model)}')

    #testing_directory = "./POP909_testing"
    #testing_data = TrainingDataProcessedInfo()
    #testing_data.load_training_data(testing_directory)

    #chord_hmm = ChordHMM()
    #chord_hmm.train_model(training_data.notes, training_data.beat_chords)
    #chord_hmm.save_model()
    #print(hsmm_log_likelihood(testing_data.notes, chord_hmm))

    #chord_hmm_first_order = ChordHMMFirstOrder()
    #chord_hmm_first_order.train_model(training_data.notes, training_data.beat_chords)
    #print(hsmm_log_likelihood_first_order(testing_data.notes, chord_hmm_first_order))
    #chord_hmm = ChordHMM.load()
    #print(f'first oder: {hsmm_log_likelihood_first_order(data.notes, chord_hmm_first_order)}')
    #print(f'second oder: {hsmm_log_likelihood(data.notes, chord_hmm)}')

    #chord_hmm = ChordHMM.load()
    #print(hsmm_log_likelihood(data.notes, chord_hmm))
    #chord_hmm = ChordHMM()
    #chord_hmm.train_model(data.notes, data.beat_chords)

    #chord_hmm.save_model()
    #chord_hmm = ChordHMM.load()
    #print(chord_hmm.transition_matrix)
    #bass = BassNoteGenerator()
    #bass.train_model()
    #print(bass.emission_probs)
    #bass.save_model()
    #print('loaded model:')
    #print(bass_loaded.emission_probs)
