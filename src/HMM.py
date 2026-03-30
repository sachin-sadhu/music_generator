from __future__ import annotations

from typing import TYPE_CHECKING
from SongInfo import TrainingDataProcessedInfo, OrnamentGrouping, OrnamentNote
from music21 import corpus, instrument

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

class BassNoteGenerator:
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

        interval_map = {
                0: "root", 
                1: "b2", 
                2: "2nd", 
                3: "b3",   # minor 3rd
                4: "3rd",  # major 3rd
                5: "4th",
                6: "b5",   # tritone
                7: "5th", 
                8: "b6",   # minor 6th
                9: "6th",  # major 6th
                10: "b7",  # minor 7th
                11: "7th"  # major 7th
            }

        return interval_map.get(interval, 'root')

class SkeletonEmission:
    def __init__(self, note_chord_tone, octave_offset, chord_function):
        self.note_chord_tone = note_chord_tone
        self.octave_offset = 4
        self.chord_function = chord_function

    def calc_midi_pitch(self, key: KeyTiming):
        chromatic_intervals_inverted = {
            "root": 0, "b2": 1, "2nd": 2, "b3": 3, "3rd": 4, "4th": 5,
            "b5": 6, "5th": 7, "b6": 8, "6th": 9, "b7": 10, "7th": 11, "octave": 12
        }

        roman_numeral_to_semitones = {
            'I': 0,   # Tonic (0 semitones above root)
            'ii': 2,  # 2 semitones above root
            'iii': 4, # 4 semitones above root
            'IV': 5,  # 5 semitones above root
            'V': 7,   # 7 semitones above root ← We need this!
            'vi': 9,  # 9 semitones above root
            'vii': 11 # 11 semitones above root
        }

        note_to_pitch_class = {
            'C': 0, 
            'C#': 1, 'Db': 1,
            'D': 2,
            'D#': 3, 'Eb': 3,
            'E': 4,
            'F': 5,
            'F#': 6, 'Gb': 6,
            'G': 7,
            'G#': 8, 'Ab': 8,
            'A': 9,
            'A#': 10, 'Bb': 10,
            'B': 11
        }

        # Get key information
        key_root_note = key.get_root_note()
        key_pitch_class = note_to_pitch_class.get(key_root_note, 0)
        chord_pitch_class = (key_pitch_class + roman_numeral_to_semitones.get(self.chord_function, 0)) % 12
        interval = chromatic_intervals_inverted.get(self.note_chord_tone, 0)

        #key_root_note_midi_pitch = 60 + note_to_pitch_class.get(key_root_note, 0)
        #chord_root_note_midi_pitch = key_root_note_midi_pitch + roman_numeral_to_semitones.get(self.chord_function, 0)
        #interval = chromatic_intervals_inverted.get(self.note_chord_tone, 0)

        note_midi_pitch = (self.octave_offset + 1) * 12 + chord_pitch_class + interval

        return note_midi_pitch

class ChordHMMThirdOrder:
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
            for i in range(len(collapsed_chord_sequence)-3):
                first_chord = collapsed_chord_sequence[i]
                second_chord = collapsed_chord_sequence[i+1]
                third_chord = collapsed_chord_sequence[i+2]
                fourth_chord = collapsed_chord_sequence[i+3]

                if first_chord == 'N' or second_chord == 'N' or third_chord == 'N' or fourth_chord == 'N': 
                    continue

                transition_count[(first_chord, second_chord, third_chord)][fourth_chord] += 1

        print(f'transition count: {transition_count}')

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
                beats.append(SkeletonEmission(chord_tone, 0, hidden_state))
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
                beats.append(SkeletonEmission(chord_tone, 0, hidden_state))
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


class ChordHMMFirstOrder:
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
            for i in range(len(collapsed_chord_sequence)-1):
                curr_chord = collapsed_chord_sequence[i]
                next_chord = collapsed_chord_sequence[i+1]

                if curr_chord == 'N' or next_chord == 'N': 
                    continue

                transition_count[curr_chord][next_chord] += 1

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

    def sample_next_hidden_state(self, curr_chord):
        if curr_chord not in self.transition_matrix:
            raise ValueError("Invalid hidden state")

        next_hidden_states = list(self.transition_matrix[curr_chord].keys())
        next_hidden_probs = list(self.transition_matrix[curr_chord].values())

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
                beats.append(SkeletonEmission(chord_tone, 0, hidden_state))
            return beats

        sampled_chord_function_beat_duration = []
        sampled_beats = []

        # Sample initial hidden state
        #hidden_states = list(self.emission_matrix.keys())
        #hidden_states_probs = list(self.initial_probabilities[state] for state in hidden_states)
        #current_state = hidden_states[np.random.choice(len(hidden_states), p=hidden_states_probs)]
        #current_emission = self.sample_emission(current_state)

        # Force first state to be 'I'
        prev_state = 'I'
        current_state_duration = self.sample_hidden_state_duration(prev_state)
        sampled_chord_function_beat_duration.append((prev_state, current_state_duration))
        sampled_beats.extend(sample_beats(prev_state, current_state_duration))

        while len(sampled_beats) < num_beats:
            # Sample next hidden state and emission.
            current_state = self.sample_next_hidden_state(prev_state)

            current_state_duration = self.sample_hidden_state_duration(current_state)

            sampled_chord_function_beat_duration.append((current_state, current_state_duration))
            sampled_beats.extend(sample_beats(current_state, current_state_duration))

            prev_state = current_state

        return sampled_beats[:num_beats]

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

def get_segments(test_data):
    segments = []
    curr_note = test_data[0]
    curr_chord = curr_note.get_chord_function()
    curr_emission = curr_note.get_chord_tone()
    curr_emissions = [curr_emission]

    for note in test_data[1:]:
        chord = note.get_chord_function()
        chord_tone = note.get_chord_tone()
        if chord == curr_chord:
            curr_emissions.append(chord_tone)
        else:
            segments.append((curr_chord, len(curr_emissions), curr_emissions))
            curr_chord = chord
            curr_emissions = [chord_tone]

    segments.append((curr_chord, len(curr_emissions), curr_emissions))
    return segments

def hsmm_chord_ll_first_order(chord_data, model):
    epsilon = 1e-9
    segments = []
    for song in chord_data:
        curr_chord = song[0]
        duration_counter = 1
        for chord in song[1:]:
            if chord == curr_chord:
                duration_counter += 1
            else:
                segments.append((curr_chord, duration_counter))
                curr_chord = chord
                duration_counter = 1
    
    log_probs = []
    for i, (curr_chord, duration_counter) in enumerate(segments):   
        try:
            if i >= 1:
                prev_chord = segments[i - 1][0]
                trans_probs = model.transition_matrix[prev_chord][curr_chord]
                log_probs.append(np.log(np.maximum(trans_probs, epsilon)))

            dur_probs = model.duration_matrix[curr_chord][duration_counter]
            log_probs.append(np.log(np.maximum(dur_probs, epsilon)))
        except Exception:
            continue

    return np.mean(log_probs)

def hsmm_chord_ll_third_order(chord_data, model):
    epsilon = 1e-9
    segments = []
    for song in chord_data:
        curr_chord = song[0]
        duration_counter = 1
        for chord in song[1:]:
            if chord == curr_chord:
                duration_counter += 1
            else:
                segments.append((curr_chord, duration_counter))
                curr_chord = chord
                duration_counter = 1
    
    log_probs = []
    for i, (curr_chord, duration_counter) in enumerate(segments):   
        try:
            if i >= 3:
                prev_chord = segments[i - 1][0]
                prev_prev_chord = segments[i - 2][0]
                prev_prev_prev_chord = segments[i - 3][0]
                trans_probs = model.transition_matrix[(prev_prev_prev_chord, prev_prev_chord, prev_chord)][curr_chord]
                log_probs.append(np.log(np.maximum(trans_probs, epsilon)))

            dur_probs = model.duration_matrix[curr_chord][duration_counter]
            log_probs.append(np.log(np.maximum(dur_probs, epsilon)))
        except Exception:
            continue

    return np.mean(log_probs)

def hsmm_chord_ll(chord_data, model):
    epsilon = 1e-9
    segments = []
    for song in chord_data:
        curr_chord = song[0]
        duration_counter = 1
        for chord in song[1:]:
            if chord == curr_chord:
                duration_counter += 1
            else:
                segments.append((curr_chord, duration_counter))
                curr_chord = chord
                duration_counter = 1
    
    log_probs = []
    for i, (curr_chord, duration_counter) in enumerate(segments):   
        try:
            if i >= 2:
                prev_chord = segments[i - 1][0]
                prev_prev_chord = segments[i - 2][0]
                trans_probs = model.transition_matrix[(prev_prev_chord, prev_chord)][curr_chord]
                log_probs.append(np.log(np.maximum(trans_probs, epsilon)))

            dur_probs = model.duration_matrix[curr_chord][duration_counter]
            log_probs.append(np.log(np.maximum(dur_probs, epsilon)))
        except Exception:
            continue

    return np.mean(log_probs)

def emission_log_likelihood(test_data, model):
    epsilon = 1e-10
    segments = get_segments(test_data)
    log_probs = []

    for i, (chord, _, emissions) in enumerate(segments):
        try:
            for emission in emissions:
                emit_probs = model.emission_matrix[chord][emission]
                log_probs.append(np.log(np.maximum(emit_probs, epsilon)))
        except Exception:
            continue
    return np.mean(log_probs)

def hsmm_log_likelihood(test_data, model):
    epsilon = 1e-10
    segments = get_segments(test_data)
    log_probs = []

    for i, (chord, duration, emissions) in enumerate(segments):
        try:
            if i >= 2:
                prev_chord = segments[i - 1][0]
                prev_prev_chord = segments[i - 2][0]
                trans_probs = model.transition_matrix[(prev_prev_chord, prev_chord)][chord]
                log_probs.append(np.log(np.maximum(trans_probs, epsilon)))

            dur_probs = model.duration_matrix[chord][duration]
            log_probs.append(np.log(np.maximum(dur_probs, epsilon)))

            for emission in emissions:
                emit_probs = model.emission_matrix[chord][emission]
                log_probs.append(np.log(emit_probs))
        except Exception:
            continue

    return np.mean(log_probs)

def hsmm_log_likelihood_first_order(test_data, model):
    epsilon = 1e-10
    segments = get_segments(test_data)
    log_probs = []

    for i, (chord, duration, emissions) in enumerate(segments):

        try:
            if i >= 1:
                prev_chord = segments[i - 1][0]
                trans_probs = model.transition_matrix[prev_chord][chord]
                log_probs.append(np.log(np.maximum(trans_probs, epsilon)))

            dur_probs = model.duration_matrix[chord][duration]
            log_probs.append(np.log(np.maximum(dur_probs, epsilon)))

            for emission in emissions:
                emit_probs = model.emission_matrix[chord][emission]
                log_probs.append(np.log(emit_probs))
        except Exception:
            continue

    return np.mean(log_probs)

def count_num_occurences_emission_notes(sequences):
    occurences = defaultdict(int)
    for note in sequences:
        occurences[note.get_chord_tone()] += 1

    occurences_probs = defaultdict(float)
    total_events = sum(occurences.values())
    for event, count in occurences.items():
        occurences_probs[event] = count / total_events

    return occurences_probs

def count_num_occurences(sequences):
    occurences = defaultdict(int)
    for sequence in sequences:
        for event in sequence:
            occurences[event] += 1

    occurences_probs = defaultdict(float)
    total_events = sum(occurences.values())
    for event, count in occurences.items():
        occurences_probs[event] = count / total_events

    return occurences_probs

def calculate_unigram_baseline(test_sequence, events_probs_dict):
    probs = []
    for sequence in test_sequence:
        for event in sequence:
            probs.append(events_probs_dict[event])
    return np.mean(np.log(probs))

def calculate_unigram_baseline_emission(test_sequence, events_probs_dict):
    probs = []
    for note in test_sequence:
        p = events_probs_dict.get(note.get_chord_tone(), 0)
        probs.append(max(p, 1e-10))
    return np.mean(np.log(probs))

def plot_emission_heat_map():
    emissions = {
    'I':   {'5th':0.223,'6th':0.079,'root':0.204,'2nd':0.152,'3rd':0.281,'7th':0.027,'4th':0.031,'b5':0.000,'b3':0.001,'b2':0.000,'b7':0.001,'b6':0.001},
    'ii':  {'6th':0.023,'b7':0.195,'root':0.265,'2nd':0.176,'b3':0.103,'5th':0.146,'4th':0.083,'b6':0.001,'3rd':0.008,'b5':0.001,'7th':0.001,'b2':0.000},
    'iii': {'b7':0.157,'5th':0.092,'b3':0.280,'b6':0.133,'4th':0.093,'root':0.201,'b2':0.029,'3rd':0.007,'2nd':0.005,'6th':0.001,'7th':0.000,'b5':0.001},
    'IV':  {'7th':0.128,'5th':0.246,'6th':0.116,'3rd':0.240,'b5':0.039,'root':0.071,'2nd':0.142,'b3':0.006,'b2':0.001,'b7':0.005,'4th':0.003,'b6':0.003},
    'V':   {'root':0.204,'5th':0.252,'4th':0.166,'3rd':0.071,'2nd':0.109,'6th':0.158,'b7':0.034,'b5':0.001,'b3':0.000,'b6':0.001,'b2':0.001,'7th':0.003},
    'vi':  {'b7':0.130,'root':0.189,'b3':0.276,'4th':0.131,'2nd':0.063,'5th':0.185,'b6':0.018,'b2':0.000,'6th':0.004,'3rd':0.004,'b5':0.000,'7th':0.000},
    'vii': {'b7':0.095,'root':0.133,'b2':0.053,'b5':0.014,'4th':0.161,'b3':0.256,'5th':0.112,'2nd':0.071,'b6':0.077,'7th':0.011,'3rd':0.011,'6th':0.006},
}

    # Chromatic order: root up to 7th
    note_order = ['root', 'b2', '2nd', 'b3', '3rd', '4th', 'b5', '5th', 'b6', '6th', 'b7', '7th']
    chord_order = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii']

    df = pd.DataFrame(emissions).T          # chords x notes
    df = df[note_order].reindex(chord_order) # reorder both axes

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(
        df,
        ax=ax,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='#e0e0e0',
        cbar_kws={'label': 'emission probability'},
        vmin=0,
        vmax=df.values.max(),
    )

    ax.set_title('ChordHMM emission probabilities by chord function', fontsize=14, pad=12)
    ax.set_xlabel('Scale degree', fontsize=11)
    ax.set_ylabel('Chord', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('emission_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_bigram_matrix(sequence, chords):
    n = len(chords)
    chord_to_idx = {c: i for i, c in enumerate(chords)}
    matrix = np.zeros((n, n))

    for i in range(len(sequence) - 1):
        curr = sequence[i]
        next_chord = sequence[i + 1]
        if curr == 'N' or next_chord == 'N':
            continue
        matrix[chord_to_idx[curr]][chord_to_idx[next_chord]] += 1

    # normalise rows to get probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    matrix = matrix / row_sums

    return matrix

def collapse_chord_sequence(chord_sequence):
    if len(chord_sequence) <= 1:
        return chord_sequence

    collapsed_sequence = [chord_sequence[0]]
    for i in range(1, len(chord_sequence)):
        if chord_sequence[i] != collapsed_sequence[-1]:
            collapsed_sequence.append(chord_sequence[i])

    return collapsed_sequence

def plot_bass_heatmap(bass_model):
    transitions = bass_model.emission_probs
    note_order = ['root','b2','2nd','b3','3rd','4th','b5','5th','b6','6th','b7','7th']

    df = pd.DataFrame(transitions).T
    df = df.reindex(index=note_order, columns=note_order).fillna(0)

    fig, ax = plt.subplots(figsize=(20,8))

    sns.heatmap(
        df,
        ax=ax,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='#e0e0e0',
        cbar_kws={'label': 'transition probability'},
        vmin=0,
        vmax=df.values.max(),
    )

    ax.set_title('Soprano Bass Note Pairings', fontsize=16, pad=12)
    ax.set_xlabel('Bass note', fontsize=12)
    ax.set_ylabel('Soprano note', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('bass_heatmap.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    #bass_gen = BassNoteGenerator()
    #bass_gen.train_model()
    #bass_gen.save_model()

    #bass_model = BassNoteGenerator.load_model()
    #plot_bass_heatmap(bass_model)

    training_directory = "./POP909_training"
    training_data = TrainingDataProcessedInfo()
    training_data.load_training_data(training_directory)
    ##print(training_data.beat_chords)

    testing_directory = "./POP909_testing"
    testing_data = TrainingDataProcessedInfo()
    testing_data.load_training_data(testing_directory)

    chord_hmm_second_order = ChordHMM()
    chord_hmm_second_order.train_model(training_data.notes, training_data.beat_chords)
    chord_hmm_second_order.save_model('models/chord_second_order.pkl')

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
    #chord_transitions = second_order.generate_chord_transitions(100000)
    #flattened_real_chords = [chord for sequence in testing_data.beat_chords for chord in sequence]
    #flattened_real_chords = collapse_chord_sequence(flattened_real_chords)
    #chords = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii']

    #gen_matrix = get_bigram_matrix(chord_transitions, chords)
    #real_matrix = get_bigram_matrix(flattened_real_chords, chords)

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
