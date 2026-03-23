from __future__ import annotations

from typing import TYPE_CHECKING
from SongInfo import TrainingDataProcessedInfo, OrnamentGrouping, OrnamentNote
from music21 import corpus, instrument

if TYPE_CHECKING:
    from Note import TrainingNote
    from Timings import KeyTiming

from collections import defaultdict
import numpy as np
import pickle
import os

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
                target_offset = soprano.offset
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
        self.octave_offset = octave_offset
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
        key_root_note_midi_pitch = 60 + note_to_pitch_class.get(key_root_note, 0)

        # Calculate chord root
        chord_root_note_midi_pitch = key_root_note_midi_pitch + roman_numeral_to_semitones.get(self.chord_function, 0)

        # Calculate final note pitch
        note_midi_pitch = chord_root_note_midi_pitch + chromatic_intervals_inverted.get(self.note_chord_tone, 0) + (self.octave_offset * 12)

        #print(f"for chord type: {chord_roman_numeral}. for chord_tone: {chord_tone}. for key: {key}. generated_midi_pitch: {note_midi_pitch}")

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

if __name__ == "__main__":
    #directory = "/home/sachin/Documents/music_generator/POP909/POP909"
    #data = TrainingDataProcessedInfo()
    #data.load_training_data(directory)

    #chord_hmm = ChordHMM()
    #chord_hmm.train_model(data.notes, data.beat_chords)

    #chord_hmm.save_model()
    #chord_hmm = ChordHMM.load()
    #print(chord_hmm.transition_matrix)
    bass = BassNoteGenerator()
    bass.train_model()
    #print(bass.emission_probs)
    #bass.save_model()
    #print('loaded model:')
    #print(bass_loaded.emission_probs)
