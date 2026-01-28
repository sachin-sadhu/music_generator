import os
from datetime import datetime
from ChordFunctions import *
from collections import defaultdict
import pickle
import numpy as np
from Preprocessing import *
from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel
from PatternMarkovChain import *
from NoteEmission import *
from Main import *

class ChordTransitionModel:
    
    def __init__(self):
        self.transition_count = defaultdict(lambda: defaultdict(int))
        self.transition_probabilities = defaultdict(lambda: defaultdict(float))

    def train(self, sequential_chord_list):
        # Count occurences
        for i in range(len(sequential_chord_list) - 1):
            prev_chord = sequential_chord_list[i] 
            current_chord = sequential_chord_list[i+1]

            if prev_chord == 'N':
                continue

            self.transition_count[prev_chord][current_chord] += 1
        
        # Convert to probabilities
        for prev_chord, next_chords in self.transition_count.items():
            total_transitions = sum(next_chords.values())

            for next_chord, count in next_chords.items():
                probability = count / total_transitions
                self.transition_probabilities[prev_chord][next_chord] = probability

    def generate_chord_sequence(self, sequence_length=4.0):
        generated_chords = ['I']
        current_chord = 'I'

        while (len(generated_chords) < sequence_length):
            current_chord_transition_probabilities = self.transition_probabilities.get(current_chord, 'I')

            next_chords = list(current_chord_transition_probabilities.keys())
            probabilities = list(current_chord_transition_probabilities.values())

            # Sample next chord
            current_chord = str(np.random.choice(next_chords, p=probabilities))
            generated_chords.append(current_chord)

        return generated_chords

    def save(self, filepath='models/chord_model.pkl'):

        if not self.is_trained:
            print("Warning: Model has not yet been trained.")

        model_data = {
            'transition_probabilities': self.transition_probabilities,
            'trained': self.is_trained,
            'saved_at': datetime.now().isoformat()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath='models/chord_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.transition_probabilities = model_data['transition_probabilities']
        model.transition_occurences = model_data['transition_occurences']
        model.is_trained = model_data['trained']
        return model

    def get_probabilities(self):
        if not self.is_trained:
            print("Warning. Model has not yet been trained.")
        return self.transition_probabilities

if __name__ == "__main__":
    directory = '/cs/home/slzys1/Documents/music_generator/test_data'
    bars, bars_chords = load_songs(directory)
    filtered_bars, filtered_bars_chords = filter_empty_bars_no_chord(bars, bars_chords)

    model = ChordTransitionModel()
    model.train(bars_chords)
    generated_chords = model.generate_chord_sequence()

    num_patterns = 6
    max_duration = 10
    obs_array = np.array(filtered_bars, dtype=object)
    durations = np.random.dirichlet(np.ones(max_duration), size=num_patterns)
    tmat = initalise_tmat(num_patterns)

    note_emission = NoteEmission(num_patterns, filtered_bars_chords)
    note_emission_hsmm = HSMMModel(
        note_emission, durations, tmat
    )

    note_emission.set_context(filtered_bars_chords, filtered_bars_chords)
    result = note_emission_hsmm.fit(obs_array)
    decoded_states = note_emission_hsmm.decode(filtered_bars)
    key = 'C:maj'

    pattern_bars = build_pattern_bars_dict(filtered_bars, filtered_bars_chords, decoded_states)
    print(pattern_bars)

    chains = build_chains(num_patterns, pattern_bars)

    number_of_bars = 10
    _, states = note_emission_hsmm.sample(number_of_bars)
    chord_model = ChordTransitionModel()
    chord_model.train(filtered_bars_chords)
    
    sampled_chord_sequence = chord_model.generate_chord_sequence(number_of_bars)
    sampled_song = []
    for i in range(number_of_bars):
        bar_pattern = states[i]
        bar_chord = sampled_chord_sequence[i]

        print(f'bar pattern: {bar_pattern}. bar chord: {bar_chord}')
        
        chain = chains[bar_pattern]
        sampled_bar = chain.sample_bar(bar_chord)

        bar_midi_note = []
        for note in sampled_bar:
            chord_tone, octave_offset, note_duration, note_onset = note
            note_midi_pitch = get_note_midi_pitch(chord_tone, bar_chord, key)
            note_formatted = (note_midi_pitch, note_duration, note_onset)
            bar_midi_note.append(note_formatted)

        sampled_song.append(bar_midi_note)

    print(sampled_song)
    save_to_midi(sampled_song)

    