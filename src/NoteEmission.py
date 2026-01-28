from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel
import numpy as np
from collections import defaultdict
from ChordFunctions import *
from PatternMarkovChain import *

class NoteEmission(AbstractEmissions):
    dtype = object

    def __init__(self, num_patterns, chord_functions):
        self.chord_functions = chord_functions
        self.num_patterns = num_patterns
        self.emission_probs = {
            pattern: {chord_func: defaultdict(lambda: 1e-10) for chord_func in self.chord_functions}
            for pattern in range(num_patterns)
        }
        self.training_bar_chord_functions = None
        self.gen_bar_chord_functions = None
        self.current_bar_index = 0

    def set_context(self, training_chord_functions, gen_chord_functions):
        self.training_bar_chord_functions = training_chord_functions
        self.gen_bar_chord_functions = gen_chord_functions
        self.current_bar_index = 0

    def likelihood(self, obs):
        """
        Compute likelihood matrix for all bars and all patterns
        """

        num_bars = len(obs)
        likelihoods = np.zeros((num_bars, self.num_patterns))

        # For each bar
        for bar_index in range(num_bars):
            bar_notes = obs[bar_index]
            chord_function = self.training_bar_chord_functions[bar_index]

            # For each playing pattern
            for pattern_index in range(self.num_patterns):
                # Compute P(bar_notes | pattern, chord_function)
                log_prob = 0.0

                for note in bar_notes:
                    print(f'chord function: {chord_function}')
                    note_prob = self.emission_probs[pattern_index][chord_function].get(note, 1e-10)
                    note_prob = np.clip(note_prob, 1e-10, 1.0)
                    log_prob += np.log(note_prob)
    
                likelihoods[bar_index, pattern_index] = log_prob

        likelihoods = np.clip(likelihoods, -700, 0)
        likelihoods = np.exp(likelihoods)

        likelihoods = np.nan_to_num(likelihoods, nan=1e-300, posinf=1.0)

        return likelihoods

    def reestimate(self, gamma, obs):
        # Given the observations need to update our self.emission_probs 
        # so need to loop through all the patterns and loop through each chord function
        # Then get all bars with the chord function equal to current iteration
        # Track how often each note type appears, weighted by the probability that the bar was that pattern type
        # Normalise to probabilities 

        for pattern_index in range(self.num_patterns):
            for chord_function in self.chord_functions:
                self.emission_probs[pattern_index][chord_function] = defaultdict(lambda: 1e-10)

        for pattern_index in range(self.num_patterns):
            for chord_function in self.chord_functions:
                note_counts = defaultdict(float)
                total_count = 0.0

                for bar_index in range(len(obs)):
                    if self.training_bar_chord_functions[bar_index] != chord_function:
                        continue

                    bar_notes = obs[bar_index]
                    weight = max(0.0, gamma[pattern_index, bar_index])

                    if weight < 0:
                        print(weight)

                    for note in bar_notes:
                        note_counts[note] += weight
                        total_count += weight

                if total_count > 0:
                    for note, count in note_counts.items():
                        self.emission_probs[pattern_index][chord_function][note] = count / total_count

        #print("Reestimated emission probabilities:")
        for pattern in range(self.num_patterns):
            for chord_func in self.chord_functions:
                learned_notes = {note: prob for note, prob in self.emission_probs[pattern][chord_func].items() if prob > 1e-9}
                if learned_notes:
                    pass
                    #print(f" Pattern {pattern}, Chord {chord_func}: {learned_notes}")

    def sample_for_state(self, state, size=None):
        if size is None:
            return self.sample_one_bar(state)
        else:
            bars = np.empty(size, dtype=object)
            for i in range(size):
                bars[i] = self.sample_one_bar(state)
            return bars

    def sample_one_bar(self, state):
        if self.gen_bar_chord_functions is None or self.current_bar_index >= len(self.gen_bar_chord_functions):
            chord_function = 0
        else:
            chord_function = self.gen_bar_chord_functions[self.current_bar_index]

        self.current_bar_index += 1

        prob_distribution = self.emission_probs[state][chord_function]

        if len(prob_distribution) == 0:
            return [('root', 0, 'crotchet', 0.0)]

        bar_notes = self._sample_by_position(prob_distribution)
        return bar_notes


    def _sample_by_position(self, dist):
        """
            Sample notes sequentially, advancing by note duration

            Process
            1. Start at beat position 0.0
            2. Sample note at current beat position
            3. Advance position by note's duration
            4. Repeat until bar is full
        """

        duration_beats = {
            'semiquaver': 0.25,
            'quaver': 0.5,
            'dotted_quaver': 0.75,
            'crotchet': 1.0,
            'dotted_crotchet': 1.5,
            'minim': 2.0,
            'dotted_minim': 3.0,
            'semibreve': 4.0,
        }

        notes_by_position = defaultdict(list)
        probs_by_position = defaultdict(list)

        for note, prob in dist.items():
            position = note[3]
            notes_by_position[position].append(note)
            probs_by_position[position].append(prob)

        bar_notes = []
        current_position = 0.0
        beats_per_bar = 4.0

        while current_position < beats_per_bar:
            tolerance = 0.01

            # Find all valid notes we can sample from at this position
            valid_notes = []
            valid_probs = []

            for position, notes in notes_by_position.items():
                if abs(position - current_position) < tolerance:
                    # Notes at this position
                    for note, prob in zip(notes, probs_by_position[position]):
                        pitch_class, octave, duration_name, pos = note
                        note_duration = duration_beats.get(duration_name, 1.0)

                        # Check if note fits in remaining bar
                        if current_position + note_duration <= beats_per_bar + tolerance:
                            valid_notes.append(note)
                            valid_probs.append(prob)

            if len(valid_notes) == 0:
                # No valid notes found at this position, skip to next position
                next_positions = [pos for pos in notes_by_position.keys() if pos > current_position]

                if len(next_positions) == 0:
                    break # No more notes, bar is done

                current_position = min(next_positions)
                continue

            # Normalise probabilites
            valid_probs = np.array(valid_probs)
            valid_probs = valid_probs / valid_probs.sum()

            # Sample single note
            sampled_index = np.random.choice(len(valid_notes), p=valid_probs)
            sampled_note = valid_notes[sampled_index]

            bar_notes.append(sampled_note)

            _, _, duration_name, position = sampled_note
            note_duration = duration_beats.get(duration_name, 1.0)
            current_position += note_duration

        return bar_notes


        # Sample notes
        notes = list(prob_distribution.keys())
        probs = np.array([prob_distribution[n] for n in notes])
        probs = probs / probs.sum()

        num_notes = np.random.poisson(6)
        sampled_indices = np.random.choice(len(notes), size=num_notes, p=probs)

        return [notes[i] for i in sampled_indices]


    def copy(self):
        return NoteEmission(self.num_patterns, self.chord_functions.copy())

observations = [
    # Bars 0-3: Arpeggio pattern on tonic (I)
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('5th', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('octave', 0, 'quaver', 3.0),
        ('5th', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    [
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 0, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
    # Bars 4-5: Block chord on subdominant (IV)
    [
        ('root', 0, 'semibreve', 0.0),
        ('3rd', 0, 'semibreve', 0.0),
        ('5th', 0, 'semibreve', 0.0),
    ],
    [
        ('root', 0, 'semibreve', 0.0),
        ('3rd', 0, 'semibreve', 0.0),
        ('5th', 0, 'semibreve', 0.0),
    ],
    
    # Bars 6-7: Block chord on dominant (V)
    [
        ('root', 0, 'minim', 0.0),
        ('3rd', 0, 'minim', 0.0),
        ('5th', 0, 'minim', 0.0),
        ('root', 0, 'minim', 2.0),
        ('3rd', 0, 'minim', 2.0),
        ('5th', 0, 'minim', 2.0),
    ],
    [
        ('root', 0, 'semibreve', 0.0),
        ('3rd', 0, 'semibreve', 0.0),
        ('5th', 0, 'semibreve', 0.0),
    ],
]

def initalise_tmat(num_patterns):
    tmat = np.zeros((num_patterns, num_patterns))

    for i in range(num_patterns):
        prob_per_transition = 1.0 / (num_patterns - 1)

        for j in range(num_patterns):
            if i != j:
                tmat[i,j] = prob_per_transition

    return tmat

def test_generate(model):
    print("\nTesting generation...")
    model.emissions.current_bar_index = 0
    obs, states = model.sample(8)
    print("Generated observations:")
    for i, obs in enumerate(obs):
        print(f"Bar: {i}: {obs}")
    print(f"Generated states: {states}")
    return obs, states

def get_song_pattern_assignments(model, observations, bar_chord_functions):
    """
        Runs Viterbi algorithm on this training song to get bar pattern assignements.

        Returns:
            pattern_bars: dict
                pattern_bars[pattern][chord_function] = list of bars from the song assigned to this pattern/function
    """

    pattern_bars = defaultdict(lambda: defaultdict(list))
    model.emissions.set_context(bar_chord_functions, None)

    state_sequence = model.decode(observations)

    for bar_index, pattern in enumerate(state_sequence):
        chord_function = bar_chord_functions[bar_index]
        bar_notes = observations[bar_index]
        pattern_bars[pattern][chord_function].append(bar_notes)

    return pattern_bars

def get_note_midi_pitch(chord_tone, chord_roman_numeral, key, octave_offset=0):
    major_scale_intervals_inverted = {
        "root": 0, "b2": 1, "2nd": 2, "b3": 3, "3rd": 4, "4th": 5,
        "b5": 6, "5th": 7, "b6": 8, "6th": 9, "b7": 10, "7th": 11, "octave": 12
    }

    minor_scale_intervals_inverted = {
        "root": 0, "b2": 1, "2nd": 2, "3rd": 3, "#3": 4, "4th": 5,
        "b5": 6, "5th": 7, "6th": 8, "#6": 9, "7th": 10, "#7": 11, "octave": 12
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
    key_root_note, key_type = get_chord_root_and_type(key)
    key_root_note_midi_pitch = 60 + note_to_pitch_class.get(key_root_note, 0)

    # Calculate chord root
    chord_root_note_midi_pitch = key_root_note_midi_pitch + roman_numeral_to_semitones.get(chord_roman_numeral, 0)
    interval_mapping = major_scale_intervals_inverted if key_type.isupper() else minor_scale_intervals_inverted

    # Calculate final note pitch
    note_midi_pitch = chord_root_note_midi_pitch + interval_mapping.get(chord_tone, 0) + (octave_offset * 12)

    return note_midi_pitch

song_chord_functions = ['I', 'I', 'I', 'I', 'iii', 'iii', 'V', 'V', 'V', 'V', 'V', 'V', 'V']
gen_functions = ['I', 'I', 'I', 'I', 'iii', 'iii', 'V', 'V']
chord_functions = ['I','iii','V']
obs_array = np.array(observations, dtype=object)

num_patterns = 6
max_duration = 5
durations = np.random.dirichlet(np.ones(max_duration), size=num_patterns)
tmat = initalise_tmat(num_patterns)

print(f'Initial durations: {durations}')
print(f'Initial tmat: {tmat}')

note_emission = NoteEmission(num_patterns, chord_functions)

note_emission_hsmm = HSMMModel(
    note_emission,
    durations, tmat
)

def build_pattern_bars_dict(bars, bars_chord_function, bars_pattern):
    pattern_bars = defaultdict(lambda: defaultdict(list))

    for i in range(len(bars)):
        print(f'current bar: {bars[i]}')
        bar_pattern = bars_pattern[i]
        bar_chord_function = bars_chord_function[i]
        pattern_bars[bar_pattern][bar_chord_function].append(bars[i])

    return pattern_bars

def generate_song(bar_patterns, markov_chains, chord_functions, song_key):
    generated_bars = []

    for i in range(len(bar_patterns)):
        bar_pattern = bar_patterns[i]
        chord_function = chord_functions[i]
        chain = markov_chains.get(bar_pattern)

        bar_notes = chain.sample_bar(chord_function)

        # Create bar notes with MIDI pitches
        midi_notes = []
        for note in bar_notes:
            chord_tone, octave_offset, duration, beat_onset = note
            note_midi_pitch = get_note_midi_pitch(chord_tone, chord_function, song_key, octave_offset)
            midi_note = (note_midi_pitch, octave_offset, duration, beat_onset)
            midi_notes.append(midi_note)

        generated_bars.append(midi_notes)

    return generated_bars

if __name__ == "__main__":
    note_emission.set_context(song_chord_functions, gen_functions)
    result = note_emission_hsmm.fit(obs_array)
    obs, states = test_generate(note_emission_hsmm)
    decoded_states = note_emission_hsmm.decode(obs)
    print(f'decoded states: {decoded_states}')
    #midi_pitch = get_note_midi_pitch("root", 'I', 'F:major', 0)
    #print(midi_pitch)
    print(f'obs: {obs} chord_functions: {song_chord_functions} decoded states: {decoded_states}')
    print(f'pattern bars: {pattern_bars}')
    chains = build_chains(num_patterns, pattern_bars)
    print(f'chains: {chains}')

    song = generate_song([0,1], chains, ['I', 'ii'], 'C:maj')
    print(song)