from collections import defaultdict
from ChordFunctions import *
import numpy as np

class PatternMarkovChain:
    """Markov chain for a given pattern and chord function"""

    def __init__(self):
        """
            self.transitions = {
                                    chord_function: {
                                                        prev_note: {
                                                                        current_note: 10
                                                                   }
                                                    }
                               }

            self.initial_matrix = {
                                    chord_function: {
                                        note: 10
                                    }
                                }
        """
        self.transition_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.start_count = defaultdict(lambda: defaultdict(int))
        self.total_bars = 0
    
    def train(self, bars_by_chord_functions):
        """
        Takes in a dictionary where key is chord function and value is a list of all the bars of that chord_function
        Then want to iterate through each chord function and the list of of bars, and for each bar, we identify the starting note (for intiall probs)
        and also we identify the transition count.

        bars_by_chord_functions[chord_function] = [bars]
        """

        for chord_function, bars in bars_by_chord_functions.items():
            for bar in bars:
                if len(bar) == 0:
                    continue

                print(f'bar: {bar}')

                sorted_notes = sorted(bar, key=lambda n: n[3])

                start_note = sorted_notes[0]
                self.start_count[chord_function][start_note] += 1

                for i in range(len(sorted_notes)-1):
                    prev_note = sorted_notes[i]
                    current_note = sorted_notes[i+1]
                    self.transition_count[chord_function][prev_note][current_note] += 1

                self.total_bars += 1

    def sample_bar(self, chord_function, num_beats=4.0):
        if chord_function not in self.start_count or len(self.start_count[chord_function]) == 0:
            return [('root', 0, 'semibreve', 0.0)]

        bar_notes = []

        # Sample first note
        start_notes = list(self.start_count[chord_function].keys())
        start_counts = [self.start_count[chord_function][note] for note in start_notes]
        start_probs = np.array(start_counts, dtype=float)
        start_probs /= start_probs.sum()

        current_note = start_notes[np.random.choice(len(start_notes), p=start_probs)]
        bar_notes.append(current_note)

        # Sample subsequent notes
        duration_beats = {
            'semiquaver': 0.25, 'quaver': 0.5, 'dotted_quaver': 0.75,
            'crotchet': 1.0, 'dotted_crotchet': 1.5,
            'minim': 2.0, 'dotted_minim': 3.0, 'semibreve': 4.0,
        }

        current_position = current_note[3]

        while current_position < 4.01:
            note_duration = duration_beats.get(current_note[2], 1.0)
            current_position += note_duration

            # Get possible next notes
            candidate_notes = list(self.transition_count[chord_function][current_note].keys())

            if len(candidate_notes) == 0:
                break

            # For each candiate note, check if choosing this note will result in over beat
            valid_notes = []
            valid_counts = []

            for candidate_note in candidate_notes:
                next_duration = duration_beats.get(candidate_note[2], 1.0)
                if current_position + next_duration < 4.01:
                    valid_notes.append(candidate_note)
                    valid_counts.append(self.transition_count[chord_function][current_note][candidate_note])

            # No valid notes
            if len(valid_notes) == 0:
                break

            probs = np.array(valid_counts, dtype=float)
            probs /= probs.sum()

            current_note = valid_notes[np.random.choice(len(valid_notes), p=probs)]
            bar_notes.append(current_note)

        return bar_notes

    # Build a markov chain for each playing pattern
def build_chains(num_patterns, pattern_bars):
    """
    pattern_bars: dict
        pattern_bars[pattern][chord_function] = list of bars
    """
    
    markov_chains = {}

    for pattern in range(num_patterns):
        chain = PatternMarkovChain()
        training_data = pattern_bars[pattern]
        chain.train(training_data)
        markov_chains[pattern] = chain

    return markov_chains

def get_note_midi_pitch(chord_tone, chord_roman_numeral, key, octave_offset=0):
    """
        Converts a note into its MIDI pitch
        chord_tone: root, 2nd etc
        chord_roman_numeral: chord relative to current key
        key: current_key
    """
    major_scale_intervals_inverted = {
        "root": 0, "b2": 1, "2nd": 2, "b3": 3, "3rd": 4, "4th": 5,
        "b5": 6, "5th": 7, "b6": 8, "6th": 9, "b7": 10, "7th": 11
    }

    minor_scale_intervals_inverted = {
        "root": 0, "b2": 1, "2nd": 2, "3rd": 3, "#3": 4, "4th": 5,
        "b5": 6, "5th": 7, "6th": 8, "#6": 9, "7th": 10, "#7": 11
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
    note_midi_pitch = chord_root_note_midi_pitch + interval_mapping.get(chord_tone) + (octave_offset * 12)

    return note_midi_pitch

sample_bars = [[
        ('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 1, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5),
    ],
        [('root', 0, 'quaver', 0.0),
        ('3rd', 0, 'quaver', 0.5),
        ('5th', 0, 'quaver', 1.0),
        ('octave', 1, 'quaver', 1.5),
        ('5th', 0, 'quaver', 2.0),
        ('3rd', 0, 'quaver', 2.5),
        ('root', 0, 'quaver', 3.0),
        ('3rd', 0, 'quaver', 3.5)]
    ]

pattern_bars = {0: {'I': sample_bars, 'II': sample_bars}}

chains = PatternMarkovChain()
chains = build_chains(1, pattern_bars)

sample_bar = chains.get(0).sample_bar('I')
print(sample_bar)
