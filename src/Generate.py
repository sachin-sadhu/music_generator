import random
from NoteTraining import NoteTraining
from ChordTraining import ChordTransitionModel
import mido
from mido import MidiFile, MidiTrack, Message

class GenerateSequence:

    def __init__(self, melody_model, chord_model, key):
        self.key = key
        self.current_chord = key
        self.current_note = {
            'pitch': 60,
            'start_time': 0,
            'duration': 1,
            'velocity': 80,
            'melodic_state': 'ascending_step'
        }
        self.chord_transition_matrix = chord_model.transition_probabilities
        self.melody_transmision_matrix = melody_model.melodic_state_transition_probabilities
        self.melody_emission_matrix = melody_model.melodic_emission_probabilities
        self.chord_to_scale_degree = {
        'C:maj': 1,
        'D:min': 2,
        'E:min': 3,
        'F:maj': 4,
        'G:maj': 5,
        'A:min': 6,
        'B:dim': 7,
        }

    # Need a way to track how many notes we've created so far 
    # Need a way to know when to sample new chord
    # Need a way to know when to sample a new note
    # Need a way to decide 
    
    def generate_sequence(self, num_notes):

        sequence = []
        beats_in_bar = 0

        while (len(sequence) < num_notes):
            
            if beats_in_bar >= 4:
                self.current_chord = self._get_next_chord()
                beats_in_bar = 0

            next_note = self._get_next_note()
            sequence.append(next_note)
            self.current_note = next_note
            beats_in_bar += 1

        return sequence

    def _get_next_chord(self):
        """
            want to convert current chord in current key to chord in C major
            Then convert chord in C major to number form (1-7).
            Then sample from transition matrix
            Then convert chord number back to chord in appropriate key
        """
        num_semitones_to_shift = self._calc_semitones_to_c(self.key)
        (chord_root_note, chord_type) = self._get_chord_root_and_type(self.current_chord)
        new_root_note = self._transpose_to_c(chord_root_note, num_semitones_to_shift)
        transposed_chord = f"{new_root_note}:{chord_type}"

        ## now have original chord relative to C major
        current_roman_numeral = self.chord_to_scale_degree[transposed_chord]
        next_roman_numeral_probabilities = self.chord_transition_matrix[current_roman_numeral]
        next_roman_numeral = self._sample_value(next_roman_numeral_probabilities)
        next_chord_name = self._get_chord_name_in_original_key(next_roman_numeral)

        return next_chord_name

    def _calc_semitones_to_c(self, original_key: str):

        key_root_note = original_key.split(':')[0]

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

        # Figure out how many semitones to shift up to transpose
        # Shift is always a number between 0-11
        shift = (0 - note_to_pitch_class[key_root_note]) % 12

        return shift

    def _transpose_to_c(self, chord_root_note: str, shift: int):
        # Convert chord_root_note to an integer
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

        pitch_class_to_note = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
            7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }

        current_semitone = note_to_pitch_class[chord_root_note]

        new_semitone = (current_semitone + shift) % 12

        return pitch_class_to_note[new_semitone]

    def _get_chord_root_and_type(self, chord: str):
        parts = chord.split(':')

        if len(parts) >= 2:
            chord_root_note = parts[0]
            chord_type = parts[1]
            return (chord_root_note, chord_type)
        else: 
            return -1

    def _transpose_song_chord(self, chord):
        original_key = self.key.split(':')[0]
        num_semitones_to_shift = self.calc_semitones_to_c(original_key)

        parts = chord.split(':')

        if len(parts) >= 2:
            chord_root_note, chord_type = self._get_chord_root_and_type(self.current_chord)
        else:
            return -1

        transposed_chord_root_note = self.transpose_to_c(chord_root_note, num_semitones_to_shift)
        transposed_chord = f'{transposed_chord_root_note}:{chord_type}'

        return transposed_chord

    def _convert_chord_name_to_scale_degree(self, chord: str):
        if chord in self.chord_to_scale_degree:
            return self.chord_to_scale_degree[chord]
        else :
            return -1

    def _get_chord_name_in_original_key(self, roman_numeral):
        scale_degree_to_semitones = {
            1: 0,   # Tonic (0 semitones above root)
            2: 2,  # 2 semitones above root
            3: 4, # 4 semitones above root
            4: 5,  # 5 semitones above root
            5: 7,   # 7 semitones above root ← We need this!
            6: 9,  # 9 semitones above root
            7: 11 # 11 semitones above root
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

        (key_root_note, chord_type) = self._get_chord_root_and_type(self.current_chord)
        key_root_note_number = note_to_pitch_class[key_root_note]
        scale_num_semitones = scale_degree_to_semitones[roman_numeral]
        chord_pitch_class = (key_root_note_number + scale_num_semitones) % 12

        pitch_class_to_note = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
            7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }

        chord_root_name = pitch_class_to_note[chord_pitch_class]
        chord_name = f"{chord_root_name}:{chord_type}"

        return chord_name


    def _get_next_note(self):

        # check if need to sample new chord
        # if no, then sample new note using current chord 
        # sample note by first sampling new melodic state,
        # once new melodic state is sampled, get the intervals represented by that melodic state
        # for each of those intervals, calculate whether the new note would be a chord note or not
        # normalise the probabilities and sample from that 

        next_melodic_state = self._sample_melodic_state()
        state_intervals = self._get_state_intervals(next_melodic_state)
        intervals_probabilities = {}

        for interval in state_intervals:
            is_ct = self._is_chord_tone(interval, self.current_chord)
            emission_probability = self.melody_emission_matrix[next_melodic_state][is_ct][interval]
            intervals_probabilities[interval] = emission_probability

        norm_probabilities = self._normalise_probabilities(intervals_probabilities)
        sampled_pitch_interval = self._sample_value(norm_probabilities)

        new_note_pitch = self.current_note['pitch'] + sampled_pitch_interval

        new_note = {
            'pitch': new_note_pitch,
            'start_time': self.current_note['start_time'] + 1,
            'duration': 1,
            'velocity': 80,
            'melodic_state': next_melodic_state
        }

        return new_note


    def _normalise_probabilities(self, probabilities: dict):
        total_prob = sum(probabilities.values())
        normalised_probabilities = {}

        for key, value in probabilities.items():
            normalised_value = value / total_prob
            normalised_probabilities[key] = normalised_value

        return normalised_probabilities

    def _is_chord_tone(self, interval, chord_name):

        new_midi_pitch = int(self.current_note['pitch']) + interval

        CHORD_TEMPLATES = {
            'C:maj': [0, 4, 7],
            'C:min': [0, 3, 7],
            'D:min': [2, 5, 9],
            'D:maj': [2, 6, 9],
            'E:min': [4, 7, 11],
            'E:maj': [4, 8, 11],
            'F:maj': [5, 9, 0],
            'F:min': [5, 8, 0],
            'G:maj': [7, 11, 2],
            'G:min': [7, 10, 2],
            'A:min': [9, 0, 4],
            'A:maj': [9, 1, 4],
            'B:min': [11, 2, 6],
            'B:maj': [11, 3, 6],
            'B:dim': [11, 2, 5],
            'G:7': [7, 11, 2, 5]
        }

        if chord_name not in CHORD_TEMPLATES:
            return False

        pitch_class = new_midi_pitch % 12

        return pitch_class in CHORD_TEMPLATES[chord_name];

    def _get_state_intervals(self, melodic_state):
        state_inteveral_mapping = {
            'ascending_step': [1,2],
            'descending_step': [-1,-2],
            'leap_up': [3,4,5,7,12],
            'leap_down': [-3,-4,-5,-7,-12],
            'repeat': [0]
        }

        if melodic_state not in state_inteveral_mapping:
            return None

        return state_inteveral_mapping[melodic_state]


    def _sample_melodic_state(self):
        """
        Sample the next melodic state based on the current state's transition probabilities.

        This method uses the melody transmission matrix to determine the probability distribution
        of possible next states given the current melodic state, then randomly samples from this
        distribution.

        Returns:
            str : The sampled melodic state based on the transition probabilities from 
                        the current melodic state.
        """

        curr_melodic_state = self.current_note['melodic_state']
        current_state_transition_probs = self.melody_transmision_matrix[curr_melodic_state]
        sample = self._sample_value(current_state_transition_probs)

        return sample

    def _sample_value(self, key_prob_dict):
        states = list(key_prob_dict.keys())
        probabilities = list(key_prob_dict.values())
        sample = random.choices(states, weights=probabilities, k=1)[0]

        return sample

    def notes_to_midi(self, notes, filename='output.mid', tempo=120):
        """
        Convert a list of note dictionaries to a MIDI file.
        
        Args:
            notes: List of dicts with 'pitch', 'start_time', 'duration', 'velocity'
            filename: Output MIDI filename
            tempo: Tempo in BPM (beats per minute)
        """
        # Create MIDI file and track
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (500000 microseconds per beat = 120 BPM)
        # Formula: microseconds_per_beat = 60,000,000 / tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Convert beats to ticks (480 ticks per beat is standard)
        ticks_per_beat = 480
        
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])
        
        current_time = 0
        
        for note in sorted_notes:
            # Calculate delta time in ticks
            note_start_ticks = int(note['start_time'] * ticks_per_beat)
            delta_time = note_start_ticks - current_time
            
            # Note on
            track.append(Message('note_on', 
                            note=note['pitch'], 
                            velocity=note['velocity'], 
                            time=delta_time))
            
            # Note off (after duration)
            duration_ticks = int(note['duration'] * ticks_per_beat)
            track.append(Message('note_off', 
                            note=note['pitch'], 
                            velocity=0, 
                            time=duration_ticks))
            
            current_time = note_start_ticks + duration_ticks
        
        # Save the file
        