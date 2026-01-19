import random
from NoteTraining import NoteTraining
from ChordTraining import ChordTransitionModel
from ChordFunctions import *
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
            'melodic_state': 'ascending_step',
            'current_chord': self.current_chord
        }
        self.chord_transition_matrix = chord_model.transition_probabilities
        self.melody_transmision_matrix = melody_model.melodic_state_transition_probabilities
        self.melody_emission_matrix = melody_model.melodic_emission_probabilities
        self.chord_note_emission_matrix = melody_model.chord_pitch_probabilities
        self.chord_name_to_roman_numeral_mapping = {
        'C:maj':'I',
        'D:min': 'ii',
        'E:min': 'iii',
        'F:maj': 'IV',
        'G:maj': 'V',
        'A:min': 'vi',
        'B:dim': 'vii',
        }

    def generate_sequence(self, num_notes):
        sequence = [self.current_note]
        beats_in_bar = 1

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
        Generate the next chord in the progression based on transition probabilities.

        This method determines the next chord by:
        1. Transposing the current chord to C major key for normalization
        2. Converting the transposed chord to its Roman numeral representation
        3. Using the chord transition matrix to probabilistically select the next Roman numeral
        4. Transposing the selected chord back to the original key

        Returns:
            str: The next chord in the progression, formatted as "note:chord_type" 
                 (e.g., "D:maj", "A:min") in the original key.
        """
        key_root_note = self.key.split(':')[0]
        num_semitones_to_shift = calc_semitones_to_c(key_root_note)
        (chord_root_note, chord_type) = get_chord_root_and_type(self.current_chord)
        new_root_note = transpose_note(chord_root_note, num_semitones_to_shift)
        transposed_chord = f"{new_root_note}:{chord_type}"

        ## now have chord relative to C major
        current_roman_numeral = self.chord_name_to_roman_numeral_mapping[transposed_chord]
        next_roman_numeral_probabilities = self.chord_transition_matrix[current_roman_numeral]
        next_roman_numeral = self._sample_value(next_roman_numeral_probabilities)
        next_chord_name = get_chord_name_in_original_key(next_roman_numeral, self.key)

        return next_chord_name

    def _get_next_note(self):
        """
        Generate the next note in the melody based on the Hidden Markov Model.

        This method samples a melodic state, calculates emission probabilities for possible
        intervals based on whether they are chord tones, normalizes these probabilities,
        and samples an interval to determine the pitch of the next note.

        Returns:
            dict: A dictionary representing the next note with the following keys:
                - pitch (int): The MIDI pitch value of the note
                - start_time (float): The start time of the note (current note's start time + 1)
                - duration (float): The duration of the note (default: 1)
                - velocity (int): The MIDI velocity of the note (default: 80)
                - melodic_state: The melodic state associated with this note

        Notes:
            - Uses the melody emission matrix to weight interval probabilities
            - Considers whether intervals are chord tones relative to the current chord
            - The pitch is calculated relative to the current note's pitch
        """

        note_pitch_class_mapping = {
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

        chord_note_probabilities = self.chord_note_emission_matrix[self.current_chord]
        next_note = self._sample_value(chord_note_probabilities)
        next_note_pitch_class = 60 + note_pitch_class_mapping[next_note]

        new_note = {
            'pitch': next_note_pitch_class,
            'start_time': self.current_note['start_time'] + 1,
            'duration': 1,
            'velocity': 80,
            'melodic_state': 'ascending',
            'current_chord': self.current_chord
        }

        return new_note

    def _normalise_probabilities(self, probabilities: dict):
        """
        Normalises a dictionary of probabilities so they sum to 1.0.

        This method takes a dictionary where keys represent events and values represent
        their probabilities, then divides each probability by the total sum to ensure
        all probabilities sum to exactly 1.0.

        Args:
            probabilities (dict): A dictionary mapping keys to probability values.

        Returns:
            dict: A new dictionary with the same keys as the input, but with
                normalised probability values that sum to 1.0.
        """
        total_prob = sum(probabilities.values())
        normalised_probabilities = {}

        for key, value in probabilities.items():
            normalised_value = value / total_prob
            normalised_probabilities[key] = normalised_value

        return normalised_probabilities

    def _get_state_intervals(self, melodic_state):
        """
        Get the possible melodic intervals for a given melodic state.

        This method maps melodic states to their corresponding interval values in semitones.
        The intervals represent the distance between consecutive notes in a melody.

        Args:
            melodic_state (str): The melodic state to look up. Valid values are:
                - 'ascending_step': Small upward movements (1-2 semitones)
                - 'descending_step': Small downward movements (-1 to -2 semitones)
                - 'leap_up': Larger upward movements (3, 4, 5, 7, or 12 semitones)
                - 'leap_down': Larger downward movements (-3, -4, -5, -7, or -12 semitones)
                - 'repeat': No movement (0 semitones)

        Returns:
            list[int]: A list of possible interval values in semitones for the given melodic state.

        Raises:
            ValueError: If the melodic_state is not one of the valid states defined in the mapping.
        """
        melodic_state_interval_mapping = {
            'ascending_step': [1,2],
            'descending_step': [-1,-2],
            'leap_up': [3,4,5,7,12],
            'leap_down': [-3,-4,-5,-7,-12],
            'repeat': [0]
        }

        if melodic_state not in melodic_state_interval_mapping:
            raise ValueError(f"Error. Invalid melodic state: {melodic_state}")

        return melodic_state_interval_mapping[melodic_state]

    def _sample_value(self, key_prob_dict):
        """
        Sample a value from a probability distribution.

        Args:
            key_prob_dict (dict): A dictionary where keys represent possible states/values
                and values represent their corresponding probabilities or weights.

        Returns:
            The sampled state/value from the distribution based on the given probabilities.
        """
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
        mid.save(filename)
        print(f"MIDI file saved as {filename}")

if __name__ == "__main__":
    melody_model = NoteTraining.load()
    chord_model = ChordTransitionModel.load()
    generator = GenerateSequence(melody_model, chord_model, "C:maj")
    notes = generator.generate_sequence(16)
    print(notes)
    generator.notes_to_midi(notes)