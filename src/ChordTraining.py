import os
from datetime import datetime
import pickle

class ChordTransitionModel:
    
    def __init__(self):
        self.is_trained = False
        self.transition_occurences = {
            1: {},
            2: {},
            3: {},
            4: {},
            5: {},
            6: {},
            7: {}
        }
        self.transition_probabilities = {}
        self.chord_mapping = {
        'C:maj': 1,
        'D:min': 2,
        'E:min': 3,
        'F:maj': 4,
        'G:maj': 5,
        'A:min': 6,
        'B:dim': 7,
        }
        self.trained_songs = []

    def calc_semitones_to_c(self, original_key: str):
        """
        Calculate the number of semitones from a given note to C.
        This function returns the chromatic distance (in semitones) from the input note
        to the reference note C.itones in an octave.
        Args:
            original_key (str): The key name. Accepts both sharp (#) and flat (b) notations.
                        Valid values include: 'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 
                        'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B'.
        Returns:
            int: The number of semitones from the input note to C, ranging from 0 to 11.
                Returns 0 for C, and increases chromatically up to 11 for B.
        Raises:
            KeyError: If the provided note is not found in the semitone dictionary.
        """
        
        num_semitones = {
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

        shift = (0 - num_semitones[original_key]) % 12

        return shift

    def transpose_to_c(self, chord_root_note: str, shift: int):
        """
        Transpose a musical note by a given number of semitones.

        This function takes a chord root note and transposes it by the specified
        number of semitones (shift), wrapping around the chromatic scale using
        modulo 12 arithmetic.

        Args:
            chord_root_note (str): The starting note to transpose. Can be any valid
                musical note name including sharps (#) and flats (b). Examples: 'C',
                'C#', 'Db', 'D', etc.
            shift (int): The number of semitones to transpose. Positive values
                transpose up, negative values transpose down.

        Returns:
            str: The transposed note name using sharp notation (e.g., 'C#' not 'Db').

        Raises:
            KeyError: If chord_root_note is not a valid note name.
        """

        # Convert chord_root_note to an integer
        note_to_num = {
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

        num_to_note = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
            7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }

        current_semitone = note_to_num[chord_root_note]

        new_semitone = (current_semitone + shift) % 12

        return num_to_note[new_semitone]

    def transpose_song_chords(self, chords: list[str], original_key: str):
        """
        Transpose all chords in a song to the key of C.

        This function takes a list of chords and the original key of the song,
        then transposes each chord to the equivalent chord in the key of C by
        calculating the required semitone shift.

        Args:
            chords (list[str]): A list of chord strings in the format "root:type"
                            (e.g., ["D:maj", "G:min", "A:7"]).
            original_key (str): The original key of the song (e.g., "D", "G", "A").

        Returns:
            list[str]: A list of transposed chords in the same format as the input,
                    with all chords transposed to the key of C.

        """

        transposed_chords = []

        original_key = original_key.split(':')[0]

        num_semitones_to_shift = self.calc_semitones_to_c(original_key)

        for chord in chords:

            parts = chord.split(':')

            if len(parts) >= 2:
                chord_root_note = parts[0]
                chord_type = parts[1]
            else:
                continue

            new_root_note = self.transpose_to_c(chord_root_note, num_semitones_to_shift)
            transposed_chord = f'{new_root_note}:{chord_type}'

            transposed_chords.append(transposed_chord)

        return transposed_chords

    def convert_names_to_numbers(self, chords: list[str]):

        chords_number = []

        for chord in chords:
            if chord in self.chord_mapping:
                chords_number.append(self.chord_mapping[chord])
            else :
                chords_number.append(-1)

        return chords_number

    def update_transition_occurences(self, chords: list[int]):
        """
        Build a dictionary tracking the occurrence count of chord transitions.

        Args:
            chords (list[int]): A list of chord identifiers (integers 1-7) representing
                            a sequence of musical chords.

        Returns:
            dict: A nested dictionary where:
                - Outer keys are chord identifiers (1-7)
                - Inner keys are the next chord identifiers that follow
                - Values are the count of how many times that transition occurs
        """

        for i in range(len(chords)-1):
            curr_chord = chords[i]
            next_chord = chords[i+1]

            if curr_chord == -1 or next_chord == -1:
                continue

            if next_chord in self.transition_occurences[curr_chord]:
                self.transition_occurences[curr_chord][next_chord] = self.transition_occurences[curr_chord][next_chord] + 1
            else:
                self.transition_occurences[curr_chord][next_chord] = 1

        return self.transition_occurences

    def build_transition_probabilities(self):
        """
        Build transition probability matrix from chord transition occurrence counts.

        This function converts raw chord transition counts into probabilities by normalizing
        each current chord's transition counts by the total number of transitions from that chord.

        Args:
            transition_occurences (dict[int, dict[int, int]]): A nested dictionary where the outer
                dictionary maps current chord IDs to inner dictionaries. Each inner dictionary
                maps next chord IDs to their occurrence counts.
                Example: {1: {2: 5, 3: 3}, 2: {1: 4, 3: 2}}

        Returns:
            dict[int, dict[int, float]]: A nested dictionary with the same structure as the input,
                but with occurrence counts replaced by probabilities (values sum to 1.0 for each
                current chord).
                Example: {1: {2: 0.625, 3: 0.375}, 2: {1: 0.667, 3: 0.333}}
        """

        transition_probabilities = {}

        for curr_chord in self.transition_occurences:
            next_chord_counts = self.transition_occurences[curr_chord]
            total_occurences = sum(next_chord_counts.values())

            transition_probabilities[curr_chord] = {}
            for next_chord, count in next_chord_counts.items():
                transition_probabilities[curr_chord][next_chord] = count / total_occurences

        self.transition_probabilities = transition_probabilities

    def extra_song_info_from_file(self, chord_path: str, key_path: str):
        chords = []
        info = {}

        with open(chord_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                curr_chord = parts[2]
                chords.append(curr_chord)
            
            info['chords'] = chords

        with open(key_path, 'r') as f:
            line = f.readline().strip()
            info['key'] = line.split('\t')[2]

        return info

    def train_model(self, dataset_path: str):

        songs_trained = 0

        for song_folder in os.listdir(dataset_path):
            song_path = os.path.join(dataset_path, song_folder)

            if not os.path.isdir(song_path):
                continue

            chord_file_path = os.path.join(song_path, 'chord_midi.txt')
            key_file_path = os.path.join(song_path, 'key_audio.txt')

            if not os.path.exists(chord_file_path) or not os.path.exists(key_file_path):
                continue

            try:
                info = self.extra_song_info_from_file(chord_file_path, key_file_path)
                chords = info['chords']
                key = info['key']

                if 'maj' not in key:
                    continue

                transposed_chords = self.transpose_song_chords(chords, key)
                chords_numbered = self.convert_names_to_numbers(transposed_chords)

                self.update_transition_occurences(chords_numbered)
                self.trained_songs.append(song_folder)
            except Exception as e:
                print(f"Error processing {song_folder}")
                continue
        
        self.build_transition_probabilities()
        self.is_trained = True
        print("Finished training.")
        print(f"Sucessfully trained on {len(self.trained_songs)} songs.")

        for song in self.trained_songs:
            print(song)

        return self.transition_probabilities

    def save(self, filepath='models/chord_model.pkl'):

        if not self.is_trained:
            print("Warning: Model has not yet been trained.")

        model_data = {
            'transition_probabilities': self.transition_probabilities,
            'transition_occurences': self.transition_occurences,
            'chord_mapping': self.chord_mapping,
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
        model.chord_mapping = model_data['chord_mapping']
        model.is_trained = model_data['trained']
        return model

    def get_probabilities(self):
        if not self.is_trained:
            print("Warning. Model has not yet been trained.")
        return self.transition_probabilities

#data_path = '/Users/sachin/Documents/music_generator/test_training'
#model = ChordTransitionModel()
#model.train_model(data_path)
#model.save()
if __name__ == "__main__":
    model = ChordTransitionModel.load()
    probs = model.get_probabilities()
    print(probs)

