import os
from datetime import datetime
from ChordFunctions import *
import pickle

class ChordTransitionModel:
    
    def __init__(self):
        self.is_trained = False
        self.transition_occurences = {
            'I': {},
            'ii': {},
            'iii': {},
            'IV': {},
            'V': {},
            'vi': {},
            'vii': {}
        }
        self.transition_probabilities = {}
        self.trained_songs = []

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
        num_semitones_to_shift = calc_semitones_to_c(original_key)

        for chord in chords:
            try:
                (chord_root_note, chord_type) = get_chord_root_and_type(chord)
            except Exception as e:
                print(f"Error caught. {e}")
                continue

            transposed_root_note = transpose_note(chord_root_note, num_semitones_to_shift)
            transposed_chord = f'{transposed_root_note}:{chord_type}'
            transposed_chords.append(transposed_chord)

        return transposed_chords

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
                self.transition_occurences[curr_chord][next_chord] += 1
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

                # Chords transposed to C major
                transposed_chords = self.transpose_song_chords(chords, key)

                roman_numerals = []
                for chord in transposed_chords:
                    try:
                        roman_numerals.append(convert_chord_name_to_roman_numeral(chord))
                    except Exception as e:
                        print(f"Error caught: {e}")
                        roman_numerals.append(-1)

                self.update_transition_occurences(roman_numerals)
                self.trained_songs.append(song_folder)
            except Exception as e:
                print(f"Error processing {song_folder}. Error: {e}")
                raise e
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
    #data_path = '/Users/sachin/Documents/music_generator/test_training'
    #model = ChordTransitionModel()
    #model.train_model(data_path)
    #model.save()

    model = ChordTransitionModel.load()
    probs = model.get_probabilities()
    print(probs)

