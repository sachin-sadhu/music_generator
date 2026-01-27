import os
from datetime import datetime
from ChordFunctions import *
from collections import defaultdict
import pickle

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
    #data_path = '/Users/sachin/Documents/music_generator/test_training'
    #model = ChordTransitionModel()
    #model.train_model(data_path)
    #model.save()

    model = ChordTransitionModel.load()
    probs = model.get_probabilities()
    print(probs)

