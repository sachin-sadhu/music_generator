from collections import defaultdict
import numpy as np
import pickle
import os

class HMM:
    
    def __init__(self):
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probabilities = None

    def save_model(self, filepath="models/hmm.pkl"):

        transition_probs = {}
        emission_probs = {}

        if self.transition_matrix:
            transition_probs = {k: dict(v) for k, v in self.transition_matrix.items()}

        if self.emission_matrix:
            emission_probs = {k: dict(v) for k, v in self.emission_matrix.items()}

        model_data = {
            'transition_probs': transition_probs,
            'emission_probs': emission_probs,
            'initial_probs': self.initial_probabilities
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}.")

    @classmethod
    def load(cls, filepath='models/hmm.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.transition_matrix = model_data['transition_probs']
        model.emission_matrix = model_data['emission_probs']
        model.initial_probabilities = model_data['initial_probs']

        return model

    def calc_initial_probabilities(self, chord_beat_notes_list):
        note_sequence = self.get_note_beat_sequence(chord_beat_notes_list)
        unique_states = set(note_sequence)

        prob = 1.0 / len(unique_states)
        initial_probs = {state: prob for state in unique_states}    

        return initial_probs

    # Want a matrix from (first strong beat, second strong beat): {'I': 0.05}
    def calc_hidden_state_transition_matrix(self, chord_beat_notes_list):
        """
            looks like 'I': {'II': 0.05, 'IV': 0.03}, 'II': {'I':0.01}
        """
        transition_count = defaultdict(lambda: defaultdict(int))

        note_sequence = self.get_note_beat_sequence(chord_beat_notes_list)
        for i in range(len(note_sequence)-1):
            curr_chord = note_sequence[i]
            next_chord = note_sequence[i+1]

            # Increment counter
            transition_count[curr_chord][next_chord] += 1

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_chord in transition_count.keys():
            total_count = sum(transition_count[curr_chord].values())
            for next_chord, count in transition_count[curr_chord].items():
                transition_probs[curr_chord][next_chord] = count / total_count

        return transition_probs

    def calc_emission_state_transition_matrix(self, chord_beat_notes_list):
        """
            want it to be like {'IV': {(2nd,3rd): 0.03}}
        """
        emission_count = defaultdict(lambda: defaultdict(int))

        for chord_note_struct in chord_beat_notes_list:
            chord, note_one, note_two = chord_note_struct
            note_pattern = (note_one, note_two)
            emission_count[chord][note_pattern] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for chord in emission_count.keys():
            total_count = sum(emission_count[chord].values())
            for note_beats, count in emission_count[chord].items():
                emission_probs[chord][note_beats] = count / total_count

        return emission_probs

    def get_note_beat_sequence(self, chord_beat_notes_list):
        return [(i[0]) for i in chord_beat_notes_list]

    def sample_emission(self, hidden_state):
        if hidden_state not in self.emission_matrix:
            raise ValueError("Invalid hidden state")

        emission_notes = list(self.emission_matrix[hidden_state].keys())
        emission_probs = list(self.emission_matrix[hidden_state].values())

        return emission_notes[np.random.choice(len(emission_notes), p=emission_probs)]

    def sample_next_hidden_state(self, current_hidden_state):
        if current_hidden_state not in self.transition_matrix:
            raise ValueError("Invalid hidden state")

        next_hidden_states = list(self.transition_matrix[current_hidden_state].keys())
        next_hidden_probs = list(self.transition_matrix[current_hidden_state].values())

        return next_hidden_states[np.random.choice(len(next_hidden_states), p=next_hidden_probs)]

    def train_model(self, chord_beat_notes_list):
        self.initial_probabilities = self.calc_initial_probabilities(chord_beat_notes_list)
        self.transition_matrix = self.calc_hidden_state_transition_matrix(chord_beat_notes_list)
        self.emission_matrix = self.calc_emission_state_transition_matrix(chord_beat_notes_list)

    def sample(self, num_samples=10):
        sampled_hidden_states = []
        sampled_emissions = []

        # Sample initial hidden state
        hidden_states = list(self.emission_matrix.keys())
        hidden_states_probs = list(self.initial_probabilities[state] for state in hidden_states)
        current_state = hidden_states[np.random.choice(len(hidden_states), p=hidden_states_probs)]
        current_emission = self.sample_emission(current_state)

        sampled_hidden_states.append(current_state)
        sampled_emissions.append(current_emission)

        while len(sampled_hidden_states) < num_samples:
            # Sample next hidden state and emission.
            current_state = self.sample_next_hidden_state(current_state)
            current_emission = self.sample_emission(current_state)

            sampled_hidden_states.append(current_state)
            sampled_emissions.append(current_emission)

        return sampled_hidden_states, sampled_emissions

    def get_hidden_states(self):
        return list(self.transition_matrix.keys())

    def viterbi(self, obs):
        V = [{}]
        path = {}
        states = self.get_hidden_states()

        for state in states:
            V[0][state] = self.initial_probabilities[state] * self.emission_matrix[state][obs[0]]
            path[state] = [state]

        for i in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in states:
                (prob, state) = max(
                    [(V[i-1][y0] * self.transition_matrix[y0][y] * self.emission_matrix[y][obs[i]], y0) for y0 in states]
                )
                V[i][y] = prob
                newpath[y] = path[state] + [y]

            path = newpath

        (prob, state) = max([(V[-1][y], y) for y in states])
        return (prob, path[state])

#observations = ('Walk', 'Shop', 'Clean')
#start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
#transition_probability = {
   #'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
   #'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
   #}
#emission_probability = {
   #'Rainy' : {'Walk': 0.1, 'Shop': 0.4, 'Clean': 0.5},
   #'Sunny' : {'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1},
   #}

if __name__ == "__main__":
    chord_beat_notes_list = [('IV', 'root', 'root'), ('I', '3rd', '3rd'), ('I', '2nd', '2nd')]

    hmm = HMM()
    hmm.train_model(chord_beat_notes_list)
    hmm.save_model()
    hmm_loaded = HMM.load()
    print(hmm_loaded.emission_matrix)
    print(hmm_loaded.initial_probabilities)
    print(hmm_loaded.transition_matrix)