from collections import defaultdict
from OrnamentGroupings import *
import numpy as np

class SecondLayerHMM:
    def __init__(self):
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probabilities = None

    def calc_initial_probabilities(self):
        initial_probabilities = {}
        print(f'hidden states: {self.transition_matrix.keys()}')
        hidden_states = list(self.transition_matrix.keys())

        if len(hidden_states) == 0:
            raise ValueError("HMM has no hidden states")

        uniform_prob = 1.0 / len(hidden_states)
        
        for hidden_state in hidden_states:
            initial_probabilities[hidden_state] = uniform_prob

        return initial_probabilities

    def calc_transition_matrix(self, ornament_groupings: list[OrnamentGrouping]):
        transition_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            # No ornament notes
            if len(grouping.ornament_notes) == 0:
                continue

            for i in range(len(grouping.ornament_notes)-1):
                curr_note_role = grouping.ornament_notes[i]
                next_note_role = grouping.ornament_notes[i+1]
                transition_count[curr_note_role][next_note_role] += 1

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_type in transition_count.keys():
            total_count = sum(transition_count[curr_type].values())
            for next_type, count in transition_count[curr_type].items():
                transition_probs[curr_type][next_type] = count / total_count

        return transition_probs

    def calc_emission_matrix(self, ornament_groupings: list[OrnamentGrouping]):
        emission_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            for note in grouping.ornament_notes:
                emission_count[note.role][(note.offset, note.duration)] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for role in emission_count.keys():
            total_count = sum(emission_count[role].values())
            for (offset, duration), count in emission_count[role].items():
                emission_probs[role][(offset, duration)] = count / total_count

        return emission_probs

    def sample_next_hidden_state(self, current_hidden_state):
        if current_hidden_state not in self.transition_matrix:
            raise ValueError("Invalid hidden state")

        next_hidden_states = list(self.transition_matrix[current_hidden_state].keys())
        next_hidden_probs = list(self.transition_matrix[current_hidden_state].values())

        return next_hidden_states[np.random.choice(len(next_hidden_states), p=next_hidden_probs)]

    def sample_emission(self, hidden_state):
        if hidden_state not in self.emission_matrix:
            raise ValueError("Invalid hidden state")

        emission_notes = list(self.emission_matrix[hidden_state].keys())
        emission_probs = list(self.emission_matrix[hidden_state].values())

        return emission_notes[np.random.choice(len(emission_notes), p=emission_probs)]

    def sample_initial_state(self):
        hidden_states = list(self.initial_probabilities.keys())
        probs = list(self.initial_probabilities.values())
        return hidden_states[np.random.choice(len(hidden_states), p=probs)]

    def train_model(self, ornament_groupings):
        try:
            self.transition_matrix = self.calc_transition_matrix(ornament_groupings)
            self.emission_matrix = self.calc_emission_matrix(ornament_groupings)
            self.initial_probabilities = self.calc_initial_probabilities()
        except ValueError as e:
            raise e

    def sample(self, num_samples=10):
        current_state = self.sample_initial_state()
        current_emission = self.sample_emission(current_state)

        sampled_hidden_states = [current_state]
        sampled_emissions = [current_emission]

        while len(sampled_emissions) < num_samples:
            current_state = self.sample_next_hidden_state(current_state)
            current_emission = self.sample_emission(current_state)
            
            sampled_hidden_states.append(current_state)
            sampled_emissions.append(current_emission)

        return sampled_hidden_states, sampled_emissions