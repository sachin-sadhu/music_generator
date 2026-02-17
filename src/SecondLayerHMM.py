from collections import defaultdict
import numpy as np

class SecondLayerHMM:
    def __init__(self):
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probabilities = None


    """
        want it to be in the form of [(s1,s2), [(ornament_role, offset), (ornament_role, offset)], chord_function]
    """
    def calc_transition_matrix(self, ornament_groupings):
        transition_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            if len(grouping) < 2: 
                continue;

            for i in range(len(grouping)-1):
                curr_note = grouping[i]
                curr_note_type = curr_note[0]

                next_note = grouping[i+1]
                next_note_type = next_note[0]

                transition_count[curr_note_type][next_note_type] += 1 

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_type in transition_count.keys():
            total_count = sum(transition_count[curr_type].values())
            for next_type, count in transition_count[curr_type].items():
                transition_probs[curr_type][next_type] = count / total_count

        return transition_probs

    def calc_emission_matrix(self, ornament_groupings):
        emission_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            for note in grouping:
                note_type, note_interval = note
                emission_count[note_type][note_interval] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for emission_type in emission_count.keys():
            total_count = sum(emission_count[emission_type].values())
            for interval, count in emission_count[emission_type].items():
                emission_probs[emission_type][interval] = count / total_count

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

    def train_model(self, ornament_groupings):
        self.transition_matrix = self.calc_transition_matrix(ornament_groupings)
        self.emission_matrix = self.calc_emission_matrix(ornament_groupings)

    def sample(self, num_samples=10):
        current_state = 'chord_tone'
        current_emission = self.sample_emission(current_state)

        sampled_hidden_states = [current_state]
        sampled_emissions = [current_emission]

        while len(sampled_emissions) < num_samples:
            # Sampled next hidden state and emission
            current_state = self.sample_next_hidden_state(current_state)
            current_emission = self.sample_emission(current_state)
            
            sampled_hidden_states.append(current_state)
            sampled_emissions.append(current_emission)

        return sampled_hidden_states, sampled_emissions
