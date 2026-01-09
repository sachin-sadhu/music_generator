from Process import DatasetProcessor, Processor
import os
import pickle

class NoteTraining:

    def __init__(self):
        self.melodic_state_transition_occurences = {}
        self.melodic_state_transition_probabilities = {}
        self.melodic_emission_occurences = {}
        self.melodic_emission_probabilities = {}

    def train_model(self, notes):
        self._update_melodic_transition_occurences(notes)
        self._build_transition_probabilities()
        self._update_emission_transition_occurences(notes)
        self._build_emission_probabilities()
        print("training complete")

    def save(self, filepath='models/melody_probabilities.pkl'):
        model_data = {
            'transition_probs': self.melodic_state_transition_probabilities,
            'emission_probs': self.melodic_emission_probabilities
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath='models/melody_probabilities.pkl'):

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.melodic_emission_probabilities = model_data['emission_probs']
        model.melodic_state_transition_probabilities = model_data['transition_probs']
        return model

    def _update_emission_transition_occurences(self, notes):
        """                    
        str: A string describing the melodic state:
        - 'repeat': interval is 0 (same note)
        - 'ascending_step': interval is 1 or 2 semitones upward
        - 'descending_step': interval is 1 or 2 semitones downward
        - 'leap_up': interval is 3 or more semitones upward
        - 'leap_down': interval is 3 or more semitones downward
        """

        """
            iterate through list as per normal 
            when we hit a ascneding_step for example
            check if +1 or +2, increment counter for appropriate one
        """
        for note in notes:
            melodic_state = note['melodic_state']
            interval = note['interval']
            is_ct = note['chord_note']

            if melodic_state not in self.melodic_emission_occurences:
                self.melodic_emission_occurences[melodic_state] = {True: {}, False: {}}

            if interval in self.melodic_emission_occurences[melodic_state][is_ct]:
                self.melodic_emission_occurences[melodic_state][is_ct][interval] = self.melodic_emission_occurences[melodic_state][is_ct][interval] + 1
            else:
                self.melodic_emission_occurences[melodic_state][is_ct][interval] = 1

    def _build_emission_probabilities(self):
        transition_probabilities = {}

        for curr_melodic_state in self.melodic_emission_occurences:
            transition_probabilities[curr_melodic_state] = {}

            for is_ct in self.melodic_emission_occurences[curr_melodic_state].keys():
                transition_probabilities[curr_melodic_state][is_ct] = {}
                interval_emission_counts = self.melodic_emission_occurences[curr_melodic_state][is_ct]
                total_occurences = sum(interval_emission_counts.values())

                for emission_interval, count in interval_emission_counts.items():
                    transition_probabilities[curr_melodic_state][is_ct][emission_interval] = count / total_occurences

        self.melodic_emission_probabilities = transition_probabilities

    ## Need functionality for transition matrix between different melodic states 
    def _update_melodic_transition_occurences(self, notes):

        for i in range(len(notes)-1):
            curr_melodic_state = notes[i]['melodic_state']
            next_melodic_state = notes[i+1]['melodic_state']

            # Create empty dict if current state not yet present.
            if curr_melodic_state not in self.melodic_state_transition_occurences:
                self.melodic_state_transition_occurences[curr_melodic_state] = {}

            if next_melodic_state in self.melodic_state_transition_occurences[curr_melodic_state]:
                self.melodic_state_transition_occurences[curr_melodic_state][next_melodic_state] = self.melodic_state_transition_occurences[curr_melodic_state][next_melodic_state] + 1
            else:
                self.melodic_state_transition_occurences[curr_melodic_state][next_melodic_state] = 1

    def _build_transition_probabilities(self):

        transition_probabilities = {}

        for curr_melodic_state in self.melodic_state_transition_occurences:
            next_melodic_state_counts = self.melodic_state_transition_occurences[curr_melodic_state]
            total_occurences = sum(next_melodic_state_counts.values())

            transition_probabilities[curr_melodic_state] = {}
            for next_state, count in next_melodic_state_counts.items():
                transition_probabilities[curr_melodic_state][next_state] = count / total_occurences

        self.melodic_state_transition_probabilities = transition_probabilities

if __name__ == "__main__":
    processor = DatasetProcessor.load()
    notes = processor.get_notes()
    trainer = NoteTraining()
    trainer.train_model(notes)
    trainer.save()

    test = NoteTraining.load()
    transition_prob = test.melodic_state_transition_probabilities
    emission_prob = test.melodic_emission_probabilities
    print(f"transition probs: {transition_prob}")
    print(f"emission probs: {emission_prob}")