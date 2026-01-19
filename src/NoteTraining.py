from Process import DatasetProcessor, Processor
import os
import pickle

class NoteTraining:

    def __init__(self):
        self.melodic_state_transition_probabilities = {}
        self.melodic_emission_probabilities = {}
        self.chord_pitch_probabilities = {}

    def train_model(self, notes):
        self._build_transition_probabilities(notes)
        self._build_emission_probabilities(notes)
        self._build_chord_note_probabilities(notes)
        print("training complete")

    def save(self, filepath='models/melody_probabilities.pkl'):
        model_data = {
            'transition_probs': self.melodic_state_transition_probabilities,
            'emission_probs': self.melodic_emission_probabilities,
            'chord_pitch_probs': self.chord_pitch_probabilities
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
        model.chord_pitch_probabilities = model_data['chord_pitch_probs']
        return model

    def _get_chord_note_occurences(self, notes):
        occurence_count = {}

        pitch_class_note_mapping = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
            7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }

        for note in notes:
            current_chord = note['chord']
            pitch_class = note['pitch'] % 12
            note_name = pitch_class_note_mapping[pitch_class]

            if current_chord not in occurence_count:
                occurence_count[current_chord] = {}

            if note_name not in occurence_count[current_chord]:
                occurence_count[current_chord][note_name] = 1
            else:
                occurence_count[current_chord][note_name] += 1

        return occurence_count

    def _build_chord_note_probabilities(self, notes):
        occurence_count = self._get_chord_note_occurences(notes)
        note_probabilities = {}

        for curr_chord in occurence_count:
            note_probabilities[curr_chord] = {}
            total_occurences = sum(occurence_count[curr_chord].values())

            for curr_note, count in occurence_count[curr_chord].items():
                note_probabilities[curr_chord][curr_note] = count / total_occurences

        self.chord_pitch_probabilities = note_probabilities

    def _get_emission_transition_occurences(self, notes):
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
        occurence_count = {}

        for note in notes:
            melodic_state = note['melodic_state']
            interval = note['interval']
            is_ct = note['chord_note']

            if melodic_state not in occurence_count:
                occurence_count[melodic_state] = {True: {}, False: {}}

            if interval in occurence_count[melodic_state][is_ct]:
                occurence_count[melodic_state][is_ct][interval] += 1
            else:
                occurence_count[melodic_state][is_ct][interval] = 1

        return occurence_count

    def _build_emission_probabilities(self, notes):
        occurence_count = self._get_emission_transition_occurences(notes)
        transition_probabilities = {}

        for curr_melodic_state in occurence_count:
            transition_probabilities[curr_melodic_state] = {}

            for is_ct in occurence_count[curr_melodic_state].keys():
                transition_probabilities[curr_melodic_state][is_ct] = {}
                interval_emission_counts = occurence_count[curr_melodic_state][is_ct]
                total_occurences = sum(interval_emission_counts.values())

                for emission_interval, count in interval_emission_counts.items():
                    transition_probabilities[curr_melodic_state][is_ct][emission_interval] = count / total_occurences

        self.melodic_emission_probabilities = transition_probabilities

    ## Need functionality for transition matrix between different melodic states 
    def _get_melodic_transition_occurences(self, notes):
        occurence_count = {}

        for i in range(len(notes)-1):
            curr_melodic_state = notes[i]['melodic_state']
            next_melodic_state = notes[i+1]['melodic_state']

            # Create empty dict if current state not yet present.
            if curr_melodic_state not in occurence_count:
                occurence_count[curr_melodic_state] = {}

            if next_melodic_state in occurence_count[curr_melodic_state]:
                occurence_count[curr_melodic_state][next_melodic_state] += 1
            else:
                occurence_count[curr_melodic_state][next_melodic_state] = 1

        return occurence_count

    def _build_transition_probabilities(self, notes):
        occurence_count = self._get_melodic_transition_occurences(notes)
        transition_probabilities = {}

        for curr_melodic_state in occurence_count:
            next_melodic_state_counts = occurence_count[curr_melodic_state]
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
    chord_prob = test.chord_pitch_probabilities
    print(f"transition probs: {transition_prob}")
    print(f"emission probs: {emission_prob}")
    print(f"chord probs: {chord_prob['C:maj']}")