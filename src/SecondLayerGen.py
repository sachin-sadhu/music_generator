from SecondLayerHMM import *

class SecondLayerGen:
    def __init__(self):
        self.hmms = {}

    def train_hmms(self, ornament_groupings_dict):
        trained_hmms = {}
        for offset_chordfunction, training_data in ornament_groupings_dict.items():
            try:
                current_hmm = SecondLayerHMM()
                current_hmm.train_model(training_data)
                trained_hmms[offset_chordfunction] = current_hmm
            except ValueError:
                continue

        self.hmms = trained_hmms

    def generate_sequence(self, skeleton_note_offset, chord_function):
        if (skeleton_note_offset, chord_function) not in self.hmms:
            print(f'{skeleton_note_offset, chord_function} not found in self.hmms')
            return []
        
        hmm: SecondLayerHMM = self.hmms[(skeleton_note_offset, chord_function)]
        _, sampled_sequence = hmm.sample(1)
        return sampled_sequence
        
    

