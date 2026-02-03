from collections import defaultdict

class HMM:
    
    def __init__(self, chord_beat_notes_list):
        self.chord_beat_notes_list = chord_beat_notes_list
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probabilities = None

    def calc_initial_probabilities(self):
        note_sequence = self.get_note_beat_sequence()
        unique_states = set(note_sequence)

        prob = 1.0 / len(unique_states)
        initial_probs = {state: prob for state in unique_states}    

        return initial_probs

    # Want a matrix from (first strong beat, second strong beat): {'I': 0.05}
    def calc_hidden_state_transition_matrix(self):
        """
            looks like 'I': {'II': 0.05, 'IV': 0.03}, 'II': {'I':0.01}
        """
        transition_count = defaultdict(lambda: defaultdict(int))

        note_sequence = self.get_note_beat_sequence()
        for i in range(len(note_sequence)-1):
            curr_note_beats = note_sequence[i]
            next_note_beats = note_sequence[i+1]

            # Increment counter
            transition_count[curr_note_beats][next_note_beats] += 1

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_note_beats in transition_count.keys():
            total_count = sum(transition_count[curr_note_beats].values())
            for next_note_beats, count in transition_count[curr_note_beats].items():
                transition_probs[curr_note_beats][next_note_beats] = count / total_count

        return transition_probs

    def calc_emission_state_transition_matrix(self):
        """
            want it to be like {'IV': {(2nd,3rd): 0.03}}

        """
        emission_count = defaultdict(lambda: defaultdict(int))

        for chord_note_struct in self.chord_beat_notes_list:
            chord, note_one, note_two = chord_note_struct
            note_pattern = (note_one, note_two)
            emission_count[note_pattern][chord] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for note_beats in emission_count.keys():
            total_count = sum(emission_count[note_beats].values())
            for chord, count in emission_count[note_beats].items():
                emission_probs[note_beats][chord] = count / total_count

        return emission_probs

    def get_note_beat_sequence(self):
        return [(i[1], i[2]) for i in self.chord_beat_notes_list]

    def train_model(self):
        self.initial_probabilities = self.calc_initial_probabilities()
        self.transition_matrix = self.calc_hidden_state_transition_matrix()
        self.emission_matrix = self.calc_emission_state_transition_matrix()

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
    observations = ('I', 'IV', 'I')
    start_probability = {('root', 'root'): 0.6, ('3rd', '3rd'): 0.4}
    transition_probability = {
    ('root', 'root') : {('root', 'root'): 0.7, ('3rd', '3rd'): 0.3},
    ('3rd', '3rd') : {('root', 'root'): 0.4, ('3rd', '3rd'): 0.6},
    }
    emission_probability = {
    ('root', 'root') : {'I': 0.1, 'II': 0.4, 'IV': 0.5},
    ('3rd', '3rd') : {'I': 0.6, 'II': 0.3, 'IV': 0.1},
    }

    hmm = HMM(None)
    hmm.transition_matrix = transition_probability
    hmm.emission_matrix = emission_probability
    hmm.initial_probabilities = start_probability
    print(hmm.viterbi(observations))