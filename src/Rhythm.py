import pretty_midi
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import KFold
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt

class RhythmHMM:
    def __init__(self) -> None:
        self.model = None
        self.training_set = None
        self.validation_set = None
        self.testing_set = None
        self.loaded_data = False
        self.trained = False

    def save_model(self, filename):
        if self.trained:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f'successfully saved model to {filename}')
        else:
            print('Please train the model first.')

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def generate_rhythm_sequence(self, num_notes):
        if not self.trained:
            print('please train the model first')
            return

        indicies_beat_mapping = {
            0: ('note', 1), # semiqauver note
            1: ('note', 2), # quaver note
            2: ('note', 4), # crotchet note
            3: ('note', 6), # dotted crotchet note
            4: ('note', 8), # minim note
            5: ('note', 12), # dotted minim note
            6: ('note', 16), # semibreve note

            7: ('rest', 1), # semiqauver rest
            8: ('rest', 2), # quaver rest
            9: ('rest', 4), # crotchet rest
            10: ('rest', 6), # dotted crotchet rest
            11: ('rest', 8), # minim rest
            12: ('rest', 12), # dotted minim rest
            13: ('rest', 16) # semibreve rest
        }

        sampled_sequence = []
        num_notes_generated = 0
        while num_notes_generated < num_notes:
            sequence = list(self.model.sample(num_notes)[0].flatten())
            sequence = [int(x) for x in sequence]
            sequence = [indicies_beat_mapping.get(indicies, ('note', 1)) for indicies in sequence]
            print(sequence)

            for note_type, beat_duration in sequence:
                if note_type == 'note':
                    note_representation = 0
                    num_notes_generated += 1
                else:
                    note_representation = 2

                if beat_duration == 1:
                    sampled_sequence.append(note_representation)
                else:
                    if note_representation == 0:
                        sampled_sequence.append(note_representation)
                        for _ in range(1, beat_duration):
                            sampled_sequence.append(1)
                    else:
                        for _ in range(beat_duration):
                            sampled_sequence.append(2)

            self.postprocess_rhythm_sequence(sampled_sequence)

        #only return the rhythm for num_notes
        new_notes_seen = 0
        index = 0 
        while (new_notes_seen < num_notes and index < len(sampled_sequence)):
            # new note seen
            if sampled_sequence[index] == 0:
                new_notes_seen += 1
            index += 1

        while (index < len(sampled_sequence) and sampled_sequence[index] == 1):
            index += 1

        return sampled_sequence[:index]

    def load_data(self, data_dir):
        print('loading data...')
        unprocessed_song_sequences = self.extract_sequences_from_dataset(data_dir)
        filtered_sequence = [self.filter_out_beginning_end_rests(sequence) for sequence in unprocessed_song_sequences]
        filtered_sequence = [sequence for sequence in filtered_sequence if len(sequence) > 0]
        processed_song_sequences = [self.convert_to_num_beats(sequence) for sequence in filtered_sequence]
        training_set, validation_set, testing_set = self.split_training_set_validation_set(processed_song_sequences)
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set
        self.loaded_data = True
        print(f'data loaded.')

    def train_model(self, num_hidden_states=20, n_iter=1000, tol=1e-7):
        if not self.loaded_data:
            print('Please load the data before attempting to train the model.')
            return

        print('training model...')
        best_model = None
        best_log_likelihood = -np.inf

        training_data_collapsed = np.concatenate(self.training_set)
        training_data_collapsed = training_data_collapsed.reshape(-1, 1)
        training_data_lengths = [len(sequence) for sequence in self.training_set]
        validation_data_collapsed = np.concatenate(self.validation_set)
        validation_data_collapsed = validation_data_collapsed.reshape(-1 ,1)
        validation_data_lengths = [len(sequence) for sequence in self.validation_set]

        for i in range(10):
            print(f'running iteration: {i}...')
            model = hmm.CategoricalHMM(
                n_components=num_hidden_states,
                n_iter=n_iter,
                random_state=42+i,
                tol=tol
            )
            model.fit(training_data_collapsed, training_data_lengths)

            val_log_likelihood = model.score(validation_data_collapsed, validation_data_lengths)
            print(f'iteration complete. log_likelihood on validation set: {val_log_likelihood}')

            if val_log_likelihood > best_log_likelihood:
                best_log_likelihood = val_log_likelihood
                best_model = model

        self.model = best_model
        self.trained = True

    def evaluate_model(self):
        if not self.trained:
            print('please train the model before attempting to evaluate it.')
            return

        test_data_collapsed = np.concatenate(self.testing_set)
        test_data_collapsed = test_data_collapsed.reshape(-1 ,1)
        test_data_lengths = [len(sequence) for sequence in self.testing_set]  

        test_score = self.model.score(test_data_collapsed, test_data_lengths)
        return test_score / sum(test_data_lengths)

    def convert_to_num_beats(self, sequence):
        beat_indices_mapping = {
            ('note', 1): 0, # semiqauver note
            ('note', 2): 1, # quaver note
            ('note', 4): 2, # crotchet note
            ('note', 6): 3, # dotted crotchet note
            ('note', 8): 4, # minim note
            ('note', 12): 5, # dotted minim note
            ('note', 16): 6, # semibreve note

            ('rest', 1): 7, # semiqauver rest
            ('rest', 2): 8, # quaver rest
            ('rest', 4): 9, # crotchet rest
            ('rest', 6): 10, # dotted crotchet rest
            ('rest', 8): 11, # minim rest
            ('rest', 12): 12, # dotted minim rest
            ('rest', 16): 13 # semibreve rest
        }

        def closest_duration(kind, duration):
            available = [d for (k, d) in beat_indices_mapping.keys() if k == kind]
            closest_duration = min(available, key=lambda x: abs(x-duration))
            return closest_duration

        beat_sequence = []
        leftPointer = 0
        while leftPointer < len(sequence):
            if sequence[leftPointer] == 0:
                if leftPointer == len(sequence) - 1:
                    beat_sequence.append(beat_indices_mapping.get(('note', 1), 0))
                    break
                else:
                    rightPointer = leftPointer + 1
                    while rightPointer < len(sequence) and sequence[rightPointer] == 1:
                        rightPointer += 1
                    note_beat_duration = rightPointer - leftPointer
                    note_beat_duration = closest_duration('note', note_beat_duration)
                    beat_sequence.append(beat_indices_mapping.get(('note', note_beat_duration), 0))
                    leftPointer = rightPointer

            elif sequence[leftPointer] == 2:
                if leftPointer == len(sequence) - 1:
                    beat_sequence.append(beat_indices_mapping.get(('rest', 1), 0))
                    break
                else:
                    rightPointer = leftPointer + 1
                    while rightPointer < len(sequence) and sequence[rightPointer] == 2:
                        rightPointer += 1
                    rest_beat_duration = rightPointer - leftPointer
                    rest_beat_duration = closest_duration('rest', rest_beat_duration)
                    beat_sequence.append(beat_indices_mapping.get(('rest', rest_beat_duration), 0))
                    leftPointer = rightPointer

            else:
                leftPointer += 1
        return beat_sequence

    def filter_out_beginning_end_rests(self, sequence):
        start_index = 0
        while start_index < len(sequence) and sequence[start_index] == 2:
            start_index += 1

        end_index = len(sequence) - 1
        while end_index >= 0 and sequence[end_index] == 2:
            end_index -=1

        return sequence[start_index:end_index+1]

    def extract_rhythm_sequence(self, midi_path, beats_per_bar=4, subdivisions=4, n_bars=16):
        """
            here subdivisions indicates each beat being divided, using semiquavers as each timestep.
        """
        pm = pretty_midi.PrettyMIDI(midi_path)

        beat_times = pm.get_beats()

        grid = []
        for i in range(len(beat_times)-1):
            beat_duration = beat_times[i+1] - beat_times[i]
            step_duration = beat_duration / subdivisions
            for s in range(subdivisions):
                grid.append(beat_times[i] + s * step_duration)
        
        m = beats_per_bar * subdivisions * n_bars
        grid = grid[:m]

        melody_track = pm.instruments[0]
        notes = []
        for note in melody_track.notes:
            if note.start < grid[0]:
                continue

            if note.start > grid[-1]:
                break

            onset_index = np.argmin(np.abs(np.array(grid) - note.start))
            offset_index = np.argmin(np.abs(np.array(grid) - note.end))
            notes.append((onset_index, offset_index))

        sequence = np.full(m, 2) # default is silence

        for onset_index, offset_index, in notes:
            if onset_index < m:
                sequence[onset_index] = 0 # note starting
                for t in range(onset_index + 1, min(offset_index, m)):
                    sequence[t] = 1 # sustatin note

        return sequence

    def find_best_num_hidden_states(self, sequences):
        if not self.loaded_data:
            print('please load the training data first')
            return

        hidden_state_candidates = [15, 18, 20, 23]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        num_hidden_states_avg_log_likelihood = {}

        # Run k-folds for each hidden_state candiate
        for num_hidden_states in hidden_state_candidates:
            print(f'parameter testing: {num_hidden_states}')
            fold_log_likelihoods = []
            for train_index, validation_index in kf.split(sequences):
                print(f'training index: {train_index}')
                print(f'validation_index: {validation_index}')

                # Split into training and validation folds
                training_data = [sequences[i] for i in train_index]
                validation_data = [sequences[i] for i in validation_index]

                training_data_collapsed = np.concatenate(training_data)
                training_data_collapsed = training_data_collapsed.reshape(-1, 1)
                lengths = [len(sequence) for sequence in training_data]

                # For each training fold run, repeat 5 times and take best likelihood incase of bad initialisation.
                current_fold_best_log_likelihood = -np.inf
                for i in range(5):
                    model = hmm.CategoricalHMM(
                        n_components=num_hidden_states,
                        n_iter=100,
                        random_state=42+i
                    )
                    model.fit(training_data_collapsed, lengths)

                    validation_data_collapsed = np.concatenate(validation_data)
                    validation_data_collapsed = validation_data_collapsed.reshape(-1, 1)
                    validations_lengths = [len(sequence) for sequence in validation_data]
                    val_log_likelihood = model.score(validation_data_collapsed, validations_lengths)

                    if val_log_likelihood > current_fold_best_log_likelihood:
                        current_fold_best_log_likelihood = val_log_likelihood

                fold_log_likelihoods.append(current_fold_best_log_likelihood)

            # Take average log-likelihood over all folds as benchmark for this parameter candidate.
            avg_log_likelihood = np.mean(fold_log_likelihoods)
            num_hidden_states_avg_log_likelihood[num_hidden_states] = avg_log_likelihood

        current_best_num_hidden_states = hidden_state_candidates[0]
        current_best_log_likelihood = -np.inf

        for num_hidden_states, log_likelihood in num_hidden_states_avg_log_likelihood.items():
            if log_likelihood > current_best_log_likelihood:
                current_best_num_hidden_states = num_hidden_states
                current_best_log_likelihood = log_likelihood

        return current_best_num_hidden_states, current_best_log_likelihood

    def extract_sequences_from_dataset(self, dirpath):
        songs_sequences = []
        for song_dir in sorted(os.listdir(dirpath)):
            song_path = os.path.join(dirpath, song_dir)

            if not os.path.isdir(song_path):
                continue

            midi_file_path = os.path.join(song_path, f"{song_dir}.mid")
            song_rhythm = self.extract_rhythm_sequence(midi_file_path)
            songs_sequences.append(song_rhythm)
        return songs_sequences

    def split_training_set_validation_set(self, sequences, train_ratio=0.8, val_ratio=0.1):
        train_end = int(len(sequences) * train_ratio)
        val_end = int(len(sequences) * (train_ratio + val_ratio))
        training_set = sequences[:train_end]
        validation_set = sequences[train_end:val_end]
        testing_set = sequences[val_end:]
        return training_set, validation_set, testing_set

        """
            generates a rhuhthm sequence, where we watn num_notes number of notes
        keeps on generating sequences until we have at least num_notes occurences of 0
        then, we find the num_notes occurence of 0, find the end of that sustain of that note and splice the index according to that
        """

    def postprocess_rhythm_sequence(self, sequence):
        for i, note_event in enumerate(sequence):
            if i % 8 == 0:
                if note_event != 0:
                    sequence[i] = 0

        # fix occurence where it goes note onset, rest, noteonset 
        for i in range(len(sequence)-2):
            if sequence[i] == 0 and sequence[i+1] == 2 and sequence[i+2] != 2:
                sequence[i+1] = 1

def count_num_occurences(sequences):
    occurences = defaultdict(int)
    for sequence in sequences:
        for event in sequence:
            occurences[event] += 1

    occurences_probs = defaultdict(float)
    total_events = sum(occurences.values())
    for event, count in occurences.items():
        occurences_probs[event] = count / total_events

    return occurences_probs

def calculate_unigram_baseline_testing(test_sequence, event_probs_dict):
    probs = [event_probs_dict[event[0]] for event in test_sequence]
    normalised_ll = np.mean(np.log(probs))
    return normalised_ll

def calculate_unigram_baseline_training(train_sequence, event_probs_dict):
    probs = []
    for sequence in train_sequence:
        for event in sequence:
            probs.append(event_probs_dict[event])
    normalised_ll = np.mean(np.log(probs))
    return normalised_ll

def train_markov_chain(train_sequences):
    counts = defaultdict(lambda: defaultdict(int))
    
    for sequence in train_sequences:
        for i in range(len(sequence) - 1):
            counts[sequence[i]][sequence[i+1]] += 1
    
    # Convert to probabilities
    transition_probs = {}
    for state, next_states in counts.items():
        total = sum(next_states.values())
        transition_probs[state] = {k: v/total for k, v in next_states.items()}
    
    return transition_probs
    
def train_markov_chain_second_order(train_sequences):
    counts = defaultdict(lambda: defaultdict(int))
    
    for sequence in train_sequences:
        for i in range(len(sequence) - 2):
            counts[(sequence[i], sequence[i+1])][sequence[i+2]] += 1
    
    # Convert to probabilities
    transition_probs = {}
    for state, next_states in counts.items():
        total = sum(next_states.values())
        transition_probs[state] = {k: v/total for k, v in next_states.items()}
    
    return transition_probs

def markov_log_likelihood_second_order(test_sequences, transition_probs, smoothing=1e-10):
    log_probs = []
    
    for sequence in test_sequences:
        sequence = np.array(sequence).flatten()
        for i in range(len(sequence) - 2):
            curr, next_chord, next_next_chord = int(sequence[i]), int(sequence[i+1]), int(sequence[i+2])
            prob = transition_probs.get((curr, next_chord), {}).get(next_next_chord, smoothing)
            log_probs.append(np.log(prob))
    
    return np.mean(log_probs)

def markov_log_likelihood(test_sequences, transition_probs, smoothing=1e-10):
    log_probs = []
    
    for sequence in test_sequences:
        sequence = np.array(sequence).flatten()
        for i in range(len(sequence) - 1):
            curr, next_chord = int(sequence[i]), int(sequence[i+1])
            prob = transition_probs.get(curr, {}).get(next_chord, smoothing)
            log_probs.append(np.log(prob))
    
    return np.mean(log_probs)

if __name__ == "__main__":
    rhyhm_hmm = RhythmHMM()
    rhyhm_hmm.load_data('./POP909/POP909')
    rhyhm_hmm.train_model()
    rhyhm_hmm.save_model('models/20_hidden_state_class.pkl')

    #unprocessed_song_sequences = extract_sequences_from_dataset('./POP909/POP909')
    #filtered_sequence = [filter_out_beginning_end_rests(sequence) for sequence in unprocessed_song_sequences]
    #filtered_sequence = [seq for seq in filtered_sequence if len(seq) > 0]
    ######print(filtered_sequence)
    #processed_song_sequences = [convert_to_num_beats(sequence) for sequence in filtered_sequence]
    ######print(filtered_sequence)

    #training_set, validation_set, testing_set = split_training_set_validation_set(processed_song_sequences)
    ##print(f'lenght of training set: {len(training_set)}')
    ##print(f'lenght of validation set: {len(validation_set)}')
    ##print(f'lenght of testing set: {len(testing_set)}')

    #####optimal_num_hidden_states, _ = find_best_num_hidden_states(training_set)
    #####print(f'optimal number of hidden states: {optimal_num_hidden_states}')

    #test_data_collapsed = np.concatenate(testing_set)
    #test_data_collapsed = test_data_collapsed.reshape(-1 ,1)
    #test_data_lengths = [len(sequence) for sequence in testing_set]   

    #training_data_collapsed = np.concatenate(training_set)
    #training_data_collapsed = training_data_collapsed.reshape(-1 ,1)
    #training_data_lengths = [len(sequence) for sequence in training_set]   

    ##occurence_probs = count_num_occurences(processed_song_sequences)
    ##print(f'testing set: {testing_set}')
    ##print(f'unigram baseline training nll: {calculate_unigram_baseline_training(training_set, occurence_probs)}')
    ##print(f'unigram baseline testing nll: {calculate_unigram_baseline_testing(test_data_collapsed, occurence_probs)}')

    ##transition_probs = train_markov_chain(training_set)
    ##print(f'markov log likelihood training: {markov_log_likelihood(training_set, transition_probs)}')
    ##print(f'markov log likelihood testing: {markov_log_likelihood(testing_set, transition_probs)}')

    ##transition_probs_second_order = train_markov_chain_second_order(training_set)
    ##print(f'markov second order log likelihood training: {markov_log_likelihood_second_order(training_set, transition_probs_second_order)}')
    ##print(f'markov second order log likelihood testing: {markov_log_likelihood_second_order(testing_set, transition_probs_second_order)}')
    #rhythm_hmm = RhythmHMM()
    #rhythm_hmm.train_model(training_set, validation_set, 20, n_iter=1000)
    #rhythm_hmm.save_model()

    ##print(f'test log-likelihood: {test_score / sum(test_data_lengths)}')
    #print(model.monitor_)
    #print(f'original model: {model}')
    #print(loglikelihood / (len(validation_data) * len(rhythm_16)))

    #model = load_model('models/5_hidden_states_rhythm_model.pkl')
    #normalised_score = np.array(model.monitor_.history) / sum(training_data_lengths)
    ##test_score = model.score(test_data_collapsed, test_data_lengths)
    ##print(f'training score: {model.monitor_.history[-1] / sum(training_data_lengths)}')
    ##print(f'test log-likelihood: {test_score / sum(test_data_lengths)}')

    #log_likelihoods = model.monitor_.history
    #plt.plot(normalised_score)
    #plt.xlabel('Iteration')
    #plt.ylabel('Negative Log likelihood')
    #plt.title("HMM Training Negative Log Likelihood (5 hidden states)")
    #plt.savefig("log_likelihood_plt_5.png")
    #test_score = model.score(test_data_collapsed, test_data_lengths)

    #X = generate_rhythm_sequence(256, model)
    #print(X)

    

    




    




