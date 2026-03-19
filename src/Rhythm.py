import pretty_midi
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import KFold
import pickle
import os

def save_model(filename, model):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f'successfully saved model to {filename}')

def load_model(filename):
    print('loading model...')
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print('model loaded...')
    return model

def preprocess_rhythm_sequence(sequence):
    start_index = 0
    while start_index < len(sequence) and sequence[start_index] == 2:
        start_index += 1

    end_index = len(sequence) - 1
    while end_index >= 0 and sequence[end_index] == 2:
        end_index -=1

    return sequence[start_index:end_index+1]

def extract_rhythm_sequence(midi_path, beats_per_bar=4, subdivisions=4, n_bars=16):
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

def train_model(training_data: list[list[int]], validation_data, num_hidden_states, n_iter, tol=1e-7):
    print('training model...')
    best_model = None
    best_log_likelihood = -np.inf

    training_data_collapsed = np.concatenate(training_data)
    training_data_collapsed = training_data_collapsed.reshape(-1, 1)
    lengths = [len(sequence) for sequence in training_data]

    validation_data_collapsed = np.concatenate(validation_data)
    validation_data_collapsed = validation_data_collapsed.reshape(-1 ,1)
    validation_data_lengths = [len(sequence) for sequence in validation_data]

    for i in range(10):
        print(f'running iteration: {i}...')
        model = hmm.CategoricalHMM(
            n_components=num_hidden_states,
            n_iter=n_iter,
            random_state=42+i,
            tol=tol
        )
        model.fit(training_data_collapsed, lengths)

        val_log_likelihood = model.score(validation_data_collapsed, validation_data_lengths)
        print(f'iteration complete. log_likelihood on validation set: {val_log_likelihood}')

        if val_log_likelihood > best_log_likelihood:
            best_log_likelihood = val_log_likelihood
            best_model = model

    return best_model, best_log_likelihood

def find_best_num_hidden_states(sequences):
    hidden_state_candidates = [14,16]

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

def convert_rhythm_to_midi(generated_sequence, output_path, beats_per_bar=4, subdivisions=4, 
                          bpm=120, pitch=60):
    """
    Convert rhythm sequence [1,2,3] back to MIDI
    
    Args:
        generated_sequence: List/array of [1,2,3] values
        output_path: Where to save the MIDI file
        beats_per_bar: Number of beats per bar
        subdivisions: Subdivisions per beat (4 = semiquavers)
        bpm: Tempo in beats per minute
        pitch: MIDI pitch for all notes (60 = middle C)
    """
    import pretty_midi
    
    # Calculate timing parameters
    seconds_per_beat = 60.0 / bpm
    seconds_per_step = seconds_per_beat / subdivisions
    
    # Create time grid
    time_grid = []
    for i in range(len(generated_sequence)):
        time_grid.append(i * seconds_per_step)
    
    # Convert sequence to note events
    notes = []
    i = 0
    while i < len(generated_sequence):
        if generated_sequence[i] == 1:  # Note onset
            start_time = time_grid[i]
            
            # Find end time by looking for end of sustain
            end_index = i + 1
            while (end_index < len(generated_sequence) and 
                   generated_sequence[end_index] == 2):  # Sustain
                end_index += 1
            
            end_time = time_grid[end_index - 1] + seconds_per_step
            
            # Create MIDI note
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch + (i % 10),
                start=start_time,
                end=end_time
            )
            notes.append(note)
            
            i = end_index  # Skip past this note
        else:
            i += 1
    
    # Create MIDI file
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=1)  # Piano
    instrument.notes.extend(notes)
    midi.instruments.append(instrument)
    
    # Save MIDI file
    midi.write(output_path)
    print(f"MIDI saved to {output_path}")
    
    return midi

def extract_sequences_from_dataset(dirpath):
    songs_sequences = []
    for song_dir in sorted(os.listdir(dirpath)):
        song_path = os.path.join(dirpath, song_dir)

        if not os.path.isdir(song_path):
            continue

        midi_file_path = os.path.join(song_path, f"{song_dir}.mid")
        song_rhythm = extract_rhythm_sequence(midi_file_path)
        songs_sequences.append(song_rhythm)
    return songs_sequences

def split_training_set_validation_set(sequences, train_ratio=0.8, val_ratio=0.1):
    train_end = int(len(sequences) * train_ratio)
    val_end = int(len(sequences) * (train_ratio + val_ratio))
    training_set = sequences[:train_end]
    validation_set = sequences[train_end:val_end]
    testing_set = sequences[val_end:]
    return training_set, validation_set, testing_set

if __name__ == "__main__":
    unprocessed_song_sequences = extract_sequences_from_dataset('./test_data')
    processed_song_sequences = [preprocess_rhythm_sequence(sequence) for sequence in unprocessed_song_sequences]
    processed_song_sequences = [seq for seq in processed_song_sequences if len(seq) > 0]
    training_set, validation_set, testing_set = split_training_set_validation_set(processed_song_sequences)
    print(f'lenght of training set: {len(training_set)}')
    print(f'lenght of validation set: {len(validation_set)}')
    print(f'lenght of testing set: {len(testing_set)}')

    #optimal_num_hidden_states, _ = find_best_num_hidden_states(training_set)
    #print(f'optimal number of hidden states: {optimal_num_hidden_states}')

    test_data_collapsed = np.concatenate(testing_set)
    test_data_collapsed = test_data_collapsed.reshape(-1 ,1)
    test_data_lengths = [len(sequence) for sequence in testing_set]   
    model, _ = train_model(training_set, validation_set, 14, n_iter=200)
    test_score = model.score(test_data_collapsed, test_data_lengths)

    print(f'test log-likelihood: {test_score / sum(test_data_lengths)}')
    save_model('rhythm_model.pkl', model)
    #print(f'original model: {model}')
    #print(model.monitor_)
    #print(loglikelihood / (len(validation_data) * len(rhythm_16)))
    #model = load_model('rhythm_model.pkl')
    #print(f'loaded model: {model}')
    #print(model)

    




    




