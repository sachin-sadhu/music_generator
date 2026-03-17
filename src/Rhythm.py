import pretty_midi
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import KFold

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

    sequence = np.full(m, 3) # default is silence

    for onset_index, offset_index, in notes:
        if onset_index < m:
            sequence[onset_index] = 1 # note starting
            for t in range(onset_index + 1, min(offset_index, m)):
                sequence[t] = 2 # sustatin note

    return sequence

def train_model(training_data: list[list[int]], validation_data, num_hidden_states, n_iter, tol=1e-7):
    best_model = None
    best_log_likelihood = -np.inf

    training_data_collapsed = np.concatenate(training_data)
    training_data_collapsed = training_data_collapsed.reshape(-1, 1)
    lengths = [len(sequence) for sequence in training_data]

    validation_data_collapsed = np.concatenate(validation_data)
    validation_data_collapsed = validation_data_collapsed.reshape(-1 ,1)
    validation_data_lengths = [len(sequence) for sequence in validation_data]

    for _ in range(10):
        model = hmm.CategoricalHMM(
            n_components=num_hidden_states,
            n_iter=n_iter,
            random_state=None,
            tol=tol
        )
        model.fit(training_data_collapsed, lengths)

        val_log_likelihood = model.score(validation_data_collapsed, validation_data_lengths)

        if val_log_likelihood > best_log_likelihood:
            best_log_likelihood = val_log_likelihood
            best_model = model

    return best_model, best_log_likelihood

def find_best_num_hidden_states(sequences):
    hidden_state_candidates = [3,4,5,6,7,8,9]

    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    num_hidden_states_avg_log_likelihood = {}

    # Run k-folds for each hidden_state candiate
    for num_hidden_states in hidden_state_candidates:
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
            for _ in range(5):
                model = hmm.CategoricalHMM(
                    n_components=num_hidden_states,
                    n_iter=100,
                    random_state=None
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


if __name__ == "__main__":
    rhythm_1 = extract_rhythm_sequence("./POP909/POP909/001/001.mid")
    rhythm_2 = extract_rhythm_sequence("./POP909/POP909/002/002.mid")
    rhythm_3 = extract_rhythm_sequence("./POP909/POP909/003/003.mid")
    rhythm_4 = extract_rhythm_sequence("./POP909/POP909/004/004.mid")
    rhythm_5 = extract_rhythm_sequence("./POP909/POP909/005/005.mid")
    rhythm_6 = extract_rhythm_sequence("./POP909/POP909/006/006.mid")
    rhythm_7 = extract_rhythm_sequence("./POP909/POP909/007/007.mid")
    rhythm_8 = extract_rhythm_sequence("./POP909/POP909/008/008.mid")
    rhythm_9 = extract_rhythm_sequence("./POP909/POP909/009/009.mid")
    rhythm_10 = extract_rhythm_sequence("./POP909/POP909/010/010.mid")
    rhythm_11 = extract_rhythm_sequence("./POP909/POP909/011/011.mid")
    rhythm_12 = extract_rhythm_sequence("./POP909/POP909/012/012.mid")
    rhythm_13 = extract_rhythm_sequence("./POP909/POP909/013/013.mid")
    rhythm_14 = extract_rhythm_sequence("./POP909/POP909/014/014.mid")
    rhythm_15 = extract_rhythm_sequence("./POP909/POP909/015/015.mid")

    rhythm_16 = extract_rhythm_sequence("./POP909/POP909/016/016.mid")

    sequences = [rhythm_1, rhythm_2, rhythm_3, rhythm_4, rhythm_5, rhythm_6, rhythm_7, rhythm_8, rhythm_9, rhythm_10]
    validation_data = [rhythm_16]

    #num_hidden_states = find_best_num_hidden_states(sequences)
    #print(num_hidden_states)
    model, loglikelihood = train_model(sequences, validation_data, 8, n_iter=200)
    print(model.monitor_)
    print(loglikelihood / (len(validation_data) * len(rhythm_16)))
    X, Z = model.sample(256)
    print(X.flatten().tolist())

    




    




