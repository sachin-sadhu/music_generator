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

def train_model(training_data, validation_data, num_hidden_states, n_iter, tol=1e-7):
    best_model = None
    best_val_score = -np.inf

    for _ in range(10):
        model = hmm.CategoricalHMM(
            n_components=num_hidden_states,
            n_iter=n_iter,
            random_state=30,
            tol=tol
        )
        model.fit(training_data)
        val_log_likelihood = model.score(validation_data)

        if val_log_likelihood > best_val_score:
            best_val_score = val_log_likelihood
            best_model = model

    return best_model

def find_best_num_hidden_states(sequences):
    hidden_state_candidates = [3,4,5,6]


    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    num_hidden_states_avg_log_likelihood = {}

    for num_hidden_states in hidden_state_candidates:
        fold_log_likelihoods = []
        for train_index, validation_index in kf.split(sequences):
            print(f'training index: {train_index}')
            print(f'validation_index: {validation_index}')

            # Split into training and validation folds
            training_data = [sequences[i] for i in train_index]
            validation_data = [sequences[i] for i in validation_index]

            model = hmm.CategoricalHMM(
                n_components=num_hidden_states,
                n_iter=100,
                random_state=30
            )
            model.fit(training_data)
            val_log_likelihood = model.score(validation_data)
            fold_log_likelihoods.append(val_log_likelihood)

        avg_log_likelihood = np.mean(fold_log_likelihoods)
        num_hidden_states_avg_log_likelihood[num_hidden_states] = avg_log_likelihood

    current_best_num_hidden_states = hidden_state_candidates[0]
    current_best_log_likelihood = -np.inf

    for num_hidden_states, log_likelihood in num_hidden_states_avg_log_likelihood.items():
        if log_likelihood > current_best_log_likelihood:
            current_best_num_hidden_states = num_hidden_states

    return current_best_num_hidden_states, current_best_log_likelihood
    
    







if __name__ == "__main__":
    rhythm_1 = extract_rhythm_sequence("./POP909/001/001.mid")
    rhythm_2 = extract_rhythm_sequence("./POP909/002/002.mid")
    rhythm_3 = extract_rhythm_sequence("./POP909/003/003.mid")
    rhythm_4 = extract_rhythm_sequence("./POP909/004/004.mid")
    rhythm_5 = extract_rhythm_sequence("./POP909/005/005.mid")

    rhythm = [rhythm_1, rhythm_2, rhythm_3, rhythm_4, rhythm_5]

    model = hmm.CategoricalHMM(n_components=8, n_iter=100)
    model.fit(rhythm)
    #print(model.monitor_)
    score = model.score(rhythm)
    #print(score)
    train_model(rhythm)


