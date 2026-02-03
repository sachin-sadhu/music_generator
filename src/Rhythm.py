import numpy as np
from Preprocessing import *
from collections import defaultdict

def extract_rhythm(notes, ticks_per_beat):
    DURATION_TO_UNITS = {
        'semiquaver': 1,       # 0.25 beats = 1 semiquaver
        'quaver': 2,           # 0.50 beats = 2 semiquavers
        'dotted_quaver': 3,    # 0.75 beats = 3 semiquavers
        'crotchet': 4,         # 1.0 beats = 4 semiquavers
        'dotted_crotchet': 6,  # 1.5 beats = 6 semiquavers
        'minim': 8,            # 2.0 beats = 8 semiquavers
        'dotted_minim': 12,    # 3.0 beats = 12 semiquavers
        'semibreve': 16        # 4.0 beats = 16 semiquavers
    }

    rhythm_sequence = []

    for note in notes:
        beat_duration = note['beat_duration']
        beat_unit_duration = DURATION_TO_UNITS.get(beat_duration, 1)

        symbol = f'N{beat_unit_duration}'

def build_markov_chain(rhythm_sequence):
    transition_counts = defaultdict(lambda: defaultdict(int))

    """
    transition_counts = {'N1': {'N2':3,'N5:1}}
    """

    states = sorted(set(rhythm_sequence))

    transition_matrix = {}

    for state in states:
        transition_matrix[state] = {}
        total_occurences = sum(transition_counts[state].values())

        if total_occurences > 0:
            for next_state in states:
                count = transition_counts[state][next_state]
                transition_matrix[state][next_state] = count / total_occurences
        else:
            for next_state in states:
                transition_matrix[state][next_state] = 1.0 / len(states)

    return states, transition_matrix

def transition_matrix_to_numpy(transition_matrix, states):
    n = len(states)
    matrix = np.zeros((n,n))

    state_to_index = {state: i for i, state in enumerate(states)}

    for i, curr_state in enumerate(states):
        for j, next_state in enumerate(states):
            print(curr_state, next_state)
            matrix[i, j] = transition_matrix[curr_state][next_state]

    return matrix, state_to_index

def get_duration(state):
    return int(state[1:])

def build_revised_chain(transition_matrix, states):
    revised_states = []

    for state in states:
        beat_duration = get_duration(state)
        for beat_counter in range(1, beat_duration + 1):
            revised_states.append((state, beat_counter))

    n = len(revised_states)
    state_to_index = {s: i for i, s in enumerate(revised_states)}

    revised_matrix = np.zeros((n,n))

    for current_state_index, (current_state, beat_counter) in enumerate(revised_states):
        state_duration = get_duration(current_state)

        # Internal transition if counter < duration
        if beat_counter < state_duration:
            # Must go to next counter of the same state
            next_state = (current_state, beat_counter+1)
            next_state_index = state_to_index[next_state]
            revised_matrix[current_state_index, next_state_index] = 1.0

        else:
            # At end of note, transition to other states
            # Use normal transition probabilities  
            for next_state in states:
                print(current_state)
                print(next_state)
                prob = transition_matrix[current_state][next_state] 
                if prob > 0:
                    # Transition to first counter of next state
                    next_state = (next_state, 1)
                    next_state_index = state_to_index[next_state]
                    revised_matrix[current_state_index, next_state_index] = prob

    return revised_matrix, revised_states, state_to_index

def calculate_onset_probabilities(revised_matrix, revised_states, initial_state, bar_length=16):
    n = len(revised_states)
    state_to_index = {s: i for i, s in enumerate(revised_states)}

    pi = np.zeros(n)
    current_state_index = state_to_index[(initial_state, 1)]
    pi[current_state_index] = 1.0

    onset_probs = np.zeros(bar_length)

    for i in range(bar_length):
        onset_prob = 0.0

        for j, (state, beat_counter) in enumerate(revised_states):
            if beat_counter == 1 and not state.startswith('R'):
                onset_prob += pi[j]

        onset_probs[i] = onset_prob

        if i < bar_length - 1:
            pi = pi @ revised_matrix

    return onset_probs

def calculate_all_onset_probabilities(revised_matrix, revised_states, states, bar_length=16):
    all_onset_probs = {}

    for initial_state in states:
        if not initial_state.startswith('R'):
            onset_probs = calculate_onset_probabilities(
                revised_matrix, revised_states, initial_state, bar_length
            )
            all_onset_probs[initial_state] = onset_probs

    return all_onset_probs

rhythm_sequence = ['N1', 'N2', 'N1', 'R1', 'N3', 'N2', 'R1', 'N1']
states, transition_matrix = build_markov_chain(rhythm_sequence)
revised_matrix, revised_states, revised_state_index = build_revised_chain(transition_matrix, states)
all_onset_probs = calculate_all_onset_probabilities(revised_matrix, revised_states, states)

reggae = np.array([
        0.5, 0.0, 0.0, 0.0,  # Weak downbeat
        0.0, 0.0, 0.9, 0.0,  # Strong off-beat (& of 2)
        0.3, 0.0, 0.0, 0.0,  # Weak beat 3
        0.0, 0.0, 0.8, 0.0   # Strong off-beat (& of 4)
    ])

print(calculate_syncopation(reggae))