from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HMM import HMM

from collections import defaultdict
from Note import OrnamentGrouping, GeneratedNote
from ChordFunctions import get_chord_name_in_original_key
from Rhythm import RhythmMC
import numpy as np

class Generator:
    def __init__(self, chord_progression_hmm: HMM, ornament_hmms: OrnamentNoteHMMs, rhythm_mc: RhythmMC):
        self.chord_progression_hmm = chord_progression_hmm
        self.ornament_note_hmm = ornament_hmms
        self.rhythm_mc = rhythm_mc

    def generate(self, key) -> list[GeneratedNote]:
        full_sequence = []
        skeleton_notes_beat_duration = 0.5

        sampled_beats: list = self.chord_progression_hmm.generate()
        print(sampled_beats)
        for i in range(len(sampled_beats)-1):
            beat_1_pitch = sampled_beats[i].calc_midi_pitch(key)
            beat_2_pitch = sampled_beats[i+1].calc_midi_pitch(key)
            offset = beat_1_pitch - beat_2_pitch
            chord_function = sampled_beats[i].chord_function
            note_chord = get_chord_name_in_original_key(chord_function, key)

            ornament_notes = self.ornament_note_hmm.generate_sequence(offset, chord_function, self.rhythm_mc)

            full_sequence.append(GeneratedNote(beat_1_pitch, skeleton_notes_beat_duration, note_chord))
            for ornament_note in ornament_notes:
                midi_pitch = full_sequence[-1].midi_pitch + ornament_note.offset
                duration = ornament_note.duration
                full_sequence.append(GeneratedNote(midi_pitch, duration, note_chord))
            full_sequence.append(GeneratedNote(beat_2_pitch, skeleton_notes_beat_duration, note_chord))

        final_note_pitch = sampled_beats[-1].calc_midi_pitch(key)
        chord_function = sampled_beats[-1].chord_function
        note_chord = get_chord_name_in_original_key(chord_function, key)
        full_sequence.append(GeneratedNote(final_note_pitch, skeleton_notes_beat_duration, note_chord))

        return full_sequence

class OrnamentEmission:
    def __init__(self, offset, duration):
        self.offset = offset
        self.duration = duration

class OrnamentNoteHMMs:
    def __init__(self, ornament_groupings: list[OrnamentGrouping]):
        self.offset_function_training_data_mapping = self.split_song_ornaments(ornament_groupings)
        self.hmms = {}
        self.num_of_each_role = defaultdict(int)

    def print_stats(self):
        print(self.num_of_each_role)

    def split_song_ornaments(self, ornament_groupings: list[OrnamentGrouping]):
        offset_function_dict = defaultdict(list)

        for grouping in ornament_groupings:
            note_offset = grouping.get_group_note_interval()
            chord_function = grouping.chord_function

            offset_function_dict[(note_offset, chord_function)].append(grouping)

        return offset_function_dict

    def train_hmms(self):
        for offset_chord_function, training_data in self.offset_function_training_data_mapping.items():
            hmm = OrnamentHMM()
            succ_train = hmm.train_model(training_data)
            if succ_train:
                self.hmms[offset_chord_function] = hmm
                for role, count in hmm.num_of_each_role.items():
                    self.num_of_each_role[role] += count

    def generate_sequence(self, offset, chord_function, rhythm_mc):
        if (offset, chord_function) not in self.hmms:
            print(f'{offset, chord_function} not found in self.hmms')
            return []

        hmm = self.hmms[(offset, chord_function)]
        _, sampled_sequence = hmm.generate(rhythm_mc)
        return sampled_sequence

class OrnamentHMM:
    def __init__(self):
        self.transition_matrix = {}
        self.emission_matrix = {}
        self.initial_probabilities = {}
        self.num_of_each_role = defaultdict(int)

    def calc_initial_probabilities(self):
        initial_probabilities = {}
        hidden_states = list(self.transition_matrix.keys())

        if len(hidden_states) == 0:
            raise ValueError("HMM has no hidden states")

        uniform_prob = 1.0 / len(hidden_states)
        
        for hidden_state in hidden_states:
            initial_probabilities[hidden_state] = uniform_prob

        return initial_probabilities

    def calc_transition_matrix(self, ornament_groupings: list[OrnamentGrouping]):
        transition_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            # No ornament notes
            if len(grouping.ornament_notes) == 0:
                continue

            for i in range(len(grouping.ornament_notes)-1):
                curr_note_role = grouping.ornament_notes[i].role
                next_note_role = grouping.ornament_notes[i+1].role
                transition_count[curr_note_role][next_note_role] += 1

        if len(transition_count) == 0:
            print('no training data')

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_type in transition_count.keys():
            total_count = sum(transition_count[curr_type].values())
            self.num_of_each_role[curr_type] += total_count
            for next_type, count in transition_count[curr_type].items():
                transition_probs[curr_type][next_type] = count / total_count

        return transition_probs

    def calc_emission_matrix(self, ornament_groupings: list[OrnamentGrouping]):
        emission_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            for note in grouping.ornament_notes:
                emission_count[note.role][(note.offset, note.duration)] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for role in emission_count.keys():
            total_count = sum(emission_count[role].values())
            for (offset, duration), count in emission_count[role].items():
                emission_probs[role][(offset, duration)] = count / total_count

        return emission_probs

    def sample_next_hidden_state(self, current_hidden_state):
        if current_hidden_state not in self.transition_matrix:
            raise ValueError("Invalid hidden state")

        next_hidden_states = list(self.transition_matrix[current_hidden_state].keys())
        next_hidden_probs = list(self.transition_matrix[current_hidden_state].values())

        return next_hidden_states[np.random.choice(len(next_hidden_states), p=next_hidden_probs)]

    def sample_emission(self, hidden_state) -> OrnamentEmission:
        if hidden_state not in self.emission_matrix:
            raise ValueError("Invalid hidden state")

        emission_notes = list(self.emission_matrix[hidden_state].keys())
        emission_probs = list(self.emission_matrix[hidden_state].values())

        offset, duration = emission_notes[np.random.choice(len(emission_notes), p=emission_probs)]
        return OrnamentEmission(offset, duration)

    def sample_initial_state(self):
        hidden_states = list(self.initial_probabilities.keys())
        probs = list(self.initial_probabilities.values())
        return hidden_states[np.random.choice(len(hidden_states), p=probs)]

    def train_model(self, ornament_groupings: list[OrnamentGrouping]):
        try:
            self.transition_matrix = self.calc_transition_matrix(ornament_groupings)
            self.emission_matrix = self.calc_emission_matrix(ornament_groupings)
            self.initial_probabilities = self.calc_initial_probabilities()
            return True
        except ValueError as e:
            print(f'Error training model: {e}')
            return False

    def generate(self, rhytm_mc: RhythmMC, remaining_beats=1.5):
        hidden_state = self.sample_initial_state()
        emission = self.sample_emission(hidden_state)

        sampled_hidden_states = [hidden_state]
        sampled_emissions = [emission]

        # so basically, this samples some beats, lets say i fix the distance between skeleton notes
        # then i know how many beats i have to sample with
        # so then, i only want to sample beats that have a duration <= the beats i have left 
        # could just repeatedly sample, lets say 10 times until i get a valid note, if after 10 samples
        # i dont get any valid notes, just take the next sampled pitch, fix it to a duration equal to remaining duration

        while remaining_beats > 0:
            hidden_state = self.sample_next_hidden_state(hidden_state)
            found_valid_emission = False
            for i in range(10):
                emission = self.sample_emission(hidden_state)
                next_duration = rhytm_mc.sample_next_duration(sampled_emissions[-1].duration)
                print(f'remaining beat duration: {remaining_beats}. found an emission with a duration {emission.duration}')
                if next_duration <= remaining_beats:
                    found_valid_emission = True
                    emission.duration = next_duration
                    break
            if not found_valid_emission:
                emission = self.sample_emission(hidden_state)
                emission.duration = remaining_beats
            remaining_beats -= emission.duration

            sampled_hidden_states.append(hidden_state)
            sampled_emissions.append(emission)

        return sampled_hidden_states, sampled_emissions