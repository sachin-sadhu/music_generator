from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HMM import HMM

from collections import defaultdict
from Note import OrnamentGrouping, GeneratedNote
from ChordFunctions import get_chord_name_in_original_key
from SongInfo import KeyTiming
from HMM import SkeletonEmission
import pretty_midi
import numpy as np

class Generator:
    def __init__(self, chord_progression_hmm: HMM, ornament_hmms: OrnamentNoteHMMs, rhythm_sequence):
        self.chord_progression_hmm = chord_progression_hmm
        self.ornament_note_hmm = ornament_hmms
        self.rhythm_sequence = rhythm_sequence
        self.fixed_ornament_note_pitches = {}

    def generate(self, key):
        subdivisions = 4
        bpm = 120
        seconds_per_beat = 60.0 / bpm
        seconds_per_step = seconds_per_beat / subdivisions

        melody = []
        sampled_beats = self.chord_progression_hmm.generate(64)
        sampled_beats = sampled_beats[::2]
        original_sampled_beats = sampled_beats.copy()
        print(sampled_beats)
        print(len(self.rhythm_sequence))


        i = 0
        while i < len(self.rhythm_sequence):
            #current_time = i * seconds_per_step
            ## Check if on strong beat position
            if i % 8 == 0:
                ## Need to start playing a fresh note
                if self.rhythm_sequence[i] == 0:
                    # Check if ornament note or skeelton note
                    end_index = i + 1
                    while (end_index < len(self.rhythm_sequence) and self.rhythm_sequence[end_index] == 1):
                        end_index += 1

                    # Skeleton note
                    # create MIDI note
                    skeleton_note = sampled_beats.pop(0)
                    midi_pitch = skeleton_note.calc_midi_pitch(key)
                    chord_function = skeleton_note.chord_function
                    note = {
                        'pitch': midi_pitch,
                        'start': i,
                        'end': end_index,
                        'chord_function': chord_function
                    }
                    melody.append(note)
                elif self.rhythm_sequence[i] == 1 or self.rhythm_sequence[i] == 2:
                    # Sustaining previous note or rest on this strong beat
                    if sampled_beats:
                        sampled_beats.pop(0)
            else:
                # need an ornament note first check to see whether we need to generate a new note or note
                if self.rhythm_sequence[i] == 0:
                    # check to see if ornament note pitch has already been decided
                    if i in self.fixed_ornament_note_pitches:
                        midi_pitch = self.fixed_ornament_note_pitches[i].calc_midi_pitch(key)
                        chord_function = self.fixed_ornament_note_pitches[i].chord_function
                    else:

                        try:
                            previous_skeleton_note = self.find_previous_skeleton_note(i, melody)
                            previous_skeleton_note_pitch = previous_skeleton_note['pitch']
                            previous_skeleton_note_chord_function = previous_skeleton_note['chord_function']
                            next_skeleton_note_pitch, _ = self.find_next_skeleton_note(i, sampled_beats, key)
                        except Exception:
                            # Default to 0 offset and setting previous chord function to 'I'
                            next_skeleton_note_pitch = 60
                            previous_skeleton_note_pitch = 60
                            previous_skeleton_note_chord_function = 'I'

                        # Check if this note is the note that sutains onto the next strong beat note
                        if i in self.fixed_ornament_note_pitches:
                            midi_pitch = self.fixed_ornament_note_pitches[i].calc_midi_pitch(key)
                            chord_function = self.fixed_ornament_note_pitches[i].chord_function
                        else:
                            offset = previous_skeleton_note_pitch - next_skeleton_note_pitch 
                            chord_function = previous_skeleton_note_chord_function
                            ornament_note = self.ornament_note_hmm.generate_sequence(offset, chord_function)[0]
                            if len(melody) > 0:
                                midi_pitch = melody[-1]['pitch'] + ornament_note.offset
                            else:
                                midi_pitch = previous_skeleton_note_pitch

                    end_index = i + 1
                    while (end_index < len(self.rhythm_sequence) and self.rhythm_sequence[end_index] == 1):
                        end_index += 1

                    note = {
                        'pitch': midi_pitch,
                        'start': i,
                        'end': end_index,
                        'chord_function': chord_function
                    }
                    melody.append(note)
            i += 1

        return melody, original_sampled_beats

    """
        Returns the pitch and chord function of the next skeleton note
    """
    def find_next_skeleton_note(self, current_index, skeleton_notes, key):
        # first find out index of next skeleton note
        next_strong_beat_index = current_index + 1
        found_strong_beat = False
        while (not found_strong_beat and next_strong_beat_index < len(self.rhythm_sequence)):
            if next_strong_beat_index % 8 == 0:
                found_strong_beat = True
                break
            else:
                next_strong_beat_index += 1

        if not found_strong_beat:
            raise ValueError("no next strong beat found")

        # Note being generated at the strong beat
        if self.rhythm_sequence[next_strong_beat_index] == 1:
            return skeleton_notes[0]
        elif self.rhythm_sequence[next_strong_beat_index] == 2:
            # note is being sustained, figure out index of note that will sustain it
            sustained_note_index = next_strong_beat_index-1
            while sustained_note_index >= 0:
                if self.rhythm_sequence[sustained_note_index] == 1:
                    break
                else:
                    sustained_note_index -= 1
            self.fixed_ornament_note_pitches[sustained_note_index] = skeleton_notes[0]
        else:
            raise ValueError("rest being played at skeleton note.")
        
        pitch = skeleton_notes[0].calc_midi_pitch(key)
        chord_function = skeleton_notes[0].chord_function

        return pitch, chord_function

    """ 
        issue right now is that sound playing at previous skeleton note might not be played 
        exactly on a skeleton note

        so to find the pitch, we should search backwards to find out which index is responsible for
        that not being played. then we can search for whichever note was generated at that index what the pitch is
        can find index by doing start / seconds_per_step
    """
    def find_previous_skeleton_note(self, current_index, current_melody):
        # search backwards and find most recent strong beat index
        recent_strong_beat_index = current_index - 1
        found_strong_beat = False
        while (not found_strong_beat and recent_strong_beat_index >= 0):
            if recent_strong_beat_index % 8 == 0:
                found_strong_beat = True
                break
            else:
                recent_strong_beat_index -= 1

        if not found_strong_beat:
            raise ValueError("no previous skeleton note found")

        # Check if this strong beat was a 1 (new note) or 2 (sustained pitch from previous note)
        if self.rhythm_sequence[recent_strong_beat_index] == 1:
            for note in reversed(current_melody):
                if note['start'] == recent_strong_beat_index:
                    return note
        elif self.rhythm_sequence[recent_strong_beat_index] == 2:
            # note was sustained from somewhere else, need to find note responsible for it 
            # by searching to the left somemore for the most recent 1 note
            recent_new_note_index = recent_strong_beat_index - 1
            found_new_note = False
            while (not found_new_note and recent_new_note_index >= 0):
                if self.rhythm_sequence[recent_new_note_index] == 1:
                    found_new_note = True
                    break
                else:
                    recent_new_note_index -= 1
            
            if not found_new_note:
                raise ValueError("no previous skeleton note found")
            
            for note in reversed(current_melody):
                if note['start'] == recent_new_note_index:
                    return note

class OrnamentEmission:
    def __init__(self, offset):
        self.offset = offset

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

    def generate_sequence(self, offset, chord_function):
        if (offset, chord_function) not in self.hmms:
            print(f'{offset, chord_function} not found in self.hmms')
            # TODO need ot change this back
            return [OrnamentEmission(0)]

        hmm = self.hmms[(offset, chord_function)]
        sampled_sequence = hmm.generate()
        print(f'for ({offset},{chord_function}): sampled: {sampled_sequence[0].offset} ')
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
                emission_count[note.role][note.offset] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for role in emission_count.keys():
            total_count = sum(emission_count[role].values())
            for offset, count in emission_count[role].items():
                emission_probs[role][offset] = count / total_count

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

        offset = emission_notes[np.random.choice(len(emission_notes), p=emission_probs)]
        return OrnamentEmission(offset)

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

    def generate(self):
        hidden_state = self.sample_initial_state()
        emission = self.sample_emission(hidden_state)

        sampled_hidden_states = [hidden_state]
        sampled_emissions = [emission]
        
        return sampled_emissions

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