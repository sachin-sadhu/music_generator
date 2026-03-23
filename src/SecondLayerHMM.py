from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HMM import HMM

from HMM import BassNoteGenerator, SkeletonEmission
from collections import defaultdict
from Note import OrnamentGrouping
from SongInfo import TrainingDataProcessedInfo
from Rhythm import *
import numpy as np
import pickle

class Generator:
    def __init__(self, chord_progression_hmm: HMM, ornament_mcs: OrnamentNoteMCs, rhythm_model, bass_generator: BassNoteGenerator):
        self.chord_progression_hmm = chord_progression_hmm
        self.ornament_note_mcs = ornament_mcs
        self.rhythm_model = rhythm_model
        self.bass_generator = bass_generator
        self.fixed_ornament_note_pitches = {}

    def generate(self, key, num_notes):
        subdivisions = 4
        bpm = 120
        seconds_per_beat = 60.0 / bpm
        seconds_per_step = seconds_per_beat / subdivisions

        melody = []
        bass = []

        sampled_beats = self.chord_progression_hmm.generate(num_notes)
        sampled_beats = sampled_beats[::2]
        sampled_rhythm = generate_rhythm_sequence(num_notes, self.rhythm_model)
        print(f'rhythm: {sampled_rhythm}')
        original_sampled_beats = sampled_beats.copy()

        i = 0
        while i < len(sampled_rhythm):
            #current_time = i * seconds_per_step
            ## Check if on strong beat position
            if i % 8 == 0:
                ## Need to start playing a fresh note
                if sampled_rhythm[i] == 0:
                    # Check if ornament note or skeelton note
                    end_index = i + 1
                    while (end_index < len(sampled_rhythm) and sampled_rhythm[end_index] == 1):
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

                    # Generate cooresponding bass note
                    bass_note_chord_tone = self.bass_generator.get_bass_note(skeleton_note.note_chord_tone)
                    bass_note_midi_pitch = SkeletonEmission(bass_note_chord_tone, 0, skeleton_note.chord_function).calc_midi_pitch(key) - 24
                    bass.append((bass_note_midi_pitch, i))
                    print(f'adding a bass note to index {i}. cooresponding treble index {i}, pithc: {midi_pitch}')
                elif sampled_rhythm[i] == 1 or sampled_rhythm[i] == 2:
                    # Sustaining previous note or rest on this strong beat
                    if sampled_beats:
                        skeleton_note = sampled_beats.pop(0)
                        # Generate cooresponding bass note
                        bass_note_chord_tone = self.bass_generator.get_bass_note(skeleton_note.note_chord_tone)
                        bass_note_midi_pitch = SkeletonEmission(bass_note_chord_tone, 0, skeleton_note.chord_function).calc_midi_pitch(key) - 24
                        bass.append((bass_note_midi_pitch, i))
            else:
                # need an ornament note first check to see whether we need to generate a new note or note
                if sampled_rhythm[i] == 0:
                    # check to see if ornament note pitch has already been decided
                    if i in self.fixed_ornament_note_pitches:
                        midi_pitch = self.fixed_ornament_note_pitches[i].calc_midi_pitch(key)
                        chord_function = self.fixed_ornament_note_pitches[i].chord_function
                    else:

                        try:
                            previous_skeleton_note = self.find_previous_skeleton_note(i, melody, sampled_rhythm)
                            previous_skeleton_note_pitch = previous_skeleton_note['pitch']
                            previous_skeleton_note_chord_function = previous_skeleton_note['chord_function']
                            next_skeleton_note = self.find_next_skeleton_note(i, sampled_beats, sampled_rhythm)
                            next_skeleton_note_pitch = next_skeleton_note.calc_midi_pitch(key)
                            print(f'found both')
                        except ValueError as e:
                            # Default to 0 offset and setting previous chord function to 'I'
                            print(e)
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
                            ornament_note_offset = self.ornament_note_mcs.generate_sequence(offset)
                            print(f'ornament note offset: {ornament_note_offset}')
                            if len(melody) > 0:
                                midi_pitch = melody[-1]['pitch'] + ornament_note_offset
                            else:
                                midi_pitch = previous_skeleton_note_pitch

                    end_index = i + 1
                    while (end_index < len(sampled_rhythm) and sampled_rhythm[end_index] == 1):
                        end_index += 1

                    note = {
                        'pitch': midi_pitch,
                        'start': i,
                        'end': end_index,
                        'chord_function': chord_function
                    }
                    melody.append(note)
            i += 1

        return melody, bass, original_sampled_beats

    """
        Returns the pitch and chord function of the next skeleton note
    """
    def find_next_skeleton_note(self, current_index, skeleton_notes, sampled_rhythm):
        # first find out index of next skeleton note
        next_strong_beat_index = current_index + 1
        found_strong_beat = False
        while (not found_strong_beat and next_strong_beat_index < len(sampled_rhythm)):
            if next_strong_beat_index % 8 == 0:
                found_strong_beat = True
                break
            else:
                next_strong_beat_index += 1

        if not found_strong_beat:
            raise ValueError("no next strong beat found")

        # Note being generated at the strong beat
        if sampled_rhythm[next_strong_beat_index] == 0:
            return skeleton_notes[0]
        elif sampled_rhythm[next_strong_beat_index] == 1:
            # note is being sustained, figure out index of note that will sustain it
            sustained_note_index = next_strong_beat_index-1
            while sustained_note_index >= 0:
                if sampled_rhythm[sustained_note_index] == 0:
                    break
                else:
                    sustained_note_index -= 1
            self.fixed_ornament_note_pitches[sustained_note_index] = skeleton_notes[0]
        else:
            raise ValueError("rest being played at skeleton note.")
        
        return skeleton_notes[0]

    """ 
        issue right now is that sound playing at previous skeleton note might not be played 
        exactly on a skeleton note

        so to find the pitch, we should search backwards to find out which index is responsible for
        that not being played. then we can search for whichever note was generated at that index what the pitch is
        can find index by doing start / seconds_per_step
    """
    def find_previous_skeleton_note(self, current_index, current_melody, sampled_rhythm):
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
        if sampled_rhythm[recent_strong_beat_index] == 0:
            for note in reversed(current_melody):
                if note['start'] == recent_strong_beat_index:
                    return note
        elif sampled_rhythm[recent_strong_beat_index] == 1:
            # note was sustained from somewhere else, need to find note responsible for it 
            # by searching to the left somemore for the most recent 1 note
            recent_new_note_index = recent_strong_beat_index - 1
            found_new_note = False
            while (not found_new_note and recent_new_note_index >= 0):
                if sampled_rhythm[recent_new_note_index] == 1:
                    found_new_note = True
                    break
                else:
                    recent_new_note_index -= 1
            
            if not found_new_note:
                raise ValueError("no previous skeleton note found")
            
            for note in reversed(current_melody):
                if note['start'] == recent_new_note_index:
                    return note

        raise ValueError("no previous skeleton note found")

class OrnamentEmission:
    def __init__(self, offset):
        self.offset = offset

class OrnamentNoteMCs:
    def __init__(self):
        self.mcs = {}

    def save_model(self, filepath='models/ornament_mcs.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath='models/ornament_mcs.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def split_song_ornaments(self, ornament_groupings: list[OrnamentGrouping]):
        offset_function_dict = defaultdict(list)

        for grouping in ornament_groupings:
            note_offset = grouping.get_group_note_interval()
            offset_function_dict[note_offset].append(grouping)

        return offset_function_dict

    def train_mcs(self, training_data: list[OrnamentGrouping]):
        offset_training_data = self.split_song_ornaments(training_data)
        for offset, training_data in offset_training_data.items():
            mc = OrnamentMC()
            succ_train = mc.train_mc(training_data)
            if succ_train:
                self.mcs[offset] = mc

    def generate_sequence(self, offset):
        if offset not in self.mcs:
            print(f'{offset} not found')
            return [OrnamentEmission(0)]

        mc = self.mcs[offset]
        sampled_sequence = mc.generate()
        return sampled_sequence

class OrnamentMC:
    def __init__(self):
        self.transition_matrix = {}
        self.initial_probabilities = {}
        self.num_of_each_role = defaultdict(int)

    def calc_initial_probabilities(self, ornament_groupings: list[OrnamentGrouping]):
        initial_count = defaultdict(int)

        for grouping in ornament_groupings:
            if len(grouping.ornament_notes) == 0:
                continue
            initial_note_offset = grouping.ornament_notes[0].offset
            initial_count[initial_note_offset] += 1

        initial_probs = {}
        total_count = sum(initial_count.values())
        for initial_emission, count in initial_count.items():
            initial_probs[initial_emission] = count / total_count

        return initial_probs

    def calc_transition_matrix(self, ornament_groupings: list[OrnamentGrouping]):
        transition_count = defaultdict(lambda: defaultdict(int))

        for grouping in ornament_groupings:
            # No ornament notes
            if len(grouping.ornament_notes) == 0:
                continue

            for i in range(len(grouping.ornament_notes)-1):
                curr_note_offset = grouping.ornament_notes[i].offset
                next_note_offset = grouping.ornament_notes[i+1].offset
                transition_count[curr_note_offset][next_note_offset] += 1

        if len(transition_count) == 0:
            print('no training data')

        transition_probs = {}
        #transition_probs = defaultdict(lambda: defaultdict(float))
        for current_offset in transition_count.keys():
            total_count = sum(transition_count[current_offset].values())
            for next_offset, count in transition_count[current_offset].items():
                if current_offset not in transition_probs:
                    transition_probs[current_offset] = {}
                transition_probs[current_offset][next_offset] = count / total_count

        return transition_probs

    def sample_next_offset(self, current_offset):
        if current_offset not in self.transition_matrix:
            raise ValueError("Invalid offset.")

        next_offsets = list(self.transition_matrix[current_offset].keys())
        offsets_probs = list(self.transition_matrix[current_offset].values())

        return np.random.choice(next_offsets, p=offsets_probs)

    def sample_initial_state(self):
        hidden_states = list(self.initial_probabilities.keys())
        probs = list(self.initial_probabilities.values())
        return np.random.choice(hidden_states, p=probs)

    def train_mc(self, ornament_groupings: list[OrnamentGrouping]):
        try:
            self.transition_matrix = self.calc_transition_matrix(ornament_groupings)
            self.initial_probabilities = self.calc_initial_probabilities(ornament_groupings)
            return True
        except ValueError as e:
            print(f'Error training model: {e}')
            return False

    def generate(self):
        emission = self.sample_initial_state()
        return emission
        # TODO need to change this
        #emission = self.sample_emission(hidden_state)

        #sampled_hidden_states = [hidden_state]
        #sampled_emissions = [emission]
        
        #return sampled_emissions

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

if __name__ == "__main__":  
    directory = "/home/sachin/Documents/music_generator/POP909/POP909"
    data = TrainingDataProcessedInfo()
    data.load_training_data(directory)

    ornament_note_mcs = OrnamentNoteMCs()
    ornament_note_mcs.train_mcs(data.ornament_groupings)
    ornament_note_mcs.save_model()
    #ornament_note_mcs = OrnamentNoteMCs.load()
    #print(ornament_note_mcs.mcs[2].transition_matrix)