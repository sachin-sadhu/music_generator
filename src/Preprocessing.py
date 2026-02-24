from mido import MidiFile, tick2second
from collections import defaultdict
from SecondLayerHMM import *
from Note import OrnamentGrouping
from Timings import BeatTiming, ChordTiming
from SongInfo import SongInfo
from SongInfo import TrainingDataProcessedInfo
import os

def load_song_info(directory: str) -> TrainingDataProcessedInfo:
    all_song_notes = []
    all_song_beat_chords = []
    all_song_ornament_groupings = []

    # Loop through all songs
    for song_dir in sorted(os.listdir(directory)):
        song_path = os.path.join(directory, song_dir)

        if not os.path.isdir(song_path):
            continue
    
        midi_file = os.path.join(song_path, f"{song_dir}.mid")
        chord_file = os.path.join(song_path, "chord_midi.txt")
        key_file = os.path.join(song_path, "key_audio.txt")
        beat_file = os.path.join(song_path, "beat_midi.txt")

        if all(os.path.exists(file) for file in [midi_file, chord_file, key_file, beat_file]):
            try:
                song_info = SongInfo(key_file, midi_file, chord_file, beat_file)

                #bars = group_notes_by_bar(notes)
                #bar_timings = get_all_bar_timings(bars, 120)
                #bars_chords_mapped = assign_chords_to_bars(bar_timings, chord_timings)
                #notes_chord_assigned = assign_chord_to_notes(notes, chord_timings)

                # Skip songs in minor keys for now
                if get_chord_root_and_type(song_info.song_key)[1] == 'min':
                    continue

                groupings = get_ornament_groupings(song_info)
                all_song_ornament_groupings.append(groupings)

                # Process beat chords association
                beats_chords = get_beats_matching_chords(song_info)
                beats_chords_function_list = []
                for matched_chord in beats_chords:
                    try:
                        if matched_chord is None:
                            beats_chords_function_list.append('N')
                        elif matched_chord.get_chord_name() == 'N':
                            beats_chords_function_list.append('N')
                        else:
                            transposed_chord = transpose_chord_to_c_major(matched_chord, song_info.song_key)
                            chord_function = convert_chord_name_to_roman_numeral(transposed_chord)
                            beats_chords_function_list.append(chord_function)
                    except Exception as e:
                        beats_chords_function_list.append('N')
                all_song_beat_chords.append(beats_chords_function_list)

                # Process song notes
                song_notes_filtered = []
                for note in song_info.notes:
                    try:
                        note.set_original_chord(song_info.chord_timings)

                        if note.get_original_chord() is None or note.get_original_chord() == 'N':
                            continue

                        note.set_chord_tone()
                        note.set_chord_function(song_info)
                        song_notes_filtered.append(note)
                    except Exception as e:
                        continue

                all_song_notes.append(song_notes_filtered)

            except Exception as e:
                print(f"Error{song_dir}: {e}")
                continue
        else:
            print(f"Missing file for song {song_dir}")

    training_data = TrainingDataProcessedInfo(all_song_notes, all_song_beat_chords, all_song_ornament_groupings)
    return training_data

def get_tempo_from_midi(midi):
    """
    Extract tempo from MIDI file (microseconds per beat).
    Default to 120 BPM if not found.
    """
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000  # Default: 120 BPM

def ticks_to_seconds(ticks, ticks_per_beat, tempo_microseconds):
    """
        Convert MIDI ticks to seconds
    """
    seconds_per_beat = tempo_microseconds / 1000000
    beats = ticks / ticks_per_beat
    seconds = beats * seconds_per_beat
    return seconds

def load_midi_notes(midi_path):
    midi = MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    tempo = get_tempo_from_midi(midi)

    notes = []
    active_notes = {}
    current_tick = 0

    for msg in midi.tracks[3]:
        current_tick += msg.time
        #current_seconds = ticks_to_seconds(current_tick, ticks_per_beat, tempo)
        current_seconds = tick2second(current_tick, ticks_per_beat, tempo)

        # Note is being played
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = {
                'pitch': msg.note,
                'start_tick': current_tick,
                'start_seconds': current_seconds
            }
        
        # Note is being turned off
        elif msg.type == 'note_on' and msg.velocity == 0:
            if msg.note in active_notes:
                curr_note = active_notes[msg.note]

                seconds_onset = curr_note['start_seconds']
                tick_onset = curr_note['start_tick']
                tick_duration = current_tick - tick_onset
                beat_duration = tick_duration / ticks_per_beat
                note_duration = quantise_beat_duration(beat_duration)
                clef = 'treble' if msg.note >= 60 else 'bass'

                note = TrainingNote(msg.note, clef, note_duration, seconds_onset)
                notes.append(note)
                del active_notes[msg.note]

    return notes

def get_beats_matching_chords(song_info: SongInfo) -> list[ChordTiming | None]:
    beat_timings: list[BeatTiming] = song_info.beat_timings
    beat_chords = []
    for beat in beat_timings:
        matched_chord = beat.get_matching_chord(song_info)
        beat_chords.append(matched_chord)

    return beat_chords

def split_beat_timings_to_bars(beat_timings):
    bars = []
    current_bar = []

    for beat_time, beat_strong_beat, beat_new_bar in beat_timings:
        # Reached the end of the bar
        if len(current_bar) != 0 and beat_new_bar == 1.0:
            bars.append(current_bar)
            current_bar = [(beat_time, beat_strong_beat)]
        else:
            current_bar.append((beat_time, beat_strong_beat))

    # Add last bar
    if current_bar:
        bars.append(current_bar)

    return bars
        
    
def calculate_beat_position(ticks_per_bar, ticks_per_beat, tick_onset):
    """
    Calculate the bar index and beat position within a bar for a given tick onset.

    Args:
        ticks_per_bar (int): The number of ticks in one bar.
        ticks_per_beat (int): The number of ticks in one beat.
        tick_onset (int): The absolute tick position to convert.

    Returns:
        tuple: A tuple containing:
            - bar_index (int): The zero-based index of the bar.
            - beat_position_bar (float): The position within the bar expressed in beats.
    """
    bar_index = tick_onset // ticks_per_bar
    tick_onset_within_bar = tick_onset % ticks_per_bar
    beat_position_bar = tick_onset_within_bar / ticks_per_beat
    return (bar_index, beat_position_bar)

def quantise_beat_position(beat_position, grid_size=0.5):
    return round(beat_position / grid_size) * grid_size

def group_notes_by_bar(notes):
    """
    Groups musical notes by their bar index.

    This function takes a list of note dictionaries and organizes them into separate
    lists based on their bar_index value. Each bar is represented as a list of notes
    that belong to that bar.

    Args:
        notes (list[dict]): A list of note dictionaries, where each dictionary must
                           contain a 'bar_index' key indicating which bar the note
                           belongs to.

    Returns:
        list[list[dict]]: A list of lists, where each inner list contains all notes
                         belonging to a specific bar. The outer list index corresponds
                         to the bar index (e.g., bars[0] contains notes from bar 0).
    """
    max_bar = max(note['bar_index'] for note in notes)
    bars = [[] for _ in range(max_bar+1)]

    for note in notes:
        current_bar_index = note['bar_index']
        bars[current_bar_index].append(note)

    return bars

def get_all_bar_timings(bars, beats_per_min, beats_per_bar=4.0):
    def get_bar_timings(bar_index):
        """
        Calculate the start and end timings of a musical bar in seconds.

        Args:
            bar_index (int): The index of the bar (0-based).
            beats_per_bar (float, optional): Number of beats in a bar. Defaults to 4.0.
            beats_per_min (float): The tempo in beats per minute (BPM).

        Returns:
            tuple: A tuple containing two float values:
                - bar_onset_seconds (float): The start time of the bar in seconds.
                - bar_finished_seconds (float): The end time of the bar in seconds.
        """
        seconds_per_beat = 60 / beats_per_min
        seconds_per_bar = beats_per_bar * seconds_per_beat
        bar_onset_seconds = bar_index * seconds_per_bar
        bar_finished_seconds = bar_onset_seconds + seconds_per_bar

        return bar_onset_seconds, bar_finished_seconds

    bar_timings = [get_bar_timings(i) for i in range(len(bars))]
    return bar_timings

def cap_notes_in_bar(bar_notes, beats_per_bar=4.0):
    for note in bar_notes:
        beats_remaining = beats_per_bar - note['beat_onset']
        if note['beat_duration'] > beats_remaining:
            note['beat_duration'] = beats_remaining
            note['note_type'] = quantise_beat_duration(beats_remaining)

def filter_empty_bars_no_chord(bars, bars_chords):
    filtered_bars = []
    filtered_bars_chords = []

    for i in range(min(len(bars), len(bars_chords))):
        if bars[i] != [] and bars_chords[i] != 'N':
            filtered_bars.append(bars[i])
            filtered_bars_chords.append(bars_chords[i])

    return filtered_bars, filtered_bars_chords

def get_note_midi_pitch(chord_tone, chord_roman_numeral, key, octave_offset=0):
    chromatic_intervals_inverted = {
        "root": 0, "b2": 1, "2nd": 2, "b3": 3, "3rd": 4, "4th": 5,
        "b5": 6, "5th": 7, "b6": 8, "6th": 9, "b7": 10, "7th": 11, "octave": 12
    }

    roman_numeral_to_semitones = {
        'I': 0,   # Tonic (0 semitones above root)
        'ii': 2,  # 2 semitones above root
        'iii': 4, # 4 semitones above root
        'IV': 5,  # 5 semitones above root
        'V': 7,   # 7 semitones above root ← We need this!
        'vi': 9,  # 9 semitones above root
        'vii': 11 # 11 semitones above root
    }

    note_to_pitch_class = {
        'C': 0, 
        'C#': 1, 'Db': 1,
        'D': 2,
        'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5,
        'F#': 6, 'Gb': 6,
        'G': 7,
        'G#': 8, 'Ab': 8,
        'A': 9,
        'A#': 10, 'Bb': 10,
        'B': 11
    }

    # Get key information
    key_root_note, key_type = get_key_root_and_type(key)
    key_root_note_midi_pitch = 60 + note_to_pitch_class.get(key_root_note, 0)

    # Calculate chord root
    chord_root_note_midi_pitch = key_root_note_midi_pitch + roman_numeral_to_semitones.get(chord_roman_numeral, 0)

    # Calculate final note pitch
    note_midi_pitch = chord_root_note_midi_pitch + chromatic_intervals_inverted.get(chord_tone, 0) + (octave_offset * 12)

    #print(f"for chord type: {chord_roman_numeral}. for chord_tone: {chord_tone}. for key: {key}. generated_midi_pitch: {note_midi_pitch}")

    return note_midi_pitch

def get_note_at_beat_timing(beat_timing, notes, threshold=0.05):
    notes = sorted(notes, key=lambda i: i['start_seconds'])
    for note in notes:
        if abs(beat_timing - note['start_seconds']) <= threshold:
            return note
        
        if note['start_seconds'] - 1.0 >= beat_timing:
            return None

    return None

#def get_closest_note_at_time(target_time, notes: list[TrainingNote], threshold=0.4) -> TrainingNote | None:
    #notes = sorted(notes, key=lambda note: note.get_start_seconds())
    #current_closest_diff = 1000000
    #current_closest_note = None
    #for note in notes:
        #print(target_time)
        #diff = abs(note.get_start_seconds() - target_time)
        #if diff < current_closest_diff:
            #current_closest_diff = diff
            #current_closest_note = note
    
    #if current_closest_diff < threshold:
        #return current_closest_note

    #return None

#def get_notes_at_bar_onset(beat_timings, notes):
    #"""
        #Gets all the notes at the start of each bar
    #"""
    #bar_onset_notes = []
    #for beat_onset_seconds, _,  new_bar_note in beat_timings:
        #if new_bar_note == 0:
            #continue
            
        #target_note = get_closest_note_at_time(beat_onset_seconds, notes)
        #bar_onset_notes.append(target_note)

    #return notes



#def get_skeleton_note_pairs(notes, beat_timings):
    #"""
        #Given the list of notes and beat timings, gets all beat note timings
    #"""
    #beat_timings.sort(key=lambda beat: beat[0])
    #pairs = []
    #bar_onset_notes = get_notes_at_bar_onset(beat_timings, notes)

    #for i in range(len(bar_onset_notes)-1):
        #curr_beat_timing = beat_timings[i][0]
        #next_beat_timing = beat_timings[i+1][0]

        #curr_beat_note = get_note_at_beat_timing(curr_beat_timing, notes)
        #next_beat_note = get_note_at_beat_timing(next_beat_timing, notes)

        #if curr_beat_note is None or next_beat_note is None:
            #continue

        #pair = (curr_beat_note, next_beat_note)
        #pairs.append((pair))

    #return pairs

""" 
    given a list of ntoes, we want to define functions that 
    - for each ornament note, determine the interval from the previous note 
    - for each ornament note, determines the note role
    - for each grouping, determine the chord function 
    - for each grouping, determine the chord notes
"""

# here the ornament_groupings_dict is the dict where song followed by a list of lists 

# each (interval offset, chord function) pairing has its HMM. so when training given the list of groupings, for each song, we want to identify
#which ornament grouping this belongs to, then we should add the grouping to some sort of list that is then iterated through when training 

def split_song_ornaments(ornament_groupings_dict):
    # want to split into (offset, chord_function): [[ornament], [ornament]]
    offset_chord_function_dict = defaultdict(list)
    
    # here the ornament_groupings_dict is the dict where song followed by a list of lists 

    # each (interval offset, chord function) pairing has its HMM. so when training given the list of groupings, for each song, we want to identify
    #which ornament grouping this belongs to, then we should add the grouping to some sort of list that is then iterated through when training 
    for song_groupings in ornament_groupings_dict.values():
        for grouping in song_groupings:
            skeleton_note_one_midi_pitch = grouping[0][0]['pitch']
            skeleton_note_two_midi_pitch = grouping[0][1]['pitch']
            offset = skeleton_note_one_midi_pitch - skeleton_note_two_midi_pitch
            chord_function = grouping[-1]
            ornament_notes = grouping[1]

            offset_chord_function_dict[(offset, chord_function)].append(ornament_notes)

    return offset_chord_function_dict

def train_all_hmms(ornament_groupings_dict):
    hmms = {}
    for offset_chordfunction, training_data in ornament_groupings_dict.values():
        current_hmm = SecondLayerHMM()
        current_hmm.train_model(training_data)
        hmms[offset_chordfunction] = current_hmm

    return hmms