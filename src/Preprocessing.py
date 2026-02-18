from mido import MidiFile, tick2second
from ChordFunctions import *
from collections import defaultdict
from SecondLayerHMM import *
import os

def load_song_info(directory):
    all_song_notes = {}
    all_song_beat_chords = {}
    all_song_groupings = {}

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
                song_key = load_key(key_file)
                song_notes = load_midi_notes(midi_file)
                chord_timings = load_chord_timings(chord_file)
                beat_timings = load_beat_timings(beat_file)

                #bars = group_notes_by_bar(notes)
                #bar_timings = get_all_bar_timings(bars, 120)
                #bars_chords_mapped = assign_chords_to_bars(bar_timings, chord_timings)
                #notes_chord_assigned = assign_chord_to_notes(notes, chord_timings)

                # Skip songs in minor keys for now
                if get_chord_root_and_type(song_key)[1] == 'min':
                    continue

                groupings = get_ornament_groupings(song_notes, beat_timings)
                prepped_groupings = prep_groupings_for_second_layer(groupings, chord_timings, song_key)
                all_song_groupings[song_dir] = prepped_groupings

                # Process beat chords association
                beats_chords = get_beats_chords(beat_timings, chord_timings)
                beats_chords_function = []
                for beat_chord in beats_chords:
                    try:
                        if beat_chord == 'N':
                            beats_chords_function.append('N')
                        else:
                            transposed_chord = transpose_chord_to_c_major(beat_chord, song_key)
                            chord_function = convert_chord_name_to_roman_numeral(transposed_chord)
                            beats_chords_function.append(chord_function)
                    except Exception as e:
                        beats_chords_function.append('N')
                all_song_beat_chords[song_dir] = beats_chords_function

                # Process song notes
                song_notes_filtered = []
                for note in song_notes:
                    try:
                        note_cooresponding_chord = get_matching_chord(note['start_seconds'], chord_timings)
                        if note_cooresponding_chord == 'N':
                            continue

                        note_chord_tone = get_note_chord_tone(note['pitch'], note_cooresponding_chord)
                        transposed_chord = transpose_chord_to_c_major(note_cooresponding_chord, song_key)
                        chord_function = convert_chord_name_to_roman_numeral(transposed_chord)

                        # Add new attributes to note dict
                        note['original_chord'] = note_cooresponding_chord
                        note['note_chord_tone'] = note_chord_tone
                        note['chord_function'] = chord_function
                        song_notes_filtered.append(note)
                    except Exception as e:
                        #print(f"Error {song_dir} skipping note: {e}")
                        continue

                all_song_notes[song_dir] = song_notes_filtered

            except Exception as e:
                print(f"Error{song_dir}: {e}")
                continue
        else:
            print(f"Missing file for song {song_dir}")

    return all_song_notes, all_song_beat_chords, all_song_groupings

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

                #tick_onset = curr_note['start_tick']
                #tick_duration = current_tick - tick_onset
                #beat_duration = tick_duration / ticks_per_beat
                #(bar_index, beat_position) = calculate_beat_position(ticks_per_bar, ticks_per_beat, tick_onset)
                #quantised_beat_position = quantise_beat_position(beat_position)
                #note_type = quantise_beat_duration(beat_duration)

                #curr_note['tick_duration'] = tick_duration
                #curr_note['beat_duration'] = beat_duration
                #curr_note['beat_onset'] = quantised_beat_position
                #curr_note['bar_index'] = bar_index
                #curr_note['note_type'] = note_type
                curr_note['end_seconds'] = current_seconds
                curr_note['clef'] = 'treble' if msg.note >= 60 else 'bass'

                notes.append(curr_note)
                del active_notes[msg.note]

    notes = sorted(notes, key=lambda note: note['start_seconds'])
    treble_clef = [note for note in notes if note['clef'] == 'treble']
    
    return treble_clef

def load_beat_timings(beat_file_path):
    beats = []
    with open(beat_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(' ')

            if len(parts) >= 3:
                beat_time = float(parts[0])
                beat_strong_beat = float(parts[1])
                beat_new_bar = float(parts[2])

                beats.append((beat_time, beat_strong_beat, beat_new_bar))

    return beats

def get_beats_chords(beat_timings, chord_timings):
    beat_chords = []
    for beat in beat_timings:
        beat_onset = beat[0]
        matched_chord = get_matching_chord(beat_onset, chord_timings)
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
        

"""
    loop through file, when see a 1.0 in last column indicates new bar
    continoulsy add until we see another 1.0
"""

def get_matching_chord(onset, chord_timings):
    for chord_start, chord_end, chord_label in chord_timings:
        if chord_start <= onset <= chord_end:
            return chord_label

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

def quantise_beat_duration(beat_duration, grid_size=0.5, max_duration=4.0):
    """
    Quantise a beat duration to the nearest musical note value.

    Args:
        beat_duration (float): The duration of the beat to quantise, in beats.
        grid_size (float, optional): The grid size for quantisation. Defaults to 0.5.
        max_duration (float, optional): The maximum allowed duration. Defaults to 4.0.

    Returns:
        str: The name of the musical note value (e.g., 'crotchet', 'quaver', 'semibreve')
             that best represents the quantised beat duration.
    """
    DURATION_BINS = {
        0.25: 'semiquaver',
        0.50: 'quaver',
        0.75: 'dotted_quaver',
        1.0: 'crotchet',
        1.5: 'dotted_crotchet',
        2.0: 'minim',
        3: 'dotted_minim',
        4: 'semibreve'
    }

    quantised = round(beat_duration / grid_size) * grid_size

    if quantised < grid_size:
        quantised = grid_size
    if quantised > max_duration:
        quantised = max_duration

    closest = min(DURATION_BINS.keys(), key=lambda x: abs(x-quantised))
    return DURATION_BINS[closest]

def get_note_chord_tone(note_midi_pitch, chord):
    """
    Determines the relationship of a note to a given chord in terms of intervals and octave offset.

    Args:
        note_midi_pitch (int): The MIDI pitch value of the note (0-127).
        chord (str): The chord name (e.g., 'Cmaj', 'Amin').

    Returns:
        tuple: A tuple containing:
            - chord_tone (str): The interval name relative to the chord root 
              (e.g., 'root', '3rd', '5th', 'b7').
            - octave_offset (int): The number of octaves the note is above or below 
              the chord root in octave 4.

    Example:
        >>> get_note_chord_tone(64, 'Cmaj')
        ('3rd', 0)
        >>> get_note_chord_tone(72, 'Cmaj')
        ('root', 1)

    Note:
        Uses major scale intervals for major chords and minor scale intervals for minor chords.
        The reference octave for the chord root is octave 4 (middle C = 60).
    """
    ## Given chord and a note, wantt o figure out note representation relative to the chord
    octave_4_note_midi_pitch_mapping = {
        'C': 60,
        'C#': 61, 'Db': 61,
        'D': 62,
        'D#': 63, 'Eb': 63,
        'E': 64,
        'F': 65,
        'F#': 66, 'Gb': 66,
        'G': 67,
        'G#': 68, 'Ab': 68,
        'A': 69,
        'A#': 70, 'Bb': 70,
        'B': 71
    }

    chromatic_intervals = {
        0: "root", 
        1: "b2", 
        2: "2nd", 
        3: "b3",   # minor 3rd
        4: "3rd",  # major 3rd
        5: "4th",
        6: "b5",   # tritone
        7: "5th", 
        8: "b6",   # minor 6th
        9: "6th",  # major 6th
        10: "b7",  # minor 7th
        11: "7th"  # major 7th
    }

    try:
        (chord_root_note, chord_type) = get_chord_root_and_type(chord)
        chord_root_midi_pitch = octave_4_note_midi_pitch_mapping[chord_root_note]
        semitone_offset = note_midi_pitch - chord_root_midi_pitch
        pitch_class_offset = semitone_offset % 12
        octave_offset = semitone_offset // 12

        chord_tone = chromatic_intervals[pitch_class_offset]
    except Exception as e:
        return ("root", 0)

    return (chord_tone, octave_offset)

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

def assign_chords_to_bars(bar_timings, chord_timings):
    """
    Assign chord labels to musical bars based on maximum temporal overlap.
    This function determines which chord best represents each bar by calculating
    the overlap duration between each bar's time span and all available chords,
    then selecting the chord with the maximum overlap for that bar.
    Args:
        bar_timings (list of tuple): A list of tuples where each tuple contains
            (bar_start, bar_end) representing the start and end times of each bar
            in the musical piece.
        chord_timings (list of tuple): A list of tuples where each tuple contains
            (chord_start, chord_end, chord_label) representing the start time,
            end time, and label of each chord.
    Returns:
        list: A list of chord labels corresponding to each bar in bar_timings.
            Each element is the chord label that has the maximum overlap with
            the corresponding bar. May contain None if no chord overlaps with a bar.
    """
    bar_chords = []

    for bar_start, bar_end in bar_timings:
        current_max_overlap = 0
        current_best_chord = None
        for chord_start, chord_end, chord_label in chord_timings:
            overlap_start = max(bar_start, chord_start)
            overlap_end = min(bar_end, chord_end)

            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > current_max_overlap:
                current_best_chord = chord_label
                current_max_overlap = overlap_duration
            
        bar_chords.append(current_best_chord)

    return bar_chords

def load_chord_timings(chord_file_path):
    """
    Load chord timings from a chord annotation file.

    This function reads a chord annotation file where each line contains tab-separated
    values representing chord timing information and returns a list of chord timing tuples.

    Args:
        chord_file_path (str): Path to the chord annotation file. The file should contain
            tab-separated values with at least 3 columns: start time, end time, and chord label.

    Returns:
        list of tuple: A list of tuples where each tuple contains:
            - chord_start (float): The start time of the chord in seconds
            - chord_end (float): The end time of the chord in seconds
            - chord_label (str): The label/name of the chord
    """
    chord_timings = []

    with open(chord_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')

            if len(parts) >= 3:
                chord_start = float(parts[0])
                chord_end = float(parts[1])
                chord_label = parts[2]
                chord_timings.append((chord_start, chord_end, chord_label))

    return chord_timings

def load_key(key_file_path):
    keys = []

    with open(key_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')

            if len(parts) >= 3:
                key = parts[2]
                keys.append(key)

    return keys[0]

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

def create_chord_beat_onset_tuple_structure(bars, bars_chords):
    length = min(len(bars), len(bars_chords))

    chord_skeleton_notes_list = []
    for i in range(length):
        current_bar = bars[i]
        current_chord = bars_chords[i]

        beat_notes = {0.0: None, 1.0: None, 2.0: None, 3.0: None}

        for note in current_bar:
            beat_position = note[3]
            if beat_position in beat_notes:
                beat_notes[beat_position] = note[0]

            # If bar is completely empty, skip
            if all(v is None for v in beat_notes.values()):
                continue

            chord_skeleton_notes = (
                current_chord,
                beat_notes[0.0],
                beat_notes[1.0],
                beat_notes[2.0],
                beat_notes[3.0]
            )

        chord_skeleton_notes_list.append(chord_skeleton_notes)

    return chord_skeleton_notes_list

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
    key_root_note, key_type = get_chord_root_and_type(key)
    key_root_note_midi_pitch = 60 + note_to_pitch_class.get(key_root_note, 0)

    # Calculate chord root
    chord_root_note_midi_pitch = key_root_note_midi_pitch + roman_numeral_to_semitones.get(chord_roman_numeral, 0)

    # Calculate final note pitch
    note_midi_pitch = chord_root_note_midi_pitch + chromatic_intervals_inverted.get(chord_tone, 0) + (octave_offset * 12)

    #print(f"for chord type: {chord_roman_numeral}. for chord_tone: {chord_tone}. for key: {key}. generated_midi_pitch: {note_midi_pitch}")

    return note_midi_pitch

def analyze_chromatic_frequency(skeleton_list):
    """Check how often chromatic notes appear on each chord type."""
    chromatic = {'b2', 'b5'}  # Most suspicious ones
    
    chord_chromatic_count = {}
    chord_total_count = {}
    
    for item in skeleton_list:
        chord = item[0]
        notes = [n for n in item[1:] if n is not None]
        
        chord_total_count[chord] = chord_total_count.get(chord, 0) + len(notes)
        
        for note in notes:
            if note in chromatic:
                chord_chromatic_count[chord] = chord_chromatic_count.get(chord, 0) + 1
    
    print("Chromatic (b2, b5) frequency by chord:")
    for chord in chord_total_count:
        count = chord_chromatic_count.get(chord, 0)
        total = chord_total_count[chord]
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {chord}: {count}/{total} ({pct:.1f}%)")

def is_note_on_beat(note_onset_seconds, beat_timings, threshold=0.05):
    for beat_timing in beat_timings:
        beat_onset_seconds = beat_timing[0]
        if abs(beat_onset_seconds - note_onset_seconds) <= threshold:
            return True
    return False

def get_note_at_beat_timing(beat_timing, notes, threshold=0.05):
    notes = sorted(notes, key=lambda i: i['start_seconds'])
    for note in notes:
        if abs(beat_timing - note['start_seconds']) <= threshold:
            return note
        
        if note['start_seconds'] - 1.0 >= beat_timing:
            return None

    return None

def get_closest_note_at_time(target_time, notes, threshold=0.4):
    notes = sorted(notes, key=lambda i: i['start_seconds'])
    current_closest_diff = 1000000
    current_closest_note = None
    for note in notes:
        print(target_time)
        diff = abs(note['start_seconds'] - target_time)
        if diff < current_closest_diff:
            current_closest_diff = diff
            current_closest_note = note
    
    if current_closest_diff < threshold:
        return current_closest_note

    return None

def get_notes_at_bar_onset(beat_timings, notes):
    """
        Gets all the notes at the start of each bar
    """
    bar_onset_notes = []
    for beat_onset_seconds, _,  new_bar_note in beat_timings:
        if new_bar_note == 0:
            continue
            
        target_note = get_closest_note_at_time(beat_onset_seconds, notes)
        bar_onset_notes.append(target_note)

    return notes

def get_ornaments(s1_onset, s2_onset, notes):
    """
        Given 2 skeleton notes, gets all the notes in between them
    """
    # Given the onset in seconds of 2 notes, want to find all the notes that fall in between them  
    notes = sorted(notes, key=lambda i: i['start_seconds'])
    inbetween_notes = [note for note in notes if s1_onset < note['start_seconds'] < s2_onset]
    return inbetween_notes

def get_strong_beat_pairs(notes, beat_timings):
    strong_bar_beats = [beat for beat in beat_timings if beat[1] == 1.0]
    strong_bar_beats = sorted(strong_bar_beats, key=lambda beat: beat[0])

    pairs = []
    for i in range(len(strong_bar_beats)-1):
        curr_beat_timing = strong_bar_beats[i][0]
        next_beat_timing = strong_bar_beats[i+1][0]

        curr_beat_note = get_closest_note_at_time(curr_beat_timing, notes)
        next_beat_note = get_closest_note_at_time(next_beat_timing, notes)

        if curr_beat_note is not None and next_beat_note is not None:
            pairs.append((curr_beat_note, next_beat_note))

    return pairs

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

def get_ornament_groupings(notes, beat_timings):
    """
        given the list of notes, groups them into skeleton notes and ornaments between those skeletons
    """
    # have in format [s1, ornament notes, s2]
    groupings = []
    strong_beat_pairs = get_strong_beat_pairs(notes, beat_timings)

    for s1, s2 in strong_beat_pairs:
        ornament_notes = get_ornaments(s1['start_seconds'], s2['start_seconds'], notes)
        grouping = [s1]
        for note in ornament_notes:
            grouping.append(note)
        grouping.append(s2)
        groupings.append(grouping)

    return groupings

""" 
    given a list of ntoes, we want to define functions that 
    - for each ornament note, determine the interval from the previous note 
    - for each ornament note, determines the note role
    - for each grouping, determine the chord function 
    - for each grouping, determine the chord notes
"""

def ornament_note_role(prev_note_pitch, target_note_pitch, next_note_pitch, chord_tones):
    step_from_prev = abs(target_note_pitch - prev_note_pitch)
    step_to_next = abs(target_note_pitch - next_note_pitch)
    same_direction = (prev_note_pitch < target_note_pitch < next_note_pitch) or (prev_note_pitch > target_note_pitch > next_note_pitch)

    #prev_note_pitch_class = prev_note_pitch % 12
    target_note_pitch_class = target_note_pitch % 12
    next_note_pitch_class = next_note_pitch % 12

    if step_to_next == 1 and next_note_pitch_class in chord_tones and target_note_pitch_class not in chord_tones:
        return "chromatic_approach"
    elif same_direction and step_from_prev <= 2:
        return "passing_tone"
    elif step_from_prev <= 2 and step_to_next <= 2 and not same_direction:
        return "neighbour_tone"
    elif target_note_pitch_class in chord_tones:
        return "chord_tone"
    else:
        return "other"

def determine_chord_function(start_note_timing, chord_timings, song_key):
    try:
        chord = get_matching_chord(start_note_timing, chord_timings)
        if chord == 'N':
            return None
        
        transposed_chord = transpose_chord_to_c_major(chord, song_key)
        chord_function = convert_chord_name_to_roman_numeral(transposed_chord)
        return chord_function
    except Exception as e:
        return None

def prep_groupings_for_second_layer(groupings, chord_timings, song_key):
    '''
        want a note to be in the format note:{
                                                'role': 'blah',
                                                'offset': '+2'
                                            }
    '''

    '''
        want list to be in the format [(s1, s2), [ornament_notes], chord_function]
    '''
    processed_groupings = []
    for grouping in groupings:
        
        ### If only no onrmanet notes inbetween, skip
        if len(grouping) <= 2:
            continue

        s1 = grouping[0]
        s2 = grouping[-1]

        chord = get_matching_chord(s1['start_seconds'], chord_timings)
        chord_function = determine_chord_function(s1['start_seconds'], chord_timings, song_key)
        chord_tone_notes = get_chord_tones(chord)
        print(f'chord_tone_notes: {chord_tone_notes}')

        processed_ornament_notes = []
        for i in range(1, len(grouping)-1):
            prev_note_midi_pitch = grouping[i-1]['pitch']
            curr_note_midi_pitch = grouping[i]['pitch']
            next_note_midi_pitch = grouping[i+1]['pitch']

            note_role = ornament_note_role(prev_note_midi_pitch, curr_note_midi_pitch, next_note_midi_pitch, chord_tone_notes)
            note_offset = curr_note_midi_pitch - prev_note_midi_pitch

            processed_ornament_notes.append((note_role, note_offset))
        
        processed_groupings.append([(s1,s2), processed_ornament_notes, chord_function])

    return processed_groupings

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