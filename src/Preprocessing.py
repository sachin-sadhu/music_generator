from mido import MidiFile
from ChordFunctions import *
import os

def load_songs(directory):
    songs_bars = []
    songs_bars_chords_roman = []

    for song_dir in sorted(os.listdir(directory)):
        song_path = os.path.join(directory, song_dir)

        if not os.path.isdir(song_path):
            continue
    
        midi_file = os.path.join(song_path, f"{song_dir}.mid")
        chord_file = os.path.join(song_path, "chord_midi.txt")
        key_file = os.path.join(song_path, "key_audio.txt")

        if all(os.path.exists(file) for file in [midi_file, chord_file, key_file]):
            try:
                song_key = load_key(key_file)[0]
                notes = load_midi(midi_file)
                bars = group_notes_by_bar(notes)
                bar_timings = get_all_bar_timings(bars, 120)
                chord_timings = load_chord_timings(chord_file)
                #bars_chords_mapped = assign_chords_to_bars(bar_timings, chord_timings)
                notes_chord_assigned = assign_chord_to_notes(notes, chord_timings)

                if get_chord_root_and_type(song_key)[1] == 'min':
                    continue

                for note in notes:
                    note_cooresponding_chord = note_which_chord(note['start_seconds'], chord_timings)
                    note_chord_tone = get_note_chord_tone(note['pitch'], note_cooresponding_chord)
                    transposed_chord = (note_cooresponding_chord, song_key)
                    chord_function = convert_chord_name_to_roman_numeral(transposed_chord)

                    # Add new attributes to note dict
                    note['chord_tone'] = note_chord_tone
                    note['chord_function'] = chord_function

                for i in range(len(bars)):
                    try:
                        current_bar = bars[i]
                        current_bar_chord = bars_chords_mapped[i]
                        current_bar_formatted = []

                        if current_bar_chord == 'N':
                            songs_bars_chords_roman.append('N')
                            songs_bars.append([])
                            continue

                        # Convert chord to its roman numeral form
                        chord_in_c = transpose_chord_to_c_major(current_bar_chord, song_key)
                        chord_roman = convert_chord_name_to_roman_numeral(chord_in_c)
                        songs_bars_chords_roman.append(chord_roman)

                        # Format notes into the HSMM learning format: (chord_tone, octave_offset, note_duration, beat onset)
                        for note in current_bar:
                            note_midi_pitch = note['pitch']
                            note_type = note['note_type']
                            note_onset = note['beat_onset']
                            
                            note_chord_tone, octave_offset = get_note_chord_tone(note_midi_pitch, current_bar_chord)
                            chord_in_c = transpose_chord_to_c_major(current_bar_chord, song_key)
                            chord_roman = convert_chord_name_to_roman_numeral(chord_in_c)
                            note_formatted = (note_chord_tone, octave_offset, note_type, note_onset)
                            current_bar_formatted.append(note_formatted)

                        songs_bars.append(current_bar_formatted)
                    except Exception as e:
                        songs_bars_chords_roman.append('I')
                        current_bar_formatted.append(('root', 0, 'semibreve', 0.0))
                        songs_bars.append(current_bar_formatted)
                        continue
            except Exception as e:
                print(f"Error loading song {song_dir}: {e}")
                raise e
        else:
            print(f"Missing file for song {song_dir}")

    return songs_bars, songs_bars_chords_roman

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

def load_midi(midi_path):

    midi = MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    tempo = get_tempo_from_midi(midi)

    notes = []
    active_notes = {}
    current_tick = 0

    for msg in midi.tracks[3]:
        current_tick += msg.time
        current_seconds = ticks_to_seconds(current_tick, ticks_per_beat, tempo)

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
    
    return notes

def load_beat(beat_file_path):
    print("hello")
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

def note_which_chord(note_onset, chord_timings):
    for chord_start, chord_end, chord_label in chord_timings:
        if chord_start <= note_onset <= chord_end:
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

    return keys

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

    print(f"for chord type: {chord_roman_numeral}. for chord_tone: {chord_tone}. for key: {key}. generated_midi_pitch: {note_midi_pitch}")

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



if __name__ == "__main__":
    directory = '/cs/home/slzys1/Documents/music_generator/short_test_data/001/001.mid'
    chord_directory = '/cs/home/slzys1/Documents/music_generator/short_test_data/001/chord_midi.txt'
    beat_directory = '/cs/home/slzys1/Documents/music_generator/short_test_data/001/beat_midi.txt'
    beat_timings = load_beat(beat_directory)
    bars = split_beat_timings_to_bars(beat_timings)
    print(bars[0:5])