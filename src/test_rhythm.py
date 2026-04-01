from Rhythm import *
from ChordFunctions import transpose_chord_to_c_major, transpose_note, calc_semitones_to_c, get_chord_name_in_original_key, get_chord_type_from_roman_numeral, convert_chord_name_to_roman_numeral
from Timings import ChordTiming, KeyTiming
from SongInfo import *

def test_filter_out_beginning_end_rests():
    rhythm_hmm = RhythmHMM()
    sequence = [NOTE_REST, NOTE_REST, NOTE_ONSET, NOTE_REST, NOTE_REST]
    assert rhythm_hmm.filter_out_beginning_end_rests(sequence) == [NOTE_ONSET]

    sequence = [NOTE_ONSET, NOTE_ONSET]
    assert rhythm_hmm.filter_out_beginning_end_rests(sequence) == [NOTE_ONSET, NOTE_ONSET]

def test_convert_to_beats():
    rhythm_hmm = RhythmHMM()
    sequence = [NOTE_ONSET, NOTE_ONSET]
    assert rhythm_hmm.convert_to_num_beats(sequence) == [0,0]

    sequence = [NOTE_ONSET, NOTE_CONTINUE, NOTE_CONTINUE, NOTE_CONTINUE, NOTE_REST, NOTE_REST, NOTE_ONSET]
    assert rhythm_hmm.convert_to_num_beats(sequence) == [2, 8, 0]

def test_postprocess_rhythm_sequence():
    rhythm_hmm = RhythmHMM()
    sequence = [NOTE_REST]
    assert rhythm_hmm.postprocess_rhythm_sequence(sequence) == [NOTE_ONSET]

def test_semitones_to_c():
    assert calc_semitones_to_c('C') == 0
    assert calc_semitones_to_c('C#') == 11
    assert calc_semitones_to_c('B') == 1

def test_transpose_note():
    assert transpose_note('C#', 11) == 'C'
    assert transpose_note('B', 1) == 'C'
    assert transpose_note('C#', 1) == 'D'

def test_transpose_chord_to_c_major():
    c_major_chord = ChordTiming(0,0, 'C:maj')
    a_major_chord = ChordTiming(0,0, 'A:maj')

    c_major_key = KeyTiming('C:maj')
    g_major_key = KeyTiming('G:maj')
    b_major_key = KeyTiming('B:maj')

    assert transpose_chord_to_c_major(c_major_chord, c_major_key) == "C:maj"
    assert transpose_chord_to_c_major(c_major_chord, g_major_key) == "F:maj"
    assert transpose_chord_to_c_major(a_major_chord, b_major_key) == "A#:maj"

def test_get_chord_name_in_original_key():
    c_major_key = KeyTiming('C:maj')
    g_major_key = KeyTiming('G:maj')

    assert get_chord_name_in_original_key('V', c_major_key) == "G:maj"
    assert get_chord_name_in_original_key('V', g_major_key) == "D:maj"

def test_get_chord_type_from_roman_numeral():
    assert get_chord_type_from_roman_numeral('I') == 'maj'
    assert get_chord_type_from_roman_numeral('ii') == 'min'
    
def test_convert_chord_name_to_roman_numera():
    assert(convert_chord_name_to_roman_numeral('C:maj')) == 'I'
    assert(convert_chord_name_to_roman_numeral('D:min')) == 'ii'
    assert(convert_chord_name_to_roman_numeral('F:maj')) == 'IV'
    assert(convert_chord_name_to_roman_numeral('G:maj')) == 'V'

def test_load_song_info():
    key_file_path = './POP909/001/key_audio.txt'
    midi_file_path = './POP909/001/001.mid'
    beat_file_path = './POP909/001/beat_midi.txt'
    chord_file_path = './POP909/001/chord_midi.txt'

    song_info = SongInfo(key_file_path, midi_file_path, chord_file_path, beat_file_path)
    assert song_info.song_key.get_name() == 'Gb:maj'

    assert song_info.chord_timings[4].chord_name == 'B:maj'
    assert song_info.chord_timings[4].chord_start == 2.721993
    assert song_info.chord_timings[4].chord_end == 4.055323

    assert song_info.beat_timings[0].beat_time == 0.05533319499999998
    assert song_info.beat_timings[0].new_bar == True
    assert song_info.beat_timings[0].strong_beat == True