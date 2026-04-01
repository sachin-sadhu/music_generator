NOTE_ONSET=0
NOTE_CONTINUE=1
NOTE_REST=2

# Note lengths
SEMIQUAVER_LENGTH=1
QUAVER_LENGTH=2
CROTCHET_LENGTH=4
DOTTED_CROTCHET_LENGTH=6
MINIM_LENGTH=8
DOTTED_MINIM_LENGTH=12
SEMIBREVE_LENGTH=16

# Note/Rest duration mapping
DEFAULT_NOTE=0
NOTE_SEMIQUAVER=0
NOTE_QAUVER=1
NOTE_CROTCHET=2
NOTE_DOTTED_CROTCHET=3
NOTE_MINIM=4
NOTE_DOTTED_MINIM=5
NOTE_SEMIBREVE=6

REST_SEMIQUAVER=7
REST_QAUVER=8
REST_CROTCHET=9
REST_DOTTED_CROTCHET=10
REST_MINIM=11
REST_DOTTED_MINIM=12
REST_SEMIBREVE=13

# Note pitch class
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

SCALE_DEGREE_NOTE_NAME_MAPPING = {
    0: "root", 
    1: "b2", 
    2: "2nd", 
    3: "b3", 
    4: "3rd",
    5: "4th",
    6: "b5",
    7: "5th", 
    8: "b6",   
    9: "6th",  
    10: "b7",  
    11: "7th"  
}

NOTE_NAME_TO_SCALE_DEGREE_MAPPING = {
    "root": 0, "b2": 1, "2nd": 2, "b3": 3, "3rd": 4, "4th": 5,
    "b5": 6, "5th": 7, "b6": 8, "6th": 9, "b7": 10, "7th": 11, "octave": 12
}



pitch_class_to_note = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
    7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
} 

roman_numeral_to_semitones = {
        'I': 0,   # Tonic (0 semitones above root)
        'ii': 2,  # 2 semitones above root
        'iii': 4, # 4 semitones above root
        'IV': 5,  # 5 semitones above root
        'V': 7,   # 7 semitones above root
        'vi': 9,  # 9 semitones above root
        'vii': 11 # 11 semitones above root
}

root_to_degree = {
        'C': 'I',
        'D': 'ii',
        'E': 'iii',
        'F': 'IV',
        'G': 'V',
        'A': 'vi',
        'B': 'vii',
}

# MIDI Processing
MIDI_TRACK=1

# KEY FILE
NUM_KEY_PARTS=3
KEY_NAME_INDEX=2

# CHORD FILE
NUM_CHORD_PARTS=3
CHORD_START_INDEX=0
CHORD_END_INDEX=1
CHORD_LABEL_INDEX=2

# BEAT FILE
NUM_BEAT_PARTS=3
BEAT_TIME_INDEX=0
BEAT_STRONG_BEAT_INDEX=1
BEAT_NEW_BAR_INDEX=2


TREBLE_MIDI_PITCH=60
BPM_120_TICKS=500000

METRONOME_80_BPM=80
MINIM_DURATION_CROTCHET=2
STEPS_PER_BEAT=4
OCTAVE_SEMITONES=12      
KEY_SCALES = {
            'A:maj': [9, 11, 1, 2, 4, 6, 8],      # A, B, C#, D, E, F#, G#
            'C:maj': [0, 2, 4, 5, 7, 9, 11],      # C, D, E, F, G, A, B
            'D:maj': [2, 4, 6, 7, 9, 11, 1],      # D, E, F#, G, A, B, C#
            'G:maj': [7, 9, 11, 0, 2, 4, 6],      # G, A, B, C, D, E, F#
}  

TRITONES = {
            'A:maj': 3,   # D#/Eb
            'C:maj': 6,   # F#/Gb  
            'D:maj': 8,   # G#/Ab
            'G:maj': 1,   # C#/Db
        }