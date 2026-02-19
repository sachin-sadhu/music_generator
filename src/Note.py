from Helper import *
from ChordFunctions import *
from BeatTiming import BeatTiming
from SongInfo import *

class TrainingNote:
    def __init__(self, midi_pitch, clef, duration, start_seconds):
        self.midi_pitch = midi_pitch
        self.clef = clef
        self.duration = duration
        self.pitch_class = midi_pitch % 12
        self.start_seconds = start_seconds

        # Optional attributes
        self.chord_function = None
        self.chord_tone = None
        self.octave_offset = None

    def get_chord(self, song_info):
        chord_timings = song_info.chord_timings
        return get_event_matching_chord(self.start_seconds, chord_timings)

    def get_midi_pitch(self):
        return self.midi_pitch

    def get_chord_function(self):
        return self.chord_function

    def get_duration(self):
        return self.duration

    def get_start_seconds(self):
        return self.start_seconds

    def set_original_chord(self, song_info):

    def set_chord_tone_octave_offset(self):
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
            (chord_root_note, _) = get_chord_root_and_type(self.original_chord)
            chord_root_midi_pitch = octave_4_note_midi_pitch_mapping[chord_root_note]
            semitone_offset = self.midi_pitch - chord_root_midi_pitch
            pitch_class_offset = semitone_offset % 12
            self.chord_tone = chromatic_intervals[pitch_class_offset]
            self.octave_offset = semitone_offset // 12
        except Exception:
            self.chord_tone = 'root'
            self.octave_offset = 0


    def is_chord_triad(self, chord_name):
        """
        Determines if a given note pitch is a chord tone of the specified chord.

        Args:
            note_pitch (int): The MIDI pitch number of the note to check.
            chord_name (str): The name of the chord in the format 'Root:quality' 
                            (e.g., 'C:maj', 'A:min', 'G:7').

        Returns:
            bool: True if the note pitch is a chord tone of the specified chord, 
                False otherwise.

        Raises:
            ValueError: If the chord_name is not found in the CHORD_TEMPLATES dictionary.

        Note:
            The function uses pitch class (pitch % 12) to determine if a note belongs
            to a chord, making it octave-invariant.
        """
        CHORD_TEMPLATES = {
            'C:maj': [0, 4, 7],
            'C:min': [0, 3, 7],
            'D:min': [2, 5, 9],
            'D:maj': [2, 6, 9],
            'E:min': [4, 7, 11],
            'E:maj': [4, 8, 11],
            'F:maj': [5, 9, 0],
            'F:min': [5, 8, 0],
            'G:maj': [7, 11, 2],
            'G:min': [7, 10, 2],
            'A:min': [9, 0, 4],
            'A:maj': [9, 1, 4],
            'B:min': [11, 2, 6],
            'B:maj': [11, 3, 6],
            'B:dim': [11, 2, 5],
            'G:7': [7, 11, 2, 5]
        }

        if chord_name not in CHORD_TEMPLATES:
            raise ValueError(f"Invalid chord name: {chord_name} ")

        return self.pitch_class in CHORD_TEMPLATES[chord_name]

    def is_note_on_beat(self, beat_timings: list[BeatTiming], threshold=0.05) -> bool:
        for beat in beat_timings:
            if abs(beat.get_onset_time() - self.start_seconds) <= threshold:
                return True
        return False

    def set_chord_function(self, song_info: SongInfo):
        if self.original_chord is None:
            self.chord_function = None
        else:
            transposed_chord = transpose_chord_to_c_major(self.original_chord, song_info.song_key)
            self.chord_function = convert_chord_name_to_roman_numeral(transposed_chord)

class GeneratedNote:
    def __init__(self, chord_tone, chord_function, octave_offset):
        self.chord_tone = chord_tone
        self.chord_function = chord_function
        self.octave_offset = octave_offset

    def get_note_midi_pitch(self, song_key):
        chromatic_intervals_inverted = {
            "root": 0, "b2": 1, "2nd": 2, "b3": 3, "3rd": 4, "4th": 5,
            "b5": 6, "5th": 7, "b6": 8, "6th": 9, "b7": 10, "7th": 11, "octave": 12
        }

        chord_function_to_semitones = {
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

        key_root_note, _ = get_key_root_and_type(song_key)
        key_root_note_midi_pitch = 60 + note_to_pitch_class.get(key_root_note, 0)

        chord_root_note_midi_pitch = key_root_note_midi_pitch + chord_function_to_semitones.get(self.chord_function, 0)
        note_midi_value = chord_root_note_midi_pitch + chromatic_intervals_inverted.get(self.chord_tone, 0) + (self.octave_offset * 12)

        return note_midi_value
