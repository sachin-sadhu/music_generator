from ChordFunctions import *
from BeatTiming import BeatTiming

class Note:
    def __init__(self, midi_pitch, clef, duration, start_seconds):
        self.midi_pitch = midi_pitch
        self.clef = clef
        self.duration = duration
        self.pitch_class = midi_pitch % 12
        self.start_seconds = start_seconds

    def get_midi_pitch(self):
        return self.midi_pitch

    def get_duration(self):
        return self.duration

    def get_start_seconds(self):
        return self.start_seconds

    def get_note_chord_tone(self, chord):
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
            semitone_offset = self.midi_pitch - chord_root_midi_pitch
            pitch_class_offset = semitone_offset % 12
            octave_offset = semitone_offset // 12
            chord_tone = chromatic_intervals[pitch_class_offset]
        except Exception:
            return ("root", 0)

        return (chord_tone, octave_offset)

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