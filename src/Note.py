from Timings import ChordTiming, KeyTiming
from config import SCALE_DEGREE_NOTE_NAME_MAPPING, OCTAVE_4_NOTE_MIDI_PITCH_MAPPING

"""
    Class that contains information about a note loaded from the POP909 dataset.
"""
class TrainingNote:
    def __init__(self, midi_pitch, clef, duration, start_seconds, chord: ChordTiming, key: KeyTiming):
        self.midi_pitch = midi_pitch
        self.clef = clef
        self.duration = duration
        self.pitch_class = midi_pitch % 12
        self.start_seconds = start_seconds
        self.chord = chord
        self.song_key = key

    def get_chord(self):
        return self.chord

    def get_midi_pitch(self):
        return self.midi_pitch

    def get_chord_function(self):
        return self.chord.get_function(self.song_key)

    def get_duration(self):
        return self.duration

    def get_start_seconds(self):
        return self.start_seconds

    def get_chord_tone(self):
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
  

        try:
            chord_root_note = self.chord.get_root_note()
            chord_root_midi_pitch = OCTAVE_4_NOTE_MIDI_PITCH_MAPPING[chord_root_note]
            semitone_offset = self.midi_pitch - chord_root_midi_pitch
            pitch_class_offset = semitone_offset % 12
            chord_tone = SCALE_DEGREE_NOTE_NAME_MAPPING[pitch_class_offset]
            return chord_tone
        except Exception:
            return 'root'

class OrnamentNote:
    def __init__(self, note_offset):
        self.offset = note_offset

"""
    Class used to represent an ornament grouping. Contains both skeelton notes and 
    list of ornament notes between them.
"""
class OrnamentGrouping:
    def __init__(self, first_skeleton_note: TrainingNote , second_skeleton_note: TrainingNote, ornament_notes: list[OrnamentNote]):
        self.first_skeleton_note = first_skeleton_note
        self.second_skeleton_note = second_skeleton_note
        self.ornament_notes = ornament_notes

    def get_group_note_interval(self):
        return self.first_skeleton_note.get_midi_pitch() - self.second_skeleton_note.get_midi_pitch()

"""
    Class that will act as ornament emission
"""
class OrnamentEmission:
    def __init__(self, offset):
        self.offset = offset