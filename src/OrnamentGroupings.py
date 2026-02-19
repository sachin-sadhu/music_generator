from Note import *

class OrnamentNote:
    def __init__(self, note_role, note_offset, note_duration):
        self.role = note_role
        self.offset = note_offset
        self.duration = note_duration

class OrnamentGrouping:
    def __init__(self, first_skeleton_note: TrainingNote , second_skeleton_note: TrainingNote, ornament_notes: list[OrnamentNote], chord_function: str):
        self.first_skeleton_note = first_skeleton_note
        self.second_skeleton_note = second_skeleton_note
        self.ornament_notes = ornament_notes
        self.chord_function = chord_function
        