from Note import Note

class OrnamentNote:
    def __init__(self, note_role, note_offset, note_duration):
        self.note_role = note_role
        self.note_offset = note_offset
        self.note_duration = note_duration

class OrnamentGrouping:
    def __init__(self, first_skeleton_note: Note, second_skeleton_note: Note, ornament_notes, grouping_chord_function: str):
        self.first_skeleton_note = first_skeleton_note
        self.second_skeleton_note = second_skeleton_note
        self.ornament_notes = ornament_notes
        self.grouping_chord_function = grouping_chord_function
        