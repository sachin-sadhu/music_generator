from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Note import TrainingNote

from ChordFunctions import get_event_matching_chord, transpose_chord_to_c_major, convert_chord_name_to_roman_numeral
from abc import ABC, abstractmethod

class KeyChordTimingAbstractClass(ABC):
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_root_note(self):
        pass

    @abstractmethod
    def get_type(self):
        pass

class KeyTiming(KeyChordTimingAbstractClass):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_root_note(self):
        parts = self.name.split(':')
        if len(parts) >= 2:
            root_note = parts[0]
            return root_note
        else:
            raise ValueError(f"Invalid format: {self.name}")

    def get_type(self):
        parts = self.name.split(':')
        if len(parts) >= 2:
            type_t = parts[1]
            return type_t
        else:
            raise ValueError(f"Invalid format: {self.name}")

class ChordTiming(KeyChordTimingAbstractClass):
    def __init__(self, chord_start, chord_end, chord_name):
        self.chord_start = float(chord_start)
        self.chord_end = float(chord_end)
        self.chord_name = chord_name
        
    def get_chord_start(self):
        return self.chord_start

    def get_chord_end(self):
        return self.chord_end

    def get_name(self):
        return self.chord_name

    def get_root_note(self):
        parts = self.chord_name.split(':')
        if len(parts) >= 2:
            root_note = parts[0]
            return root_note
        else:
            raise ValueError(f"Invalid chord format: {self.chord_name}")

    def get_type(self):
        parts = self.chord_name.split(':')
        if len(parts) >= 2:
            chord_type = parts[1]
            return chord_type
        else:
            raise ValueError(f"Invalid chord format: {self.chord_name}")

    def get_function(self, song_key: KeyTiming) -> str:
        try:
            if self.chord_name == 'N':
                return 'N'
            
            transposed_chord = transpose_chord_to_c_major(self, song_key)
            chord_function = convert_chord_name_to_roman_numeral(transposed_chord)
            return chord_function
        except Exception:
            return 'N'

class BeatTiming:
    def __init__(self, beat_time, beat_strong_beat, beat_new_bar):
        self.beat_time = beat_time
        self.strong_beat: bool = float(beat_strong_beat) == 1.0
        self.new_bar: bool = float(beat_new_bar) == 1.0
        
    def get_onset_time(self):
        return self.beat_time

    def is_strong_beat(self):
        return self.new_bar

    def is_new_bar(self):
        return self.new_bar

    def get_matching_chord(self, chord_timings: list[ChordTiming]):
        return get_event_matching_chord(self.beat_time, chord_timings)

    def get_closest_note(self, notes: list[TrainingNote], threshold=0.4) -> TrainingNote | None:
        notes = sorted(notes, key=lambda note: note.get_start_seconds())
        current_closest_diff = 1000000
        current_closest_note = None
        for note in notes:
            diff = abs(note.get_start_seconds() - self.beat_time)
            if diff < current_closest_diff:
                current_closest_diff = diff
                current_closest_note = note
        
        if current_closest_diff < threshold:
            return current_closest_note

        return None