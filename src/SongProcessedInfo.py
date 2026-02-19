from Note import *
from OrnamentGroupings import *

class TrainingDataProcessedInfo:
    def __init__(self, notes: list[list[TrainingNote]], beat_chords: list[list[str]], ornament_groupings: list[list[OrnamentGrouping]]):
        self.notes = notes
        self.beat_chords = beat_chords
        self.ornament_groupings = ornament_groupings