class BeatTiming:
    def __init__(self, beat_time, beat_strong_beat, beat_new_bar):
        self.beat_time = beat_time
        self.strong_beat: bool = float(beat_strong_beat) == 1.0
        self.new_bar: bool = float(beat_new_bar) == 1.0
        
    def get_onset_time(self):
        return self.beat_time

    def is_strong_beat(self):
        return self.strong_beat

    def is_new_bar(self):
        return self.new_bar