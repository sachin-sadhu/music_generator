class ChordTiming:
    def __init__(self, chord_start, chord_end, chord_name):
        self.chord_start = float(chord_start)
        self.chord_end = float(chord_end)
        self.chord_name = chord_name
        
    def get_chord_start(self):
        return self.chord_start

    def get_chord_end(self):
        return self.chord_end

    def get_chord_name(self):
        return self.chord_name
