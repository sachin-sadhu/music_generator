from mido import MidiFile, tick2second
from ChordTiming import ChordTiming
from BeatTiming import BeatTiming
from Helper import *
from Note import Note

class SongInfo:
    def __init__(self, key_file_path, midi_file_path, chord_file_path, beat_file_path):
        self.song_key = self.load_key(key_file_path)
        self.chord_timings = self.load_chord_timings(chord_file_path)
        self.notes = self.load_midi_notes(midi_file_path)
        self.beat_timings = self.load_beat_timings(beat_file_path) 

    def load_key(self, key_file_path):
        keys = []

        with open(key_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')

                if len(parts) >= 3:
                    key = parts[2]
                    keys.append(key)

        return keys[0]

    def load_chord_timings(self, chord_file_path):
        chord_timings = []

        with open(chord_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')

                if len(parts) >= 3:
                    chord_start = parts[0]
                    chord_end = parts[1]
                    chord_label = parts[2]

                    chord = ChordTiming(chord_start, chord_end, chord_label)
                    chord_timings.append(chord)

        return chord_timings

    def load_midi_notes(self, midi_path):
        midi = MidiFile(midi_path)
        ticks_per_beat = midi.ticks_per_beat
        tempo = self.get_tempo_from_midi(midi)

        notes = []
        active_notes = {}
        current_tick = 0

        for msg in midi.tracks[3]:
            current_tick += msg.time
            #current_seconds = ticks_to_seconds(current_tick, ticks_per_beat, tempo)
            current_seconds = tick2second(current_tick, ticks_per_beat, tempo)

            # Note is being played
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {
                    'pitch': msg.note,
                    'start_tick': current_tick,
                    'start_seconds': current_seconds
                }
            
            # Note is being turned off
            elif msg.type == 'note_on' and msg.velocity == 0:
                if msg.note in active_notes:
                    curr_note = active_notes[msg.note]

                    seconds_onset = curr_note['start_seconds']
                    tick_onset = curr_note['start_tick']
                    tick_duration = current_tick - tick_onset
                    beat_duration = tick_duration / ticks_per_beat
                    note_duration = quantise_beat_duration(beat_duration)
                    clef = 'treble' if msg.note >= 60 else 'bass'

                    note = Note(msg.note, clef, note_duration, seconds_onset)
                    notes.append(curr_note)
                    del active_notes[msg.note]

        return notes

    def load_beat_timings(self, beat_file_path):
        beats = []
        with open(beat_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(' ')

                if len(parts) >= 3:
                    beat_time = float(parts[0])
                    beat_strong_beat = float(parts[1])
                    beat_new_bar = float(parts[2])

                    beat = BeatTiming(beat_time, beat_strong_beat, beat_new_bar)
                    beats.append(beat)

        return beats

    def get_tempo_from_midi(self, midi):
        """
        Extract tempo from MIDI file (microseconds per beat).
        Default to 120 BPM if not found.
        """
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return msg.tempo
        return 500000  # Default: 120 BPM
