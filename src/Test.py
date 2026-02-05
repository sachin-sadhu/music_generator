from mido import MidiFile
from ChordFunctions import *
from Preprocessing import *
import os

def load_midi(midi_path):
    midi = MidiFile(midi_path)

    ticks_per_beat = midi.ticks_per_beat
    beats_per_bar = 4
    ticks_per_bar = ticks_per_beat * beats_per_bar

    notes = []
    active_notes = {}
    current_tick = 0

    for msg in midi.tracks[3]:
        print(msg)
        current_tick += msg.time

        # Note is being played
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = {
                'pitch': msg.note,
                'start_tick': current_tick,
                'velocity': msg.velocity
            }
        
        # Note is being turned off
        elif msg.type == 'note_on' and msg.velocity == 0:
            if msg.note in active_notes:
                curr_note = active_notes[msg.note]

                tick_onset = curr_note['start_tick']
                tick_duration = current_tick - tick_onset
                beat_duration = tick_duration / ticks_per_beat
                (bar_index, beat_position) = calculate_beat_position(ticks_per_bar, ticks_per_beat, tick_onset)
                quantised_beat_position = quantise_beat_position(beat_position)
                note_type = quantise_beat_duration(beat_duration)

                curr_note['tick_duration'] = tick_duration
                curr_note['beat_duration'] = beat_duration
                curr_note['beat_onset'] = quantised_beat_position
                curr_note['bar_index'] = bar_index
                curr_note['note_type'] = note_type

                notes.append(curr_note)
                del active_notes[msg.note]

    sorted_notes = sorted(notes, key=lambda note: (note['bar_index'], note['beat_onset']))
    return sorted_notes

def split_by_clef(notes, split_pitch=60):
    treble_clef = []
    bass_clef = []

    for note in notes:
        if note['pitch'] >= split_pitch:
            treble_clef.append(note)
        else:
            bass_clef.append(note)

    return treble_clef, bass_clef

if __name__ == "__main__":
    path = "/home/sachin/Documents/music_generator/POP909/POP909/001/001.mid"
    notes = load_midi(path)
    treble_clef, bass_clef = split_by_clef(notes)
    print(f'treble clef: {treble_clef[0:3]}')
    print('\n')
    print(f'bass clef: {bass_clef[0:3]}')