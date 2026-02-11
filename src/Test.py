from mido import MidiFile

def load_beat(beat_file_path):
    beats = []

    with open(beat_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')

            if len(parts) >= 3:
                beat_time = parts[0]
                beat_strong_beat = parts[1]
                beat_new_bar = parts[2]

                beats.append((beat_time, beat_strong_beat, beat_new_bar))

    return beats

def note_which_chord(note_onset, chord_timings):
    for chord_start, chord_end, chord_label in chord_timings:
        if chord_start <= note_onset <= chord_end:
            return chord_label

def load_midi(midi_path):
    midi = MidiFile(midi_path)

    ticks_per_beat = midi.ticks_per_beat
    beats_per_bar = 4
    ticks_per_bar = ticks_per_beat * beats_per_bar

    notes = []
    active_notes = {}
    current_tick = 0

    for msg in midi.tracks[3]:
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
                curr_note['clef'] = 'treble' if msg.note >= 60 else 'bass'

                notes.append(curr_note)
                del active_notes[msg.note]

    notes = sorted(notes, key=lambda note: (note['bar_index'], note['beat_onset']))
    
    return notes

if __name__ == "__main__":
    chord_timings = [(0,2,'F:maj'), ((2,4,'C:maj'))]
    print(note_which_chord(5.5, chord_timings))