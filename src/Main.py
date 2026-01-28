import mido
from mido import MidiFile, MidiTrack, Message

def save_to_midi(bars, output_path='output.mid', tempo=500000, ticks_per_beat=480):
    duration_to_beats_map = {
        'semiquaver': 0.25,
        'quaver': 0.50,
        'dotted_quaver': 0.75,
        'crotchet': 1.0,
        'dotted_crotchet': 1.5,
        'minim': 2.0,
        'dotted_minim': 3,
        'semibreve': 4
    }

    # for each note, we need to track when to send the note_on 
    # for each note, we send out a note_on and note_off message
    # note_off should be sent with a time_delta equal to note_on + note_duration in ticks

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    current_tick = 0
    bar_length_beats = 4
    note_events = []

    for bar_index, bar in enumerate(bars):
        bar_start_beat = bar_index * bar_length_beats

        for note in bar:
            note_pitch, note_duration, note_bar_onset = note

            # Tracks where in the bar the note should appear
            beat_onset = bar_start_beat + note_bar_onset
            bar_tick_onset = int(ticks_per_beat * beat_onset)

            # Calculate duration to ticks
            duration_beats = duration_to_beats_map.get(note_duration, 1.0)
            duration_ticks = int(ticks_per_beat * duration_beats)

            # Store note_on and note_off events
            note_events.append(('note_on', bar_tick_onset, note_pitch))
            note_events.append(('note_off', bar_tick_onset + duration_ticks, note_pitch))

    # Sort all events by time
    note_events.sort(key=lambda x: x[1])
    
    # Write events with correct delta times
    current_tick = 0
    for event_type, event_tick, note_pitch in note_events:
        delta_time = event_tick - current_tick
        
        if delta_time < 0:
            print(f"Warning: negative delta time {delta_time}, setting to 0")
            delta_time = 0
        
        if event_type == 'note_on':
            track.append(Message('note_on', note=note_pitch, velocity=64, time=delta_time))
        else:
            track.append(Message('note_off', note=note_pitch, velocity=64, time=delta_time))
        
        current_tick = event_tick

    mid.save(output_path)