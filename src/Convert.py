import mido
from mido import MidiFile, MidiTrack, Message

notes = [{'pitch': 61, 'start_time': 1, 'duration': 1, 'velocity': 80, 'melodic_state': 'ascending_step'}, ...]

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM

ticks_per_beat = 480
current_time = 0

for note in sorted(notes, key=lambda x: x['start_time']):
    note_start = int(note['start_time'] * ticks_per_beat)
    delta = note_start - current_time
    
    track.append(Message('note_on', note=note['pitch'], velocity=note['velocity'], time=delta))
    
    duration = int(note['duration'] * ticks_per_beat)
    track.append(Message('note_off', note=note['pitch'], velocity=0, time=duration))
    
    current_time = note_start + duration

mid.save('output.mid')