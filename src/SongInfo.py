from mido import MidiFile, tick2second
from Helper import quantise_beat_duration
from Note import TrainingNote, OrnamentGrouping, OrnamentNote
from ChordFunctions import *
from Timings import BeatTiming, ChordTiming, KeyTiming
import os

class SongInfo:
    def __init__(self, key_file_path, midi_file_path, chord_file_path, beat_file_path):
        self.song_key = self.load_key(key_file_path)
        self.chord_timings = self.load_chord_timings(chord_file_path)
        self.beat_timings = self.load_beat_timings(beat_file_path) 
        self.notes = self.load_midi_notes(midi_file_path)

    def load_key(self, key_file_path) -> KeyTiming:
        with open(key_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')

                if len(parts) >= 3:
                    key_name = parts[2]
                    return KeyTiming(key_name)

        raise ValueError("Song has invalid key.")

    def load_chord_timings(self, chord_file_path) -> list[ChordTiming]:
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

    def load_midi_notes(self, midi_path) -> list[TrainingNote]:
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
                    chord = get_event_matching_chord(seconds_onset, self.chord_timings)

                    if clef == 'treble':
                        note = TrainingNote(msg.note, clef, note_duration, seconds_onset, chord, self.song_key)
                        notes.append(note)
                    del active_notes[msg.note]

        return notes

    def load_beat_timings(self, beat_file_path) -> list[BeatTiming]:
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

class TrainingDataProcessedInfo:
    def __init__(self):
        self.notes: list[TrainingNote] = []
        self.beat_chords: list[list[str]] = []
        self.ornament_groupings: list[OrnamentGrouping] = []

    def load_training_data(self, directory: str) -> None:
        all_song_notes = []
        all_song_beat_chords = []
        all_song_ornament_groupings = []

        for song_dir in sorted(os.listdir(directory)):
            song_path = os.path.join(directory, song_dir)

            if not os.path.isdir(song_path):
                continue
        
            midi_file = os.path.join(song_path, f"{song_dir}.mid")
            chord_file = os.path.join(song_path, "chord_midi.txt")
            key_file = os.path.join(song_path, "key_audio.txt")
            beat_file = os.path.join(song_path, "beat_midi.txt")

            if all(os.path.exists(file) for file in [midi_file, chord_file, key_file, beat_file]):
                try:
                    song_info = SongInfo(key_file, midi_file, chord_file, beat_file)

                    #bars = group_notes_by_bar(notes)
                    #bar_timings = get_all_bar_timings(bars, 120)
                    #bars_chords_mapped = assign_chords_to_bars(bar_timings, chord_timings)
                    #notes_chord_assigned = assign_chord_to_notes(notes, chord_timings)

                    # Skip songs in minor keys for now
                    if song_info.song_key.get_type() == 'min':
                        continue

                    groupings = self.create_ornament_groupings(song_info)
                    all_song_ornament_groupings.extend(groupings)

                    # Process beat chords association
                    beats_chords_function_list = []
                    for beat in song_info.beat_timings:
                        try:
                            chord = beat.get_matching_chord(song_info.chord_timings)
                            if chord.get_name() == 'N':
                                beats_chords_function_list.append('N')
                            else:
                                transposed_chord = transpose_chord_to_c_major(chord, song_info.song_key)
                                chord_function = convert_chord_name_to_roman_numeral(transposed_chord)
                                beats_chords_function_list.append(chord_function)
                        except Exception:
                            beats_chords_function_list.append('N')

                    all_song_beat_chords.append(beats_chords_function_list)
                    all_song_notes.extend(song_info.notes)

                except Exception as e:
                    print(f"Error{song_dir}: {e}")
                    continue
            else:
                print(f"Missing file for song {song_dir}")

        self.notes = all_song_notes
        self.beat_chords = all_song_beat_chords
        self.ornament_groupings = all_song_ornament_groupings

    def create_ornament_groupings(self, song_info: SongInfo) -> list[OrnamentGrouping]:
        '''
            want a note to be in the format note:{
                                                    'role': 'blah',
                                                    'offset': '+2',
                                                    'note_duration': 'quaver'
                                                }
        '''

        '''
            want list to be in the format [(s1, s2), [ornament_notes], chord_function]
        '''
        groupings = []

        notes = song_info.notes
        song_key = song_info.song_key

        strong_beat_pairs = self.group_strong_beat_pairs(notes, song_info)
        for skeleton_one, skeleton_two in strong_beat_pairs:
            ornament_notes = self.find_ornament_notes(skeleton_one, skeleton_two, notes)

            # skip if no ornament notes between these 2 skeleton notes
            if len(ornament_notes) == 0:
                continue

            chord: ChordTiming = skeleton_one.get_chord()

            if chord is None:
                continue

            chord_function = chord.get_function(song_key)

            if chord_function == 'N':
                continue

            processed_ornament_notes = []
            ornament_notes.insert(0, skeleton_one)
            ornament_notes.append(skeleton_two)

            for i in range(1, len(ornament_notes)-1):
                prev_note_midi_pitch = ornament_notes[i-1].get_midi_pitch()
                curr_note_midi_pitch = ornament_notes[i].get_midi_pitch()
                note_offset = curr_note_midi_pitch - prev_note_midi_pitch
                ornament_note = OrnamentNote(note_offset)
                processed_ornament_notes.append(ornament_note)

            ornament_grouping = OrnamentGrouping(skeleton_one, skeleton_two, processed_ornament_notes)
            groupings.append(ornament_grouping)

        return groupings

    def find_ornament_notes(self, note1: TrainingNote, note2: TrainingNote, notes: list[TrainingNote]) -> list[TrainingNote]:
        """
            Given 2 skeleton notes, gets all the notes in between them
        """
        # Given the onset in seconds of 2 notes, want to find all the notes that fall in between them  
        notes = sorted(notes, key=lambda note: note.get_start_seconds())
        inbetween_notes = [note for note in notes if note1.get_start_seconds() < note.get_start_seconds() < note2.get_start_seconds()]
        return inbetween_notes

    def group_strong_beat_pairs(self, notes, song_info: SongInfo) -> list[tuple[TrainingNote, TrainingNote]]:
        strong_bar_beats = [beat for beat in song_info.beat_timings if beat.is_strong_beat()]
        strong_bar_beats = sorted(strong_bar_beats, key=lambda beat: beat.get_onset_time())

        pairs = []
        for i in range(len(strong_bar_beats)-1):
            curr_beat = strong_bar_beats[i]
            next_beat = strong_bar_beats[i+1]

            curr_beat_note = curr_beat.get_closest_note(notes)
            next_beat_note = next_beat.get_closest_note(notes)

            if curr_beat_note is not None and next_beat_note is not None:
                pairs.append((curr_beat_note, next_beat_note))

        return pairs