## File for converting POP909 music into something usable for training
## First we need to load file
import pretty_midi
import os
import pickle
from collections import Counter

class Processor:

    def __init__(self, chord_file_path, midi_file_path):
        self.chord_file_path = chord_file_path
        self.midi_file_path = midi_file_path
        self.melody_track = None
        self.chords = []
        self.processed_notes = []

        self._load_raw_song_data()

    def get_chord_tones_only(self):
        return [note for note in self.processed_notes if note['chord_note']]

    def get_non_chord_tones_only(self):
        return [note for note in self.processed_notes if not note['chord_note']]

    def get_chords_info(self):
        return self.chords

    def process(self):
        """
        Extract and process melody notes with contextual information.

        This function processes a sequence of melody notes by extracting their temporal
        and pitch information, along with the active chord at each note's timeframe and
        the melodic interval from the previous note.

        Args:
            melody_track: A track object containing a sequence of notes. Each note must have
                        'start', 'end', and 'pitch' attributes.
            chords: A collection of chord objects used to determine which chord is active
                    during each note's duration.

        Returns:
            list: A list of dictionaries, where each dictionary contains:
                - start_time (float): The start time of the note
                - end_time (float): The end time of the note
                - duration (float): The duration of the note (end_time - start_time)
                - pitch (int/float): The MIDI pitch value of the note
                - active_chord: The chord that is active during this note's duration
                - interval (int/float): The pitch interval from the previous note
                - melodic_state (str): The melodic interval of the note
                - chord_note (bool): Whether the pitch of the note is part of the current active_chord 

        Note:
            The first note in melody_track.notes is skipped since interval calculation
            requires a previous note. The returned list will contain len(melody_track.notes) - 1
            elements.
        """

        for i in range(1, len(self.melody_track.notes)):

            current_note = self.melody_track.notes[i]
            active_chord_name = self._find_active_chord([current_note.start, current_note.end])
            previous_note_pitch = self.melody_track.notes[i-1].pitch
            pitch_interval = current_note.pitch - previous_note_pitch
            melodic_state = self._classify_melodic_state(pitch_interval)
            chord_note = self._is_note_chord(current_note.pitch, active_chord_name)

            processed_note = {
                'start_time': current_note.start,
                'end_time': current_note.end,
                'duration': current_note.end - current_note.start,
                'pitch': current_note.pitch,
                'chord': active_chord_name,
                'chord_note': chord_note,
                'interval': pitch_interval,
                'melodic_state': melodic_state
            }

            self.processed_notes.append(processed_note)

        return self.processed_notes

    def _load_raw_song_data(self):
        """
        Load raw song data from MIDI and chord files.

        Returns:
            dict: A dictionary containing:
                - 'melody_track': PrettyMIDI Instrument object for the melody track
                - 'chords': List of dictionaries, each containing:
                    - 'start_time': Start time of the chord
                    - 'end_time': End time of the chord
                    - 'chord_name': Name of the chord

        Raises:
            FileNotFoundError: If the MIDI file or chord_midi.txt file does not exist.
            IOError: If there are issues reading the files.
        """

        # Load Midi file
        midi = pretty_midi.PrettyMIDI(self.midi_file_path)

        # Extract melody track
        for instrument in midi.instruments:
            if instrument.name == 'MELODY':
                self.melody_track = instrument

        # Load chord information
        chords = []

        with open(self.chord_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    chords.append({
                        'start_time': float(parts[0]),
                        'end_time': float(parts[1]),
                        'chord_name': parts[2],
                    })

        chords.sort(key=lambda c: c['start_time'])
        self.chords = chords

    def _classify_melodic_state(self, interval: int):
        """
        Classify the melodic movement based on the interval between two notes.

        Args:
            interval (int): The interval between two notes, measured in semitones.
                        Positive values indicate upward movement, negative values indicate downward movement.

        Returns:
            str: A string describing the melodic state:
                - 'repeat': interval is 0 (same note)
                - 'ascending_step': interval is 1 or 2 semitones upward
                - 'descending_step': interval is 1 or 2 semitones downward
                - 'leap_up': interval is 3 or more semitones upward
                - 'leap_down': interval is 3 or more semitones downward
        """
        if interval == 0:
            return 'repeat'
        elif 1 <= interval <= 2:
            return 'ascending_step'
        elif -2 <= interval <= -1:
            return 'descending_step'
        elif interval >= 3:
            return 'leap_up'
        elif interval <= -3:
            return 'leap_down'

    def _is_note_chord(self, pitch, chord_name):
        """
        Check if a given pitch belongs to a specified chord.

        This function determines whether a MIDI pitch note is part of a given chord
        by comparing the pitch class (pitch modulo 12) against the chord's template
        of intervals.

        Args:
            pitch (int): A MIDI pitch value (0-127) representing the note to check.
            chord_name (str): The name of the chord in the format 'Root:quality'
                            (e.g., 'C:maj', 'A:min', 'G:7'). Must match one of the
                            predefined chord templates.

        Returns:
            bool: True if the pitch belongs to the specified chord, False otherwise.
                Returns False if the chord_name is not found in CHORD_TEMPLATES.

        Note:
            The function uses a predefined dictionary of chord templates where each
            chord is represented by pitch classes (0-11) corresponding to the notes
            in the chord.
        """
        CHORD_TEMPLATES = {
            'C:maj': [0, 4, 7],
            'C:min': [0, 3, 7],
            'D:min': [2, 5, 9],
            'D:maj': [2, 6, 9],
            'E:min': [4, 7, 11],
            'E:maj': [4, 8, 11],
            'F:maj': [5, 9, 0],
            'F:min': [5, 8, 0],
            'G:maj': [7, 11, 2],
            'G:min': [7, 10, 2],
            'A:min': [9, 0, 4],
            'A:maj': [9, 1, 4],
            'B:min': [11, 2, 6],
            'B:maj': [11, 3, 6],
            'B:dim': [11, 2, 5],
            'G:7': [7, 11, 2, 5]
        }

        if chord_name not in CHORD_TEMPLATES:
            return False

        pitch_class = pitch % 12

        return pitch_class in CHORD_TEMPLATES[chord_name];


    def _find_active_chord(self, note_times: list[float, float]):
        """
        Find the chord that is active during a given note's start time.

        Args:
            chords: A collection of chord objects with start_time and end_time attributes.
            note_times (list[float, float]): A list containing two floats representing 
                the start and end times of a note. Only the first element (start time) 
                is used.

        Returns:
            The chord name that overlaps with the note's start time, or -1 if no 
            active chord is found.

        Note:
            Currently uses linear search. Binary search implementation is planned for 
            better performance with large chord collections.
        """
        ## TODO do binary search instead

        start_time = note_times[0]

        for chord in self.chords:
            if start_time >= chord['start_time']  and start_time <= chord['end_time']:
                return chord['chord_name']

        return None

class DatasetProcessor:
    """Process all songs/chords in the POP909 dataset directory"""

    def __init__(self):
        self.processed_notes = []
        self.chords = []
        self.song_summaries = []

    def get_summary(self):
        return {
            'total_songs': len(self.song_summaries),
            'total_notes': len(self.processed_notes),
            'song_details': self.song_summaries
        }

    def get_chords(self):
        return self.chords

    def get_notes(self):
        return self.processed_notes

    def get_notes_by_song(self, song_id):
        return [note for note in self.processed_notes if note['song_id'] == song_id]

    def get_chords_by_song(self, song_id):
        return [chord for chord in self.chords if chord['song_id'] == song_id]

    def save_notes(self, filepath='models/notes_processed.pkl'):

        model_data = {
            'songs_processed': len(self.song_summaries),
            'notes': self.processed_notes
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def save_chords(self, filepath='models/chord_processed.pkl'):

        model_data = {
            'songs_processed': len(self.song_summaries),
            'chords': self.chords
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, chordfilepath='models/chord_processed.pkl', notefilepath='models/notes_processed.pkl'):

        with open(chordfilepath, 'rb') as f:
            chord_model_data = pickle.load(f)

        with open(notefilepath, 'rb') as f:
            note_model_data = pickle.load(f)

        model = cls()
        model.processed_notes = note_model_data['notes']
        model.chords = chord_model_data['chords']
        return model

    def process_all(self, dataset_path):
        for song_folder in os.listdir(dataset_path):
            song_path = os.path.join(dataset_path, song_folder)

            # Skip if not directory
            if not os.path.isdir(song_path):
                continue
                
            midi_path = os.path.join(song_path, f"{song_folder}.mid")
            chord_path = os.path.join(song_path, "chord_midi.txt")

            if not os.path.exists(midi_path) or not os.path.exists(chord_path):
                continue

            try:
                processor = Processor(chord_path, midi_path)
                notes = processor.process()
                chords = processor.get_chords_info()

                for note in notes:
                    note['song_id'] = song_folder

                for chord in chords:
                    chord['song_id'] = song_folder

                self.processed_notes.extend(notes)
                self.chords.extend(chords)

            except Exception as e:
                print(f"Error processing {song_folder}: {e}")
                continue

if __name__ == "__main__":
    dataset_path = '/Users/sachin/Documents/music_generator/POP909'
    #processor = DatasetProcessor()
    #processor.process_all(dataset_path)
    #processor.save_chords()
    #processor.save_notes()
    #processor = DatasetProcessor.load()
    #chords = processor.get_chords()
    #notes = processor.get_notes()
    #print(f"chords: {chords}")
    #print(f"notes: {notes}")