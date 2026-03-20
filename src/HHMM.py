from music21 import corpus, roman, environment
from collections import defaultdict
import numpy as np

class Song:
    def __init__(self, notes, sections):
        self.notes = notes
        self.section = section

class MC:
    def __init__(self):
        self.initial_probs = {}
        self.transition_probs = {}

    def sample_notes(self):
        initial_notes = list(self.initial_probs.keys())
        initial_probs = list(self.initial_probs.values())

        # Sample initial note
        current_note = np.random.choice(initial_notes, p=initial_probs)
        sampled_notes = [str(current_note)]

        while current_note != '#':
            next_notes = list(self.transition_probs[current_note].keys())
            next_probs = list(self.transition_probs[current_note].values())
            current_note = np.random.choice(next_notes, p=next_probs)
            print(current_note)
            sampled_notes.append(str(current_note))

        return sampled_notes

    def train_model(self, notes: list[list[str]]):
        transition_count = defaultdict(lambda: defaultdict(int))
        transition_probs = defaultdict(lambda: defaultdict(float))
        initial_count = defaultdict(int)
        initial_probs = defaultdict(float)
        for current_section in notes:
            first_note = current_section[0]
            initial_count[first_note] += 1

            for i in range(len(current_section)-1):
                current_note = current_section[i]
                next_note = current_section[i+1]
                transition_count[current_note][next_note] += 1

        # Normalise transition probs
        for note in transition_count.keys():
            total_counts = sum(transition_count[note].values())
            for next_note, count in transition_count[note].items():
                transition_probs[note][next_note] = count / total_counts

        # Normalise inital probs
        total_count = sum(initial_count.values())
        for note, count in initial_count.items():
            initial_probs[note] = count / total_count

        self.initial_probs = initial_probs
        self.transition_probs = transition_probs

class PitchGenerator:
    def __init__(self, start_mc: MC, middle_mc: MC, end_mc: MC):
        self.start_section_mc = start_mc
        self.middle_section_mc = middle_mc
        self.end_section_mc = end_mc

    def generate(self):
        notes = []
        sampled_start_notes = self.start_section_mc.sample_notes()
        notes.extend(sampled_start_notes)
        print(f'finished generating starting section...')

        sampled_middle_notes = self.middle_section_mc.sample_notes()
        notes.extend(sampled_middle_notes)
        print(f'finished generating middle section...')

        sampled_end_notes = self.end_section_mc.sample_notes()
        notes.extend(sampled_end_notes)
        print(f'finished generating end section...')

        return notes

bach_corupus_song = {'bwv2.6': 0}

bwv = corpus.parse('bach/bwv2.6')
key = bwv.analyze('key')

soprano = bwv.parts[0]

chords = bwv.chordify()

chord_offset_dict = {chord.offset: chord for chord in chords.flatten().getElementsByClass('Chord')}

def get_chord_tone(note, chord):
    if chord is None:
        return 'root'

    note_pc = note.pitch.pitchClass
    root_pc = chord.root().pitchClass

    interval = (note_pc - root_pc) % 12

    interval_map = {
            0: "root", 
            1: "b2", 
            2: "2nd", 
            3: "b3",   # minor 3rd
            4: "3rd",  # major 3rd
            5: "4th",
            6: "b5",   # tritone
            7: "5th", 
            8: "b6",   # minor 6th
            9: "6th",  # major 6th
            10: "b7",  # minor 7th
            11: "7th"  # major 7th
        }

    return interval_map.get(interval, 'root')

# Takes in a list of list of notes, where inner list is notes of a song
def preprocess_notes(notes):
    note_chord_tone = []
    for note in notes:
        chord_offset = max([o for o in chord_offset_dict if o <= note.offset])
        chord = chord_offset_dict.get(chord_offset, None)
        if chord:
            try:
                chord_tone = get_chord_tone(note, chord)
                note_chord_tone.append(chord_tone)
            except Exception:
                note_chord_tone.append('root')
        else:
            note_chord_tone.append('root')

def extract_sequences_from_dataset(bach_chorals_dict: dict[str,int]):
    song_sequences = {}
    for song in bach_chorals_dict.keys():
        song_chord_tones = []
        bwv = corpus.parse(song)
        soprano = bwv.parts[0]
        song_notes = soprano.flatten().notes
        chords = bwv.chordify()
        for note in song_notes:
            chord_offset_dict = {chord.offset: chord for chord in chords.flatten().getElementsByClass('Chord')}
            chord_offset = max([o for o in chord_offset_dict if o <= note.offset])
            chord = chord_offset_dict.get(chord_offset, None)
            if chord:
                try:
                    chord_tone = get_chord_tone(note, chord)
                    song_chord_tones.append(chord_tone)
                except Exception:
                    print('erro')
                    song_chord_tones.append('root')
            else:
                song_chord_tones.append('root')
        song_sequences[song] = song_chord_tones

    return song_sequences

back_chorals_dict = {'bwv2.6': 0, 'bwv3.6': 0}
song_sequences = extract_sequences_from_dataset(back_chorals_dict)

song_phrases = {'bwv2.6': [[1,9], [10,17], [18,26], [27,35], [36,44]],
                'bwv3.6': [[1,10], [11,20], [21,29], [30,37]]
               }

start_sections = []
middle_sections = []
end_sections = []

for song_id in song_phrases.keys():
    current_song_phrases = song_phrases[song_id]
    current_song_sequence = song_sequences[song_id]
    for i, section in enumerate(current_song_phrases):
        start_index = section[0] - 1
        end_index = section[1] - 1

        song_notes = song_sequences[song_id]
        section_notes = [song_notes[i] for i in range(start_index, end_index+1)]
        # Indicates that section has ended (allows loop to exit)
        section_notes.append('#')

        if i == 0:
            start_sections.append(section_notes)
        elif i == len(song_phrases) - 1:
            end_sections.append(section_notes)
        else:
            middle_sections.append(section_notes)

start_section_mc = MC()
start_section_mc.train_model(start_sections)
print(start_section_mc.transition_probs)

middle_section_mc = MC()
middle_section_mc.train_model(middle_sections)

end_section_mc = MC()
end_section_mc.train_model(end_sections)

pitch_generator = PitchGenerator(start_section_mc, middle_section_mc, end_section_mc)
print(pitch_generator.generate())