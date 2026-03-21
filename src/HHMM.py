from music21 import corpus, roman, environment
from collections import defaultdict
import numpy as np

class Song:
    def __init__(self, notes, sections):
        self.notes = notes
        self.section = section

class BassProbs:
    def __init__(self):
        self.emission_probs = {}       

    def train_model(self, soprano_bass_pairs: list[tuple[str, str]]):
        emission_count = defaultdict(lambda: defaultdict(int))

        for pair in soprano_bass_pairs:
            soprano, bass = pair
            if soprano is None or bass is None:
                continue

            emission_count[soprano][bass] += 1

        emission_probs = defaultdict(lambda: defaultdict(float))
        for soprano in emission_count:
            total_count = sum(emission_count[soprano].values())
            for bass, count in emission_count[soprano].items():
                emission_probs[soprano][bass] = count / total_count

        self.emission_probs = emission_probs

    def get_bass_note(self, soprano_chord_tone):
        if soprano_chord_tone not in self.emission_probs:
            return 'root'
        
        bass_notes = list(self.emission_probs[soprano_chord_tone].keys())
        bass_probs = list(self.emission_probs[soprano_chord_tone].values())

        return np.random.choice(bass_notes, p=bass_probs)

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
            if current_note != '#':
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

    # Generates a minimum of {min_notes}
    def generate(self, min_notes=64):
        notes = []
        sampled_start_notes = self.start_section_mc.sample_notes()
        notes.extend(sampled_start_notes)
        print(f'finished generating starting section...')

        while len(notes) < min_notes:
            sampled_middle_notes = self.middle_section_mc.sample_notes()
            notes.extend(sampled_middle_notes)
        print(f'finished generating middle section...')

        sampled_end_notes = self.end_section_mc.sample_notes()
        notes.extend(sampled_end_notes)
        print(f'finished generating end section...')

        return notes

# Creates a list of tupels
def create_soprano_bass_pairs(song, soprano_track, bass_track):
    bwv = corpus.parse(song)
    soprano = bwv.parts[soprano_track]
    soprano_notes = soprano.flatten().notes
    bass = bwv.parts[bass_track]
    bass_notes = bass.flatten().notes
    chords = bwv.chordify()
    chord_offset_dict = {chord.offset: chord for chord in chords.flatten().getElementsByClass('Chord')}

    pairs = []
    for soprano_note in soprano_notes:
        target_offset = soprano_note.offset
        cooresponding_bass_note = None

        # Find cooresponding bass note
        for bass_note in bass_notes:
            if bass_note.offset == target_offset:
                cooresponding_bass_note = bass_note

        if cooresponding_bass_note is not None:
            chord_offset = max([o for o in chord_offset_dict if o <= soprano_note.offset])
            chord = chord_offset_dict.get(chord_offset, None)
            if chord:
                try:
                    soprano_chord_tone = get_chord_tone(soprano_note, chord)
                    bass_chord_tone = get_chord_tone(cooresponding_bass_note, chord)
                    pairs.append((soprano_chord_tone, bass_chord_tone))
                except Exception:
                    continue
            else:
                continue

    return pairs

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
def preprocess_notes(notes, chord_offset_dict):
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

if __name__ == "__main__":
    back_chorals_dict = {'bwv2.6': 0,
                        'bwv3.6': 0,
                        'bwv5.7':0,    
                        'bwv6.6': 0,    
                        'bwv9.7': 0,    
                        'bwv10.7':0,    
                        'bwv11.6': 0,
                        'bwv12.7': 1,
                        'bwv14.5': 0,
                        'bwv16.6': 0,
                        'bwv17.7': 0,
                        'bwv19.7': 3,
                        'bwv25.6': 0,
                        'bwv26.6': 0,
                        'bwv28.6': 0,
                        'bwv30.6': 0,
                        'bwv32.6': 0,
                        'bwv33.6': 0
                        }
    
    soprano_bass_pairs = []
    for song, soprano_track_index in back_chorals_dict.items():
        soprano_bass_pairs.extend(create_soprano_bass_pairs(song, soprano_track_index, soprano_track_index+2))

    soprano_bass_model = BassProbs()
    soprano_bass_model.train_model(soprano_bass_pairs)
    print(soprano_bass_model.get_bass_note('root'))

    #song_sequences = extract_sequences_from_dataset(back_chorals_dict)

    #song_phrases = {
        #'bwv2.6': [[1,9], [10,17], [18,26], [27,35], [36,44]],
        #'bwv3.6': [[1,10], [11,20], [21,29], [30,37]],
        #'bwv5.7': [[1,6], [7,12], [13,19], [20,26], [27,33], [34,40]],
        #'bwv6.6': [[1,11], [12,19], [20,30], [31,38]],
        #'bwv9.7': [[1,9], [10,20], [21,29], [30,39], [41,47]],
        #'bwv10.7': [[1,11], [12,20], [21,31], [32,46]],
        #'bwv11.6': [[1,9], [10,14], [15,25], [26,34], [35,41], [42,50]],
        #'bwv12.7': [[1,8], [9,15], [16,45], [46,52]],
        #'bwv14.5': [[1,9], [10,21], [22,27]],
        #'bwv16.6': [[1,8], [9,14], [15,20], [21,27], [28,35], [36,43]],
        #'bwv17.7': [[1,8], [9,17], [18,26], [27,33], [34,42], [43,49], [50,63], [64,71], [72,82]],
        #'bwv19.7': [[1,8], [9,15], [16,22], [23,29], [30,37], [38,45]],
        #'bwv25.6': [[1,8], [9,18], [19,25], [26,32], [33,41], [42,49]],
        #'bwv26.6': [[1,8], [9,15], [16,23], [24,31], [32,40]],
        #'bwv28.6': [[1,8], [9,14], [15,20], [21,27], [28,36], [37,43]],
        #'bwv30.6': [[1,8], [9,15], [16,22], [23,29], [30,39], [40,49]],
        #'bwv32.6': [[1,8], [9,15], [16,22], [23,29], [30,38], [39,47]],
        #'bwv33.6': [[1,11], [12,26], [27,36], [37,46], [47,55], [56,60], [61,71]],
    #}

    #start_sections = []
    #middle_sections = []
    #end_sections = []

    #for song_id in song_phrases.keys():
        #current_song_phrases = song_phrases[song_id]
        #current_song_sequence = song_sequences[song_id]
        #for i, section in enumerate(current_song_phrases):
            #start_index = section[0] - 1
            #end_index = section[1] - 1
            #print(f'start index: {start_index} end_index: {end_index}')

            #song_notes = song_sequences[song_id]
            #section_notes = [song_notes[i] for i in range(start_index, end_index+1)]
            ## Indicates that section has ended (allows loop to exit)
            #section_notes.append('#')

            #if i == 0:
                #start_sections.append(section_notes)
            #elif i == len(current_song_phrases) - 1:
                #end_sections.append(section_notes)
            #else:
                #middle_sections.append(section_notes)

    #print(f'end sections: {end_sections}')

    #start_section_mc = MC()
    #start_section_mc.train_model(start_sections)

    #middle_section_mc = MC()
    #middle_section_mc.train_model(middle_sections)

    #end_section_mc = MC()
    #end_section_mc.train_model(end_sections)

    #pitch_generator = PitchGenerator(start_section_mc, middle_section_mc, end_section_mc)
    #print(pitch_generator.generate())