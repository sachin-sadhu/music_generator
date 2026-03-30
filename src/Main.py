from music21 import stream, note, tempo, instrument, chord
import numpy as np
from Preprocessing import *
from ChordFunctions import *
from HMM import ChordHMM, ChordHMMFirstOrder
from SecondLayerHMM import Generator
from Timings import KeyTiming
from Rhythm import *
from HHMM import *

CHORD_TEMPLATES = {
    'C:maj': [0, 4, 7],
    'C:min': [0, 3, 7],
    'D:min': [2, 5, 9], 
    'D:maj': [2, 6, 9],
    'E:min': [4, 7, 11],
    'E:maj': [4, 8, 11],
    'F:maj': [5, 9, 12],     # F-A-C
    'F:min': [5, 8, 12],     # F-Ab-C
    'G:maj': [7, 11, 14],    # G-B-D
    'G:min': [7, 10, 14],    # G-Bb-D
    'A:min': [9, 12, 16],    # A-C-E
    'A:maj': [9, 13, 16],    # A-C#-E
    'B:min': [11, 14, 18],   # B-D-F#
    'B:maj': [11, 15, 18],   # B-D#-F#
    'B:dim': [11, 14, 17],   # B-D-F
    'G:7': [7, 11, 14, 17]   # G-B-D-F
}

def notes_to_midi(notes, filename='a_fuck this.mid', bpm=80):
    CHORD_TEMPLATES = {
        'C:maj': [0, 4, 7],
        'C:min': [0, 3, 7],
        'D:min': [2, 5, 9], 
        'D:maj': [2, 6, 9],
        'E:min': [4, 7, 11],
        'E:maj': [4, 8, 11],
        'F:maj': [5, 9, 12],     # F-A-C
        'F:min': [5, 8, 12],     # F-Ab-C
        'G:maj': [7, 11, 14],    # G-B-D
        'G:min': [7, 10, 14],    # G-Bb-D
        'A:min': [9, 12, 16],    # A-C-E
        'A:maj': [9, 13, 16],    # A-C#-E
        'B:min': [11, 14, 18],   # B-D-F#
        'B:maj': [11, 15, 18],   # B-D#-F#
        'B:dim': [11, 14, 17],   # B-D-F
        'G:7': [7, 11, 14, 17]   # G-B-D-F
    }

    score = stream.Score()

    # Treble clef part
    treble = stream.Part()
    treble.append(instrument.Piano())
    treble.append(tempo.MetronomeMark(number=bpm))

    # Bass clef part
    bass = stream.Part()
    bass.append(instrument.Piano())

    for curr_note in notes:
        # Treble melody note
        pitch = curr_note.midi_pitch
        n = note.Note(pitch)

        n.quarterLength = curr_note.duration  # Duration in quarter notes
        treble.append(n)
        print(curr_note.chord)

        ## Bass arpeggio for the current chord
        #if curr_note.chord in CHORD_TEMPLATES:
            #chord_tones = CHORD_TEMPLATES[curr_note.chord]
            
            ## Calculate arpeggio note duration (divide melody duration by number of chord tones)
            #arpeggio_note_duration = curr_note.duration / len(chord_tones)
            
            #for i, tone in enumerate(chord_tones):
                #bass_pitch = 48 + tone  # Bass register
                #bass_note = note.Note(bass_pitch)
                #bass_note.quarterLength = curr_note.duration
                ##bass.append(bass_note)

        # Steady bass line - quarter note roots
        #if curr_note.chord != current_chord and curr_note.chord in CHORD_TEMPLATES:
            ## Only play bass when chord changes
            #root_tone = CHORD_TEMPLATES[curr_note.chord][0]  # Root note
            #bass_pitch = 48 + root_tone
            #bass_note = note.Note(bass_pitch)
            #bass_note.quarterLength = 1.0  # Steady quarter note
            #bass.append(bass_note)
            #current_chord = curr_note.chord
        #else:
            ## Add rest to maintain steady rhythm
            #rest = note.Rest()
            #rest.quarterLength = curr_note.duration
            #bass.append(rest)

        #if curr_note.chord != current_chord and curr_note.chord in CHORD_TEMPLATES:
            ## chord has changed
            #chord_tones = CHORD_TEMPLATES[curr_note.chord]
            #bass_notes = []
            #for tone in chord_tones:
                #bass_pitch = 48 + tone
                #bass_notes.append(bass_pitch)

            #bass_chord = chord.Chord(bass_notes)
            #bass_chord.quarterLength = 1.0
            #bass.append(bass_chord)

            #current_chord = curr_note.chord

    score.append(treble)
    score.append(bass)
    score.write('midi', fp=filename)

def fill_chord_triad(notes, key):
    CHORD_TEMPLATES = {
        'C:maj': [0, 4, 7],
        'C:min': [0, 3, 7],
        'D:min': [2, 5, 9], 
        'D:maj': [2, 6, 9],
        'E:min': [4, 7, 11],
        'E:maj': [4, 8, 11],
        'F:maj': [5, 9, 12],     # F-A-C
        'F:min': [5, 8, 12],     # F-Ab-C
        'G:maj': [7, 11, 14],    # G-B-D
        'G:min': [7, 10, 14],    # G-Bb-D
        'A:min': [9, 12, 16],    # A-C-E
        'A:maj': [9, 13, 16],    # A-C#-E
        'B:min': [11, 14, 18],   # B-D-F#
        'B:maj': [11, 15, 18],   # B-D#-F#
        'B:dim': [11, 14, 17],   # B-D-F
        'G:7': [7, 11, 14, 17]   # G-B-D-F
    }

    chord_tones = []

    for count, note in enumerate(notes):
        chord = get_chord_name_in_original_key(note[1], key)
        print(f'chord: {chord}')
        if chord not in CHORD_TEMPLATES:
            print('coudl not find chord')
            continue
        else:
            chord_triad_offsets = CHORD_TEMPLATES[chord]
            beat_onset = count % 4
            for triad in chord_triad_offsets:
                midi_pitch = 48 + triad
                chord_tones.append((midi_pitch, 1.0, beat_onset))
                print(f'beat onset: {beat_onset}')

def postprocess_bass(notes, key):
    """
    Clean up generated melody by avoiding really bad notes.
    """

    cleaned_up_notes = []
    key_name = key.name
    # Define key scales (notes to keep)
    KEY_SCALES = {
        'A:maj': [9, 11, 1, 2, 4, 6, 8],      # A, B, C#, D, E, F#, G#
        'C:maj': [0, 2, 4, 5, 7, 9, 11],      # C, D, E, F, G, A, B
        'D:maj': [2, 4, 6, 7, 9, 11, 1],      # D, E, F#, G, A, B, C#
        'G:maj': [7, 9, 11, 0, 2, 4, 6],      # G, A, B, C, D, E, F#
        # Add more keys as needed
    }
    
    # Define tritones from tonic (notes to absolutely avoid)
    TRITONES = {
        'A:maj': 3,   # D#/Eb
        'C:maj': 6,   # F#/Gb  
        'D:maj': 8,   # G#/Ab
        'G:maj': 1,   # C#/Db
    }
    
    if key_name not in KEY_SCALES:
        return notes  # Skip processing if key not defined
    
    allowed_notes = set(KEY_SCALES[key_name])
    tritone = TRITONES.get(key_name)
    
    for note in notes:
        pitch_class = note[0] % 12  # Get note within one octave
        
        # Check if its tritone
        if pitch_class == tritone:
            # Move tritone to nearest safe note
            if tritone + 1 in allowed_notes:
                corrected_pitch = note[0] + 1
            elif tritone - 1 in allowed_notes:
                corrected_pitch = note[0] - 1
            else:
                corrected_pitch = note[0] + 2  # Fallback
            
            cleaned_up_notes.append((corrected_pitch, note[1]))
            
        # Check if it's outside the key scale
        elif pitch_class not in allowed_notes:
            # Find nearest note in scale
            distances = [(abs(pitch_class - allowed), allowed) for allowed in allowed_notes]
            distances.sort(key=lambda x: x[0])
            
            nearest_note = distances[0][1]
            pitch_adjustment = nearest_note - pitch_class
            
            # Handle octave wrapping
            if abs(pitch_adjustment) > 6:
                pitch_adjustment = 12 - abs(pitch_adjustment)
                if pitch_adjustment < 0:
                    pitch_adjustment = -pitch_adjustment
                    
            corrected_pitch = note[0] + pitch_adjustment
            cleaned_up_notes.append((corrected_pitch, note[1]))

def postprocess_melody(notes, key):
    """
    Clean up generated melody by avoiding really bad notes.
    """

    key_name = key.name
    # Define key scales (notes to keep)
    KEY_SCALES = {
        'A:maj': [9, 11, 1, 2, 4, 6, 8],      # A, B, C#, D, E, F#, G#
        'C:maj': [0, 2, 4, 5, 7, 9, 11],      # C, D, E, F, G, A, B
        'D:maj': [2, 4, 6, 7, 9, 11, 1],      # D, E, F#, G, A, B, C#
        'G:maj': [7, 9, 11, 0, 2, 4, 6],      # G, A, B, C, D, E, F#
        # Add more keys as needed
    }
    
    # Define tritones from tonic (notes to absolutely avoid)
    TRITONES = {
        'A:maj': 3,   # D#/Eb
        'C:maj': 6,   # F#/Gb  
        'D:maj': 8,   # G#/Ab
        'G:maj': 1,   # C#/Db
    }
    
    if key_name not in KEY_SCALES:
        return notes  # Skip processing if key not defined
    
    allowed_notes = set(KEY_SCALES[key_name])
    tritone = TRITONES.get(key_name)
    
    for note in notes:
        pitch_class = note['pitch'] % 12
        # Check if its tritone
        if pitch_class == tritone:
            # Move tritone to nearest safe note
            if tritone + 1 in allowed_notes:
                corrected_pitch = note['pitch'] + 1
            elif tritone - 1 in allowed_notes:
                corrected_pitch = note['pitch'] - 1
            else:
                corrected_pitch = note['pitch'] + 2  # Fallback
            
            print(f"Fixed tritone: {note['pitch']} -> {corrected_pitch}")
            note['pitch'] = corrected_pitch
            
        # Check if it's outside the key scale
        elif pitch_class not in allowed_notes:
            # Find nearest note in scale
            distances = [(abs(pitch_class - allowed), allowed) for allowed in allowed_notes]
            distances.sort(key=lambda x: x[0])

            nearest_note = distances[0][1]
            note['pitch'] = nearest_note

def generate_score(key: KeyTiming, rhythm_sequence: list[int], soprano_notes: list[str], bass_notes: list[str]):
    score = stream.Score()

    # Treble clef part
    treble = stream.Part()
    treble.append(instrument.Piano())
    treble.append(tempo.MetronomeMark(number=80))

    print(f'rhythm_sequence: {rhythm_sequence}')
    rhythm_sequence_index = 0
    while rhythm_sequence_index < len(rhythm_sequence):
        # Encountered new note
        if rhythm_sequence[rhythm_sequence_index] == 0:
            if len(soprano_notes) > 0:
                note_chord_tone = soprano_notes.pop(0)
            else:
                note_chord_tone = 'root'
            
            midi_pitch = SkeletonEmission(note_chord_tone, 0, 'I').calc_midi_pitch(key)
            print(f'note chord tone: {note_chord_tone}, midi pitch: {midi_pitch}')

            end_duration_pointer = rhythm_sequence_index + 1
            # Increment rhythm sequence index
            while (end_duration_pointer < len(rhythm_sequence) and rhythm_sequence[end_duration_pointer] == 1):
                end_duration_pointer += 1
            
            duration = end_duration_pointer - rhythm_sequence_index
            rhythm_sequence_index = end_duration_pointer

            treble_note = note.Note(midi_pitch)
            treble_note.quarterLength = duration / 4
            treble.append(treble_note)

        elif rhythm_sequence[rhythm_sequence_index] == 2:
            end_duration_pointer = rhythm_sequence_index + 1
            while (end_duration_pointer < len(rhythm_sequence) and rhythm_sequence[end_duration_pointer] == 2):
                end_duration_pointer += 1

            duration = end_duration_pointer - rhythm_sequence_index
            rhythm_sequence_index = end_duration_pointer

            rest = note.Rest()
            rest.quarterLength = duration / 4
            treble.append(rest)

    score.append(treble)
    score.write('midi', fp='a_major.mid')
    #score.write('lilypond.pdf', fp='mmb')

if __name__ == "__main__":
    #directory = "/home/sachin/Documents/music_generator/POP909/POP909"
    #data = TrainingDataProcessedInfo()
    #data.load_training_data(directory)

    #chord_beat_hmm = HMM()
    #chord_beat_hmm.train_model(data.notes, data.beat_chords)
    second_order_chord_hmm = ChordHMM.load('models/chord_second_order.pkl')
    #first_order_chord_hmm = ChordHMMFirstOrder.load('models/first_order_hmm.pkl')
    ornament_hmms = OrnamentNoteMCs.load()
    bass_model = BassNoteGenerator.load_model()
    rhythm_model = load_model('models/rhythm_model.pkl')

    num_notes = 256
    song = Generator(second_order_chord_hmm, ornament_hmms, rhythm_model, bass_model)
    key = KeyTiming('C:maj')
    melody, bass_notes, sampled_beats = song.generate(key, num_notes)

    postprocess_melody(melody, key)

    score = stream.Score()

    # Treble clef part
    treble = stream.Part()
    treble.append(instrument.Piano())
    treble.append(tempo.MetronomeMark(number=80))

    # Bass clef part
    bass = stream.Part()
    bass.append(instrument.Piano())

    for i in range(len(bass_notes)-1):
        pitch, onset = bass_notes[i]
        n = note.Note(pitch)
        n.quarterLength = 2
        bass.append(n)

        _, next_note_onset = bass_notes[i+1]
        if next_note_onset != (onset + 8):
            rest_duration = next_note_onset - onset - 8
            r = note.Rest()
            r.quarterLength = rest_duration / 4
            bass.append(r)

    pitch, _ = bass_notes[-1]
    n = note.Note(pitch)
    n.quarterLength = 2  # Duration in quarter notes
    bass.append(n)

    for i in range(len(melody)-1):
        curr_note = melody[i]
        pitch = curr_note['pitch']
        n = note.Note(pitch)
        duration = curr_note['end'] - curr_note['start']
        n.quarterLength = duration / 4  # Duration in quarter notes
        treble.append(n)

        #curr_chord = get_chord_name_in_original_key(curr_note['chord_function'], key)
        #print(curr_chord)

        ##if curr_chord in CHORD_TEMPLATES:
            ##chord_tones = CHORD_TEMPLATES[curr_chord]
            ##bass_pitches = [48 + tone for tone in chord_tones]
            ##bass_chord = chord.Chord(bass_pitches)
            ##bass_chord.quarterLength = duration / 4
            ##bass.append(bass_chord)

        next_note = melody[i+1]
        if next_note['start'] != curr_note['end']:
            # Need a rest!
            rest_duration = next_note['start'] - curr_note['end']
            r = note.Rest()
            r.quarterLength = rest_duration / 4
            treble.append(r)

    # Need to add last note
    last_note = melody[-1]
    pitch = last_note['pitch']
    n = note.Note(pitch)
    duration = last_note['end'] - last_note['start']
    n.quarterLength = duration / 4  # Duration in quarter notes
    treble.append(n)
    
    score.append(treble)
    score.append(bass)
    score.write('midi', fp='good_rhythm.mid')
    #score.write('lilypond.pdf', fp='asdflj')

    ##cleaned_melody = postprocess_melody(melody_sequence, key)
    ##print(cleaned_melody)

    ##notes_to_midi(cleaned_melody)
        ##converted_notes = []
    ##beat_counter = 0
    ##for count, note in enumerate(sequence):
        ##beat_onset = count % 4
        ##converted_notes.append((note, 'crotchet', beat_onset))
        ##beat_counter += 1

    ##print(converted_notes)
    ##save_to_midi(converted_notes, "tone.mid")

    ##print(split)

    ##chord_beat_hmm = HMM()
    ##chord_beat_hmm.train_model(song_notes, all_song_beat_chords)
    ##sampled_beats = chord_beat_hmm.sample(5)

    ##sampled_beats = [(get_note_midi_pitch(note[0], note[-1], 'C:maj', note[1]), note[-1]) for note in sampled_beats]
    ##fill_chord_triad(sampled_beats, 'C:maj')
    ##second_layer = SecondLayerGen()
    ##second_layer.train_hmms(split)
    ##filled = fill_ornament_notes(sampled_beats, second_layer)
    ##print(filled)

    ##converted_notes = []
    ##beat_counter = 0
    ##for count, note in enumerate(filled):
        ##beat_onset = count % 4
        ##converted_notes.append((note, 'crotchet', beat_onset))
        ##beat_counter += 1

    ##print(converted_notes)
    ##save_to_midi(converted_notes, "what.mid")

    ##num_patterns = 3
    ##max_duration = 10
    ##obs_array = np.array(filtered_bars, dtype=object)
    ##durations = np.random.dirichlet(np.ones(max_duration), size=num_patterns)
    ##tmat = initalise_tmat(num_patterns)

    ##note_emission = NoteEmission(num_patterns, filtered_bars_chords)
    ##note_emission_hsmm = HSMMModel(
        ##note_emission, durations, tmat
    ##)

    ##note_emission.set_context(filtered_bars_chords, filtered_bars_chords)
    ##result = note_emission_hsmm.fit(obs_array)
    ##decoded_states = note_emission_hsmm.decode(filtered_bars)
    ##key = 'C:maj'

    ##pattern_bars = build_pattern_bars_dict(filtered_bars, filtered_bars_chords, decoded_states)
    ##chains = build_chains(num_patterns, pattern_bars)

    ##num_bars = 10

    ##_, states = note_emission_hsmm.sample(num_bars)

   
    ##sampled_chords = ['I', 'V', 'vi', 'IV', 'I', 'V', 'vi', 'IV', 'I', 'I']
    ##sampled_skeleton_notes = [('root', 'root') for _ in range(num_bars)]

    ##print(f'generated chords: {sampled_chords}')
    ##print(f'skeleton notes: {sampled_skeleton_notes}')
    ### Want to use skeleton notes provided by emitted bars as the start in the markov chain
    ##fully_sampled_bars = []
    ##for i, bar_skeleton_note in enumerate(sampled_skeleton_notes):
        ##skeleton_note_tone = bar_skeleton_note[0]
        ##bar_markov_chain: PatternMarkovChain = chains[states[i]]
        ##sampled_bar = bar_markov_chain.sample_bar(skeleton_note_tone, sampled_chords[i])
        ##fully_sampled_bars.append(sampled_bar)

    ###converted_bars = []
    ###for bar in emitted_bars:
        ###bar_notes = []
        ###bar_notes.append((bar[0], 0, 'crotchet', 1.0))
        ###bar_notes.append((bar[1], 0, 'crotchet', 3.0))
        ###converted_bars.append(bar_notes)

    ##bars_midi_pitch = []
    ##for i, bar in enumerate(fully_sampled_bars):
        ##bar_midi_notes = []
        ##bar_chord = sampled_chords[i]
        ##for note in bar:
            ##chord_tone, _, note_duration, note_onset = note
            ##note_midi_pitch = get_note_midi_pitch(chord_tone, bar_chord, key)
            ##note_formatted = (note_midi_pitch, note_duration, note_onset)
            ##bar_midi_notes.append(note_formatted)
        ##bars_midi_pitch.append(bar_midi_notes)

    ##print(bars_midi_pitch)
