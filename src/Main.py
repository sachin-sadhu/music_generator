from music21 import stream, note, tempo, instrument, chord
import numpy as np
from Preprocessing import *
from ChordFunctions import *
from HMM import HMM
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

    print(chord_tones)

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
        pitch_class = note['pitch'] % 12  # Get note within one octave
        
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
            pitch_adjustment = nearest_note - pitch_class
            
            # Handle octave wrapping
            if abs(pitch_adjustment) > 6:
                pitch_adjustment = 12 - abs(pitch_adjustment)
                if pitch_adjustment < 0:
                    pitch_adjustment = -pitch_adjustment
                    
            corrected_pitch = note['pitch'] + pitch_adjustment
            note['pitch'] = corrected_pitch
            print(f"Fixed chromatic note: {note['pitch']} -> {corrected_pitch}")

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
            if len(notes) > 0:
                note_chord_tone = notes.pop(0)
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
    score.write('midi', fp='64yeet.mid')
    #score.write('lilypond.pdf', fp='mmb')

if __name__ == "__main__":
    #directory = "/home/sachin/Documents/music_generator/POP909/POP909"
    #data = TrainingDataProcessedInfo()
    #data.load_training_data(directory)

    #chord_beat_hmm = HMM()
    #chord_beat_hmm.train_model(data.notes, data.beat_chords)

    #ornament_hmms = OrnamentNoteHMMs(data.ornament_groupings)
    #ornament_hmms.train_hmms()
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

    song_phrases = {
        'bwv2.6': [[1,9], [10,17], [18,26], [27,35], [36,44]],
        'bwv3.6': [[1,10], [11,20], [21,29], [30,37]],
        'bwv5.7': [[1,6], [7,12], [13,19], [20,26], [27,33], [34,40]],
        'bwv6.6': [[1,11], [12,19], [20,30], [31,38]],
        'bwv9.7': [[1,9], [10,20], [21,29], [30,39], [41,47]],
        'bwv10.7': [[1,11], [12,20], [21,31], [32,46]],
        'bwv11.6': [[1,9], [10,14], [15,25], [26,34], [35,41], [42,50]],
        'bwv12.7': [[1,8], [9,15], [16,45], [46,52]],
        'bwv14.5': [[1,9], [10,21], [22,27]],
        'bwv16.6': [[1,8], [9,14], [15,20], [21,27], [28,35], [36,43]],
        'bwv17.7': [[1,8], [9,17], [18,26], [27,33], [34,42], [43,49], [50,63], [64,71], [72,82]],
        'bwv19.7': [[1,8], [9,15], [16,22], [23,29], [30,37], [38,45]],
        'bwv25.6': [[1,8], [9,18], [19,25], [26,32], [33,41], [42,49]],
        'bwv26.6': [[1,8], [9,15], [16,23], [24,31], [32,40]],
        'bwv28.6': [[1,8], [9,14], [15,20], [21,27], [28,36], [37,43]],
        'bwv30.6': [[1,8], [9,15], [16,22], [23,29], [30,39], [40,49]],
        'bwv32.6': [[1,8], [9,15], [16,22], [23,29], [30,38], [39,47]],
        'bwv33.6': [[1,11], [12,26], [27,36], [37,46], [47,55], [56,60], [61,71]],
    }


    model = load_model('rhythm_model.pkl')

    song_sequences = extract_sequences_from_dataset(back_chorals_dict)

    start_sections = []
    middle_sections = []
    end_sections = []

    for song_id in song_phrases.keys():
        current_song_phrases = song_phrases[song_id]
        current_song_sequence = song_sequences[song_id]
        for i, section in enumerate(current_song_phrases):
            start_index = section[0] - 1
            end_index = section[1] - 1
            print(f'start index: {start_index} end_index: {end_index}')

            song_notes = song_sequences[song_id]
            section_notes = [song_notes[i] for i in range(start_index, end_index+1)]
            # Indicates that section has ended (allows loop to exit)
            section_notes.append('#')

            if i == 0:
                start_sections.append(section_notes)
            elif i == len(current_song_phrases) - 1:
                end_sections.append(section_notes)
            else:
                middle_sections.append(section_notes)

    start_section_mc = MC()
    start_section_mc.train_model(start_sections)

    middle_section_mc = MC()
    middle_section_mc.train_model(middle_sections)

    end_section_mc = MC()
    end_section_mc.train_model(end_sections)

    num_notes = 64
    pitch_generator = PitchGenerator(start_section_mc, middle_section_mc, end_section_mc)
    soprano_pitches = pitch_generator.generate(num_notes)

    soprano_bass_pairs = []
    for song, soprano_track_index in back_chorals_dict.items():
        soprano_bass_pairs.extend(create_soprano_bass_pairs(song, soprano_track_index, soprano_track_index+2))
    soprano_bass_model = BassProbs()
    soprano_bass_model.train_model(soprano_bass_pairs)

    bass_pitches = [soprano_bass_model.get_bass_note(soprano_tone) for soprano_tone in soprano_pitches]
    rhythm_sequence = generate_rhythm_sequence(len(soprano_pitches), model)
    key = KeyTiming("D:maj")
    generate_score(key, rhythm_sequence, soprano_pitches, bass_pitches)

    #song = Generator(chord_beat_hmm, ornament_hmms, X)
    #key = KeyTiming('A:maj')
    #melody, sampled_beats = song.generate(key)
    #print(f'melody: {melody}')

    #postprocess_melody(melody, key)

    #score = stream.Score()

    ## Treble clef part
    #treble = stream.Part()
    #treble.append(instrument.Piano())
    #treble.append(tempo.MetronomeMark(number=80))

    ## Bass clef part
    #bass = stream.Part()
    #bass.append(instrument.Piano())

    #for beat in sampled_beats:
        #print(beat)
        #chord_function = beat.chord_function
        #chord_name = get_chord_name_in_original_key(chord_function, key)
        #print(f'chord name: {chord_name}')
        #if chord_name in CHORD_TEMPLATES:
            #print(f'chord present')
            #chord_note = CHORD_TEMPLATES[chord_name]
            ##bass_pitches = [48 + tone for tone in chord_tones]
            #pitch = 48 + chord_note[0]
            #bass_note = note.Note(pitch)
            ##bass_chord = chord.Chord(bass_pitches)
            #bass_note.quarterLength = 2
            #bass.append(bass_note)
        #else:
            #print(f'chord not present')

    #for i in range(len(melody)-1):
        #curr_note = melody[i]
        #pitch = curr_note['pitch']
        #n = note.Note(pitch)
        #duration = curr_note['end'] - curr_note['start']
        #n.quarterLength = duration / 4  # Duration in quarter notes
        #treble.append(n)

        #curr_chord = get_chord_name_in_original_key(curr_note['chord_function'], key)
        #print(curr_chord)

        ##if curr_chord in CHORD_TEMPLATES:
            ##chord_tones = CHORD_TEMPLATES[curr_chord]
            ##bass_pitches = [48 + tone for tone in chord_tones]
            ##bass_chord = chord.Chord(bass_pitches)
            ##bass_chord.quarterLength = duration / 4
            ##bass.append(bass_chord)

        #next_note = melody[i+1]
        #if next_note['start'] != curr_note['end']:
            ## Need a rest!
            #rest_duration = next_note['start'] - curr_note['end']
            #print(f'need a rest: {rest_duration}')
            #r = note.Rest()
            #r.quarterLength = rest_duration / 4
            #treble.append(r)
            ##bass.append(r)

        ## Steady bass line - quarter note roots
        ##if curr_note.chord != current_chord and curr_note.chord in CHORD_TEMPLATES:
            ### Only play bass when chord changes
            ##root_tone = CHORD_TEMPLATES[curr_note.chord][0]  # Root note
            ##bass_pitch = 48 + root_tone
            ##bass_note = note.Note(bass_pitch)
            ##bass_note.quarterLength = 1.0  # Steady quarter note
            ##bass.append(bass_note)
            ##current_chord = curr_note.chord
        ##else:
            ### Add rest to maintain steady rhythm
            ##rest = note.Rest()
            ##rest.quarterLength = curr_note.duration
            ##bass.append(rest)

        ##if curr_note.chord != current_chord and curr_note.chord in CHORD_TEMPLATES:
            ### chord has changed
            ##chord_tones = CHORD_TEMPLATES[curr_note.chord]
            ##bass_notes = []
            ##for tone in chord_tones:
                ##bass_pitch = 48 + tone
                ##bass_notes.append(bass_pitch)

            ##bass_chord = chord.Chord(bass_notes)
            ##bass_chord.quarterLength = 1.0
            ##bass.append(bass_chord)

            ##current_chord = curr_note.chord

    ## Need to add last note
    #last_note = melody[-1]
    #pitch = last_note['pitch']
    #n = note.Note(pitch)
    #duration = last_note['end'] - last_note['start']
    #n.quarterLength = duration / 4  # Duration in quarter notes
    #treble.append(n)

    #score.append(treble)
    #score.append(bass)
    #score.write('midi', fp='poloo.mid')
    ##score.write('lilypond.pdf', fp='mmb')

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
