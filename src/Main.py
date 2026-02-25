from music21 import stream, note, tempo, instrument, chord
from Preprocessing import *
from ChordFunctions import *
from HMM import HMM
from SecondLayerHMM import Generator
from Timings import KeyTiming

def notes_to_midi(notes, filename='jabooboo.mid', bpm=80):
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
    score = stream.Score()

    # Treble clef part
    treble = stream.Part()
    treble.append(instrument.Piano())
    treble.append(tempo.MetronomeMark(number=bpm))

    # Bass clef part
    bass = stream.Part()
    bass.append(instrument.Piano())

    current_chord = None
    
    for curr_note in notes:
        # Treble melody note
        pitch = curr_note.midi_pitch
        n = note.Note(pitch)

        duration = duration_to_beats_map.get(curr_note.duration, 1.0)
        n.quarterLength = duration  # Duration in quarter notes
        treble.append(n)

        if curr_note.chord != current_chord and curr_note.chord in CHORD_TEMPLATES:
            # chord has changed
            chord_tones = CHORD_TEMPLATES[curr_note.chord]
            bass_notes = []
            for tone in chord_tones:
                bass_pitch = 48 + tone
                bass_notes.append(bass_pitch)

            bass_chord = chord.Chord(bass_notes)
            bass_chord.quarterLength = 1.0
            bass.append(bass_chord)

            current_chord = curr_note.chord

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
                chord_tones.append((midi_pitch, 'crotchet', beat_onset))
                print(f'beat onset: {beat_onset}')

    print(chord_tones)

if __name__ == "__main__":
    directory = "/home/sachin/Documents/music_generator/POP909/POP909"
    data = TrainingDataProcessedInfo()
    data.load_training_data(directory)

    chord_beat_hmm = HMM()
    chord_beat_hmm.train_model(data.notes, data.beat_chords)

    ornament_hmms = OrnamentNoteHMMs(data.ornament_groupings)
    ornament_hmms.train_hmms()

    song = Generator(chord_beat_hmm, ornament_hmms)
    key = KeyTiming('C:maj')
    melody_sequence = song.generate(key)
    print(melody_sequence)

    notes_to_midi(melody_sequence)
    
    #converted_notes = []
    #beat_counter = 0
    #for count, note in enumerate(sequence):
        #beat_onset = count % 4
        #converted_notes.append((note, 'crotchet', beat_onset))
        #beat_counter += 1

    #print(converted_notes)
    #save_to_midi(converted_notes, "tone.mid")

    #print(split)

    #chord_beat_hmm = HMM()
    #chord_beat_hmm.train_model(song_notes, all_song_beat_chords)
    #sampled_beats = chord_beat_hmm.sample(5)

    #sampled_beats = [(get_note_midi_pitch(note[0], note[-1], 'C:maj', note[1]), note[-1]) for note in sampled_beats]
    #fill_chord_triad(sampled_beats, 'C:maj')
    #second_layer = SecondLayerGen()
    #second_layer.train_hmms(split)
    #filled = fill_ornament_notes(sampled_beats, second_layer)
    #print(filled)

    #converted_notes = []
    #beat_counter = 0
    #for count, note in enumerate(filled):
        #beat_onset = count % 4
        #converted_notes.append((note, 'crotchet', beat_onset))
        #beat_counter += 1

    #print(converted_notes)
    #save_to_midi(converted_notes, "what.mid")

    #num_patterns = 3
    #max_duration = 10
    #obs_array = np.array(filtered_bars, dtype=object)
    #durations = np.random.dirichlet(np.ones(max_duration), size=num_patterns)
    #tmat = initalise_tmat(num_patterns)

    #note_emission = NoteEmission(num_patterns, filtered_bars_chords)
    #note_emission_hsmm = HSMMModel(
        #note_emission, durations, tmat
    #)

    #note_emission.set_context(filtered_bars_chords, filtered_bars_chords)
    #result = note_emission_hsmm.fit(obs_array)
    #decoded_states = note_emission_hsmm.decode(filtered_bars)
    #key = 'C:maj'

    #pattern_bars = build_pattern_bars_dict(filtered_bars, filtered_bars_chords, decoded_states)
    #chains = build_chains(num_patterns, pattern_bars)

    #num_bars = 10

    #_, states = note_emission_hsmm.sample(num_bars)

   
    #sampled_chords = ['I', 'V', 'vi', 'IV', 'I', 'V', 'vi', 'IV', 'I', 'I']
    #sampled_skeleton_notes = [('root', 'root') for _ in range(num_bars)]

    #print(f'generated chords: {sampled_chords}')
    #print(f'skeleton notes: {sampled_skeleton_notes}')
    ## Want to use skeleton notes provided by emitted bars as the start in the markov chain
    #fully_sampled_bars = []
    #for i, bar_skeleton_note in enumerate(sampled_skeleton_notes):
        #skeleton_note_tone = bar_skeleton_note[0]
        #bar_markov_chain: PatternMarkovChain = chains[states[i]]
        #sampled_bar = bar_markov_chain.sample_bar(skeleton_note_tone, sampled_chords[i])
        #fully_sampled_bars.append(sampled_bar)

    ##converted_bars = []
    ##for bar in emitted_bars:
        ##bar_notes = []
        ##bar_notes.append((bar[0], 0, 'crotchet', 1.0))
        ##bar_notes.append((bar[1], 0, 'crotchet', 3.0))
        ##converted_bars.append(bar_notes)

    #bars_midi_pitch = []
    #for i, bar in enumerate(fully_sampled_bars):
        #bar_midi_notes = []
        #bar_chord = sampled_chords[i]
        #for note in bar:
            #chord_tone, _, note_duration, note_onset = note
            #note_midi_pitch = get_note_midi_pitch(chord_tone, bar_chord, key)
            #note_formatted = (note_midi_pitch, note_duration, note_onset)
            #bar_midi_notes.append(note_formatted)
        #bars_midi_pitch.append(bar_midi_notes)

    #print(bars_midi_pitch)
