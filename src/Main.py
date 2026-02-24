import mido
from mido import MidiFile, MidiTrack, Message
from Preprocessing import *
from ChordFunctions import *
from HMM import HMM
from SecondLayerGen import *

def save_to_midi(notes, output_path='output.mid', tempo=500000, ticks_per_beat=480):
    """
    Save notes to MIDI file.
    
    Args:
        notes: List of tuples (pitch, duration, bar_position)
               - pitch: MIDI pitch (0-127)
               - duration: duration name (e.g., 'crotchet', 'quaver') 
               - bar_position: position within bar (0-3 for 4/4 time)
    """
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

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Group notes by their absolute time position
    time_groups = {}
    current_bar = 0
    last_max_position = -1
    
    for note in notes:
        note_pitch, note_duration, bar_position = note
        
        # Detect new bar when bar_position cycles back or decreases after reaching a higher value
        if bar_position == 0 and last_max_position >= 2:
            current_bar += 1
        
        last_max_position = max(last_max_position, bar_position)
        if bar_position == 0:
            last_max_position = 0
            
        # Calculate absolute beat position
        absolute_beats = (current_bar * 4) + bar_position  # 4 beats per bar
        
        if absolute_beats not in time_groups:
            time_groups[absolute_beats] = []
        
        time_groups[absolute_beats].append((note_pitch, note_duration))
    
    # Create MIDI events
    note_events = []
    
    for absolute_beats, note_group in time_groups.items():
        onset_ticks = int(ticks_per_beat * absolute_beats)
        
        for note_pitch, note_duration in note_group:
            # Calculate duration
            duration_beats = duration_to_beats_map.get(note_duration, 1.0)
            duration_ticks = int(ticks_per_beat * duration_beats)

            # Store note_on and note_off events
            note_events.append(('note_on', onset_ticks, note_pitch))
            note_events.append(('note_off', onset_ticks + duration_ticks, note_pitch))

    # Sort all events by time, then by event type (note_off before note_on for same time)
    note_events.sort(key=lambda x: (x[1], x[0] == 'note_on'))
    
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

def fill_ornament_notes(skeleton_notes, ornament_generator: SecondLayerGen):
    ### skeelton notes should be in format (midi_pitch, duration, chord_function) 
    full_sequence = []
    for i in range(len(skeleton_notes)-1):
        note_1_pitch = skeleton_notes[i][0]
        note_2_pitch = skeleton_notes[i+1][0]
        offset = note_1_pitch - note_2_pitch
        chord_function = skeleton_notes[i][-1]

        ornament_notes = ornament_generator.generate_sequence(offset, chord_function)
        full_sequence.append(note_1_pitch)

        for ornament_note in ornament_notes:
            print('yeet')
            note_midi = full_sequence[-1] + ornament_note
            full_sequence.append(note_midi)

    final_note_pitch = skeleton_notes[-1][0]
    full_sequence.append(final_note_pitch)
    return full_sequence

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

    save_to_midi(chord_tones, 'doon.mid')

if __name__ == "__main__":
    directory = "/cs/home/slzys1/Documents/music_generator/short_test_data/"
    data = TrainingDataProcessedInfo()
    data.load_training_data(directory)
    split = split_song_ornaments(groupings)
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
