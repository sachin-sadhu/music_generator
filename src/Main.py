import mido
from mido import MidiFile, MidiTrack, Message
from Preprocessing import *
from ChordFunctions import *
from ChordTraining import *
from NoteEmission import initalise_tmat, get_note_midi_pitch
from HMM import HMM
import numpy as np

def generate_song(num_bars):
    directory = '/cs/home/slzys1/Documents/music_generator/test_data'
    bars, bars_chords = load_songs(directory)
    filtered_bars, filtered_bars_chords = filter_empty_bars_no_chord(bars, bars_chords)

    num_patterns = 2
    max_duration = 10
    obs_array = np.array(filtered_bars, dtype=object)
    durations = np.random.dirichlet(np.ones(max_duration), size=num_patterns)
    tmat = initalise_tmat(num_patterns)

    note_emission = NoteEmission(num_patterns, filtered_bars_chords)
    note_emission_hsmm = HSMMModel(
        note_emission, durations, tmat
    )

    note_emission.set_context(filtered_bars_chords, filtered_bars_chords)
    result = note_emission_hsmm.fit(obs_array)
    decoded_states = note_emission_hsmm.decode(filtered_bars)
    key = 'C:maj'

    pattern_bars = build_pattern_bars_dict(filtered_bars, filtered_bars_chords, decoded_states)
    chains = build_chains(num_patterns, pattern_bars)

    _, states = note_emission_hsmm.sample(num_bars)
    chord_model = ChordTransitionModel()
    chord_model.train(filtered_bars_chords)
    
    sampled_chord_sequence = chord_model.generate_chord_sequence(num_bars)
    sampled_song = []
    for i in range(num_bars):
        bar_pattern = states[i]
        bar_chord = sampled_chord_sequence[i]

        print(f'bar pattern: {bar_pattern}. bar chord: {bar_chord}')
        
        chain = chains[bar_pattern]
        sampled_bar = chain.sample_bar(bar_chord)

        bar_midi_note = []
        for note in sampled_bar:
            chord_tone, _, note_duration, note_onset = note
            note_midi_pitch = get_note_midi_pitch(chord_tone, bar_chord, key)
            note_formatted = (note_midi_pitch, note_duration, note_onset)
            bar_midi_note.append(note_formatted)

        sampled_song.append(bar_midi_note)

    return sampled_song

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

    note_events = []
    bar_length_beats = 4  # Assuming 4/4 time
    current_bar = 0
    last_bar_position = -1

    for note in notes:
        note_pitch, note_duration, bar_position = note
        
        # Detect when we move to a new bar (bar_position resets to 0 or goes backward)
        if bar_position <= last_bar_position:
            current_bar += 1
        
        last_bar_position = bar_position
        
        # Calculate absolute beat position
        absolute_beats = (current_bar * bar_length_beats) + bar_position
        onset_ticks = int(ticks_per_beat * absolute_beats)

        # Calculate duration
        duration_beats = duration_to_beats_map.get(note_duration, 1.0)
        duration_ticks = int(ticks_per_beat * duration_beats)

        # Store note_on and note_off events
        note_events.append(('note_on', onset_ticks, note_pitch))
        note_events.append(('note_off', onset_ticks + duration_ticks, note_pitch))

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

if __name__ == "__main__":
    notes_file = '/home/sachin/Documents/music_generator/short_test_data/004/004.mid'
    beat_file = '/home/sachin/Documents/music_generator/short_test_data/004/beat_midi.txt'
    chord_file = '/home/sachin/Documents/music_generator/short_test_data/004/chord_midi.txt'
    key_file = '/home/sachin/Documents/music_generator/short_test_data/004/key_audio.txt'

    beat_timing = load_beat_timings(beat_file)
    notes = load_midi_notes(notes_file)
    chord_timings = load_chord_timings(chord_file)
    song_key = load_key(key_file)

    groupings = get_ornament_groupings(notes, beat_timing)
    first_one = None
    for group in groupings:
        if len(group[2]) >= 3:
            first_one = group
            break
    groupings = [first_one]
    processed = prep_groupings_for_second_layer(groupings, chord_timings, song_key)
    print(groupings)
    print('\n')
    print(processed)
    
    
    #hmm = HMM()
    #hmm.train_model(notes, beat_chords)
    #print(hmm.duration_matrix)
    #print(hmm.transition_matrix)

    #sampled_beats = hmm.sample(5)
    #notes = [(beat[0], beat[1], 'crotchet', 0.0) for beat in sampled_beats]
    #key = "C:maj"

    #converted_notes = []
    #beat_counter = 0
    #for beat in sampled_beats:
        #beat_onset = beat_counter % 4 
        #chord_tone, octave_offset, note_chord_function = beat
        #note_midi_pitch = get_note_midi_pitch(chord_tone, note_chord_function, key, octave_offset)
        #converted_notes.append((note_midi_pitch, 'crotchet', beat_onset))
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
