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

def save_to_midi(bars, output_path='output.mid', tempo=500000, ticks_per_beat=480):
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

    # for each note, we need to track when to send the note_on 
    # for each note, we send out a note_on and note_off message
    # note_off should be sent with a time_delta equal to note_on + note_duration in ticks

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    current_tick = 0
    bar_length_beats = 4
    note_events = []

    for bar_index, bar in enumerate(bars):
        bar_start_beat = bar_index * bar_length_beats

        for note in bar:
            note_pitch, note_duration, note_bar_onset = note

            # Tracks where in the bar the note should appear
            beat_onset = bar_start_beat + note_bar_onset
            bar_tick_onset = int(ticks_per_beat * beat_onset)

            # Calculate duration to ticks
            duration_beats = duration_to_beats_map.get(note_duration, 1.0)
            duration_ticks = int(ticks_per_beat * duration_beats)

            # Store note_on and note_off events
            note_events.append(('note_on', bar_tick_onset, note_pitch))
            note_events.append(('note_off', bar_tick_onset + duration_ticks, note_pitch))

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
    #song = generate_song(20)
    #print(song)
    #save_to_midi(song)
    directory = '/home/sachin/Documents/music_generator/test_data'
    bars, bars_chords = load_songs(directory)
    filtered_bars, filtered_bars_chords = filter_empty_bars_no_chord(bars, bars_chords)
    beat_onset_tuple_list = create_chord_beat_onset_tuple_structure(filtered_bars, filtered_bars_chords)
    print(beat_onset_tuple_list)

    chord_model = ChordTransitionModel()
    chord_model.train(filtered_bars_chords)
    sampled_chord_sequence = chord_model.generate_chord_sequence(10)

    num_patterns = 3
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

    _, states = note_emission_hsmm.sample(10)

    hmm = HMM()
    hmm.train_model(beat_onset_tuple_list)
    sampled_chords, sampled_skeleton_notes = hmm.sample(10)

    print(f'generated chords: {sampled_chords}')
    print(f'skeleton notes: {sampled_skeleton_notes}')
    # Want to use skeleton notes provided by emitted bars as the start in the markov chain
    fully_sampled_bars = []
    for i, bar_skeleton_note in enumerate(sampled_skeleton_notes):
        skeleton_note_tone = bar_skeleton_note[0]
        bar_markov_chain: PatternMarkovChain = chains[states[i]]
        sampled_bar = bar_markov_chain.sample_bar(skeleton_note_tone, sampled_chords[i])
        fully_sampled_bars.append(sampled_bar)

    #converted_bars = []
    #for bar in emitted_bars:
        #bar_notes = []
        #bar_notes.append((bar[0], 0, 'crotchet', 1.0))
        #bar_notes.append((bar[1], 0, 'crotchet', 3.0))
        #converted_bars.append(bar_notes)

    bars_midi_pitch = []
    for i, bar in enumerate(fully_sampled_bars):
        bar_midi_notes = []
        bar_chord = sampled_chord_sequence[i]
        for note in bar:
            chord_tone, _, note_duration, note_onset = note
            note_midi_pitch = get_note_midi_pitch(chord_tone, bar_chord, key)
            note_formatted = (note_midi_pitch, note_duration, note_onset)
            bar_midi_notes.append(note_formatted)
        bars_midi_pitch.append(bar_midi_notes)

    print(bars_midi_pitch)
    save_to_midi(bars_midi_pitch, "yote.mid")
