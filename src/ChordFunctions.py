def calc_semitones_to_c(note: str):
    """
    Calculate the number of semitones needed to transpose a given note to C.

    Args:
        note (str): A musical note name (e.g., 'C', 'C#', 'Db', 'D', etc.)

    Returns:
        int: The number of semitones (0-11) to shift up to transpose the note to C

    Raises:
        KeyError: If the provided note is not a valid note name
    """
    note_to_pitch_class = {
        'C': 0, 
        'C#': 1, 'Db': 1,
        'D': 2,
        'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5,
        'F#': 6, 'Gb': 6,
        'G': 7,
        'G#': 8, 'Ab': 8,
        'A': 9,
        'A#': 10, 'Bb': 10,
        'B': 11
    }

    shift = (0 - note_to_pitch_class[note]) % 12
    return shift

def transpose_note(note: str, semitones_to_shift: int):
    """
    Transpose a note to a new pitch by shifting semitones.

    Args:
        note (str): The note to transpose (e.g., 'C', 'D#', 'Gb').
        semitones_to_shift (int): Number of semitones to shift (positive for up, negative for down).

    Returns:
        str: The transposed note name using sharp notation (e.g., 'C#', 'F#').

    Raises:
        KeyError: If chord_root_note is not a valid note name.
    """
    note_to_pitch_class = {
        'C': 0, 
        'C#': 1, 'Db': 1,
        'D': 2,
        'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5,
        'F#': 6, 'Gb': 6,
        'G': 7,
        'G#': 8, 'Ab': 8,
        'A': 9,
        'A#': 10, 'Bb': 10,
        'B': 11
    }

    pitch_class_to_note = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
        7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    current_pitch_class = note_to_pitch_class[note]

    shifted_pitch_class = (current_pitch_class + semitones_to_shift) % 12

    return pitch_class_to_note[shifted_pitch_class]

def get_chord_root_and_type(chord: str):
    """
    Extracts the root note and chord type from a chord string.

    Args:
        chord (str): A chord string in the format "root:type" (e.g., "C:maj", "G:min").

    Returns:
        tuple: A tuple containing (chord_root_note, chord_type) where both are strings.

    Raises:
        ValueError: If the chord string is an invalid format.
    """
    parts = chord.split(':')
    if len(parts) >= 2:
        chord_root_note = parts[0]
        chord_type = parts[1]
        return (chord_root_note, chord_type)
    else: 
        raise ValueError(f"Invalid chord format: {chord}")

def transpose_chord_to_c_major(chord, original_key):
    """
    Transpose a chord from its original key to C major.

    This function takes a chord and the original key it belongs to, then transposes
    the chord to the equivalent chord in C major by calculating the number of semitones
    needed to shift from the original key to C.

    Args:
        chord (str): The chord to transpose, typically in the format "note:type"
                     (e.g., "D:maj", "F#:min").
        original_key (str): The original key of the chord, which may include additional
                            information after a colon (e.g., "G:major", "D:min"). Only
                            the key note before the colon is used.

    Returns:
        str: The transposed chord in C major, formatted as "note:type"
             (e.g., "C:maj", "A:min").
    """
    original_key = original_key.split(':')[0]
    num_semitones_to_shift = calc_semitones_to_c(original_key)

    (chord_root_note, chord_type) = get_chord_root_and_type(chord)
    transposed_chord_root_note = transpose_note(chord_root_note, num_semitones_to_shift)
    transposed_chord = f'{transposed_chord_root_note}:{chord_type}'

    return transposed_chord

def get_chord_name_in_original_key(roman_numeral: str, key: str):
    roman_numeral_to_semitones = {
        'I': 0,   # Tonic (0 semitones above root)
        'ii': 2,  # 2 semitones above root
        'iii': 4, # 4 semitones above root
        'IV': 5,  # 5 semitones above root
        'V': 7,   # 7 semitones above root ← We need this!
        'vi': 9,  # 9 semitones above root
        'vii': 11 # 11 semitones above root
    }

    note_to_pitch_class = {
        'C': 0, 
        'C#': 1, 'Db': 1,
        'D': 2,
        'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5,
        'F#': 6, 'Gb': 6,
        'G': 7,
        'G#': 8, 'Ab': 8,
        'A': 9,
        'A#': 10, 'Bb': 10,
        'B': 11
    }

    (key_root_note, _) = get_chord_root_and_type(key)
    key_root_note_number = note_to_pitch_class[key_root_note]
    scale_num_semitones = roman_numeral_to_semitones[roman_numeral]
    chord_pitch_class = (key_root_note_number + scale_num_semitones) % 12

    pitch_class_to_note = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#',
        7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    chord_root_name = pitch_class_to_note[chord_pitch_class]
    chord_type = get_chord_type_from_roman_numeral(roman_numeral)
    transposed_chord_name = f"{chord_root_name}:{chord_type}"

    return transposed_chord_name

def get_chord_type_from_roman_numeral(roman_numeral: str):
    if roman_numeral.isupper():
        return 'maj'
    else:
        return 'min'

print(get_chord_name_in_original_key('vii', 'A:maj'))