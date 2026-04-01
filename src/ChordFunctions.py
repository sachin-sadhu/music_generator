from __future__ import annotations
from typing import TYPE_CHECKING
from config import note_to_pitch_class, pitch_class_to_note, roman_numeral_to_semitones, root_to_degree

if TYPE_CHECKING:
    from Timings import ChordTiming, KeyTiming

def get_event_matching_chord(onset_seconds, chord_timings: list[ChordTiming]) -> ChordTiming:
    for chord in chord_timings:
        if chord.get_chord_start() <= onset_seconds <= chord.get_chord_end():
            return chord
    raise ValueError("No matching chord found.")

def calc_semitones_to_c(note: str):
    """
    Calculate the number of semitones up to the next C note.

    Args:
        note (str): A musical note name (e.g., 'C', 'C#', 'Db', 'D', etc.)

    Returns:
        int: The number of semitones (0-11) to shift up to transpose the note to C

    Raises:
        KeyError: If the provided note is not a valid note name
    """
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

    current_pitch_class = note_to_pitch_class[note]
    shifted_pitch_class = (current_pitch_class + semitones_to_shift) % 12
    return pitch_class_to_note[shifted_pitch_class]

def get_key_root_and_type(key: str):
    """
    Extracts the root note and chord type from a key .

    Args:
        key (str): A chord string in the format "root:type" (e.g., "C:maj", "G:min").

    Returns:
        tuple: A tuple containing (key_root_note, key_type) where both are strings.

    Raises:
        ValueError: If the chord string is an invalid format.
    """
    parts = key.split(':')
    if len(parts) >= 2:
        key_root_note = parts[0]
        key_type = parts[1]
        return (key_root_note, key_type)
    else: 
        raise ValueError(f"Invalid key format: {key}")

def transpose_chord_to_c_major(chord: ChordTiming, song_key: KeyTiming):
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
    try:
        key_root_note = song_key.get_root_note()
        num_semitones_to_shift = calc_semitones_to_c(key_root_note)

        chord_root_note = chord.get_root_note()
        chord_type = chord.get_type()
        transposed_chord_root_note = transpose_note(chord_root_note, num_semitones_to_shift)
        transposed_chord = f'{transposed_chord_root_note}:{chord_type}'
    except Exception as e:
        raise ValueError(f"Error transposing: {e}")

    return transposed_chord

def get_chord_name_in_original_key(roman_numeral: str, key: KeyTiming):
    """
    Convert a Roman numeral chord notation to its actual chord name in a given key.

    This function takes a Roman numeral representation of a chord (e.g., 'V', 'ii', 'vii') 
    and a key signature, then calculates the actual chord name by transposing the Roman 
    numeral to the specified key using semitone intervals.

    Args:
        roman_numeral (str): The Roman numeral representation of the chord (e.g., 'I', 'ii', 
                            'iii', 'IV', 'V', 'vi', 'vii'). Case indicates chord quality.
        key (str): The key signature to transpose to (e.g., 'C:maj', 'D#:maj', 'Bb:min').

    Returns:
        str: The transposed chord name in the format "root:type" (e.g., "G:maj", "A:min").

    Example:
        >>> get_chord_name_in_original_key('V', 'C')
        'G:maj'
        >>> get_chord_name_in_original_key('ii', 'G')
        'A:min'
    """
    key_root_note = key.get_root_note()
    key_root_note_number = note_to_pitch_class[key_root_note]
    scale_num_semitones = roman_numeral_to_semitones[roman_numeral]
    chord_pitch_class = (key_root_note_number + scale_num_semitones) % 12

    chord_root_name = pitch_class_to_note[chord_pitch_class]
    chord_type = get_chord_type_from_roman_numeral(roman_numeral)
    transposed_chord_name = f"{chord_root_name}:{chord_type}"

    return transposed_chord_name

def get_chord_type_from_roman_numeral(roman_numeral: str):
    """
    Determines the chord type (major or minor) based on the case of a Roman numeral.

    Args:
        roman_numeral (str): A Roman numeral string representing a chord (e.g., 'I', 'iv', 'V', 'ii')

    Returns:
        str: 'maj' if the Roman numeral is uppercase (indicating a major chord),
             'min' if the Roman numeral is lowercase (indicating a minor chord)
    """
    if roman_numeral.isupper():
        return 'maj'
    else:
        return 'min'

def convert_chord_name_to_roman_numeral(chord_name: str):
    """
    Convert a chord name to its corresponding Roman numeral notation.

    This function maps chord names in the format 'Note:quality' to their 
    Roman numeral equivalents in music theory, specifically for the key of C major.

    Args:
        chord_name (str): The chord name in the format 'Note:quality' 
                         (e.g., 'C:maj', 'D:min', 'E:min').

    Returns:
        str: The Roman numeral representation of the chord.
             - Uppercase Roman numerals (I, IV, V) represent major chords
             - Lowercase Roman numerals (ii, iii, vi, vii) represent minor/diminished chords

    Raises:
        ValueError: If the provided chord_name is not found in the mapping.

    Example:
        >>> convert_chord_name_to_roman_numeral('C:maj')
        'I'
        >>> convert_chord_name_to_roman_numeral('D:min')
        'ii'
        >>> convert_chord_name_to_roman_numeral('X:maj')
        ValueError: Invalid chord name: X:maj
    """

    if chord_name == 'N':
        return 'N'

    root_note = chord_name.split(':')[0]
    if root_note not in root_to_degree:
        raise ValueError(f"Invalid root note: {root_note}")

    return root_to_degree[root_note]