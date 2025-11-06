from music21 import converter

def midi_to_musicxml(midi_file_path):
    # Load the MIDI file
    midi_file = converter.parse(midi_file_path)
    
    # Export as MusicXML
    musicxml_file_path = midi_file_path.replace('.mid', '.xml')
    midi_file.write('musicxml', fp=musicxml_file_path)
    
    print(f'MusicXML file saved as {musicxml_file_path}')
    return musicxml_file_path

def musicxml_to_pdf(musicxml_file_path):
    # Use MuseScore CLI to convert MusicXML to PDF
    import os
    pdf_file_path = musicxml_file_path.replace('.xml', '.pdf')
    os.system(f'musescore3 {musicxml_file_path} -o {pdf_file_path}')
    
    print(f'PDF file saved as {pdf_file_path}')

def midi_to_pdf(midi_file_path):
    # Convert MIDI to MusicXML
    musicxml_file_path = midi_to_musicxml(midi_file_path)
    
    # Convert MusicXML to PDF
    musicxml_to_pdf(musicxml_file_path)

# Example usage:
midi_file = 'magenta_output/2025-11-05_164121_1.mid'
midi_to_pdf(midi_file)
