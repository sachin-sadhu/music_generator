import music21 as m21

us = m21.environment.UserSettings()

us['lilypondPath'] = "/usr/bin/lilypond"

s = m21.stream.Stream()

notes = ['C4']

s.append(m21.note.Note('C4', quarterLength=1))
s.write('lilypond.png', fp='output.png')