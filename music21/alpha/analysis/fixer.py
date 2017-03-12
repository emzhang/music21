# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/fixer.py
# Purpose:      Aligns two streams
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2016 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------

class OMRMidiPitchFixer(object):
    '''
    Given MIDI and OMR stream and list of changes between the two, corrects the OMR note pitch
    '''
    def __init__(self, omrStream, midiStream, changes):
        self.omrStream = omrStream
        self.midiStream = midiStream
        self.changes = changes
# 
# class OMRMidiRhythmFixer(object):
#     '''
#     Given MIDI and OMR stream and list of changes between the two, corrects the OMR note rhythm
#     '''
#         self.omrStream = omrStream
#         self.midiStream = midiStream
#         self.changes = changes