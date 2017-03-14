# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/fixer.py
# Purpose:      Fixes two streams given a list of changes between them
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2016 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
from music21.alpha.analysis import aligner
import unittest

class OMRMidiFixer(object):
    '''
    Base class for future fixers
    '''
    def __init__(self, omrStream, midiStream, changes):
        self.omrStream = omrStream
        self.midiStream = midiStream
        self.changes = changes
        
    def getChunks(self):
        '''
        Method for identifying "chunks" of streams to work with.
        Identifies and splits up chunks in the stream that are flanked by No Change Change Tuples
        '''
        pass
        
class StaccatoFixer(OMRMidiFixer):
    '''
    Fixer for passages that are misaligned because of staccato phrasings.
    
    The idea is that misaligned staccato passages will have repeated note, rest patterns in the 
    MIDI and longer sustained notes in the OMR. 
    
    '''
    def __init__(self, omrStream, midiStream, changes):
        super().__init__(omrStream, midiStream, changes)
        self.staccatoChunks = []
                
    def getStaccatoChunks(self):
        '''
        populates self.staccatoChunks with pairs of stream excerpts that have staccato mismatch
        
        >>> midiPart = converter.parse("tinyNotation: b- trip{a4 r8} trip{a4 r8} trip{a4 r8} c'4 trip{b-4 r8} trip{a4 r8} trip{b-4 r8}")
        >>> omrPart = converter.parse("tinyNotation: b-4 a a a c' b- a b-")
        >>> midiStream = stream.Part()
        >>> omrStream = stream.Part()
        >>> midiStream.append(midiPart)
        >>> omrStream.append(omrPart)
        >>> omc = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream, omrStream)
        >>> omc.processRunner()
        >>> changes = omc.changes
        >>> changes
        >>> omsf = alpha.analysis.fixer.StaccatoFixer(omrStream, midiStream, changes)
        >>> omsf.getStaccatoChunks()
        []
        '''
        for (midiNoteRef, omrNoteRef, changeOp) in self.changes:
            if changeOp == aligner.ChangeOps.NoChange:
                pass
        pass
                
        
    
    def fixStaccatoChunks(self):
        '''
        >>> midiPart = converter.parse("tinyNotation: b- trip{a4 r8} trip{a4 r8} trip{a4 r8} c trip{c4 r8} trip{b-4 r8} trip{a4 r8}")
        >>> omrPart = converter.parse("tinyNotation: b-4 a a a c' b- a b-")
        >>> midiStream = stream.Part()
        >>> omrStream = stream.Part()
        >>> midiStream.append(midiPart)
        >>> omrStream.append(omrPart)
        >>> omc = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream, omrStream)
        >>> omc.processRunner()
        >>> changes = omc.changes
        >>> omsf = alpha.analysis.fixer.StaccatoFixer(omrStream, midiStream, changes)
        '''
        pass

class Test(unittest.TestCase):
    pass

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)