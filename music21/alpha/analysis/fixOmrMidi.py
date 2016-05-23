# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/fixOmrMidi.py
# Purpose:      use MIDI score data to fix OMR scores
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2013-16 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------

from music21 import exceptions21
from music21 import note
from music21 import interval

from music21.alpha.analysis import hasher
from music21.common import numberTools


from collections import Counter
import copy
import numpy as np
import unittest
import os
import operator
import inspect
from __builtin__ import True

pathName = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

K525xmlShortPath = pathName + os.sep + 'k525short3.xml'
K525midiShortPath = pathName + os.sep + 'k525short.mid'
K525omrShortPath = pathName + os.sep + 'k525omrshort.xml'

class OmrMidiException(exceptions21.Music21Exception):
    pass

class AlignmentTracebackException(OmrMidiException):
    pass

class StreamAligner(object):
    
    def __init__(self, targetStream, sourceStream):
        self.targetStream = targetStream
        self.sourceStream = sourceStream
             
        self.h = hasher.Hasher()
        
        self.n = 0
        self.m = 0
        
        self.distMatrix = None
        
        self.changes = []
        self.percentageSimilar = 0
        
    def align(self):
        '''
        main function
        '''
        self.setup()
        self.calculateDistMatrix()
        self.calculateChanges()
        
    def setup(self):
        '''
        populates the hasher object
        creates the matrix of the right size after hashing
        '''
        self.hashedTargetStream = self.h.hashStream(self.targetStream)
        self.hashedSourceStream = self.h.hashStream(self.sourceStream)
        
        self.n = len(self.hashedTargetStream)
        self.m = len(self.hashedSourceStream)
        
        self.distMatrix = np.zeros((self.n+1, self.m+1), dtype=int)
        
        
    def insertCost(self, tup):
        '''
        cost of inserting an extra hashed item.
        
        for now, it's just the size of the tuple
        '''
        return len(tup)
    
    def deleteCost(self, tup):
        '''
        cost of deleting an extra hashed item.
        
        for now, it's just the size of the tuple
        '''
        return len(tup)
        
    def substCost(self, hashedItem1, hashedItem2):
        # don't make this a float yet maybe make the quantization of costs bigger?
        # i.e. 
        if hashedItem1 == hashedItem2:
            return 0 
        
        total = len(hashedItem1)
        for (item1, item2) in zip(hashedItem1, hashedItem2):
            if item1 == item2:
                total -= 1
            elif type(item1) == type(item2) and type(item1) is float:
                if numberTools.almostEquals(item1, item2, grain=.01):
                    total -=1
            else:
                # cost increases 
                total +=1
        for idx, item in enumerate(hashedItem1):
            if hashedItem2[idx] == hashedItem1:
                total += 2
            elif type(item) is float or type(item) is int:
                # check if this is necessary 
                if numberTools.almostEquals(item, hashedItem2[idx], grain=.01):
                    total +=1
            else:
                total -= 1
        return total
    
    def calculateDistMatrix(self):
        '''
        setup and calculations of all the entries
        '''
        # setup all the entries in the first column
        for i in range(1, self.n+1):
            self.distMatrix[i][0] = self.distMatrix[i-1][0] + self.insertCost(self.hashedTargetStream[i-1])
            
        
        # setup all the entries in the first row
        for j in range(1, self.m+1):
            self.distMatrix[0][j] = self.distMatrix[0][j-1] + self.deleteCost(self.hashedSourceStream[j-1])
            
        
        # fill in rest of matrix   
        for i in range(1, self.n+1):
            for j in range(1, self.m+1):
                insertCost = self.insertCost(self.hashedTargetStream[i-1])
                deleteCost = self.deleteCost(self.hashedSourceStream[j-1])
                substCost = self.substCost(self.hashedTargetStream[i-1], self.hashedSourceStream[j-1])
                previousValues = [self.distMatrix[i-1][j] + insertCost, 
                                   self.distMatrix[i][j-1] + deleteCost, 
                                   self.distMatrix[i-1][j-1] + substCost]  

                self.distMatrix[i][j] = min(previousValues)
           
    def calculateChanges(self):
        n = self.n
        m = self.m
        
        while (n != 0 and m != 0):
            currentCost = self.distMatrix[n][m]
            possibleMoves = [self.distMatrix[n-1][m], self.distMatrix[n][m-1], self.distMatrix[n-1][m-1]]
            minIndex, minNewCost = min(enumerate(possibleMoves), key=operator.itemgetter(1))
        
            # 0: insertion, 1: deletion, 2: substitution/nothing
            if minIndex == 0:
                self.changes.insert(0, (self.hashedTargetStream[n-1], self.hashedSourceStream[m-1], 'insertion'))
                n -= 1
                
            elif minIndex == 1:
                self.changes.insert(0, (self.hashedTargetStream[n-1], self.hashedSourceStream[m-1], 'deletion'))
                m -= 1
            elif minIndex == 2 and currentCost == minNewCost:
                self.changes.insert(0, (self.hashedTargetStream[n-1], self.hashedSourceStream[m-1], 'no change'))
                n -= 1
                m -= 1
            else:
                self.changes.insert(0, (self.hashedTargetStream[n-1], self.hashedSourceStream[m-1], 'substitution'))
                n -= 1
                m -= 1
        
        if n != 0 or m != 0:
            raise AlignmentTracebackException('Traceback of best alignment did not end properly')
        
        changesCount = Counter(elem[2] for elem in self.changes)
        self.percentageSimilar = float(changesCount['no change'])/len(self.changes)
        print self.changes
        print self.percentageSimilar
        
    
class OMRmidiNoteFixer(object):
    '''
    Fixes OMR stream according to MIDI information
    '''
    def __init__(self, omrStream, midiStream):
        self.omrStream = omrStream
        self.midiStream = midiStream
        self.correctedStream = copy.deepcopy(self.omrStream)
        
        self.bassDoublesCello = False
        
    def fixStreams(self):
        if self.check_parts():
            pass

        for omrNote, midiNote in zip(self.omrStream, self.midiStream):
            fixerRhythm = OMRmidiNoteRhythmFixer(omrNote, midiNote)
            fixerRhythm.fix()
            fixerPitch = OMRmidiNotePitchFixer(omrNote, midiNote)
            fixerPitch.fix()
    
    def check_parts(self):
        num_midi_parts = len(self.midiStream.parts)
        num_omr_parts = len(self.omrStream.parts)
        
        
        if num_midi_parts == num_omr_parts:
            if num_midi_parts == num_omr_parts + 1:
                if self.check_bass_doubles_cello():
                    return True
                
        else:
            return False
    
    def checkBassDoublesCello(self):
        '''
        check if Bass part doubles Cello 
        '''
        # assume bass part is last part
        bassPart = self.midiStream [-1]
        # assume cello part is penultimate part
        celloPart = self.midiStream[-2]
        
        sa = StreamAligner(bassPart, celloPart)
        sa.h.hashMIDI = False
        sa.h.hashNoteNameOctave = True
        sa.align()
        
        print sa.percentageSimilar
        if sa.percentageSimilar >= .8:
            self.bassDoublesCello = True
        
        return self.bassDoublesCello
        '''
        h = hasher.Hasher()
        h.validTypes = [note.Note, note.Rest]
        h.validTypes = [note.Note, note.Rest]
        h.hasherMIDI = False
        h.hasherNoteName = True
        hasherBass = h.hasher(bassPart)
        hasherCello = h.hasher(celloPart)
        self.bassDoublesCello = hasherBass == hasherCello
        return self.bassDoublesCello
        '''
    
    def alignStreams(self):

        '''
        try a variety of mechanisms to get midiStream to align with omrStream
        '''
        #if self.approxequal(self.omrStream.highestTime, self.midiStream.highestTime):
        #    pass

        # TODO: more ways of checking if stream is aligned
        
        # find the part that aligns the best? or assume already aligned?
        part_pairs = {}
        for omr_part_index, omr_part in enumerate(self.omrStream):
            midi_part = omr_part_index, self.midiStream(omr_part_index)
            part_pairs[omr_part_index] = (omr_part, midi_part)
            
            
        pass
    
    def cursoryCheck(self):
        '''
        check if both rhythm and pitch are close enough together
        '''
        pass
    
class OMRmidiNoteRhythmFixer(object):
    '''
    Fixes an OMR Note pitch according to information from MIDI Note
    '''
    
    def __init__(self, omrNote, midiNote):
        self.omrNote = omrNote
        self.midiNote = midiNote
        self.isPossiblyMisaligned = False
        
    def fix(self):
        pass
    
    
class OMRmidiNotePitchFixer(object):
    '''
    Fixes an OMR Note pitch according to information from MIDI Note
    '''

    def __init__(self, omrNote, midiNote):
        self.omrNote = omrNote
        self.midiNote = midiNote
        self.measure_accidentals = []
        self.isPossiblyMisaligned = False 

    def fix(self):
        # keySignature = self.omrNote.getContextByClass('KeySignature')
        # curr_measure = self.midiNote.measureNumber
        if self.intervalTooBig(self.omrNote, self.midiNote):
            self.isPossiblyMisaligned = True
        else:    
            self.setOMRacc()

    def setOMRacc(self):
        if self.isEnharmonic():
            pass

        if self.hasNatAcc():
            if self.isEnharmonic():
                self.omrNote.pitch.accidental= None
            if len(self.measure_accidentals) == 0:
                self.omrNote.pitch.accidental= self.midiNote.pitch.accidental         
            else:
                self.measure_accidentals.append(self.omrNote.pitch)
        elif self.hasSharpFlatAcc() and self.stepEq():
            if self.hasAcc():
                self.omrNote.pitch.accidental= self.midiNote.pitch.accidental
            else: 
                self.omrNote.pitch.accidental= None

    def isEnharmonic(self):
        return self.omrNote.pitch.isEnharmonic(self.midiNote.pitch)

    def hasAcc(self):
        return self.omrNote.pitch.accidental is not None

    def hasNatAcc(self):
        return self.hasAcc() and self.omrNote.pitch.accidental.name == "natural"

    def hasSharpFlatAcc(self):
        return self.hasAcc() and self.omrNote.pitch.accidental.name != "natural"

    def stepEq(self):
        return self.omrNote.step == self.midiNote.step
    
    def intervalTooBig(self, aNote, bNote, setint = 5):
        if interval.notesToChromatic(aNote, bNote).intervalClass > setint:
            return True
        return False

        
class Test(unittest.TestCase):
    def testEnharmonic(self):
        from music21 import note
        omrNote = note.Note('A#4')
        midiNote = note.Note('B-4')
    
        fixer = OMRmidiNotePitchFixer(omrNote, midiNote)
        fixer.fix()
        self.assertEqual(omrNote.nameWithOctave, 'A#4')
        self.assertEqual(midiNote.nameWithOctave, 'B-4')

    def testSameStep(self):
        from music21 import note, pitch
        omrNote = note.Note('Bn4')
        midiNote = note.Note('B-4')
        self.assertEqual(omrNote.nameWithOctave, 'B4')
        self.assertIsNotNone(omrNote.pitch.accidental)
    
        fixer = OMRmidiNotePitchFixer(omrNote, midiNote)
        fixer.fix()
        
        self.assertEqual(omrNote.nameWithOctave, 'B-4')
        self.assertEqual(midiNote.nameWithOctave, 'B-4')
       
        midiNote.pitch.accidental= pitch.Accidental('sharp')

        
        self.assertEqual(omrNote.nameWithOctave, 'B-4')
        self.assertEqual(midiNote.nameWithOctave, 'B#4')


    def testIntervalNotTooBig(self):
        from music21 import note
        omrNote = note.Note('G-4')
        midiNote = note.Note('A#4')
    
        self.assertIsNotNone(omrNote.pitch.accidental)
    
        fixer = OMRmidiNotePitchFixer(omrNote, midiNote)
        fixer.fix()
        self.assertEqual(omrNote.nameWithOctave, 'G-4')
        self.assertEqual(midiNote.nameWithOctave, 'A#4')
        self.assertFalse(fixer.isPossiblyMisaligned)
        
    def testNotSameStep(self):
        from music21 import note
        omrNote = note.Note('En4')
        midiNote = note.Note('B-4')
    
        self.assertIsNotNone(omrNote.pitch.accidental)
        fixer = OMRmidiNotePitchFixer(omrNote, midiNote)
        fixer.fix()
        self.assertEqual(omrNote.nameWithOctave, 'E4')
        self.assertEqual(midiNote.nameWithOctave, 'B-4')
        self.assertTrue(fixer.isPossiblyMisaligned)
        
    def testK525BassCelloDouble(self):
        '''
        K525's bass part doubles the cello part. don't hash the octave
        '''
        from music21 import converter
        from music21.alpha.analysis import hasher
        
        midiFP = K525midiShortPath
        omrFP = K525omrShortPath
        midiStream = converter.parse(midiFP)
        omrStream = converter.parse(omrFP)
        
        fixer = OMRmidiNoteFixer(omrStream, midiStream)
        celloBassAnalysis = fixer.checkBassDoublesCello()
        self.assertEqual(celloBassAnalysis, True)
        
    def testSameSimpleStream(self):
        '''
        two streams of the same notes should have 1.0 percentage similarity
        '''
        from music21 import stream
        from music21 import note
          
        target = stream.Stream()
        source = stream.Stream()
          
        note1 = note.Note("C4")
        note2 = note.Note("D4")
        note3 = note.Note("E4")
        note4 = note.Note("F4")
          
        target.append([note1, note2, note3, note4])
        source.append([note1, note2, note3, note4])
          
        sa = StreamAligner(target, source)
        sa.align()
        
        self.assertEqual(sa.percentageSimilar, 1.0)
         
    def testSameOneOffStream(self):
        '''
        two streams with just 1 note different should have .75 percentage similarity
        '''
        from music21 import stream
        from music21 import note
         
        target = stream.Stream()
        source = stream.Stream()
         
        note1 = note.Note("C4")
        note2 = note.Note("D4")
        note3 = note.Note("E4")
        note4 = note.Note("F4")
        note5 = note.Note("G4")
         
        target.append([note1, note2, note3, note4])
        source.append([note1, note2, note3, note5])
         
        sa = StreamAligner(target, source)
        sa.align()
        
        self.assertEqual(sa.percentageSimilar, .75)
        
    def testOneOffDeletionStream(self):
        '''
        two streams, both the same, but one has an extra note should have .75 percentage similarity
        '''
        from music21 import stream
        from music21 import note
         
        target = stream.Stream()
        source = stream.Stream()
         
        note1 = note.Note("C4")
        note2 = note.Note("D4")
        note3 = note.Note("E4")
        note4 = note.Note("F4")
         
        target.append([note1, note2, note3, note4])
        source.append([note1, note2, note3])
         
        sa = StreamAligner(target, source)
        sa.align()
        
        self.assertEqual(sa.percentageSimilar, .75)

## this test is included in the quarterLengthDivisor PR in the converter.py tests
# class ParseTestExternal(unittest.TestCase):
#     def testParseMidi(self):
#         from music21 import converter
#         midiStream = converter.parse(K525midiShortPath, forceSource=True, quarterLengthDivisors=[4])
#         midiStream.show()

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)