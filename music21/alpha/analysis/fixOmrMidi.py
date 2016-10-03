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
'''
requires numpy
'''
from music21 import base as base
from music21 import exceptions21
from music21 import interval

from music21.alpha.analysis import hasher
from music21.common import numberTools

from collections import Counter
import copy
import inspect
import os
import operator
import unittest

pathName = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

K525xmlShortPath = pathName + os.sep + 'k525short3.xml'
K525midiShortPath = pathName + os.sep + 'k525short.mid'
K525omrShortPath = pathName + os.sep + 'k525omrshort.xml'   

class OmrMidiException(exceptions21.Music21Exception):
    pass

class AlignmentTracebackException(OmrMidiException):
    pass


class StreamAligner(object):
    """
    Stream Aligner object for two streams
    
    """
    
    def __init__(self, targetStream, sourceStream):
        self.targetStream = targetStream
        self.sourceStream = sourceStream
             
        self.h = hasher.Hasher()
        self.hashedTargetStream = self.h.hashStream(self.targetStream)
        self.hashedSourceStream = self.h.hashStream(self.sourceStream)
        
        # n and m will be the dimensions of the Distance Matrix we set up
        self.n = len(self.hashedTargetStream)
        self.m = len(self.hashedSourceStream)
        
        self.distMatrix = None
        
        self.changes = []
        self.percentageSimilar = 0
        
    def align(self):
        '''
        main function
        '''
        self.setupDistMatrix()
        self.populateDistMatrix()
#         self.calculateDistMatrix()
        self.calculateChanges()
        
    def setupDistMatrix(self):
        # TODO: why setup and setupDistMatrix?
        '''
        populates the hasher object
        creates the matrix of the right size after hashing
        '''
        if ('numpy' in base._missingImport):
            raise OmrMidiException("Cannot run OmrMidiFix without numpy ")
        import numpy as np
        self.distMatrix = np.zeros((self.n+1, self.m+1), dtype=int)
        
        
    def populateDistMatrix(self):
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
                
    
    def insertCost(self, tup):
        '''
        Cost of inserting an extra hashed item.
        For now, it's just the size of the tuple
        '''
        return len(tup)
    
    def deleteCost(self, tup):
        '''
        Cost of deleting an extra hashed item.
        For now, it's just the size of the tuple
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
        
    def getPossibleMoves(self, i, j):
        '''
        i and j are current row and column index in self.distMatrix
        returns all possible moves (0 up to 3) 
        vertical, horizontal, diagonal costs of adjacent entries in self.distMatrix
        
        >>> target = stream.Stream()
        >>> source = stream.Stream()
          
        >>> note1 = note.Note("C4")
        >>> note2 = note.Note("D4")
        >>> note3 = note.Note("C4")
        >>> note4 = note.Note("E4")
          
        >>> target.append([note1, note2, note3, note4])
        >>> source.append([note1, note2, note3])
        >>> sa = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa.setupDistMatrix()
        >>> for i in range(4+1):
        ...     for j in range(3+1):
        ...         sa.distMatrix[i][j] = i * j
        >>> sa.distMatrix
        array([[ 0,  0,  0,  0],
               [ 0,  1,  2,  3],
               [ 0,  2,  4,  6],
               [ 0,  3,  6,  9],
               [ 0,  4,  8, 12]])        

        >>> sa.getPossibleMoves(0, 0)
        [None, None, None]
        
        >>> sa.getPossibleMoves(1, 1)
        [0, 0, 0]
        
        >>> sa.getPossibleMoves(4, 3)
        [9, 8, 6]
        
        >>> sa.getPossibleMoves(2, 2)
        [2, 2, 1]
        
        >>> sa.getPossibleMoves(0, 2)
        [None, 0, None]
        
        >>> sa.getPossibleMoves(3, 0)
        [0, None, None]
        
        '''
        verticalCost = self.distMatrix[i-1][j] if i >= 1 else None
        horizontalCost = self.distMatrix[i][j-1] if j >= 1 else None
        diagonalCost = self.distMatrix[i-1][j-1] if (i >= 1 and j >= 1) else None
        
        possibleMoves = [verticalCost, horizontalCost, diagonalCost]
        return possibleMoves
    
    def getMovementDirection(self, i, j):
        '''
        return the direction that traceback moves
        0: vertical movement, insertion
        1: horizontal movement, deletion
        2: diagonal movement, substitution
        3: diagonal movement, no change
        
        raises a ValueError if i == 0 and j == 0.
        >>> target = stream.Stream()
        >>> source = stream.Stream()
          
        >>> note1 = note.Note("C4")
        >>> note2 = note.Note("D4")
        >>> note3 = note.Note("C4")
        >>> note4 = note.Note("E4")
          
        >>> target.append([note1, note2, note3, note4])
        >>> source.append([note1, note2, note3])
        
        >>> sa = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa.setupDistMatrix()
        >>> for i in range(4+1):
        ...     for j in range(3+1):
        ...         sa.distMatrix[i][j] = i * j
        >>> sa.distMatrix
        array([[ 0,  0,  0,  0],
               [ 0,  1,  2,  3],
               [ 0,  2,  4,  6],
               [ 0,  3,  6,  9],
               [ 0,  4,  8, 12]])        
        
        
        >>> sa.getMovementDirection(4, 3)
        2
        
        >>> sa.getMovementDirection(2, 2)
        2
        
        >>> sa.getMovementDirection(0, 2)
        1
        
        >>> sa.getMovementDirection(3, 0)
        0
        
        >>> sa.getMovementDirection(0, 0)
        Traceback (most recent call last):
        ValueError: No movement possible from the origin
        '''
        possibleMoves = self.getPossibleMoves(i, j)
        
        if possibleMoves[0] is None:
            if possibleMoves[1] is None:
                raise ValueError('No movement possible from the origin')
            else:
                return 1
        elif possibleMoves[1] is None:
            return 0
        
        currentCost = self.distMatrix[i][j]
        minIndex, minNewCost = min(enumerate(possibleMoves), key=operator.itemgetter(1))
        if currentCost == minNewCost:
            minIndex = 3
        
        return minIndex
    
    def calculateChanges(self):
        '''
        TODO: sanity check of manhattan distance
        check if possible moves are in index
        change n to i, m to j
        '''
        i = self.n
        j = self.m
        
        #and?
        while (i != 0 or j != 0):
            ## check if possible moves are indexable
            minIndex = self.getMovementDirection(i, j)
        
            # minIndex : 0: insertion, 1: deletion, 2: substitution; 3: nothing
            if minIndex == 0:
                self.changes.insert(0, (self.hashedTargetStream[i-1], self.hashedSourceStream[j-1], 'insertion'))
                i -= 1
                
            elif minIndex == 1:
                self.changes.insert(0, (self.hashedTargetStream[i-1], self.hashedSourceStream[j-1], 'deletion'))
                j -= 1
                
            elif minIndex == 2:
                self.changes.insert(0, (self.hashedTargetStream[i-1], self.hashedSourceStream[j-1], 'substitution'))
                i -= 1
                j -= 1
                
            else: # 3: nothing
                self.changes.insert(0, (self.hashedTargetStream[i-1], self.hashedSourceStream[j-1], 'no change'))
                i -= 1
                j -= 1
        
        if (i != 0 and j != 0):
            raise AlignmentTracebackException('Traceback of best alignment did not end properly')
        
        changesCount = Counter(elem[2] for elem in self.changes)
        self.percentageSimilar = float(changesCount['no change'])/len(self.changes)
        print(self.changes)
        
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
        
        print (sa.percentageSimilar)
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
        
#     def testK525BassCelloDouble(self):
#         '''
#         K525's bass part doubles the cello part. don't hash the octave
#         '''
#         from music21 import converter
#         from music21.alpha.analysis import hasher
#         
#         midiFP = K525midiShortPath
#         omrFP = K525omrShortPath
#         midiStream = converter.parse(midiFP)
#         omrStream = converter.parse(omrFP)
#     
#         fixer = OMRmidiNoteFixer(omrStream, midiStream)
#         celloBassAnalysis = fixer.checkBassDoublesCello()
#         self.assertEqual(celloBassAnalysis, True)
        
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
        
class OneTest(unittest.TestCase):     
    def testK525Streams(self):
        '''
        two mini streams
        '''
        from music21 import converter
        from music21.alpha.analysis import hasher
        
        midiFP = K525midiShortPath
        omrFP = K525omrShortPath
        midiStream = converter.parse(midiFP)
        omrStream = converter.parse(omrFP)
        
        sa = StreamAligner(omrStream, midiStream)
        sa.align()
        
    def testPossibleMoves(self):
        pass
#         fixer = OMRmidiNoteFixer(omrStream, midiStream)
#         celloBassAnalysis = fixer.checkBassDoublesCello()
#         self.assertEqual(celloBassAnalysis, True)

## this test is included in the quarterLengthDivisor PR in the converter.py tests
# class ParseTestExternal(unittest.TestCase):
#     def testParseMidi(self):
#         from music21 import converter
#         midiStream = converter.parse(K525midiShortPath, forceSource=True, quarterLengthDivisors=[4])
#         midiStream.show()

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)