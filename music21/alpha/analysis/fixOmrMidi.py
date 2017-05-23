# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/fixOmrMidi.py
# Purpose:      To be deprecated; use MIDI score data to fix OMR scores 
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2013-16 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
import itertools
'''
requires numpy
'''
from music21 import base as base
from music21 import exceptions21
from music21 import interval
from music21 import metadata
from music21 import stream

from music21.alpha.analysis import hasher
from music21.common import numberTools

from collections import Counter
import copy
import inspect
import os
import operator
import unittest

try:
    import enum
except ImportError:
    from music21.ext import enum

pathName = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

K525xmlShortPath = pathName + os.sep + 'k525short3.xml'
K525midiShortPath = pathName + os.sep + 'k525short.mid'
K525omrShortPath = pathName + os.sep + 'k525omrshort.xml'   

class OmrMidiException(exceptions21.Music21Exception):
    pass

class AlignmentTracebackException(OmrMidiException):
    pass

class ChangeOps(enum.IntEnum):
    '''
    >>> ins = alpha.analysis.fixOmrMidi.ChangeOps.Insertion
    >>> ins.color
    'green'
    
    >>> dele = alpha.analysis.fixOmrMidi.ChangeOps.Deletion
    >>> dele.color
    'red'
    
    >>> subs = alpha.analysis.fixOmrMidi.ChangeOps.Substitution
    >>> subs.color
    'purple'
    
    >>> noC = alpha.analysis.fixOmrMidi.ChangeOps.NoChange
    >>> noC.color
    
    '''
    Insertion = 0
    Deletion = 1
    Substitution = 2
    NoChange = 3
    
    def __init__(self, changeOpNum):
        self.changeOpNum = changeOpNum
        self.colorDict = {0:"green", 1:"red", 2:"purple", 3:None}
        
    @property
    def color(self):
        return self.colorDict[self.changeOpNum]
    
class StreamAligner(object):
    """
    Stream Aligner object for two streams
    Target is the string being aligned against, usually the MIDI stream
    Source is the string that is being corrected, usually the OMR stream
    
    These terms are associated with the MIDI stream are:
    - target
    - n, the number of rows in the distance matrix, the left-most column of the matrix
    - i, the index into rows in the distance matrix
    - the first element of tuple
    
    These terms are associated with the OMR stream are:
    - source
    - m, the number of columns in the distance matrix, the top-most row of the matrix
    - j, the index into columns in the distance matrix
    - the second element of tuple
    """
    
    def __init__(self, targetStream, sourceStream):
        self.targetStream = targetStream
        self.sourceStream = sourceStream
             
        self.distMatrix = None
        
        self.h = hasher.Hasher()
        self.h.hashOffset = False
        self.h.includeReference = True
        
        # True => match parts to parts, False => match entire stream to entire Stream
        self.discretizeParts = True 
        
        self.changes = []
        self.similarityScore = 0
        
        self.bassDoublesCello = False
        
    def checkPartAlignment(self, midiStream, omrStream):
        numMidiParts = len(midiStream.getElementsByClass(stream.Part))
        numOmrParts = len(omrStream.getElementsByClass(stream.Part))
        if  numMidiParts == numOmrParts:
            return True
        
        # check the case if bass doubles cello
        elif numMidiParts - numOmrParts == 1:
            sa = StreamAligner(midiStream[-1], midiStream[-2])
            sa.discretizeParts = False
            sa.align()
            
            if sa.similarityScore > .8:
                self.bassDoublesCello = True
                return True
            return False
        else: 
            return False
        
    def align(self):
        '''
        main function:
        setupDistMatrix() hashes the two streams and creates a matrix of the right size 
        populateDistMatrix() enters in all the correct values in the distance matrix
        calculateChanges() does a backtrace of the distance matrix to find the best path
        
        >>> target0 = stream.Stream()
        >>> source0 = stream.Stream()
        >>> p1 = stream.Part()
        >>> p2 = stream.Part()
        >>> p3 = stream.Part()
        >>> p4 = stream.Part()
        >>> target0.append([p1, p2])
        >>> source0.append([p3, p4])
        >>> sa0 = alpha.analysis.fixOmrMidi.StreamAligner(target0, source0)
        '''
        if self.discretizeParts:
            if self.checkPartAlignment(self.targetStream, self.sourceStream):
                listOfSimilarityScores = []
                listOfPartChanges = []
                
                targetParts = self.targetStream.getElementsByClass(stream.Part)
                sourceParts = self.sourceStream.getElementsByClass(stream.Part)
                for targetPart, sourcePart in zip(targetParts, sourceParts):
                    partStreamAligner = StreamAligner(targetPart.flat, sourcePart.flat)
                    partStreamAligner.discretizeParts = False
                    partStreamAligner.align()
                    listOfSimilarityScores.append(partStreamAligner.similarityScore)
                    listOfPartChanges.append(partStreamAligner.changes)
                self.similarityScore = sum(listOfSimilarityScores) / len(listOfSimilarityScores)
                self.changes = [change for subPartList in listOfPartChanges for change in subPartList]
#                 self.changes = list(itertools.chain(listOfPartChanges))
            else:
                pass # what now?
            # TODO: partStreamAligner never gets added back into original context
        else:          
            self.setupDistMatrix()
            self.populateDistMatrix()
            self.calculateChanges()
        
    def setupDistMatrix(self):
        '''
        creates the matrix of the right size after hashing
        
        >>> note1 = note.Note("C4")
        >>> note2 = note.Note("D4")
        >>> note3 = note.Note("C4")
        >>> note4 = note.Note("E4")
        
        >>> # test for streams of length 3 and 4
        >>> target0 = stream.Stream()
        >>> source0 = stream.Stream()
          
        >>> target0.append([note1, note2, note3, note4])
        >>> source0.append([note1, note2, note3])
        
        >>> sa0 = alpha.analysis.fixOmrMidi.StreamAligner(target0, source0)
        >>> sa0.setupDistMatrix()
        >>> sa0.distMatrix.size
        20
        >>> sa0.distMatrix.shape
        (5, 4)
        
        >>> # test for empty target stream
        >>> target1 = stream.Stream()
        >>> source1 = stream.Stream()
        >>> source1.append(note1)
        >>> sa1 = alpha.analysis.fixOmrMidi.StreamAligner(target1, source1)
        >>> sa1.setupDistMatrix()
        Traceback (most recent call last):
        music21.alpha.analysis.fixOmrMidi.OmrMidiException: 
        Cannot perform alignment with empty target stream.
        
        >>> # test for empty source stream
        >>> target2 = stream.Stream()
        >>> source2 = stream.Stream()
        >>> target2.append(note3)
        >>> sa2 = alpha.analysis.fixOmrMidi.StreamAligner(target2, source2)
        >>> sa2.setupDistMatrix()
        Traceback (most recent call last):
        music21.alpha.analysis.fixOmrMidi.OmrMidiException: 
        Cannot perform alignment with empty source stream.
        
        '''
        self.hashedTargetStream = self.h.hashStream(self.targetStream)
        self.hashedSourceStream = self.h.hashStream(self.sourceStream)
        
        # n and m will be the dimensions of the Distance Matrix we set up
        self.n = len(self.hashedTargetStream)
        self.m = len(self.hashedSourceStream)
        
        if self.n == 0:
            raise OmrMidiException("Cannot perform alignment with empty target stream.")
        
        if self.m == 0:
            raise OmrMidiException("Cannot perform alignment with empty source stream.")
        
        if ('numpy' in base._missingImport):
            raise OmrMidiException("Cannot run OmrMidiFix without numpy.")
        import numpy as np
        
        
        self.distMatrix = np.zeros((self.n + 1, self.m + 1), dtype=int)
        
        
    def populateDistMatrix(self):
        '''
        >>> # sets up the distance matrix for backtracing
        
        >>> note1 = note.Note("C#4")
        >>> note2 = note.Note("C4")
        
        >>> # test 1: similar streams
        >>> targetA = stream.Stream()
        >>> sourceA = stream.Stream()
        >>> targetA.append([note1, note2])
        >>> sourceA.append([note1, note2])
        >>> saA = alpha.analysis.fixOmrMidi.StreamAligner(targetA, sourceA)
        >>> saA.setupDistMatrix()
        >>> saA.populateDistMatrix()
        >>> saA.distMatrix
        array([[0, 2, 4],
               [2, 0, 2],
               [4, 2, 0]])
        
        >>> # test 2
        >>> targetB = stream.Stream()
        >>> sourceB = stream.Stream()
        >>> targetB.append([note1, note2])
        >>> sourceB.append(note1)
        >>> saB = alpha.analysis.fixOmrMidi.StreamAligner(targetB, sourceB)
        >>> saB.setupDistMatrix()
        >>> saB.populateDistMatrix()
        >>> saB.distMatrix
        array([[0, 2],
               [2, 0],
               [4, 2]])
               
        >>> # test 3 
        >>> note3 = note.Note("D5")
        >>> note3.quarterLength = 3
        >>> note4 = note.Note("E3")
        >>> targetC = stream.Stream()
        >>> sourceC = stream.Stream()
        >>> targetC.append([note1, note2, note4])
        >>> sourceC.append([note3, note1, note4])
        >>> saC = alpha.analysis.fixOmrMidi.StreamAligner(targetC, sourceC)
        >>> saC.setupDistMatrix()
        >>> saC.populateDistMatrix()
        >>> saC.distMatrix
        array([[0, 2, 4, 6],
           [2, 2, 2, 4],
           [4, 4, 3, 3],
           [6, 6, 5, 3]])
               
        '''
        # setup all the entries in the first column, target, midi stream
        for i in range(1, self.n + 1):
            insertCost = self.insertCost(self.hashedTargetStream[i - 1])
            self.distMatrix[i][0] = self.distMatrix[i - 1][0] + insertCost
            
        
        # setup all the entries in the first row, source, omr stream
        for j in range(1, self.m + 1):
            deleteCost = self.deleteCost(self.hashedSourceStream[j - 1])
            self.distMatrix[0][j] = self.distMatrix[0][j - 1] + deleteCost
        
        # fill in rest of matrix   
        for i in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                insertCost = self.insertCost(self.hashedTargetStream[i - 1])
                deleteCost = self.deleteCost(self.hashedSourceStream[j - 1])
                substCost = self.substCost(self.hashedTargetStream[i - 1], 
                                           self.hashedSourceStream[j - 1])
                previousValues = [self.distMatrix[i - 1][j] + insertCost,
                                   self.distMatrix[i][j - 1] + deleteCost,
                                   self.distMatrix[i - 1][j - 1] + substCost]  

                self.distMatrix[i][j] = min(previousValues)
                
    
    def insertCost(self, tup):
        '''
        Cost of inserting an extra hashed item.
        For now, it's just the size of the keys of the NoteHashWithReference
        
        >>> target = stream.Stream()
        >>> source = stream.Stream()
          
        >>> note1 = note.Note("C4")
        >>> note2 = note.Note("D4")
        >>> note3 = note.Note("C4")
        >>> note4 = note.Note("E4")
          
        >>> target.append([note1, note2, note3, note4])
        >>> source.append([note1, note2, note3])
        
        >>> # This is a StreamAligner with default hasher settings
        >>> sa0 = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa0.align()
        >>> tup0 = sa0.hashedTargetStream[0]
        >>> sa0.insertCost(tup0)
        2
        
        >>> # This is a StreamAligner with a modified hasher that doesn't hash pitch at all
        >>> sa1 = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa1.h.hashPitch = False
        >>> sa1.align()
        >>> tup1 = sa1.hashedTargetStream[0]
        >>> sa1.insertCost(tup1)
        1
        
        >>> # This is a StreamAligner with a modified hasher that hashes 3 additional properties
        >>>
        >>> sa2 = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa2.h.hashOctave = True
        >>> sa2.h.hashIntervalFromLastNote = True
        >>> sa2.h.hashIsAccidental = True
        >>> sa2.align()
        >>> tup2 = sa2.hashedTargetStream[0]
        >>> sa2.insertCost(tup2)
        5
        '''
        keyDictSize = len(tup.hashItemsKeys)
        return keyDictSize
    
    def deleteCost(self, tup):
        '''
        Cost of deleting an extra hashed item.
        For now, it's just the size of the keys of the NoteHashWithReference
        
        >>> target = stream.Stream()
        >>> source = stream.Stream()
          
        >>> note1 = note.Note("C4")
        >>> note2 = note.Note("D4")
        >>> note3 = note.Note("C4")
        >>> note4 = note.Note("E4")
          
        >>> target.append([note1, note2, note3, note4])
        >>> source.append([note1, note2, note3])
        
        >>> # This is a StreamAligner with default hasher settings
        >>> sa0 = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa0.align()
        >>> tup0 = sa0.hashedTargetStream[0]
        >>> sa0.deleteCost(tup0)
        2
        
        >>> # This is a StreamAligner with a modified hasher that doesn't hash pitch at all
        >>> sa1 = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa1.h.hashPitch = False
        >>> sa1.align()
        >>> tup1 = sa1.hashedTargetStream[0]
        >>> sa1.deleteCost(tup1)
        1
        
        >>> # This is a StreamAligner with a modified hasher that hashes 3 additional properties
        >>> sa2 = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        >>> sa2.h.hashOctave = True
        >>> sa2.h.hashIntervalFromLastNote = True
        >>> sa2.h.hashIsAccidental = True
        >>> sa2.align()
        >>> tup2 = sa2.hashedTargetStream[0]
        >>> sa2.deleteCost(tup2)
        5
        
        '''
        keyDictSize = len(tup.hashItemsKeys)
        return keyDictSize
        
    def substCost(self, hashedItem1, hashedItem2):
        '''
        - hashedItem1 is a midi note
        - hashedItem2 is an omr note
        
        >>> # equality testing, both streams made from same note
        >>> # targetA will not have the same reference as sourceA
        >>> # but their hashes will be equal, which makes for their hashed objects to be 
        >>> # able to be equal.
        
        >>> note1 = note.Note("C4")
        >>> targetA = stream.Stream()
        >>> sourceA = stream.Stream()
        >>> targetA.append(note1)
        >>> sourceA.append(note1)
        >>> targetA == sourceA
        False
        
        >>> saA = alpha.analysis.fixOmrMidi.StreamAligner(targetA, sourceA)
        >>> saA.align()
        >>> hashedItem1A = saA.hashedTargetStream[0]
        >>> hashedItem2A = saA.hashedSourceStream[0]
        >>> print(hashedItem1A)
        NoteHash(Pitch=60, Duration=1.0)
        >>> print(hashedItem2A)
        NoteHash(Pitch=60, Duration=1.0)
        >>> saA.equalsWithoutReference(hashedItem1A, hashedItem2A)
        True

        >>> saA.substCost(hashedItem1A, hashedItem2A)
        0
        
        >>> note2 = note.Note("D4")
        >>> targetB = stream.Stream()
        >>> sourceB = stream.Stream()
        >>> targetB.append(note1)
        >>> sourceB.append(note2)
        >>> saB = alpha.analysis.fixOmrMidi.StreamAligner(targetB, sourceB)
        >>> saB.align()
        >>> hashedItem1B = saB.hashedTargetStream[0]
        >>> hashedItem2B = saB.hashedSourceStream[0]
        
        >>> # hashed items only differ in 1 spot
        >>> print(hashedItem1B)
        NoteHash(Pitch=60, Duration=1.0)
        
        >>> print(hashedItem2B)
        NoteHash(Pitch=62, Duration=1.0)
        
        >>> saB.substCost(hashedItem1B, hashedItem2B)
        1
        
        >>> note3 = note.Note("E4")
        >>> note4 = note.Note("E#4")
        >>> note4.duration = duration.Duration('half')
        >>> targetC = stream.Stream()
        >>> sourceC = stream.Stream()
        >>> targetC.append(note3)
        >>> sourceC.append(note4)
        >>> saC = alpha.analysis.fixOmrMidi.StreamAligner(targetC, sourceC)
        >>> saC.align()
        >>> hashedItem1C = saC.hashedTargetStream[0]
        >>> hashedItem2C = saC.hashedSourceStream[0]
        
         >>> # hashed items should differ in 2 spot
        >>> print(hashedItem1C)
        NoteHash(Pitch=64, Duration=1.0)
        
        >>> print(hashedItem2C)
        NoteHash(Pitch=65, Duration=2.0)
        
        >>> saC.substCost(hashedItem1C, hashedItem2C)
        2
        '''
        if self.equalsWithoutReference(hashedItem1, hashedItem2):
            return 0 
        
        totalPossibleDifferences = len(hashedItem1.hashItemsKeys)
        
        numSimilaritiesInTuple = self.calculateNumSimilarities(hashedItem1, hashedItem2)
        
        totalPossibleDifferences -= numSimilaritiesInTuple
                
        return totalPossibleDifferences
    
    def calculateNumSimilarities(self, hashedItem1, hashedItem2):
        '''
        - hashedItem1 is a midi note
        - hashedItem2 is an omr note
        
        returns the number of attributes that two tuples have that are the same
        
        >>> target = stream.Stream()
        >>> source = stream.Stream()
          
        >>> note1 = note.Note("D1")
        >>> target.append([note1])
        >>> source.append([note1])
        >>> sa = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        
        >>> from collections import namedtuple
        >>> NoteHash = namedtuple('NoteHash', ["Pitch", "Duration"])
        >>> nh1 = NoteHash(60, 4)
        >>> nhwr1 = alpha.analysis.hasher.NoteHashWithReference(nh1)
        >>> nhwr1.reference = note.Note('C4')
        >>> nhwr1
        NoteHash(Pitch=60, Duration=4)
        
        >>> nh2 = NoteHash(60, 4)
        >>> nhwr2 = alpha.analysis.hasher.NoteHashWithReference(nh2)
        >>> nhwr2.reference = note.Note('C4')
        >>> nhwr2
        NoteHash(Pitch=60, Duration=4)
        
       
        >>> sa.calculateNumSimilarities(nhwr1, nhwr2)
        2
        
        >>> nh3 = NoteHash(61, 4)
        >>> nhwr3 = alpha.analysis.hasher.NoteHashWithReference(nh3)
        >>> nhwr3.reference = note.Note('C#4')
        >>> nhwr3
        NoteHash(Pitch=61, Duration=4)
        
        >>> sa.calculateNumSimilarities(nhwr1, nhwr3)
        1
        
        >>> nh4 = NoteHash(59, 1)
        >>> nhwr4 = alpha.analysis.hasher.NoteHashWithReference(nh4)
        >>> nhwr4.reference = note.Note('B3')
        >>> nhwr4
        NoteHash(Pitch=59, Duration=1)
        
        >>> sa.calculateNumSimilarities(nhwr2, nhwr4)
        0
        '''
        
        count = 0
        for val in hashedItem1.hashItemsKeys:
            if getattr(hashedItem1, val) == getattr(hashedItem2, val):
                count += 1
        return count
    
    def equalsWithoutReference(self, hashedItem1, hashedItem2):
        '''
        returns whether two hashed items have the same attributes, 
        
        >>> target = stream.Stream()
        >>> source = stream.Stream()
          
        >>> note1 = note.Note("D1")
        >>> target.append([note1])
        >>> source.append([note1])
        >>> sa = alpha.analysis.fixOmrMidi.StreamAligner(target, source)
        
        >>> from collections import namedtuple
        >>> NoteHash = namedtuple('NoteHash', ["Pitch", "Duration"])
        >>> nh1 = NoteHash(60, 4)
        >>> nhwr1 = alpha.analysis.hasher.NoteHashWithReference(nh1)
        >>> nhwr1.reference = note.Note('C4')
        >>> nhwr1
        NoteHash(Pitch=60, Duration=4)
        
        >>> nh2 = NoteHash(60, 4)
        >>> nhwr2 = alpha.analysis.hasher.NoteHashWithReference(nh2)
        >>> nhwr2.reference = note.Note('C4')
        >>> nhwr2
        NoteHash(Pitch=60, Duration=4)
        
       
        >>> sa.equalsWithoutReference(nhwr1, nhwr2)
        True
        
        >>> nh3 = NoteHash(61, 4)
        >>> nhwr3 = alpha.analysis.hasher.NoteHashWithReference(nh3)
        >>> nhwr3.reference = note.Note('C#4')
        >>> nhwr3
        NoteHash(Pitch=61, Duration=4)
        
        >>> sa.equalsWithoutReference(nhwr1, nhwr3)
        False
        
        '''
        for val in hashedItem1.hashItemsKeys:
            if getattr(hashedItem1, val) != getattr(hashedItem2, val):
                return False
        return True
        
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
        verticalCost = self.distMatrix[i - 1][j] if i >= 1 else None
        horizontalCost = self.distMatrix[i][j - 1] if j >= 1 else None
        diagonalCost = self.distMatrix[i - 1][j - 1] if (i >= 1 and j >= 1) else None
        
        possibleMoves = [verticalCost, horizontalCost, diagonalCost]
        return possibleMoves
    
    def getOpFromLocation(self, i, j):
        '''
        Insert, Delete, Substitution, No Change = range(4) 
        
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
        >>> sa.populateDistMatrix()
        >>> sa.distMatrix
        array([[0, 2, 4, 6],
               [2, 0, 2, 4],
               [4, 2, 0, 2],
               [6, 4, 2, 0],
               [8, 6, 4, 2]])    
        
        
        >>> sa.getOpFromLocation(4, 3)
        <ChangeOps.Insertion: 0>
        
        >>> sa.getOpFromLocation(2, 2)
        <ChangeOps.NoChange: 3>
        
        >>> sa.getOpFromLocation(0, 2)
        <ChangeOps.Deletion: 1>
        
        >>> sa.distMatrix[0][0] = 1
        >>> sa.distMatrix
        array([[1, 2, 4, 6],
               [2, 0, 2, 4],
               [4, 2, 0, 2],
               [6, 4, 2, 0],
               [8, 6, 4, 2]])    
        
        >>> sa.getOpFromLocation(1, 1)
        <ChangeOps.Substitution: 2>
        
        >>> sa.getOpFromLocation(0, 0)
        Traceback (most recent call last):
        ValueError: No movement possible from the origin
        '''
        possibleMoves = self.getPossibleMoves(i, j)
        
        if possibleMoves[0] is None:
            if possibleMoves[1] is None:
                raise ValueError('No movement possible from the origin')
            else:
                return ChangeOps.Deletion
        elif possibleMoves[1] is None:
            return ChangeOps.Insertion
        
        currentCost = self.distMatrix[i][j]
        minIndex, minNewCost = min(enumerate(possibleMoves), key=operator.itemgetter(1))
        if currentCost == minNewCost:
            return ChangeOps.NoChange 
        else:
            return ChangeOps(minIndex)
    
    def calculateChanges(self):
        '''
        >>> note1 = note.Note("C#4")
        >>> note2 = note.Note("C4")
        
        >>> # test 1: one insertion, one no change. Target stream has one more note than
        >>> # source stream, so source stream needs an insertion to match target stream.
        >>> # should be .5 similarity between the two
        >>> targetA = stream.Stream()
        >>> sourceA = stream.Stream()
        >>> targetA.append([note1, note2])
        >>> sourceA.append(note1)
        >>> saA = alpha.analysis.fixOmrMidi.StreamAligner(targetA, sourceA)
        >>> saA.setupDistMatrix()
        >>> saA.populateDistMatrix()
        >>> saA.calculateChanges()
        >>> saA.changesCount
        Counter({<ChangeOps.Insertion: 0>: 1, <ChangeOps.NoChange: 3>: 1})
        >>> saA.similarityScore
        0.5
        
        >>> # test 2: one deletion, one no change. Target stream has one fewer note than
        >>> # source stream, so source stream needs a deletion to match target stream.
        >>> # should be .5 similarity between the two
        >>> targetB = stream.Stream()
        >>> sourceB = stream.Stream()
        >>> targetB.append(note1)
        >>> sourceB.append([note1, note2])
        >>> saB = alpha.analysis.fixOmrMidi.StreamAligner(targetB, sourceB)
        >>> saB.setupDistMatrix()
        >>> saB.populateDistMatrix()
        >>> saB.calculateChanges()
        >>> saB.changesCount
        Counter({<ChangeOps.Deletion: 1>: 1, <ChangeOps.NoChange: 3>: 1})
        >>> saB.similarityScore
        0.5
        
        >>> # test 3: no changes
        >>> targetC = stream.Stream()
        >>> sourceC = stream.Stream()
        >>> targetC.append([note1, note2])
        >>> sourceC.append([note1, note2])
        >>> saC = alpha.analysis.fixOmrMidi.StreamAligner(targetC, sourceC)
        >>> saC.setupDistMatrix()
        >>> saC.populateDistMatrix()
        >>> saC.calculateChanges()
        >>> saC.changesCount
        Counter({<ChangeOps.NoChange: 3>: 2})
        >>> saC.similarityScore
        1.0
        
        >>> # test 4: 1 no change, 1 substitution
        >>> targetD = stream.Stream()
        >>> sourceD = stream.Stream()
        >>> note3 = note.Note("C4") 
        >>> note3.quarterLength = 2 # same pitch and offset as note2
        >>> targetD.append([note1, note2])
        >>> sourceD.append([note1, note3])
        >>> saD = alpha.analysis.fixOmrMidi.StreamAligner(targetD, sourceD)
        >>> saD.setupDistMatrix()
        >>> saD.populateDistMatrix()
        >>> saD.calculateChanges()
        >>> saD.changesCount
        Counter({<ChangeOps.Substitution: 2>: 1, <ChangeOps.NoChange: 3>: 1})
        >>> saD.similarityScore
        0.5
        
        '''
        i = self.n 
        j = self.m
        
        # and?
        while (i != 0 or j != 0):
            # # check if possible moves are indexable
            bestOp = self.getOpFromLocation(i, j)
            
            self.changes.insert(0, (self.hashedTargetStream[i - 1].reference,
                                        self.hashedSourceStream[j - 1].reference,
                                        bestOp))
            # bestOp : 0: insertion, 1: deletion, 2: substitution; 3: nothing
            if bestOp == ChangeOps.Insertion:
                i -= 1
                
            elif bestOp == ChangeOps.Deletion:
                j -= 1
                
            elif bestOp == ChangeOps.Substitution:
                i -= 1
                j -= 1
                
            else:  # 3: ChangeOps.NoChange
                i -= 1
                j -= 1
        
        if (i != 0 and j != 0):
            raise AlignmentTracebackException('Traceback of best alignment did not end properly')
        
        self.changesCount = Counter(elem[2] for elem in self.changes)
        self.similarityScore = float(self.changesCount[ChangeOps.NoChange]) / len(self.changes)
        
    def showChanges(self, show=True):
        for (idx, (midiNoteRef, omrNoteRef, change)) in enumerate(self.changes):
            if change == ChangeOps.NoChange:
                pass
            else: # change is Insertion, Deletion, Substitution
                midiNoteRef.color = change.color
                midiNoteRef.addLyric(idx)
                omrNoteRef.color = change.color
                omrNoteRef.addLyric(idx)
                
         
        self.targetStream.metadata = metadata.Metadata() 
        self.sourceStream.metadata = metadata.Metadata()  
        
        self.targetStream.metadata.title = "Target/MIDI " + str(self.targetStream.id)
        self.sourceStream.metadata.title = "Source/OMR " + str(self.targetStream.id)
        
        self.targetStream.metadata.movementName = self.targetStream.metadata.title
        self.sourceStream.metadata.movementName = self.sourceStream.metadata.title
        
        if show:
            self.targetStream.show()
            self.sourceStream.show()
        
        
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
    
        if sa.similarityScore >= .8:
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
        # if self.approxequal(self.omrStream.highestTime, self.midiStream.highestTime):
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
                self.omrNote.pitch.accidental = None
            if len(self.measure_accidentals) == 0:
                self.omrNote.pitch.accidental = self.midiNote.pitch.accidental         
            else:
                self.measure_accidentals.append(self.omrNote.pitch)
        elif self.hasSharpFlatAcc() and self.stepEq():
            if self.hasAcc():
                self.omrNote.pitch.accidental = self.midiNote.pitch.accidental
            else: 
                self.omrNote.pitch.accidental = None

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
    
    def intervalTooBig(self, aNote, bNote, setint=5):
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
       
        midiNote.pitch.accidental = pitch.Accidental('sharp')

        
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
        
    def testSimpleStreamOneNote(self):
        '''
        two streams of the same note should have 1.0 similarity
        '''
        from music21 import stream
        from music21 import note
          
        target = stream.Stream()
        source = stream.Stream()
          
        note1 = note.Note("C4")
        note2 = note.Note("C4")
          
        target.append(note1)
        source.append(note2)
    
        
        sa = StreamAligner(target, source)
        sa.align()
        
        self.assertEqual(sa.similarityScore, 1.0)
        
    def testSimpleStreamOneNoteDifferent(self):
        '''
        two streams of two different notes should have 0.0 similarity
        '''
        from music21 import stream
        from music21 import note
          
        target = stream.Stream()
        source = stream.Stream()
          
        note1 = note.Note("C4")
        note2 = note.Note("C#4")
        note2.quarterLength = 4
          
        target.append(note1)
        source.append(note2)
          
        sa = StreamAligner(target, source)
        sa.align()
        
        self.assertEqual(sa.similarityScore, 0.0)
        
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
        
        self.assertEqual(sa.similarityScore, 1.0)
    
    def testSameSimpleStream2(self):
        '''
        two streams of the 2/3 same notes should have 2/3 similarity
        '''
        from music21 import stream
        from music21 import note
          
        target = stream.Stream()
        source = stream.Stream()
          
        note1 = note.Note("C4")
        note2 = note.Note("D#4")
        note3 = note.Note("D-4")
        note4 = note.Note("C4")
          
        target.append([note1, note2, note4])
        source.append([note1, note3, note4])
          
        sa = StreamAligner(target, source)
        sa.align()
        
        self.assertEqual(sa.similarityScore, 2/3)
         
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
        
        self.assertEqual(sa.similarityScore, .75)
        
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
        sa.showChanges(show=False)
        
        self.assertEqual(sa.similarityScore, .75)
    
    def testChordSimilarityStream(self):
        '''
        two streams, one with explicit chord
        '''
        from music21 import stream
        from music21 import chord
         
        target = stream.Stream()
        source = stream.Stream()
        
        cMajor = chord.Chord(["E3", "C4", "G4"])
        target.append(cMajor)
        source.append(cMajor)
        
        sa = StreamAligner(target, source)
        sa.align()
        self.assertEqual(sa.similarityScore, 1.)
        
    
    def testBWV137MultiStreams(self):
        from music21 import stream, converter
            
        bwv137midifp = '/Users/Emily/Research/MEng/testfiles/bwv137.mid'
        bwv137omrfp = '/Users/Emily/Research/MEng/testfiles/bwv137emily.xml'
        bwv137midistream = converter.parse(bwv137midifp, forceSource=True, quarterLengthDivisors=[4])
        bwv137omrstream = converter.parse(bwv137omrfp)
        
        sa1 = StreamAligner(bwv137midistream, bwv137omrstream)
        sa1.discretizeParts = False
        sa1.align()
        print("sa1 sim score: ", sa1.similarityScore)
#         sa1.showChanges(show=True)
        
        bwv137midistream
        bwv137omrstream
        sa2 = StreamAligner(bwv137midistream, bwv137omrstream)
        sa2.discretizeParts = True
        sa2.align()
        print("sa2 sim score: ", sa2.similarityScore)
        sa2.showChanges(show=True)
        
        # self.assertGreater(sa1.similarityScore, )
        
    def testBWV137twoparts(self):
        '''
        this test shows that even with just the tenor and bass part the
        first note of the bass part is still marked as aligning with the last note
        of the tenor part
        '''
        from music21 import stream, converter
            
        bwv137midifp = '/Users/Emily/Research/MEng/testfiles/bwv137.mid'
        bwv137omrfp = '/Users/Emily/Research/MEng/testfiles/bwv137emily.xml'
        bwv137midistream = converter.parse(bwv137midifp, forceSource=True, quarterLengthDivisors=[4])
        bwv137omrstream = converter.parse(bwv137omrfp)
        
        midipart3 = bwv137midistream.parts[2]
        midipart4 = bwv137midistream.parts[3]
        
        omrpart3 = bwv137omrstream.parts[2]
        omrpart4 = bwv137omrstream.parts[3]
        
        bwv137midiparts34 = stream.Score()
        bwv137midiparts34.append([midipart3, midipart4])
        
        bwv137omrparts34 = stream.Score()
        bwv137omrparts34.append([omrpart3, omrpart4]) 
        
        sa2 = StreamAligner(bwv137midiparts34, bwv137omrparts34)
        sa2.discretizeParts = True
        sa2.align()
        sa2.showChanges(show=True)
        
    def testBWV137MultiStreamsBassPart(self):
        from music21 import stream, converter
            
        bwv137midifp = '/Users/Emily/Research/MEng/testfiles/bwv137.mid'
        bwv137omrfp = '/Users/Emily/Research/MEng/testfiles/bwv137emily.xml'
        bwv137midistream = converter.parse(bwv137midifp, forceSource=True, quarterLengthDivisors=[4])
        bwv137omrstream = converter.parse(bwv137omrfp)
        
        sa2 = StreamAligner(bwv137midistream.parts[-1], bwv137omrstream.parts[-1])
        sa2.discretizeParts = True
        sa2.align()
        sa2.showChanges(show=False) 
    
    def testStreamsWithRestAtEnd(self):
        '''
        part of debugging process for figuring out why discretize parts isn't
        behaving as expected
        '''
        from music21 import stream, note, meter
        s = stream.Stream()
        p1 = stream.Part()
        p2 = stream.Part()
        m1 = stream.Measure()
        m2 = stream.Measure()
        bnote = note.Note("B3")
        m1.timeSignature = meter.TimeSignature('4/4')
        m1.repeatAppend(note.Note("C4"), 4)
        m2.repeatAppend(note.Note("C4"), 2)
        m2.append(bnote)
        p1.append([m1, m2])
        s.append(p1)
        m3 = stream.Measure()
        m4 = stream.Measure()
        m3.timeSignature = meter.TimeSignature('4/4')
        m3.repeatAppend(note.Note("A4"), 4)
        m4.repeatAppend(note.Note("A4"), 2)
        m4.append(note.Note("E4"))
        p2.append([m3, m4])
        s.append(p2)

        s2 = stream.Stream()
        p2_1 = stream.Part()
        p2_2 = stream.Part()
        m2_1 = stream.Measure()
        m2_2 = stream.Measure()
        m2_1.repeatAppend(note.Note("C4"), 4)
        m2_2.repeatAppend(note.Note("C4"), 2)
        m2_2.append(note.Note("B3"))
        m2_2.append(note.Rest())
        p2_1.append([m2_1, m2_2])
        s2.append(p2_1)
        
        m2_3 = stream.Measure()
        m2_4 = stream.Measure()
        m2_3.repeatAppend(note.Note("A4"), 4)
        m2_4.repeatAppend(note.Note("A4"), 2)
        m2_4.append(note.Note("E4"))
        m2_4.append(note.Rest())
        p2_2.append([m2_3, m2_4])
        s2.append(p2_2)
        bnote.con
        saDiscretizedWithRest = StreamAligner(s, s2)
        saDiscretizedWithRest.align()
        saDiscretizedWithRest.showChanges()
        
    def testShowNoChanges(self):
        '''
        two exact streams, makes sure that the show functions behaves predictably
        '''
        from music21 import stream
        from music21 import note
         
        target = stream.Stream()
        source = stream.Stream()
         
        note1 = note.Note("C4")
        note2 = note.Note("D4")
        note3 = note.Note("E4")
        
        note4 = note.Note("C4")
        note5 = note.Note("D4")
        note6 = note.Note("E4")
         
        target.append([note1, note2, note3])
        source.append([note4, note5, note6])
         
        sa = StreamAligner(target, source)
        sa.align()
        sa.showChanges(show=False)
        
    def testShowInsertion(self):
        '''
        two streams:
        MIDI is CCCB
        OMR is CCC
        
        Therefore there needs to be an insertion to get from OMR to MIDI
        '''
        from music21 import stream
        from music21 import note
         
        target = stream.Stream()
        source = stream.Stream()
         
        noteC1 = note.Note("C4")
        noteC2 = note.Note("C4")
        noteC3 = note.Note("C4")
        noteC4 = note.Note("C4")
        noteC5 = note.Note("C4")
        noteC6 = note.Note("C4")
        noteB = note.Note("B3")
         
        target.append([noteC1, noteC2, noteC3, noteB])
        source.append([noteC4, noteC5, noteC6])
         
        sa = StreamAligner(target, source)
        sa.align()
        sa.showChanges(show=True)
        
        self.assertEqual(target.getElementById(sa.changes[3][0].id).color, 'green')
        self.assertEqual(target.getElementById(sa.changes[3][0].id).lyric, '3')
        self.assertEqual(source.getElementById(sa.changes[3][1].id).color, 'green')
        self.assertEqual(source.getElementById(sa.changes[3][1].id).lyric, '3')
    
    def testShowDeletion(self):
        '''
        two streams:
        MIDI is CCC
        OMR is CCCB
        
        Therefore there needs to be an deletion to get from OMR to MIDI
        '''
        from music21 import stream
        from music21 import note
         
        target = stream.Stream()
        source = stream.Stream()
         
        noteC1 = note.Note("C4")
        noteC2 = note.Note("C4")
        noteC3 = note.Note("C4")
        noteC4 = note.Note("C4")
        noteC5 = note.Note("C4")
        noteC6 = note.Note("C4")
        noteB = note.Note("B3")
         
        target.append([noteC1, noteC2, noteC3])
        source.append([noteC4, noteC5, noteC6, noteB])
         
        sa = StreamAligner(target, source)
        sa.align()
        sa.showChanges(show=False)
        
        self.assertEqual(target.getElementById(sa.changes[3][0].id).color, 'red')
        self.assertEqual(target.getElementById(sa.changes[3][0].id).lyric, '3')
        self.assertEqual(source.getElementById(sa.changes[3][1].id).color, 'red')
        self.assertEqual(source.getElementById(sa.changes[3][1].id).lyric, '3')
        
    def testShowSubstitution(self):
        '''
        two streams:
        MIDI is CCC
        OMR is CCB
        
        Therefore there needs to be an substitution to get from OMR to MIDI
        '''
        from music21 import stream
        from music21 import note
         
        target = stream.Stream()
        source = stream.Stream()
         
        noteC1 = note.Note("C4")
        noteC2 = note.Note("C4")
        noteC3 = note.Note("C4")
        noteC4 = note.Note("C4")
        noteC5 = note.Note("C4")
        noteB = note.Note("B3")
         
        target.append([noteC1, noteC2, noteC3])
        source.append([noteC4, noteC5, noteB])
         
        sa = StreamAligner(target, source)
        sa.align()
        sa.showChanges(show=False)
        
        self.assertEqual(target.getElementById(sa.changes[2][0].id).color, 'purple')
        self.assertEqual(target.getElementById(sa.changes[2][0].id).lyric, '2')
        self.assertEqual(source.getElementById(sa.changes[2][1].id).color, 'purple')
        self.assertEqual(source.getElementById(sa.changes[2][1].id).lyric, '2')
        
    '''
    This test is failing
    '''
#     def testBWV137BassDoublesCello(self):
#         from music21 import stream, converter
#             
#         bwv137midifp = '/Users/Emily/Research/MEng/testfiles/bwv137midibass.mid'
#         bwv137omrfp = '/Users/Emily/Research/MEng/testfiles/bwv137emily.xml'
#         bwv137midistream = converter.parse(bwv137midifp, forceSource=True, 
#                                            quarterLengthDivisors=[4])
#         bwv137omrstream = converter.parse(bwv137omrfp)
#         
#         sa = StreamAligner(bwv137midistream, bwv137omrstream)
#         sa.discretizeParts = True
#         sa.align()
#         
#         self.assertTrue(sa.bassDoublesCello)
        

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)
