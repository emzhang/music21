# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/aligner.py
# Purpose:      Hash musical notation
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2015 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
from music21 import exceptions21
from music21 import stream
from music21.alpha.analysis import hasher

import unittest

try:
    import enum
except ImportError:
    from music21.ext import enum

class AlignerException(exceptions21.Music21Exception):
    pass

class AlignmentTracebackException(AlignerException):
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
    Stream Aligner is a dumb object that takes in two streams and forces them to align
    without any thought to any external variables
    
    These terms are associated with the Target stream are:
    - n, the number of rows in the distance matrix, the left-most column of the matrix
    - i, the index into rows in the distance matrix
    - the first element of tuple
    
    These terms are associated with the Source stream are:
    - m, the number of columns in the distance matrix, the top-most row of the matrix
    - j, the index into columns in the distance matrix
    - the second element of tuple
    """
    
    def __init__(self, targetStream=None, sourceStream=None, hasher=None):
        self.targetStream = targetStream
        self.sourceStream = sourceStream
        
        self.distanceMatrix = None
        
        if hasher is None:
            hasher = self.getDefaultHasher() 

        self.hasher = hasher    
        
        self.changes = []
        self.similarityScore = 0
    
    def getDefaultHasher(self):
        '''
        returns a default hasher.Hasher object
        which does not hashOffset or include the reference.
        
        called by __init__ if no hasher is passed in.
        
        >>> sa = alpha.analysis.aligner.StreamAligner()
        >>> h = sa.getDefaultHasher()
        >>> h
        <music21.alpha.analysis.hasher.Hasher object at 0x1068cf6a0>
        >>> h.hashOffset
        False
        >>> h.includeReference
        True
        '''
        h = hasher.Hasher()
        h.hashOffset = False
        h.includeReference = True
        return h
        
    def align(self):
        pass
    
    def setupDistanceMatrix(self):
        pass
    
    def populateDistanceMatrix(self):
        pass
    
    def getPossibleMovesFromLocation(self, i, j):
        pass
    
    def getOpFromLocation(self, i , j):
        pass
    
    def insertCost(self, tup):
        pass
    
    def deleteCost(self, tup):
        pass
    
    def substitutionCost(self, tup):
        pass
    
    def calculateNumSimilarities(self, hashedItem1, hashedItem2):
        pass
    
    def equalsWithoutReference(self, hashedItem1, hashedItem2):
        pass
    
    def calculateChangesList(self):
        pass
    
    def showChanges(self):
        pass
        
class ScoreAligner(StreamAligner):
    """
    Take two scores and aligns them
    Does things including:
    - determining whether the bass part doubles the cello
    - fixes repeats
    """
    def __init__(self, targetScore=None, sourceScore=None, hasher=None):
        super().__init__(targetScore, sourceScore, hasher)
        self.targetScore = self.targetStream  # an alias to be less confusing 
        self.sourceScore = self.sourceStream  # an alias to be less confusing 
        self.discretizeParts = True
        
    def checkPartAlignment(self):
        """
        First checks if there are the same number of parts, if not, 
        then checks if bass line in source score doubles what would be a cello line
        
        TODO:
        add in checks for measure repeats
        
        >>> score1 =  stream.Score()
        >>> score2 = stream.Score()
        >>> part1_1 = stream.Part()
        >>> part1_2 = stream.Part()
        >>> part1_3 = stream.Part()
        >>> part2_1 = stream.Part()
        >>> part2_2 = stream.Part()
        
        """
        numTargetParts = len(self.targetScore.getElementsByClass(stream.Part))
        numSourceParts = len(self.sourceScore.getElementsByClass(stream.Part))
        
        if  numTargetParts == numSourceParts:
            return True
        # checks the case if bass doubles cello
        elif numTargetParts - numSourceParts == 1:
            celloPart = self.targetScore.getElementsByClass(stream.Part)[-2]
            bassPart = self.targetScore.getElementsByClass(stream.Part)[-1]
            celloBassAligner = StreamAligner(celloPart, bassPart)
            celloBassAligner.align()
            
            if celloBassAligner.similarityScore > .8:
                return True
        else:
            return False
    
    def align(self):
        """
        Main function here. Checks if parts can be aligned and aligns them if possible.
        
        Returns Nothing.
        
        >>> midiToAlign = converter.parse(alpha.analysis.fixOmrMidi.K525midiShortPath)
        >>> omrToAlign = converter.parse(alpha.analysis.fixOmrMidi.K525omrShortPath)
        
        >>> scA = alpha.analysis.aligner.ScoreAligner(midiToAlign, omrToAlign)
        
        When discretizeParts is False then the .changes should be the same as for a 
        StreamAligner
        
        >>> scA.discretizeParts = False
        >>> scA.align()
        >>> stA = alpha.analysis.aligner.StreamAligner(midiToAlign, omrToAlign)
        >>> stA.align()
        >>> scA.changes == stA.changes
        True
        
        """
        if not self.checkPartAlignment():
            raise ValueError('Scores not similar enough to perform alignment.')
        
        if self.discretizeParts:
            self.alignDiscreteParts()
        else:
            super().align()
            
    def alignDiscreteParts(self):
        listOfSimilarityScores = []
        listOfPartChanges = []
        
        targetParts = self.targetScore.getElementsByClass(stream.Part)
        sourceParts = self.sourceScore.getElementsByClass(stream.Part)
        for targetPart, sourcePart in zip(targetParts, sourceParts):
            partStreamAligner = StreamAligner(targetPart.flat, sourcePart.flat, hasher=self.hasher)
            partStreamAligner.align()
            listOfSimilarityScores.append(partStreamAligner.similarityScore)
            listOfPartChanges.append(partStreamAligner.changes)
            self.similarityScore = sum(listOfSimilarityScores) / len(listOfSimilarityScores)
            self.changes = [change for subPartList in listOfPartChanges for change in subPartList]

            
class Test(unittest.TestCase):
    pass

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)