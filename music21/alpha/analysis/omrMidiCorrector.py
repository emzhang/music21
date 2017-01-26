# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/omrMidiCorrector.py
# Purpose:      Corrects OMR stream with corresponding MIDI stream
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2016 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
from music21 import exceptions21
from music21 import stream
from music21.alpha.analysis import aligner
from music21.alpha.analysis import hasher

import unittest

class OMRMIDICorrectorException(exceptions21.Music21Exception):
    pass

class OMRMIDICorrector(object):
    """
    Takes two streams, one MIDI, one OMR and 
    1) preprocesses them and checks that they are fit to be aligned
    2) hashes the streams
    3) aligns them
        - caveat, should only align matching parts with each other
    4) fixes the OMR stream based on changes and best alignment output from aligner
    """
    def __init__(self, midiStream, omrStream, hasher=None):
        self.midiStream = midiStream  # an alias to be less confusing 
        self.omrStream = omrStream  # an alias to be less confusing 
        self.hasher = hasher
        self.discretizeParts = True
        self.midiParts = []
        self.omrParts = []
        self.hashedMidiParts = []
        self.hashedOmrParts = []
        
    def processRunner(self):
        '''
        TODO: change this function to a better name?
        main function that all other functions (hashing, preprocessing, aligning are called from)
        '''
        self.preprocessStreams()
        self.setupHasher()
        self.hashOmrMidiStreams()
        self.alignStreams()
        self.fixStreams()
    
    def preprocessStreams(self):
        '''
        Checks if parts ought to be discretized, populates self.midiParts and self.omrParts based
        on this answer
        Checks that the number of parts are equal
        Checks that if lengths are off by one, if bass doubles cello in the MIDI score
        
        TODO: match each part in to the most likely matching part in other stream, no repeats
        '''
        if self.discretizeParts == True:
            self.midiParts = self.midiStream.getElementsByClass(stream.Part).flat
            self.omrParts = self.omrStream.getElementsByClass(stream.Part).flat
        else:
            self.midiParts = [self.midiStream.flat]
            self.omrParts = [self.omrStream.flat]
            
        numMidiParts = len(self.midiParts)
        numOmrParts = len(self.omrParts)
        if numMidiParts == numOmrParts:
            pass
            #TODO: something smarter?
        elif numMidiParts - numOmrParts == 1:
            if self.checkBassDoublesCello(self.midiStream[-1], self.midiStream[-2]):
                pass
            else:
                raise OMRMIDICorrectorException("Streams have uneven number of parts.")
        
        
            
        
    def checkBassDoublesCello(self, bassPart, celloPart):
        '''
        Creates a StreamAligner that checks if bassPart and celloPart are similar enough to be 
        considered 'doubled' or the same.
        '''
        bassCelloHasher = hasher.Hasher()
        bassCelloHasher.hashOffset = False
        bassCelloHasher.hashPitch = False
        bassCelloHasher.hashIntervalFromLastNote = True
        bassCelloAligner = aligner.StreamAligner(bassPart, celloPart, bassCelloHasher)
        bassCelloAligner.align()
        if bassCelloAligner.similarityScore > .8:
            return True
        return False
    
    def setupHasher(self):
        '''
        Sets up the hasher that is to be used to hashed the streams. If one is passed in, 
        uses that one, otherwise uses a default one with settings we have found to be generally
        effective
        
        TODO: test that self.hasher should never be None after calling this function
        
        '''
        if self.hasher is None:
            return self.setDefaultHasher()
    
    def setDefaultHasher(self):
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
        self.hasher = h
    
    def hashOmrMidiStreams(self):
        '''
        takes MIDI and OMR streams and hashes them with the hasher 
        '''
        for midiPart in self.midiParts:
            self.hashedMidiParts.append(self.hasher.hashStream(midiPart))
        
        for omrPart in self.omrParts:
            self.hashedOmrParts.append(self.hasher.hashStream(omrPart))  
    
    def alignStreams(self):
        pass
    
    def fixStreams(self):
        pass      
#     def checkPartAlignment(self):
#         """
#         First checks if there are the same number of parts, if not, 
#         then checks if bass line in source score doubles what would be a cello line
#         
#         TODO:
#         add in checks for measure repeats
#         
#         >>> score1 =  stream.Score()
#         >>> score2 = stream.Score()
#         >>> part1_1 = stream.Part()
#         >>> part1_2 = stream.Part()
#         >>> part1_3 = stream.Part()
#         >>> part2_1 = stream.Part()
#         >>> part2_2 = stream.Part()
#         
#         """
#         numTargetParts = len(self.targetScore.getElementsByClass(stream.Part))
#         numSourceParts = len(self.sourceScore.getElementsByClass(stream.Part))
#         
#         if  numTargetParts == numSourceParts:
#             return True
#         # checks the case if bass doubles cello
#         elif numTargetParts - numSourceParts == 1:
#             celloPart = self.targetScore.getElementsByClass(stream.Part)[-2]
#             bassPart = self.targetScore.getElementsByClass(stream.Part)[-1]
#             celloBassAligner = StreamAligner(celloPart, bassPart)
#             celloBassAligner.align()
#             
#             if celloBassAligner.similarityScore > .8:
#                 return True
#         else:
#             return False
#     
#     def align(self):
#         """
#         Main function here. Checks if parts can be aligned and aligns them if possible.
#         
#         Returns Nothing.
#         
#         >>> midiToAlign = converter.parse(alpha.analysis.fixOmrMidi.K525midiShortPath)
#         >>> omrToAlign = converter.parse(alpha.analysis.fixOmrMidi.K525omrShortPath)
#         
#         >>> scA = alpha.analysis.aligner.ScoreAligner(midiToAlign, omrToAlign)
#         
#         When discretizeParts is False then the .changes should be the same as for a 
#         StreamAligner
#         
#         >>> scA.discretizeParts = False
#         >>> scA.align()
#         >>> stA = alpha.analysis.aligner.StreamAligner(midiToAlign, omrToAlign)
#         >>> stA.align()
#         >>> scA.changes == stA.changes
#         True
#         
#         """
#         if not self.checkPartAlignment():
#             raise ValueError('Scores not similar enough to perform alignment.')
#         
#         if self.discretizeParts:
#             self.alignDiscreteParts()
#         else:
#             super().align()
#             
#     def alignDiscreteParts(self):
#         listOfSimilarityScores = []
#         listOfPartChanges = []
#         
#         targetParts = self.targetScore.getElementsByClass(stream.Part)
#         sourceParts = self.sourceScore.getElementsByClass(stream.Part)
#         for targetPart, sourcePart in zip(targetParts, sourceParts):
#             partStreamAligner = StreamAligner(targetPart.flat, sourcePart.flat, hasher=self.hasher)
#             partStreamAligner.align()
#             listOfSimilarityScores.append(partStreamAligner.similarityScore)
#             listOfPartChanges.append(partStreamAligner.changes)
#             self.similarityScore = sum(listOfSimilarityScores) / len(listOfSimilarityScores)
#             self.changes = [change for subPartList in listOfPartChanges for change in subPartList]


class Test(unittest.TestCase):
    pass

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)