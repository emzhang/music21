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
        self.midiStream = midiStream 
        self.omrStream = omrStream
        self.hasher = hasher
        self.discretizeParts = True
        self.midiParts = []
        self.omrParts = []
        self.hashedMidiParts = []
        self.hashedOmrParts = []
        
    def processRunner(self):
        '''
        TODO: change this function to a better name?
        main method that all other method (hashing, preprocessing, aligning are called from)
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
        
        >>> midiStream0 = stream.Stream()
        >>> omrStream0 = stream.Stream()
        >>> p01 = stream.Part()
        >>> p02 = stream.Part()
        >>> p03 = stream.Part()
        >>> p04 = stream.Part()
        >>> midiStream0.append([p01, p02])
        >>> omrStream0.append([p03, p04])
        >>> omc0 = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream0, omrStream0)
        >>> omc0.discretizeParts = True
        >>> omc0.preprocessStreams()
        >>> len(omc0.midiParts)
        2
        >>> len(omc0.omrParts)
        2

        >>> midiStream1 = stream.Stream()
        >>> omrStream1 = stream.Stream()
        >>> p11 = stream.Part()
        >>> p12 = stream.Part()
        >>> p13 = stream.Part()
        >>> p14 = stream.Part()
        >>> midiStream1.append([p11, p12])
        >>> omrStream1.append([p13, p14])
        >>> omc1 = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream1, omrStream1)
        >>> omc1.discretizeParts = False
        >>> omc1.preprocessStreams()
        >>> len(omc1.midiParts)
        1
        >>> len(omc1.omrParts)
        1
        
        >>> midiStream2 = stream.Stream()
        >>> omrStream2 = stream.Stream()
        >>> omc2 = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream2, omrStream2)
        >>> omc2.preprocessStreams()
        Traceback (most recent call last):
        music21.alpha.analysis.omrMidiCorrector.OMRMIDICorrectorException: 
        Streams must contain at least one part.
        
        >>> midiStream3 = stream.Stream()
        >>> omrStream3 = stream.Stream()
        >>> p31 = stream.Part()
        >>> p32 = stream.Part()
        >>> p33 = stream.Part()
        >>> p34 = stream.Part()
        >>> midiStream3.append([p31])
        >>> omrStream3.append([p32, p33, p34])
        >>> omc3 = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream3, omrStream3)
        >>> omc3.preprocessStreams()
        Traceback (most recent call last):
        music21.alpha.analysis.omrMidiCorrector.OMRMIDICorrectorException: 
        Streams have uneven number of parts.
        '''
        if self.discretizeParts == True:
            self.midiParts = [p.flat for p in self.midiStream.getElementsByClass(stream.Part)]
            self.omrParts = [p.flat for p in self.omrStream.getElementsByClass(stream.Part)]
        else:
            self.midiParts = [self.midiStream.flat]
            self.omrParts = [self.omrStream.flat]
            
        numMidiParts = len(self.midiParts)
        numOmrParts = len(self.omrParts)
        
        if numMidiParts == 0 or numOmrParts == 0:
            raise OMRMIDICorrectorException("Streams must contain at least one part.")
        
        if numMidiParts == numOmrParts:
            pass
            #TODO: something smarter?
        elif numMidiParts - numOmrParts == 1:
            if self.checkBassDoublesCello(self.midiStream[-1], self.midiStream[-2]):
                pass
            else:
                raise OMRMIDICorrectorException("Streams have uneven number of parts.")
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
    
        >>> midiStream = stream.Stream()
        >>> omrStream = stream.Stream()
        >>> omc = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream, omrStream)
        >>> omc.hasher
        >>> None
        >>> omc.setupHasher()
        >>> omc.hasher
        <music21.alpha.analysis.hasher.Hasher object...>
        '''
        if self.hasher is None:
            self.setDefaultHasher()
    
    def setDefaultHasher(self):
        '''
        returns a default hasher.Hasher object
        which does not hashOffset or include the reference.
        
        called by __init__ if no hasher is passed in.
        
        >>> midiStream = stream.Stream()
        >>> omrStream = stream.Stream()
        >>> omc = alpha.analysis.omrMidiCorrector.OMRMIDICorrector(midiStream, omrStream)
        >>> omc.setDefaultHasher()
        >>> omc.hasher
        <music21.alpha.analysis.hasher.Hasher object at 0x1068cf6a0>
        >>> omc.hasher.hashOffset
        False
        >>> omc.hasher.includeReference
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
    def testDiscretization(self):
        from music21 import stream
        
        midiStream = stream.Stream()
        omrStream = stream.Stream()
        p1 = stream.Part()
        p2 = stream.Part()
        p3 = stream.Part()
        p4 = stream.Part()
        midiStream.append([p1, p2])
        omrStream.append([p3, p4])
        omc = OMRMIDICorrector(midiStream, omrStream)
        omc.discretizeParts = True
        omc.preprocessStreams()

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)