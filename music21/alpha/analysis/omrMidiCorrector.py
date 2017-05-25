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
    2) creates the hasher object to be used
    3) fixes the OMR stream based on changes and best alignment output from aligner
    """
    def __init__(self, midiStream, omrStream, hasher=None):
        self.midiStream = midiStream 
        self.omrStream = omrStream
        self.hasher = hasher
        self.discretizeParts = True
        self.midiParts = []
        self.omrParts = []
        self.changes = []
        self.similarityScores = []
        
        self.debugShow = False
        
    def processRunner(self):
        '''
        TODO: change this function to a better name?
        main method that all other method (preprocessing, setting up hasher, aligning are called from)
        '''
        self.preprocessStreams()
        self.setupHasher()
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
            self.midiPartsUnflattened = [pm for pm in self.midiStream.getElementsByClass(stream.Part)]
            self.midiParts = [pm1.flat for pm1 in self.midiPartsUnflattened]
            self.omrPartsUnflattened = [po for po in self.omrStream.getElementsByClass(stream.Part)]
            self.omrParts = [po1.flat for po1 in self.omrPartsUnflattened]

        else:
            self.midiPartsUnflattened = [self.midiStream]
            self.midiParts = [self.midiStream.flat]
            self.omrPartsUnflattened = [self.omrStream]
            self.omrParts = [self.omrStream.flat]
            
        numMidiParts = len(self.midiParts)
        numOmrParts = len(self.omrParts)
        
        if numMidiParts == 0 or numOmrParts == 0:
            raise OMRMIDICorrectorException("Streams must contain at least one part.")
        
        if numMidiParts == numOmrParts:
            pass
            #TODO: something smarter?
        elif numMidiParts - numOmrParts == 1:
            if self.checkBassDoublesCello(self.midiParts[-1], self.midiParts[-2]):
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
        bassCelloHasher.includeReference = True
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
    
#
    
    def alignStreams(self):
        '''
        Creates and aligner object for each of the pairwise aligned midi/omr streams
        If self.debugShow is set, then pairwise aligned streams should show up in MuseScore
        '''
        for (midiPart, omrPart) in zip(self.midiParts, self.omrParts):
            partAligner = aligner.StreamAligner(midiPart, omrPart, hasher=self.hasher)
            partAligner.align()
            self.changes.append(partAligner.changes)
            self.similarityScores.append(partAligner.similarityScore)
            
            if self.debugShow:
                partAligner.showChanges(show=True)
                
    def fixStreams(self):
        '''
        Creates a fixer object for each of the pairwise aligned omr/midi streams
        '''
        pass


class Test(unittest.TestCase):
    
    def testk525(self):
        import os
        from music21 import omr
        pass
        
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
    
    def testSimpleOmrMidik525(self):
        from music21 import converter
        from music21.alpha.analysis import testFiles
        from music21.alpha.analysis import fixer
        from music21 import metadata
        k525shortmidifp = testFiles.K525_short_midi_path
        k525shortomrfp = testFiles.K525_short_omr_path
      
        k525midistream = converter.parse(k525shortmidifp)
        k525omrstream = converter.parse(k525shortomrfp)
      
        k525omc = OMRMIDICorrector(k525midistream, k525omrstream)
        k525omc.debugShow = False
          
        k525omc.processRunner()
        
        print(k525omc.similarityScores)
#         k525violaChanges = k525omc.changes[2]
#         k525violaMidiStream = k525omc.midiPartsUnflattened[2]
#         k525violaOmrStream = k525omc.omrPartsUnflattened[2]
#         k525DeleteFixer = fixer.DeleteFixer(k525violaChanges, k525violaMidiStream, k525violaOmrStream)
#         k525DeleteFixer.fix()
#         k525violaOmrStream.metadata = metadata.Metadata() 
#         k525violaOmrStream.metadata.title = "K.525 Viola Part OMR Stream w/ Delete Fixer 2" 
#         k525violaOmrStream.show()
        
    def testSimpleOmrMidik160(self):
             
        from pprint import pprint
        from music21 import converter
        from music21.alpha.analysis import testFiles
        from music21.alpha.analysis import fixer
        from music21 import metadata
         
        K160iMidiFP = testFiles.K160_mvmt_i_midi_ms_path
        K160iOmrFP = testFiles.K160_mvmt_i_omr_path
         
        midiStream = converter.parse(K160iMidiFP)
        omrStream = converter.parse(K160iOmrFP)
         
#         midiStreamParts = [p for p in midiStream.getElementsByClass(stream.Part)]
#         omrStreamParts = [p for p in omrStream.getElementsByClass(stream.Part)]
#         midiStream = stream.Score()
#         omrStream = stream.Score()
#         midiStream.append(midiStreamParts[2])
#         omrStream.append(omrStreamParts[2])
        K160omc = OMRMIDICorrector(midiStream, omrStream)
        K160omc.debugShow = False
        K160omc.processRunner()
        
        K160omcV1Changes = K160omc.changes[0]
        K160omcV1MidiStream = K160omc.midiParts[0]
        K160omcV1OmrStream = K160omc.omrParts[0]
        K160omcV1OmrStream.metadata = metadata.Metadata() 
        K160omcV1OmrStream.metadata.title = "K.160 Violin 1 Part OMR Stream NO FIXER " 
        K160omcV1OmrStream.show()
        K160EnharmonicFixer = fixer.EnharmonicFixer(K160omcV1Changes, K160omcV1MidiStream, K160omcV1OmrStream)
        K160EnharmonicFixer.fix()
        K160omcV1OmrStream.metadata.title = "K.160 Violin 1 Part OMR Stream w/ Enharmonic Fixer " 
        K160omcV1OmrStream.show()
# #         pprint(K160omc.similarityScores)
        
#   

if __name__ == '__main__':
    import music21
    music21.mainTest(Test)