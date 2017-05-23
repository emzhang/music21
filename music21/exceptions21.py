# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         exceptions21.py
# Purpose:      music21 Exceptions (called out to not require import music21 to access)
#
# Authors:      Michael Scott Cuthbert
#
# Copyright:    Copyright © 2012 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
'''
module to hold exceptions generated by music21, particularly the ``Music21Exception``
which all other m21 exceptions should derive from.

Do not import anything within this module.  Needs to be import free so other modules
can freely import from it.
'''
# This one is a very general exception that is here because it's very general

class Music21Exception(Exception):
    pass

# The rest of these are here because they are imported by more than one module
# which cannot import the other module because of circular imports.
# 
# if Circular imports have not been a problem and/or you don't plan to use
# an exception in multiple modules (i.e., you're not going to catch a particular
# exception in a different module, then define that exception in the module itself
# (e.g., ClefException is defined in clef) ).


class StreamException(Music21Exception):
    pass

class ImmutableStreamException(StreamException):
    def __init__(self, msg='An immutable Stream cannot be changed'): # pylint: disable=useless-super-delegation
        super(ImmutableStreamException, self).__init__(msg)


class MetadataException(Music21Exception):
    pass

class AnalysisException(Music21Exception):
    pass

class TreeException(Music21Exception):
    pass

class InstrumentException(Music21Exception):
    pass

class Music21CommonException(Music21Exception):
    pass

class CorpusException(Music21Exception):
    pass

# should be renamed because what does Group mean here? and it's "base.Groups" not "base.Group".
class GroupException(Music21Exception):
    pass



# warnings
class Music21DeprecationWarning(UserWarning):
    # Do not subclass Deprecation warning, because these
    # warnings need to be passed to users...
    pass

