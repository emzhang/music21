# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         alpha/analysis/testFiles.py
# Purpose:      consolidated testFiles
#
# Authors:      Emily Zhang
#
# Copyright:    Copyright © 2015 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
import os
import inspect

pathName = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

K525xmlShortPath = pathName + os.sep + 'k525short3.xml'
K525midiShortPath = pathName + os.sep + 'k525short.mid'
K525omrShortPath = pathName + os.sep + 'k525omrshort.xml' 