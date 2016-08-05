# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         testDocumentation.py
# Purpose:      tests from or derived from the Documentation
#
# Authors:      Michael Scott Cuthbert
#
# Copyright:    Copyright © 2010-2012 Michael Scott Cuthbert and the music21 Project
# License:      LGPL or BSD, see license.txt
#-------------------------------------------------------------------------------
'''
Module to test all the code excerpts in the .rst files in the music21 documentation
and those generated by Jupyter Notebook.

Run only on PY3.  Documentation only is designed to run there.
'''
from __future__ import print_function

import time
import re
import os.path
import sys
import doctest
import io

from collections import namedtuple
from docutils.core import publish_doctree

from music21.exceptions21 import Music21Exception
from music21.test import testRunner


ModTuple = namedtuple('ModTuple', 'module fullModulePath moduleNoExtension autoGen')

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
    
class NoOutput(object):
    def __init__(self, streamSave):
        self.stream = streamSave
        
    def write(self, data):
        pass
    
    def release(self):
        return self.stream
    
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


_MOD = "test.testDocumentation.py"  

skipModules = [
               'documenting.rst', # contains info that screws up testing
               ]

def getDocumentationFromAutoGen(fullModulePath):
    def is_code_or_literal_block(node):
        if node.tagname != 'literal_block':
            return False
        classes = node.attributes['classes']
        if 'ipython-result' in classes:
            return True
        if 'code' in classes and 'python' in classes:
            return True
        return False
        
        
    with io.open(fullModulePath, 'r', encoding='utf-8') as f:
        contents = f.read()
    sys.stderr = NoOutput(sys.stderr)
    doctree = publish_doctree(contents)
    sys.stderr = sys.stderr.release()
    allCodeExpects = []
    lastCode = None
        
    for child in doctree.traverse(is_code_or_literal_block):
        childText = child.astext()
        if '#_DOCS_SHOW' in childText:
            continue
        if 'ipython-result' in child.attributes['classes']:
            childText = childText.strip()
            childText = testRunner.stripAddresses(childText, '...')
            if lastCode is not None:
                allCodeExpects.append((lastCode, childText))
                lastCode = None
        else:
            if lastCode not in (None, ""):
                allCodeExpects.append((lastCode, ""))
            lastCode = None # unneeded but clear
            childTextSplit = childText.split('\n')
            if len(childTextSplit) == 0:
                continue
            childTextArray = [childTextSplit[0]]
            matchesShow = re.search(r'\.show\((.*)\)', childTextSplit[0])
            if matchesShow is not None and not matchesShow.group(1).startswith('t'):
                childTextArray = []
            if re.search(r'.plot\(.*\)', childTextSplit[0]):
                childTextArray = []
                
            if '#_RAISES_ERROR' in childTextSplit[0]:
                childTextArray = []
            if childTextSplit[0].startswith('%'):
                childTextArray = []
                
            for l in childTextSplit[1:]: # split into multiple examples unless indented
                if '#_RAISES_ERROR' in childTextSplit[0]:
                    childTextArray = []
                elif re.search(r'.plot\(.*\)', childTextSplit[0]):
                    continue
                elif l.startswith('%'):
                    childTextArray = []
                elif l.startswith(' '):        
                    matchesShow = re.search(r'\.show\((.*)\)', l)
                    if matchesShow is not None and not matchesShow.group(1).startswith('t'):
                        continue
                    else:
                        childTextArray.append(l)                
                else:
                    lastCode = '\n'.join(childTextArray)
                    if lastCode not in (None, ""):
                        allCodeExpects.append((lastCode, ""))
                        lastCode = None
                    childTextArray = [l]
            lastCode = '\n'.join(childTextArray)
            
    return allCodeExpects

def getDocumentationFiles(runOne=False):
    '''
    returns a list of namedtuples for each module that should be run
    
    >>> from music21.test import testDocumentation
    >>> testDocumentation.getDocumentationFiles()
    [ModTuple(module='index.rst', fullModulePath='...music21/documentation/autogenerated/index.rst', 
    moduleNoExtension='index', autoGen=False),
    ..., 
    ModTuple(module='usersGuide_03_pitches.rst', 
      fullModulePath='...music21/documentation/autogenerated/usersGuide/usersGuide_03_pitches.rst', 
       moduleNoExtension='usersGuide_03_pitches', autoGen=True),
    ...]
    '''
    from music21 import common
    music21basedir = common.getSourceFilePath()
    builddocRstDir = os.path.join(music21basedir,
                                  'documentation',
                                  'source')
    if not os.path.exists(builddocRstDir):
        raise Music21Exception(
            "Cannot run tests on documentation because the rst files " + 
            "in documentation/source do not exist")
    
    allModules = []
    for root, unused_dirnames, filenames in os.walk(builddocRstDir):
        for module in filenames:
            fullModulePath = os.path.join(root, module)
            if not module.endswith('.rst'):
                continue
            if module.startswith('module'): # we have this already...
                continue
            if module in skipModules:
                continue
            if runOne is not False:
                if not module.endswith(runOne):
                    continue
            
            with io.open(fullModulePath, 'r', encoding='utf-8') as f:
                incipit = f.read(1000)
                if 'AUTOMATICALLY GENERATED' in incipit:
                    autoGen = True
                else:
                    autoGen = False
            
            moduleNoExtension = module[:-4]
            modTuple = ModTuple(module, fullModulePath, moduleNoExtension, autoGen)
            allModules.append(modTuple)
    return allModules

def main(runOne=False):
    totalTests = 0
    totalFailures = 0
    
    timeStart = time.time()
    unused_dtr = doctest.DocTestRunner(doctest.OutputChecker(), 
                                verbose=False, 
                                optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
    
    for mt in getDocumentationFiles(runOne):
        #if 'examples' in mt.module:
        #    continue
        print(mt.module + ":", end="")
        try:
            if mt.autoGen is False:
                (failcount, testcount) = doctest.testfile(mt.fullModulePath, module_relative=False, 
                                        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
            else:
                print('ipython/autogenerated; no tests')
                continue
                ### this was an attempt to run the ipynb through the doctest, but
                ### it required too many compromises in how we'd like to write a user's
                ### guide -- i.e., dicts can change order, etc.  better just to
                ### monthly run through the User's Guide line by line and update.
#                 examples = getDocumentationFromAutoGen(mt.fullModulePath)
#                 dt = doctest.DocTest([doctest.Example(e[0], e[1]) for e in examples], {}, 
#                                      mt.moduleNoExtension, mt.fullModulePath, 0, None)
#                 (failcount, testcount) = dtr.run(dt)
            
            
            if failcount > 0:
                print("%s had %d failures in %d tests" % (mt.module, failcount, testcount))
            elif testcount == 0:
                print("no tests")
            else:
                print("all %d tests ran successfully" % (testcount))
            totalTests += testcount
            totalFailures += failcount
        except Exception as e: # pylint: disable=broad-except
            print("failed miserably! %s" % str(e))
            import traceback
            tb = traceback.format_exc()
            print("Here's the traceback for the exception: \n%s" % (tb))


    elapsedTime = time.time() - timeStart
    print("Ran %d tests (%d failed) in %.4f seconds" % (totalTests, totalFailures, elapsedTime))


if __name__ == '__main__':
    #import music21
    #music21.mainTest()
    #getDocumentationFromAutoGen(
    #    '/Users/cuthbert/git/music21base/music21/documentation/autogenerated/usersGuide/usersGuide_02_notes.rst')
    main()
    #main('usersGuide_02_notes.rst')
    #main('overviewPostTonal.rst')
