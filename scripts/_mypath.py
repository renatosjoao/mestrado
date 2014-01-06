# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 15:24:48 2014
"""

import os, sys

thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)