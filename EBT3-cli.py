#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Filippo Falezza
<filippo dot falezza at outlook dot it>
<fxf802 at student dot bham dot ac dot uk>

Released under GPLv3 and followings
"""

from EBT3 import Calibrate, Fitting
import argparse

parser = argparse.ArgumentParser(description='Code to run calibrations and exdtract data from scanned GafChromic films.')
parser.add_argument('calibrate')
