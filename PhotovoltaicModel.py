#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================
# Photovoltaic Solar Cell Two-Diode Model
# Code written by:
#   Sidi Hamady
#   Université de Lorraine, France
#   sidi.hamady@univ-lorraine.fr
# See Copyright Notice in COPYRIGHT
# HowTo in README.md and README.pdf
# https://github.com/sidihamady/Photovoltaic-Model
# http://www.hamady.org/photovoltaics/PhotovoltaicModel.zip
# ======================================================================================================

# PhotovoltaicModel.py
#   implements the program interface used to start he calculator
#       execute PhotovoltaicModel.py from the command line prompt by typing:
#           python -u PhotovoltaicModel.py
#       or by double clicking on it (depending on the operating system settings)
#       or from within your editor, if possible.
#       in the graphical interface, change the parameters you want and press 'Calculate'.

# import the program core class in PhotovoltaicModelCore.py
from PhotovoltaicModelCore import *

PVM = PhotovoltaicModelCore(verbose = False)

PVM.calculate(
    Temperature             = 300.0,                        # Temperature in K
    Isc                     = 35.0e-3,                      # Short-cicruit current in A
    Is1                     = 1e-9,                         # Reverse saturation current in A for diode 1
    n1                      = 1.5,                          # Ideality factor for diode 1
    Is2                     = 1e-9,                         # Reverse saturation current in A for diode 2
    n2                      = 2.0,                          # Ideality factor for diode 2
    Diode2                  = True,                         # Enable/Disable diode 2
    Rs                      = 1.0,                          # Series resistance in Ohms
    Rp                      = 10000.0,                      # Parallel resistance in Ohms
    Vstart                  = 0.0,                          # Voltage start value in V
    Vend                    = 1.0,                          # Voltage end value in V
    InputFilename           = None,                         # current-voltage characteristic filename (e.g. containing experimental data)
                                                            #   two columns (voltage in V  and current in A):
                                                            #   0.00	-20.035e-3
                                                            #   0.05	-20.035e-3
                                                            #   ...
                                                            #   0.55	-1.5e-8
    Fit                     = False,                        # Fit the current-voltage characteristic contained in InputFilename
    OutputFilename          = './PhotovoltaicModelOutput'   # Output file name without extension
                                                            #   (used to save figure in PDF format if in GUI mode, and the text output data).
                                                            #   set to None to disable.
    )
