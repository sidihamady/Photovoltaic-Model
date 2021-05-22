#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================
# Photovoltaic Solar Cell Two-Diode Model
# Code written by:
#   Pr. Sidi Hamady
#   Université de Lorraine, France
#   sidi.hamady@univ-lorraine.fr
# See Copyright Notice in COPYRIGHT
# HowTo in README.md and README.pdf
# https://github.com/sidihamady/Photovoltaic-Model
# http://www.hamady.org/photovoltaics/PhotovoltaicModel.zip
# ======================================================================================================

# PhotovoltaicModelCore.py
#   the class PhotovoltaicModelCore implements the program core functionality
#   only the constructor and the calculate function are to be called from outside the class
#   example (to put in a test.py file, for instance):
#
#       #!/usr/bin/env python
#       #-*- coding: utf-8 -*-
#
#       from PhotovoltaicModelCore import *
#
#       PVM = PhotovoltaicModelCore(verbose = False)
#
#       PVM.calculate(
#           Temperature             = 300.0,                        # Temperature in K
#           Isc                     = 35.0e-3,                      # Short-cicruit current in A
#           Is1                     = 1e-9,                         # Reverse saturation current in A for diode 1
#           n1                      = 1.5,                          # Ideality factor for diode 1
#           Is2                     = 1e-9,                         # Reverse saturation current in A for diode 2
#           n2                      = 2.0,                          # Ideality factor for diode 2
#           Diode2                  = True,                         # Enable/Disable diode 2
#           Rs                      = 1.0,                          # Series resistance in Ohms
#           Rp                      = 10000.0,                      # Parallel resistance in Ohms
#           Vstart                  = 0.0,                          # Voltage start value in V
#           Vend                    = 1.0,                          # Voltage end value in V
#           InputFilename           = None,                         # current-voltage characteristic filename (e.g. containing experimental data)
#                                                                   #   two columns (voltage in V  and current in A):
#                                                                   #   0.00	-20.035e-3
#                                                                   #   0.05	-20.035e-3
#                                                                   #   ...
#                                                                   #   0.55	-1.5e-8
#           Fit                     = False,                        # Fit the current-voltage characteristic contained in InputFilename
#           OutputFilename          = './PhotovoltaicModelOutput'   # Output file name without extension
#                                                                   #   (used to save figure in PDF format if in GUI mode, and the text output data).
#                                                                   #   set to None to disable.
#           )
#

# import as usual
import math
import numpy as np
import scipy as sp
import scipy.optimize as spo
import distutils.version as dver
import sys, os, time
import threading

TkFound = False
TkRet   = ''

# try to load the tkinter and matplotlib modules
# should be always installed in any Linux distribution
# (for Windows, just use some ready-to-use packages such as anaconda (https://www.anaconda.com/distribution/))
try:

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as pl
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.backends.backend_tkagg
    from matplotlib.font_manager import FontProperties
    if sys.version_info[0] < 3:
        # Python 2.7.x
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
        import Tkinter as Tk
        import ttk
        import tkFileDialog
        import tkFont
        import tkMessageBox
    else:
        # Python 3.x
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
        import tkinter as Tk
        import tkinter.ttk as ttk
        import tkinter.filedialog as tkFileDialog
        import tkinter.font as tkFont
        import tkinter.messagebox as tkMessageBox
    # end if

    class NavigationToolbar(NavigationToolbar2TkAgg):
        """ custom Tk toolbar """
        def __init__(self, chart):
            NavigationToolbar2TkAgg.__init__(self, chart.canvas, chart.root)
            self.chart = chart
        # end __init__
        try:
            toolitems = [tt for tt in NavigationToolbar2TkAgg.toolitems if tt[0] in ('Home', 'Zoom')]
            toolitems.append(('AutoScale', 'Auto scale the plot', 'hand', 'onAutoScale'))
            toolitems.append(('Save', 'Save the plot', 'filesave', 'onSave'))
        except:
            pass
        # end try
        def onAutoScale(self):
            self.chart.onAutoScale()
        # end onAutoScale
        def onSave(self):
            self.chart.onSave()
        # end onSave
    # end NavigationToolbar

    TkFound = True

except ImportError as ierr:
    # if Tkinter is not found, just install or update python/numpy/scipy/matplotlib/tk modules
    TkRet = "\n! cannot load Tkinter:\n  " + ("{0}".format(ierr)) + "\n"
    pass
except Exception as excT:
    TkRet = "\n! cannot load Tkinter:\n  %s\n" % str(excT)
    pass
# end try

# suppress a nonrelevant warning from matplotlib and scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# calculations done in a secondary thread, not on UI
class CalculationThread(threading.Thread):
    def __init__(self, id, func):
        threading.Thread.__init__(self)
        self.id     = id
        self.func   = func
    # end __init__
    def run(self):
        self.func()
    # end run
# end CalculationThread

# the core class
class PhotovoltaicModelCore(object):
    """ the PhotovoltaicModel core class """

    def __init__(self, verbose = True):
        """ the PhotovoltaicModel class constructor """

        self.name               = "Photovoltaic Solar Cell Two-Diode Model"
        self.__version__        = "Version 1.0 Build 1811"

        # Basic constants
        self.VT300              = 0.02585202874091      # kT/q at T = 300 K
        
        # current-voltage characteristic filename (e.g. containing experimental data)
        # two columns (voltage in V  and current in A):
        # 0.00	-20.035e-3
        # 0.05	-20.035e-3
        # ...
        # 0.55	-1.5e-8
        self.InputFilename      = None
        self.ModifTime          = 0
        # the used data delimiter (usually TAB) in the I-V ASCII file
        self.DataDelimiter      = '\t'
        # number of rows to skip from the file (by default the first two rows are skipped)
        self.SkipRows           = 2

        self.Temperature        = 300.0                 # in K
        self.VT                 = self.VT300 * self.Temperature / 300.0
        self.Isc                = 35.0e-3               # Short-cicruit current in A
        self.Is1                = 1e-9                  # Reverse saturation current in A for diode 1
        self.n1                 = 1.5                   # Ideality factor for diode 1
        self.Is2                = 1e-9                  # Reverse saturation current in A for diode 2
        self.n2                 = 2.0                   # Ideality factor for diode 2
        self.Diode2             = True                  # Enable/Disable diode 2
        self.Rs                 = 10.0                  # Series resistance in Ohms
        self.Rp                 = 10000.0               # Parallel resistance in Ohms
        self.Vstart             = 0.0                   # Voltage start value in V
        self.Vend               = 1.0                   # Voltage end value in V
        self.OutputFilename     = './PhotovoltaicModelOutput'

        # current-voltage characteristic loaded from a text file (e.g. experimental data)
        self.VoltageX           = None
        self.CurrentX           = None
        self.PVpoints           = None
        self.VOCX               = None
        self.ISCX               = None
        self.VmX                = None
        self.ImX                = None
        self.FFX                = None
        self.PVguess            = True
        #

        # calculated parameters
        self.VoltageY           = None
        self.CurrentY           = None
        self.VOCY               = None
        self.ISCY               = None
        self.VmY                = None
        self.ImY                = None
        self.FFY                = None
        #

        self.running            = False
        self.threadfinish       = None
        self.thread             = None
        self.actionbutton       = None
        self.timerduration      = 100       # in milliseconds

        self.report             = None

        self.root               = None
        self.GUIstarted         = False
        self.PlotInitialized    = False

        self.FileLoaded         = False

        self.tic                = 0.0

        self.nPointsMin         =   20
        self.nPoints            =  100
        self.nPointsDef         =  100
        self.nPointsMax         = 1000

        # one can set verbose to False to disable printing output
        self.verbose            = verbose
        if not self.verbose:
            print("\nverbose set to False: printing output disabled")
        # end if

        self.reportMessage      = None

        return

    # end __init__

    def calculate(self, 
        Temperature             = 300.0,
        Isc                     = 35.0e-3,
        Is1                     = 1e-9,
        n1                      = 1.5,
        Is2                     = 1e-9,
        n2                      = 2.0,
        Diode2                  = True,
        Rs                      = 10.0,
        Rp                      = 10000.0,
        Vstart                  = 0.0,
        Vend                    = 1.0,
        InputFilename           = None,
        Fit                     = False,
        OutputFilename          = './PhotovoltaicModelOutput'):
        """ the PhotovoltaicModel main function """

        if not TkFound:
            # if Tkinter is not found, just install or update python/numpy/scipy/matplotlib/tk modules
            print(TkRet)
        # end if

        # Temperature: in Kelvin (from 100 K to 500 K)
        self.Temperature    = Temperature   if ((Temperature >= 100.0)  and (Temperature <= 500.0))     else 300.0
        self.VT             = self.VT300 * self.Temperature / 300.0     # in V

        # Short-cicruit current in A
        self.Isc            = Isc           if ((Isc    >= 0.0)         and (Isc    <= 100.0))          else 20.0e-3

        # Reverse saturation current in A for diode 1
        self.Is1            = Is1           if ((Is1    > 0.0)          and (Is1    <= 1e-3))           else 1e-9
        self.n1             = n1            if ((n1     >= 1.0)         and (n1     <= 10.0))           else 1.0

        # Reverse saturation current in A for diode 2
        self.Is2            = Is1           if ((Is1    >= 0.0)         and (Is1    <= 1e-3))           else 1e-9

        # Ideality factor for diode 2
        self.n2             = n2            if ((n2     >= 1.0)         and (n2     <= 20.0))           else 2.0

        # Enable/Disable diode 2
        self.Diode2         = Diode2

        # Series resistance in Ohms
        self.Rs             = Rs            if ((Rs     >= 1e-6)        and (Rs     <= 1e6))            else 10.0

        # Parallel resistance in Ohms
        self.Rp             = Rp            if ((Rp     >= 1e-3)        and (Rp     <= 1e9))            else 10000.0

        # Voltage start value in V
        self.Vstart         = Vstart        if ((Vstart >= 0.0)         and (Vstart <= 10.0))           else 0.0

        # Voltage end value in V
        self.Vend           = Vend          if ((Vend   >= 0.0)         and (Vend   <= 10.0))           else 0.0

        # Output file name without extension (used to save figure in PDF format if in GUI mode, and the text output data).
        #   set to None to disable.
        self.OutputFilename         = OutputFilename
        if self.OutputFilename and (not self.OutputFilename.endswith('.pdf')):
            self.OutputFilename     = self.OutputFilename + '.pdf'
        # end if

        # current-voltage characteristic filename (e.g. containing experimental data)
        # two columns (voltage in V  and current in A):
        # 0.00	-20.035e-3
        # 0.05	-20.035e-3
        # ...
        # 0.55	-1.5e-8
        self.InputFilename = InputFilename
        self.ModifTime     = self.getModifTime(self.InputFilename)

        Fit = False     # Fit the current-voltage characteristic contained in InputFilename

        if TkFound:
            # GUI mode: calculation done in a working thread
            self.startGUI(Fit = Fit)
        else:
            # command-line mode
            self.start(Fit = Fit)
        # end if

        return

    # end calculate

    def getModifTime(self, InputFilename):
        mt = 0
        try:
            mt = os.stat(InputFilename).st_mtime
        except:
            pass
        # end try
        return mt
    # end getModifTime

    def isRunning(self):
        if (self.thread is None):
            return self.running
        # end if
        if (not self.thread.isAlive()):
            self.thread  = None
            self.running = False
        # end if
        return self.running
    # end isRunning

    def setRunning(self, running = True):
        self.running = running
        if self.actionbutton is not None:
            self.actionbutton["text"] = self.actionbuttonText
            if self.running:
                self.actionbutton.configure(style='Red.TButton')
            else:
                self.actionbutton.configure(style='Black.TButton')
                self.actionbutton = None
            # end if
        # end if
    # end setRunning

    # init the Tkinter GUI
    def startGUI(self, Fit = False):

        if self.GUIstarted or (not TkFound):
            return
        # end if

        try:
            self.plotcount      = 1
            self.curvecount     = 2
            self.xLabel         = {}
            self.yLabel         = {}
            self.xLabel[0]      = '$Voltage\ (V)$'
            self.yLabel[0]      = '$Current\ (A)$'

            self.root = Tk.Tk()
            self.root.bind_class("Entry","<Control-a>", self.onEntrySelectAll)
            self.root.bind_class("Entry","<Control-z>", self.onEntryUndo)
            self.root.bind_class("Entry","<Control-y>", self.onEntryRedo)
            self.root.withdraw()
            self.root.wm_title(self.name)

            self.figure = matplotlib.figure.Figure(figsize=(10,8), dpi=100, facecolor='#F1F1F1', linewidth=1.0, frameon=True)

            self.figure.subplots_adjust(top = 0.9, bottom = 0.1, left = 0.09, right = 0.95, wspace = 0.25, hspace = 0.25)

            self.plot       = {}
            self.plot[0]    = self.figure.add_subplot(111)
            self.plot[0].set_xlim( 0.0,     0.7)
            self.plot[0].set_ylim(-0.035,   0.005)

            self.line0a     = None
            self.line0b     = None

            self.datax      = {}
            self.datay      = {}
            self.datax[0]   = None
            self.datay[0]   = None
            self.datax[1]   = None
            self.datay[1]   = None

            spx  = 6
            spy  = 12
            spxm = 1
            parFrameA = Tk.Frame(self.root)
            parFrameA.pack(fill=Tk.X, side=Tk.TOP, padx=spx, pady=spx)
            parFrameB = Tk.Frame(self.root)
            parFrameB.pack(fill=Tk.X, side=Tk.TOP, padx=spx, pady=spx)

            FloatValidate = (parFrameA.register(self.onFloatValidate), '%P')

            self.LLabel = Tk.Label(parFrameA, text=" ")
            self.LLabel.pack(fill=Tk.X, side=Tk.LEFT, expand=True, padx=(spxm, spxm), pady=spy)

            self.TemperatureLabel = Tk.Label(parFrameA, text="T (K): ")
            self.TemperatureLabel.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.TemperatureEdit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.TemperatureEdit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.TemperatureEdit.insert(0, ("%.1f" % self.Temperature) if (self.Temperature is not None) else "")
            self.TemperatureEdit.prev = None
            self.TemperatureEdit.next = None

            self.IscLabel = Tk.Label(parFrameA, text="Isc (A): ")
            self.IscLabel.pack(side=Tk.LEFT, padx=(spxm, spxm), pady=spy)
            self.IscEdit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.IscEdit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.IscEdit.insert(0, ("%.4g" % self.Isc) if (self.Isc is not None) else "")
            self.IscEdit.prev = None
            self.IscEdit.next = None

            self.Is1Label = Tk.Label(parFrameA, text="Is1 (A): ")
            self.Is1Label.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.Is1Edit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.Is1Edit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.Is1Edit.insert(0, ("%.4g" % self.Is1) if (self.Is1 is not None) else "")
            self.Is1Edit.prev = None
            self.Is1Edit.next = None
            self.n1Label = Tk.Label(parFrameA, text="n1: ")
            self.n1Label.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.n1Edit = Tk.Entry(parFrameA, width=7, validate="key", vcmd=FloatValidate)
            self.n1Edit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.n1Edit.insert(0, ("%.4f" % self.n1) if (self.n1 is not None) else "")
            self.n1Edit.prev = None
            self.n1Edit.next = None

            self.Is2Label = Tk.Label(parFrameA, text="Is2 (A): ")
            self.Is2Label.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.Is2Edit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.Is2Edit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.Is2Edit.insert(0, ("%.4g" % self.Is2) if (self.Is2 is not None) else "")
            self.Is2Edit.prev = None
            self.Is2Edit.next = None
            self.Diode2Var = Tk.BooleanVar()
            self.Diode2Opt = Tk.Checkbutton(parFrameA, text="", variable=self.Diode2Var, command=self.onDiode2Opt)
            self.Diode2Opt.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.Diode2Opt.select()
            self.n2Label = Tk.Label(parFrameA, text="n2: ")
            self.n2Label.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.n2Edit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.n2Edit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.n2Edit.insert(0, ("%.4f" % self.n2) if (self.n2 is not None) else "")
            self.n2Edit.prev = None
            self.n2Edit.next = None

            self.RsLabel = Tk.Label(parFrameA, text="Rs (Ohms): ")
            self.RsLabel.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.RsEdit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.RsEdit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.RsEdit.insert(0, ("%.4g" % self.Rs) if (self.Rs is not None) else "")
            self.RsEdit.prev = None
            self.RsEdit.next = None
            self.RpLabel = Tk.Label(parFrameA, text="Rp (Ohms): ")
            self.RpLabel.pack(side=Tk.LEFT, padx=(spx, spxm), pady=spy)
            self.RpEdit = Tk.Entry(parFrameA, width=10, validate="key", vcmd=FloatValidate)
            self.RpEdit.pack(side=Tk.LEFT, padx=(spxm, spx), pady=spy)
            self.RpEdit.insert(0, ("%.4g" % self.Rp) if (self.Rp is not None) else "")
            self.RpEdit.prev = None
            self.RpEdit.next = None

            self.RLabel = Tk.Label(parFrameA, text=" ")
            self.RLabel.pack(fill=Tk.X, side=Tk.LEFT, expand=True, padx=(spxm, spxm), pady=spy)

            self.InputFilenameLabel = Tk.Label(parFrameB, width=16, text="Input Filename: ")
            self.InputFilenameLabel.pack(side=Tk.LEFT)
            inputFilenameValidate = (parFrameB.register(self.onInputFilenameValidate), '%P')
            self.InputFilenameEdit = Tk.Entry(parFrameB, validate="key", vcmd=inputFilenameValidate)
            self.InputFilenameEdit.pack(side=Tk.LEFT, fill=Tk.X, expand=1)
            self.InputFilenameEdit.insert(0, self.InputFilename if (self.InputFilename is not None) else "")
            self.InputFilenameEdit.prev = None
            self.InputFilenameEdit.next = None
            self.inputFilenameBrowse = ttk.Button(parFrameB, width=4, text="...", command=self.onBrowse)
            self.inputFilenameBrowse.pack(side=Tk.LEFT, padx=(2, 2))

            self.btnstyle_red = ttk.Style()
            self.btnstyle_red.configure("Red.TButton", foreground="#DE0015")
            self.btnstyle_black = ttk.Style()
            self.btnstyle_black.configure("Black.TButton", foreground="black")

            self.btnFit = ttk.Button(parFrameB, width=18, text="Fit", compound=Tk.LEFT, command=self.onFit)
            self.btnFit.pack(side=Tk.LEFT, padx=spx, pady=spy)
            self.btnFit.configure(style="Black.TButton")
            self.btnFit.configure(state="disabled")

            self.btnCalculate = ttk.Button(parFrameB, width=18, text="Calculate", compound=Tk.LEFT, command=self.onStart)
            self.btnCalculate.pack(side=Tk.LEFT, padx=spx, pady=spy)
            self.btnCalculate.configure(style="Black.TButton")
            self.root.bind('<Return>', self.onEnter)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
            self.canvas._tkcanvas.config(highlightthickness=0)

            try:
                self.toolbar = NavigationToolbar(self)
            except:
                self.toolbar = None
                pass
            # end try
            self.toolbar.pack(side=Tk.BOTTOM, fill=Tk.X)

            self.toolbar.update()
            self.canvas._tkcanvas.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

            if sys.version_info[0] < 3:
                self.canvas.show()
            else:
                self.canvas.draw()
            #

            self.root.protocol('WM_DELETE_WINDOW', self.onClose)

            self.linecolor          = ['r', 'b']
            self.linestyle          = ['-', 'o']
            self.linesize           = [2.0, 1.0]
            self.markersize         = [2.0, 4.0]
            self.line               = {}
            self.scatter            = {}
            for idc in range(0, self.curvecount):
                self.line[idc]      = None
                self.scatter[idc]   = None
            # end for

            # center the window
            iw = self.root.winfo_screenwidth()
            ih = self.root.winfo_screenheight()
            isize = (1020, 680)
            ix = (iw - isize[0]) / 2
            iy = (ih - isize[1]) / 2
            self.root.geometry("%dx%d+%d+%d" % (isize + (ix, iy)))

            self.root.minsize(800, 600)

            self.fontsize = 10

            for idp in range(0, self.plotcount):
                try:
                    self.plot[idp].tick_params(axis='x', labelsize=self.fontsize)
                    self.plot[idp].tick_params(axis='y', labelsize=self.fontsize)
                except:
                    [tx.label.set_fontsize(self.fontsize) for tx in self.plot[idp].xaxis.get_major_ticks()]
                    [ty.label.set_fontsize(self.fontsize) for ty in self.plot[idp].yaxis.get_major_ticks()]
                    pass
                # end try

                self.plot[idp].set_xlabel(self.xLabel[idp], fontsize=self.fontsize)
                self.plot[idp].set_ylabel(self.yLabel[idp], fontsize=self.fontsize)
            # end for

            if (os.name == "nt"):
                self.root.iconbitmap(r'iconmain.ico')
            else:
                iconmain = Tk.PhotoImage(file='iconmain.gif')
                self.root.tk.call('wm', 'iconphoto', self.root._w, iconmain)
            # end if

            self.popmenu = Tk.Menu(self.root, tearoff=0)
            self.popmenu.add_command(label="Fit", command=self.onFit)
            self.popmenu.add_command(label="Calculate", command=self.onStart)
            self.popmenu.add_separator()
            self.popmenu.add_command(label="Auto scale", command=self.onAutoScale)
            self.popmenu.add_separator()
            self.popmenu.add_command(label="Close", command=self.onClose)
            self.popmenu.add_separator()
            self.popmenu.add_command(label="About...", command=self.onAbout)
            self.root.bind("<Button-3>", self.onPopmenu)

            self.root.deiconify()
            self.setFocus()

            self.GUIstarted = True

            self.start(Fit = Fit)

            self.root.mainloop()

        except Exception as excT:
            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            strErr  = "\n! cannot initialize GUI:\n  %s\n  in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            print(strErr)
            os._exit(1)
            # never reached
            pass
        # end try

    # end startGUI

    def isIncSorted(self, arr):
        for ii in range(arr.size - 1):
            if arr[ii + 1] <= arr[ii]:
                return False
            # end if
        # end for
        return True
    # end isIncSorted

    # load the I-V characteristic file
    def loadFile(self):

        if not self.GUIstarted:
            self.FileLoaded = False
        else:
            try:
                strT = self.InputFilenameEdit.get().strip("\r\n\t")
                if os.path.isfile(strT):
                    if (self.InputFilename != strT):
                        self.InputFilename  = strT
                        self.FileLoaded     = False
                    else:
                        modifTime = self.getModifTime(strT)
                        if (modifTime > self.ModifTime):
                            self.FileLoaded = False
                        # end if
                    # end if
                else:
                    self.InputFilenameEdit.delete(0, Tk.END)
                    self.InputFilenameEdit.insert(0, self.InputFilename if (self.InputFilename is not None) else "")
                # end if
            except Exception as excT:
                pass
            # end try
        # endif

        if (self.InputFilename is None) or (not os.path.isfile(self.InputFilename)):
            return False
        # end if

        if self.FileLoaded:
            return True
        # end if

        try:

            self.PVpoints  = 0
            self.ModifTime = self.getModifTime(self.InputFilename)

            # load the current-voltage characteristic from file (e.g. containing experimental data)
            IVXData                 = np.loadtxt(self.InputFilename, delimiter=self.DataDelimiter, skiprows=self.SkipRows, usecols=(0,1))
            self.VoltageX           = IVXData[:,0]                   # V
            self.CurrentX           = IVXData[:,1]                   # A
            # check the data consistency
            if ((len (self.VoltageX)                < 5)                        or 
                (len (self.CurrentX)                < 5)                        or 
                (len (self.VoltageX)                != len(self.CurrentX))      or 
                (not (self.VoltageX                 >= -10.0).all())            or 
                (not (self.VoltageX                 <=  10.0).all())            or 
                (not (self.CurrentX                 >= -10.0).all())            or 
                (not (self.CurrentX                 <=  10.0).all())            or
                (not self.isIncSorted(self.VoltageX))):
                raise Exception('invalid voltage/current data')
            # end if
            nPoints                 = len(self.VoltageX)

            self.VOCX               = self.VoltageX[nPoints - 1]
            self.ISCX               = self.CurrentX[0]
            self.VmX                = 0.0
            self.ImX                = 0.0
            self.FFX                = 0.0
            aPm                     = 0.0
            aIprev                  = None
            aVprev                  = None
            countPV                 = 0
            # check quadrant
            # V > 0 and I > 0: to convert to Q4
            # V < 0 and I > 0: to convert to Q4
            # V < 0 and I < 0: to convert to Q4
            # V > 0 and I < 0: the used quadrant
            nQ  = [0, 0, 0, 0]
            for aV, aI in zip(self.VoltageX, self.CurrentX):
                if      (aI > 0.0) and (aV > 0.0):
                    nQ[0] += 1
                elif    (aI > 0.0) and (aV < 0.0):
                    nQ[1] += 1
                elif    (aI < 0.0) and (aV < 0.0):
                    nQ[2] += 1
                elif    (aI < 0.0) and (aV > 0.0):
                    nQ[3] += 1
                # end if
            # end for
            nQmax   = max(nQ)
            nQindex = nQ.index(nQmax) + 1
            if (nQmax < (nPoints / 2)):
                raise Exception('invalid voltage/current data: no photovoltaic quadrant identified')
            # end if
            ii = 0
            aVX             = np.copy(self.VoltageX)
            aIX             = np.copy(self.CurrentX)
            self.VoltageX   = np.array([])
            self.CurrentX   = np.array([])

            # keep only the photovoltaic part of the current-voltage characteristic
            # and convert to the fourth quadrant (V > 0 and I < 0) if necessary

            for aV, aI in zip(aVX, aIX):

                if (nQindex != 4):
                    # always use the solar cell current-voltage characteristic in the fourth quadrant 4 (V > 0 and I < 0)
                    if      (nQindex == 1):
                        aI  = -aI
                    elif    (nQindex == 2):
                        aI  = -aI
                        aV  = -aV
                    elif    (nQindex == 3):
                        aV  = -aV
                    # end if
                    aVX[ii] = aV
                    aIX[ii] = aI
                # end if

                if (aIprev is not None) and (aI >= 0.0) and (aIprev <= 0.0):
                    self.VOCX       = aV
                # end if
                if (aVprev is not None) and (aV >= 0.0) and (aVprev <= 0.0):
                    self.ISCX       = aI
                # end if
                if (aI <= 0.0) and (aV >= 0.0):
                    countPV += 1
                    self.VoltageX   = np.append(self.VoltageX, aV)
                    self.CurrentX   = np.append(self.CurrentX, aI)
                # end if
                if (aI < 0.0) and (aV > 0.0) and (math.fabs(aI * aV) > aPm):
                    aPm             = math.fabs(aI * aV)
                    self.VmX        = aV
                    self.ImX        = aI
                # end if
                aVprev = aV
                aIprev = aI
                ii    += 1

            # end for

            if (countPV < (nPoints / 5)) or (countPV < self.nPointsMin):
                raise Exception('invalid voltage/current: insufficient number of photovoltaic points (%d): should be greater than %d' % (countPV, self.nPointsMin))
            # end if
            nPoints = len(self.VoltageX)

            if self.verbose and (nQindex != 4):
                print('\ncurrent-voltage characteristic converted from quadrant %d to quadrant 4' % nQindex)
            # end if
            self.PVpoints = countPV

            self.ISCX       = math.fabs(self.ISCX)
            if (self.ISCX > 0.0) and (self.VOCX > 0.0):
                self.FFX    = math.fabs((self.VmX * self.ImX) / (self.VOCX * self.ISCX))
                if (self.FFX > 1.0):
                    # should never happen
                    self.FFX = 0.0
                # end if
                self.Isc = self.ISCX
                if self.GUIstarted:
                    self.IscEdit.delete(0, Tk.END)
                    self.IscEdit.insert(0, "%.4g" % self.Isc)
                # end if
            # end if

            self.nPoints        = nPoints if (nPoints >= self.nPointsMin) else self.nPointsDef
            Vstep               = (self.VoltageX[nPoints - 1] - self.VoltageX[0]) / float(self.nPoints)
            Vstart              = self.VoltageX[0]
            if (Vstart > 0.0):
                Vstart = 0.0
            # end if
            Vend                = self.VoltageX[nPoints - 1]
            if (self.CurrentX[nPoints - 1] < 0.0):
                VendC = Vend
                for aV in np.arange(VendC, VendC + 100.0 * Vstep, Vstep):
                    (aI, aIZ) = self.calculateCurrent(aV)
                    if aI >= 0.0:
                        Vend    = VendC
                        break
                     # end if
                # end for
            # end if
            self.Vstart         = Vstart
            self.Vend           = Vend
            self.VoltageY       = np.arange(Vstart, Vend + Vstep, Vstep)
            self.CurrentY       = None

            if (self.ISCX > 0.0) and (self.VOCX > 0.0) and (self.FFX > 0.0):
                self.reportMessage = "Isc = %.4g A ; Voc = %.4g V ; FF = %.4g %% ; Pm = %.4g W" % (self.ISCX, self.VOCX, 100.0 * self.FFX, self.FFX * self.ISCX * self.VOCX)
            else:
                self.reportMessage = "! The photovoltaic parameters cannot be extracted"
            # end if
            if self.verbose:
                print("\n" + self.reportMessage)
            # end if

            if TkFound:
                self.datax[1]   = self.VoltageX
                self.datay[1]   = self.CurrentX
            # end if

            self.FileLoaded     = True
            return True

        except Exception as excT:
            self.VoltageX = None
            self.CurrentX = None
            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            strErr  = "\n! cannot load the current-voltage characteristic:\n  %s\n  in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            self.reportMessage = '! cannot load the current-voltage characteristic'
            if self.verbose:
                print(strErr)
            # end if
            return False
            # never reached
            pass
        # end try

    # end loadFile

    def FitFunc(self, aVoltage, Isc, Is1, n1, Is2, n2, Rs, Rp):
        self.Isc        = Isc
        self.Is1        = Is1
        self.n1         = n1
        if self.Diode2:
            self.Is2    = Is2
            self.n2     = n2
        # end if
        self.Rs         = Rs
        self.Rp         = Rp
        aCurrent        = np.array([])
        for aV in aVoltage:
            (aI, aIZ)   = self.calculateCurrent(aV)
            aCurrent    = np.append(aCurrent, aI)
        # end for
        return aCurrent
    # end FitFunc

    def FitFuncD(self, aVoltage, Isc, Is1, n1, Rs, Rp):
        return self.FitFunc(aVoltage, Isc, Is1, n1, 0.0, self.n2, Rs, Rp)
    # end FitFuncD

    def CurrentFunc(self, I, V):
        Vp   = V - (self.Rs * I)
        Isc  = -self.Isc
        Iph  = Isc
        Iph += (self.Is1 * (math.exp(self.Rs * Isc / (self.n1 * self.VT)) - 1.0))
        if self.Diode2:
            Iph += (self.Is2 * (math.exp(self.Rs * Isc / (self.n2 * self.VT)) - 1.0))
        # end if
        Iph += ((self.Rs / self.Rp) * Isc)
        fI   = Iph
        fI  += (self.Is1 * (math.exp(Vp / (self.n1 * self.VT)) - 1.0))
        if self.Diode2:
            fI  += (self.Is2 * (math.exp(Vp / (self.n2 * self.VT)) - 1.0))
        # end if
        fI  += (Vp / self.Rp)
        fI  -= I
        return fI
    # end CurrentFunc

    def calculateCurrent(self, V):
        fIZ  = -self.Isc
        fIZ += (self.Is1 * (math.exp(V / (self.n1 * self.VT)) - 1.0))
        if self.Diode2:
            fIZ += (self.Is2 * (math.exp(V / (self.n2 * self.VT)) - 1.0))
        # end if
        fIZ += (V / self.Rp)
        fI   = spo.fsolve(self.CurrentFunc, x0=fIZ, args=(V,))
        return (fI, fIZ)
    # end calculateCurrent

    # calculate the current-voltage characteristic
    def calculateCharacteristic(self):
        try:

            # check the voltage range
            if (self.VoltageY is None) or (self.Vend <= self.Vstart) or ((self.Vstart <= 0.0) and (self.Vend <= 0.0)):
                self.Vstart     = 0.0
                self.Vend       = 1.0
            # end if
            if (self.nPoints   < self.nPointsMin):
                self.nPoints = self.nPointsMin
            elif (self.nPoints > self.nPointsMax):
                self.nPoints = self.nPointsMax
            # end if
            if (self.Vstart > 0.0):
                self.Vstart  = 0.0
            # end if
            (aI, aIZ) = self.calculateCurrent(self.Vend)
            if (aI < 0.0):
                Vend    = self.Vend if (self.Vend > self.VOCX) else self.VOCX
                Vstep   = (Vend - self.Vstart) / float(self.nPoints)
                for aV in np.arange(Vend, Vend + 100.0 * Vstep, Vstep):
                    (aI, aIZ) = self.calculateCurrent(aV)
                    if aI > 0.0:
                        self.Vend = aV + Vstep
                        break
                     # end if
                # end for
            # end if
            #

            Vstep               = (self.Vend - self.Vstart) / float(self.nPoints)
            self.VoltageY       = np.arange(self.Vstart, self.Vend + Vstep, Vstep)

            self.VOCY           = 0.0
            self.ISCY           = 0.0
            self.VmY            = 0.0
            self.ImY            = 0.0
            self.FFY            = 0.0
            aIprev              = None
            aVprev              = None
            aPm                 = 0.0
            aVoltage            = np.copy(self.VoltageY)
            self.VoltageY       = np.array([])
            self.CurrentY       = np.array([])
            countPV             = 0

            for aV in aVoltage:
                (aI, aIZ) = self.calculateCurrent(aV)
                if (aI <= 0.0) and (aV >= 0.0):
                    self.VoltageY   = np.append(self.VoltageY, aV)
                    self.CurrentY   = np.append(self.CurrentY, aI)
                    countPV += 1
                # end if
                if (aI < 0.0) and (aV > 0.0) and (math.fabs(aI * aV) > aPm):
                    aPm             = math.fabs(aI * aV)
                    self.VmY        = aV
                    self.ImY        = aI
                # end if
                if (aIprev is not None) and (aI >= 0.0) and (aIprev <= 0.0):
                    self.VOCY   = aV
                # end if
                if (aVprev is not None) and (aV >= 0.0) and (aVprev <= 0.0):
                    self.ISCY   = aI
                # end if
                if (aI > 0.0):
                    if (aIprev is not None) and (aIprev <= 0.0):
                        self.VoltageY   = np.append(self.VoltageY, aV)
                        self.CurrentY   = np.append(self.CurrentY, aI)
                        countPV += 1
                    # end if
                    break
                # end if
                aVprev = aV
                aIprev = aI
            # end for

            if (countPV < self.nPointsMin) and (math.fabs(self.VOCY) > 0.0) and (math.fabs(self.ISCY) > 0.0):
                # recalculate with more points
                Vstep = math.fabs(self.VOCY) / float(self.nPoints)
                self.VoltageY = np.arange(0.0, self.VOCY + Vstep, Vstep)
                self.CurrentY = np.array([])
                for aV in self.VoltageY:
                    (aI, aIZ)  = self.calculateCurrent(aV)
                    self.CurrentY   = np.append(self.CurrentY, aI)
                    if (aI < 0.0) and (aV > 0.0) and (math.fabs(aI * aV) > aPm):
                        aPm             = math.fabs(aI * aV)
                        self.VmY        = aV
                        self.ImY        = aI
                    # end if
                # end for
            # end if

            self.ISCY = math.fabs(self.ISCY)
            if (self.ISCY > 0.0) and (self.VOCY > 0.0):
                self.FFY = math.fabs((self.VmY * self.ImY) / (self.VOCY * self.ISCY))
                if (self.FFY > 1.0):
                    # should never happen
                    self.FFY = 0.0
                # end if
            # end if

            if (self.ISCY > 0.0) and (self.VOCY > 0.0) and (self.FFY > 0.0):
                self.reportMessage = "Isc = %.4g A ; Voc = %.4g V ; FF = %.4g %% ; Pm = %.4g W" % (self.ISCY, self.VOCY, 100.0 * self.FFY, self.FFY * self.ISCY * self.VOCY)
            else:
                self.reportMessage = "! The photovoltaic parameters cannot be calculated"
            # end if
            if self.verbose:
                print("\n" + self.reportMessage)
            # end if

            return True

        except Exception as excT:

            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            strErr  = "\n! cannot calculate the current-voltage characteristic:\n  %s\n  in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            self.reportMessage = '! cannot calculate the current-voltage characteristic'
            if self.verbose:
                print(strErr)
            # end if
            return False
            # never reached
            pass

        # end try

    # end calculateCharacteristic

    def monitorCalculation(self):
        running = self.isRunning()
        try:
            if not running:
                self.setRunning(running = False)
                if self.threadfinish is not None:
                    self.threadfinish()
                    self.threadfinish = None
                # end if
                return
            # end if
            if self.root:
                self.root.after(self.timerduration if ((self.timerduration >= 100) and (self.timerduration <= 1000)) else 200, self.monitorCalculation)
            # end if
        except Exception as excT:
            pass
        # end try
    # end monitorCalculation

    def getFloatValue(self, ValueEdit, ValueDef, ValueMin, ValueMax, ValueFormat):
        try:
            strT = ValueEdit.get().strip("\r\n\t")
            fT = float(strT)
            if (fT >= ValueMin) and (fT <= ValueMax):
                return fT
            else:
                ValueEdit.delete(0, Tk.END)
                ValueEdit.insert(0, ValueFormat % ValueDef)
                return ValueDef
            # end if
        except:
            return ValueDef
            # never reached
            pass
        # end try
    # end getFloatValue

    # start the current-voltage characteristic calculation
    def start(self, Fit = False):

        if self.isRunning():
            return False
        # end if

        self.reportMessage = None

        self.loadFile()

        if self.GUIstarted:

            if self.report is not None:
                self.report.set_text(" ")
            # end if

            # get the input parameters

            self.Temperature    = self.getFloatValue(self.TemperatureEdit, self.Temperature, 100.0, 500.0, "%.1f")
            self.VT             = self.VT300 * self.Temperature / 300.0

            self.Isc            = self.getFloatValue(self.IscEdit,  self.Isc,   0.0,    100.0,  "%.3g")

            self.Is1            = self.getFloatValue(self.Is1Edit,  self.Is1,   1e-15,  1e-3,   "%.3g")
            self.n1             = self.getFloatValue(self.n1Edit,   self.n1,    1.0,    10.0,   "%.3f")

            self.Is2            = self.getFloatValue(self.Is2Edit,  self.Is2,   0.0,    1e-3,   "%.3g")
            self.n2             = self.getFloatValue(self.n2Edit,   self.n2,    1.0,    10.0,   "%.3f")
            self.Diode2         = self.Diode2Var.get()

            self.Rs             = self.getFloatValue(self.RsEdit,   self.Rs,    1e-6,   1e6,    "%.3g")
            self.Rp             = self.getFloatValue(self.RpEdit,   self.Rp,    1e-3,   1e9,    "%.3g")

            try:
                strT = self.InputFilenameEdit.get().strip("\r\n\t")
                if os.path.isfile(strT):
                    self.InputFilename = strT
                else:
                    self.InputFilenameEdit.delete(0, Tk.END)
                    self.InputFilenameEdit.insert(0, self.InputFilename if (self.InputFilename is not None) else "")
                # end if
            except:
                pass
            # end try

            # start calculations
            self.actionbutton = self.btnFit if Fit else self.btnCalculate
            self.actionbuttonText = "Fit" if Fit else "Calculate"
            self.setRunning(running = True)
            self.thread = CalculationThread(id=1, func=self.fit if Fit else self.run)
            self.threadfinish = self.updatePlot
            self.thread.start()
            self.monitorCalculation()

        else:
            # GUI initialization step
            if TkFound:
                self.actionbutton = self.btnFit if Fit else self.btnCalculate
                self.actionbuttonText = "Fit" if Fit else "Calculate"
                self.setRunning(running = True)
                self.thread = CalculationThread(id=1, func=self.fit if Fit else self.run)
                self.threadfinish = self.updatePlot
                self.thread.start()
                self.monitorCalculation()
            else:
                self.setRunning(running = True)
                self.run()
                self.setRunning(running = False)
                self.doSave(self.OutputFilename)
            # end if
        # end if

     # end start

    # run the current-voltage characteristic calculation
    def run(self):

        try:
            if self.verbose:
                print("\ncalculating...")
            # end if

            # to determine the calculation duration
            ticT = time.time()

            # calculate the current-voltage characteristic
            bRet = self.calculateCharacteristic()
            if not bRet:
                if self.verbose:
                    print("\ndone.")
                # end if
                return False
            # endif

            if TkFound:
                self.datax[0] = self.VoltageY
                self.datay[0] = self.CurrentY
            # endif

            self.tic = float(time.time() - ticT)

            if self.verbose:
                print("\ndone. elapsed time = %.6f sec." % self.tic)
            # end if

            return True

        except Exception as excT:

            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            strErr  = "\n! cannot calculate the current-voltage characteristic:\n  %s\n  in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            self.reportMessage = '! cannot calculate the current-voltage characteristic'
            if self.verbose:
                print(strErr)
            # end if
            return False
            # never reached
            pass

        # end try

    # end run

    def fit(self):

        try:
            if self.verbose:
                print("\nfitting...")
            # end if

            if not self.FileLoaded:
                if self.verbose:
                    print("\ndone.")
                # end if
                return False
            # end if

            if TkFound:
                self.datax[1] = self.VoltageX
                self.datay[1] = self.CurrentX
            # end if

            # to determine the calculation duration
            ticT = time.time()

            bRet = False

            try:
                if dver.StrictVersion(sp.__version__) >= dver.StrictVersion("0.19.1"):
                    popt, pcov = spo.curve_fit(self.FitFunc if self.Diode2 else self.FitFuncD, self.VoltageX, self.CurrentX,
                        bounds=([1e-15, 1e-15, 1.0, 0.0, 1.0, 1e-9, 1e-6] if self.Diode2 else [1e-15, 1e-15, 1.0, 1e-9, 1e-6],
                        [10.0, 1e-3, 10.0, 1e-3, 10.0, 1e6, 1e9] if self.Diode2 else [10.0, 1e-3, 10.0, 1e6, 1e9]),
                        p0=np.array([self.Isc, self.Is1, self.n1, self.Is2, self.n2, self.Rs, self.Rp] if self.Diode2 else [self.Isc, self.Is1, self.n1, self.Rs, self.Rp]),
                        maxfev=100)
                else:
                    popt, pcov = spo.curve_fit(self.FitFunc if self.Diode2 else self.FitFuncD, self.VoltageX, self.CurrentX,
                        bounds=([1e-15, 1e-15, 1.0, 0.0, 1.0, 1e-9, 1e-6] if self.Diode2 else [1e-15, 1e-15, 1.0, 1e-9, 1e-6],
                        [10.0, 1e-3, 10.0, 1e-3, 10.0, 1e6, 1e9] if self.Diode2 else [10.0, 1e-3, 10.0, 1e6, 1e9]),
                        p0=np.array([self.Isc, self.Is1, self.n1, self.Is2, self.n2, self.Rs, self.Rp] if self.Diode2 else [self.Isc, self.Is1, self.n1, self.Rs, self.Rp]))
                # end if
                self.Isc        = popt[0]
                self.Is1        = popt[1]
                self.n1         = popt[2]
                if self.Diode2:
                    self.Is2    = popt[3]
                    self.n2     = popt[4]
                    self.Rs     = popt[5]
                    self.Rp     = popt[6]
                else:
                    self.Rs     = popt[3]
                    self.Rp     = popt[4]
                # end if

                # calculate the current-voltage characteristic
                bRet = self.calculateCharacteristic()
                if not bRet:
                    if self.verbose:
                        print("\ndone.")
                    # end if
                    return False
                # endif

                if TkFound:
                    self.datax[0] = self.VoltageY
                    self.datay[0] = self.CurrentY
                # endif

            except Exception as excT:
                self.reportMessage = '! fitting error: ' + str(excT)
                if self.verbose:
                    print("\nfitting error: " + str(excT))
                # end if
                bRet = False
                pass
            # end try

            self.tic = float(time.time() - ticT)

            if self.verbose:
                print("\ndone. elapsed time = %.6f sec." % self.tic)
            # end if

            return bRet

        except Exception as excT:

            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            strErr  = "\n! cannot fit the current-voltage characteristic:\n  %s\n  in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            self.reportMessage = '! cannot fit the current-voltage characteristic'
            if self.verbose:
                print(strErr)
            # end if
            return False
            # never reached
            pass

        # end try

    # end run

    def setFocus(self):
        if (not TkFound) or (not self.root):
            return
        # end if
        self.root.attributes('-topmost', 1)
        self.root.attributes('-topmost', 0)
        self.root.after(10, lambda: self.root.focus_force())
    # end setFocus

    # plot the current-voltage characteristic
    def updatePlot(self):

        if (not TkFound):
            return
        # end if

        try:
 
            if not self.PlotInitialized:

                aLabel = ["Calculated", None]
                for idc in range(0, self.curvecount):
                    self.line[idc], = self.plot[0].plot(np.array([]), np.array([]), self.linestyle[idc], label=aLabel[idc], linewidth=self.linesize[idc], zorder=4)
                    self.line[idc].set_color(self.linecolor[idc])
                # end for

                self.scatter[0], = self.plot[0].plot(np.array([]),      np.array([]),   'go', zorder=4, label=None)
                self.scatter[0].set_markerfacecolor('g')
                self.scatter[0].set_markeredgecolor('g')
                self.scatter[0].set_markersize(7)

                self.line0a = self.plot[0].axhline(y=0,                 xmin=0, xmax=1, linewidth=2, color='k')
                self.line0b = self.plot[0].axvline(x=0,                 ymin=0, ymax=1, linewidth=2, color='k')

                for idp in range(0, self.plotcount):
                    self.plot[idp].get_xaxis().set_visible(True)
                    self.plot[idp].get_yaxis().set_visible(True)
                # end for

                self.plot[0].legend(numpoints=1, fontsize='small', loc='upper center')

                afont = FontProperties()
                tfont = afont.copy()
                tfont.set_style('normal')
                tfont.set_weight('bold')
                tfont.set_size('small')
                self.report = self.plot[0].text(0.5, 1.05, ' ', color='red', horizontalalignment='center', verticalalignment='center', fontproperties=tfont, transform = self.plot[0].transAxes)

                self.PlotInitialized = True

            # end if

            for idc in range(0, self.curvecount):
                self.line[idc].set_xdata(self.datax[idc] if (self.datax[idc] is not None) else np.array([]))
                self.line[idc].set_ydata(self.datay[idc] if (self.datay[idc] is not None) else np.array([]))
            # end for

            self.scatter[0].set_xdata(self.VmY)
            self.scatter[0].set_ydata(self.ImY)

            for idp in range(0, self.plotcount):
                self.plot[idp].relim()
                self.plot[idp].autoscale()
            # end for

            if (self.reportMessage is not None):
                self.report.set_color('red' if self.reportMessage.startswith('!') else 'green')
                self.report.set_text(self.reportMessage)
            # end if

            self.setFocus()
            self.canvas.draw()

            if self.GUIstarted:
                self.IscEdit.delete(0, Tk.END)
                self.IscEdit.insert(0, "%.4g" % self.Isc)
                self.Is1Edit.delete(0, Tk.END)
                self.Is1Edit.insert(0, "%.4g" % self.Is1)
                self.n1Edit.delete (0, Tk.END)
                self.n1Edit.insert (0, "%.4f" % self.n1)
                if self.Diode2:
                    self.Is2Edit.delete(0, Tk.END)
                    self.Is2Edit.insert(0, "%.4g" % self.Is2)
                    self.n2Edit.delete (0, Tk.END)
                    self.n2Edit.insert (0, "%.4f" % self.n2)
                # end if
                self.RsEdit.delete (0, Tk.END)
                self.RsEdit.insert (0, "%.4g" % self.Rs)
                self.RpEdit.delete (0, Tk.END)
                self.RpEdit.insert (0, "%.4g" % self.Rp)
            # end if

            self.btnFit.configure(state="normal" if self.FileLoaded else "disabled")

        except Exception as excT:

            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            strErr  = "\n! cannot plot data:\n  %s\n  in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            if self.verbose:
                print(strErr)
            # end if
            pass

        # end try

    # end updatePlot

    def onFloatValidate(self, sp):
        try:
            if (not sp):
                return True
            # end if
            if (len(sp) <= 12):
                try:
                    spr = sp + '0'
                    float(spr)
                except ValueError:
                    return False
                # end try
                self.TargetEdit.prev = sp
                return True
            # end if
            return False
        except:
            return True
        # end try
    # end onFloatValidate

    def onInputFilenameValidate(self, sp):
        try:
            if (not sp) or (len(sp) <= 255):
                self.dataFilenameEdit.prev = sp
                return True
            # end if
            return False
        except:
            return True
        # end try
    # end onInputFilenameValidate

    def onBrowse(self):
        if self.isRunning():
            return
        # end if

        fileopt = {}
        fileopt['defaultextension'] = 'txt'
        fileopt['filetypes'] = [('Current-Voltage Characteristic File', '*.txt')]
        fileopt['initialfile'] = self.InputFilename
        fileopt['parent'] = self.root
        fileopt['title'] = 'Open the Current-Voltage Characteristic File'
        inputFilename = tkFileDialog.askopenfilename(**fileopt)
        if inputFilename:
            try:
                self.InputFilenameEdit.delete(0, Tk.END)
                self.InputFilenameEdit.insert(0, inputFilename)
                self.ModifTime  = self.getModifTime(inputFilename)
                self.FileLoaded = False
                self.loadFile()
                self.updatePlot()
            except:
                pass
            # end try
        # end if
    # end onBrowse

    def onEnter(self, tEvent):
        return self.start(Fit = False)
    # end onEnter

    def onDiode2Opt(self):
        try:
            self.Diode2 = self.Diode2Var.get()
            self.Is2Edit.configure(state="normal" if self.Diode2 else "disabled")
            self.n2Edit.configure (state="normal" if self.Diode2 else "disabled")
        except:
            pass
    # end onDiode2Opt

    def onStart(self):
        return self.start(Fit = False)
    # end onStart

    def onFit(self):
        return self.start(Fit = True)
    # end onFit

    def doSave(self, strFilename, savePDF = False):

        if (not strFilename) or self.isRunning():
            return
        # end if

        try:

            if savePDF and self.GUIstarted:
                # save figure in PDF format
                pdfT = PdfPages(strFilename)
                pdfT.savefig(self.figure)
                pdfT.close()
                # and in PNG format
                strPNG = os.path.splitext(strFilename)[0]
                strPNG = strPNG + '.png'
                pl.savefig(strPNG, dpi=600)
            # end if

            # save output data in text format
            strF = os.path.splitext(strFilename)[0]
            fileIV = strF + '_IV.txt'          # current-voltage characteristic
            np.savetxt(fileIV, np.c_[self.VoltageY, self.CurrentY],
                fmt='%.6f\t%.8g', delimiter=self.DataDelimiter, newline='\n',
                header=("Current-voltage characteristic for:\n  ISC = %.4g A ; VOC = %.4g V ; FF = %.4g %% ; \n  Is1 = %.4g A ; n1 = %.4g ; Is2 = %.4g A ; n2 = %.4g ; \n  RS = %.4g Ohms ; RP = %.4g Ohms\n" % (self.ISCY, self.VOCY, 100.0 * self.FFY, self.Is1, self.n1, self.Is2, self.n2, self.Rs, self.Rp))
                )

        except Exception as excT:

            strErr = "\n! cannot save output data:\n  %s\n" % str(excT)
            self.reportMessage = '! cannot save output data'
            if self.verbose:
                print(strErr)
            # end if
            return False
            # never reached
            pass

        # end try

    # end doSave

    def onSave(self):
        if (not self.GUIstarted) or self.isRunning():
            return
        # end if
        fileopt = {}
        fileopt['defaultextension'] = '.pdf'
        fileopt['filetypes'] = [('PDF files', '.pdf')]
        fileopt['initialfile'] = self.OutputFilename
        fileopt['parent'] = self.root
        fileopt['title'] = 'Save figure'
        pdfFilename = tkFileDialog.asksaveasfilename(**fileopt)
        if pdfFilename:
            self.OutputFilename = pdfFilename
            self.doSave(self.OutputFilename, savePDF = True)
        # end if
    # end onSave

    def onAutoScale(self):
        if (not self.GUIstarted) or self.isRunning():
            return
        # end if
        for idp in range(0, self.plotcount):
            self.plot[idp].relim()
            self.plot[idp].autoscale()
        # end for
        self.canvas.draw()
    # end onAutoScale

    def onEntryUndo(self, event):
        if not self.GUIstarted:
            return
        # end if
        try:
            if event.widget.prev is not None:
                event.widget.next = event.widget.get()
                strT = event.widget.prev
                idx = event.widget.index(Tk.INSERT)
                event.widget.delete(0, Tk.END)
                event.widget.insert(0, strT)
                event.widget.prev = strT
                event.widget.icursor(idx + 1)
        except:
            pass
        # end try
    # end onEntryUndo

    def onEntryRedo(self, event):
        if not self.GUIstarted:
            return
        # end if
        try:
            if event.widget.next is not None:
                idx = event.widget.index(Tk.INSERT)
                strT = event.widget.prev
                event.widget.delete(0, Tk.END)
                event.widget.insert(0, event.widget.next)
                event.widget.prev = strT
                event.widget.icursor(idx + 1)
        except:
            pass
        # end try
    # end onEntryRedo

    def onEntrySelectAll(self, event):
        if not self.GUIstarted:
            return
        # end if
        try:
            event.widget.select_range(0, Tk.END)
        except:
            pass
        # end try
    # end onEntrySelectAll

    def onAbout(self):
        if not self.GUIstarted:
            return
        # end if
        tkMessageBox.showinfo(self.name,
                             (self.name                                                         +
                              "\n"                                                              +
                              self.__version__                                                  +
                              "\nCopyright(C) 2018-2019 Pr. Sidi OULD SAAD HAMADY \n"           +
                              "Université de Lorraine, France \n"                               +
                              "sidi.hamady@univ-lorraine.fr \n"                                 +
                              "https://github.com/sidihamady/Photovoltaic-Model \n"             +
                              "http://www.hamady.org/photovoltaics/PhotovoltaicModel.zip \n"    +
                              "Under MIT license \nSee Copyright Notice in COPYRIGHT"),
                              parent=self.root)
    # end onAbout

    def onPopmenu(self, event):
        if (not TkFound):
            return
        # end if
        try:
            self.popmenu.entryconfig("Fit", state="normal" if self.FileLoaded else "disabled")
            self.popmenu.post(event.x_root, event.y_root)
        except:
            pass
        # end try
    # end onPopmenu

    def onClose(self):
        if (not self.GUIstarted) or (self.root is None):
            return
        # end if
        try:
            if self.isRunning():
                tkMessageBox.showinfo(self.name, "Please wait until calculations done.", parent=self.root)
                return
            # end if
            if not tkMessageBox.askyesno(self.name, "Close " + self.name + "?", default=tkMessageBox.NO, parent=self.root):
                return
            # end if
            self.root.quit()
            self.root.destroy()
            self.root = None
        except:
            pass
        # end try
    # end onClose

# end PhotovoltaicModelCore class
