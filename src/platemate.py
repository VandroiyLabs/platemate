## Standard libraries
import copy
import glob
#import chardet # really necessary?
import io

## Key libraries
import pylab as pl
import numpy as np
import pandas as pd
import openpyxl as xl
import string
import itertools

## Statistical libraries
import scipy.stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd




# PlateMate Internal libraries
from extrafunctions import *
import plot
reload(plot)




class PlateMate:
    """
    missing doc
    """

    def __init__(self, colonyMap = {}, controlMap = {}):
        """
        missing doc
        """

        self.colonyMap = colonyMap
        self.controlMap = controlMap
        self.colonyNames  = colonyMap.keys()
        self.controlNames = controlMap.keys()

        # putting everything together
        self.map = self.colonyMap.copy()
        self.map.update( controlMap )

        # getting class for each
        self.colClasses = { k : "colony" for k, v in self.colonyMap.items()}
        self.colClasses.update( { k : "control" for k, v in self.controlMap.items()} )


        # associating a color to each colony
        self.colors = {}
        pointers = { "control" : 0 , "colony" : 0 }
        for cname in self.colonyNames:
            cclass = self.colClasses[cname]
            self.colors[cname] = plot.pallet[cclass][ pointers[cclass] ]
            pointers[cclass] += 1


        self.setupVariables()


        return




    def setupVariables(self):
        """
        missing doc
        """

        ## Plotting variables

        # Color of lines used when connecting points
        self.plot_connline_color = (0.2,0.2,0.2)
        # Color of the edge of markers
        self.plot_markeredge_color = (0.2,0.2,0.2)

        return




    ##
    ## API
    ## Interface that connects the users with the low-level
    ## dataframes.
    ##

    def getColonyNames(self):
        """ Get the names of all colonies considered """
        return self.colonyNames

    def getControlNames(self):
        """ Get the names of all control colonies considered """
        return self.controlNames

    def summary(self, pop = "", nrows = 3):
        """
        missing doc
        """

        if pop == "" : pop = self.ColumnNames['A']
        return self.fldata[self.allCols(pop)].head(nrows)


    def getFluorescence(self, pop):
        """
        missing doc
        """
        return self.fldata[self.allCols(pop)]

    def getOpticalDensity(self, pop):
        """
        missing doc
        """
        return self.oddata[self.allCols(pop)]

    ## temperature

    def getTemperature(self):
        """
        missing doc
        """
        return self.fldata["T(oC)"]




    ##
    ## Plotting
    ##

    def plotTemperature(self):
        """
        missing doc
        """

        pl.figure(figsize=(4,3))

        plot.simplePlot( self.getTemperature() )

        # setting labels and axes
        pl.xlabel('Time (h)')
        pl.xlim(-0.5,8.5)
        pl.ylabel(r'Temperature ($^o$C)')
        pl.ylim(22.6,26.4)

        pl.tight_layout()

        return


    def plotIt(self, listPops, colors = [], ylabel = "Fluorescence (a.u.)"):
        """
        missing doc
        """


        if type(listPops) != type([]): listPops = [listPops]

        maxf = 0.
        minf = 1.e10

        # iterating colors
        for pop in listPops:
            F = self.getFluorescence(pop)

            plot.simplePlot( F, fillcolor=self.colors[pop] )

            if maxf < F.max().max() : maxf = F.max().max()
            if minf > F.min().min() : minf = F.min().min()


        # setting labels and axes
        pl.xlabel('Time (h)')
        pl.ylabel(ylabel)
        pl.ylim(0.3*minf, 1.2*maxf)

        #pl.tight_layout()

        return



    def plotMean(self, listPops, colors = [], ylabel = "Fluorescence (a.u.)"):
        """
        missing doc
        """

        if type(listPops) != type([]): listPops = [listPops]

        maxf = 0.
        minf = 1.e10

        for pop in listPops:
            F = np.array( self.getFluorescence(pop).mean(axis=1) )
            plot.simplePlot( F, fillcolor=self.colors[pop] )

            if maxf < F.max().max() : maxf = F.max().max()
            if minf > F.min().min() : minf = F.min().min()


        # setting labels and axes
        pl.xlabel('Time (h)')
        pl.ylabel(ylabel)
        pl.ylim(0.3*minf, 1.2*maxf)

        #pl.tight_layout()

        return


    def plotFuzzyMean(self, listPops, colors = [], ylabel = "Fluorescence (a.u.)",
                      fill_alpha = 0.6, lw = 2.0, markersize = 12):
        """
        missing doc
        """

        if type(listPops) != type([]): listPops = [listPops]

        maxf = 0.
        minf = 1.e10

        for pop in listPops:
            F  = np.array( self.getFluorescence(pop).mean(axis=1) )
            dF = np.array( self.getFluorescence(pop).std(axis=1) )

            pl.plot(F, "-o", linewidth=lw, markersize=markersize,
                    color=self.plot_connline_color,
                    markerfacecolor=self.colors[pop],
                    markeredgecolor=self.plot_markeredge_color )

            x = np.arange(0, F.shape[0],1 )# needs fixing!
            pl.fill_between(x, F - dF, F + dF, alpha = fill_alpha,
                            edgecolor='none', facecolor=self.colors[pop])

            if maxf < F.max().max() : maxf = F.max().max()
            if minf > F.min().min() : minf = F.min().min()


        # setting labels and axes
        pl.xlabel('Time (h)')
        pl.ylabel(ylabel)
        pl.ylim(0.3*minf, 1.2*maxf)

        #pl.tight_layout()

        return


    def plotBars(self, listPops, time, binwidth = 0.15):

        error_config = {'ecolor': '0.', 'width': 10.0, 'linewidth' : 2.}

        if type(listPops) != type([]): listPops = [listPops]

        # estimating the upper boung for plotting
        maxf = 0.

        # Positioning each colony
        colonies = np.arange(1., 4., 1)

        linen = 1

        for pop in listPops:
            F = self.getFluorescence(pop).iloc[time]
            maxf = max( maxf, F.max() )

            vals, stds = biolrepl( F )
            pl.bar(colonies + linen*0.2 - 0.5, vals, binwidth,
                   color = self.colors[pop],
                   label = listPops[linen-1],
                   yerr = stds, error_kw=error_config)

            linen += 1

        pl.xlabel("Biological replicate")
        pl.xticks( np.array(colonies, dtype=int) )
        pl.xlim(0.5, colonies.shape[0]+0.5)

        pl.ylabel("Fluorescence (a.u.)")
        pl.ylim(0., maxf*1.2)

        pl.legend( bbox_to_anchor=(1.32, 0.7), numpoints = 1 )

        return


    ##
    ## Statistical analyses
    ##

    def compareFluorescence(self, Pop1, Pop2):
        """
        It would be better to change it to a Kruskal-Wallis + Dunn
        """

        #if type(listPops) != type([]): listPops = [listPops]

        P1 = np.array( self.getFluorescence(Pop1) )
        data1 = np.reshape( P1, (P1.shape[0]*P1.shape[1]) )

        P2 = np.array( self.getFluorescence(Pop2) )
        data2 = np.reshape( P2, (P2.shape[0]*P2.shape[1]) )

        U, p = scipy.stats.mannwhitneyu(data1, data2)

        return U, p



    def ANOVA(self, listPops):
        """
        Anova
        """

        if type(listPops) != type([]): listPops = [listPops]

        # Gathering all data required.
        Paux = {}
        for pop in listPops:
            p = np.array( self.getFluorescence(pop) )
            data1 = np.reshape( p, (p.shape[0]*p.shape[1]) )
            Paux[pop] = data1

        # Evaluating anova
        U, p = scipy.stats.f_oneway( *Paux.values() )

        return U, p

    def TukeyHSD(self, listPops, confidence = 0.95):
        """
        Post-hoc Tukey HSD test.
        """

        if type(listPops) != type([]): listPops = [listPops]

        ## Gathering all data required.
        Paux = np.zeros((0,2))
        popid = 0
        for pop in listPops:
            p = np.array( self.getFluorescence(pop) )
            data1 = np.zeros( (p.shape[0]*p.shape[1],2) )
            data1[:,0] = np.reshape( p, (p.shape[0]*p.shape[1]) )
            data1[:,1] = popid
            popid += 1

            Paux = np.concatenate([Paux, data1], axis=0)

        ## Evaluating Tukey's HSD
        tukey_res = pairwise_tukeyhsd(Paux[:,0], Paux[:,1], alpha= 1 - confidence)

        ## Creating a dataframe with the results
        Groups = np.array( list( itertools.combinations(listPops, 2) ) )
        data = {'Group 1' : Groups[:,0], 'Group 2': Groups[:,1],
                'Mean diff': tukey_res.meandiffs, 'Reject H0?' : tukey_res.reject}
        df = pd.DataFrame(data=data)

        return df


    ##
    ## Data filtering
    ##

    def allCols(self, labels, r0 = 1, rf = 9):
        """
        missing doc
        """

        if type(labels) != list:
            labels = [labels]
        cols = []
        for label in labels:
            row  = self.MeaningColNames[label]
            for j in range(r0,rf+1):
                cols.append(row + '0' + str(j))

        return cols


    def normalizeByOD(self, Readings, OD):
        """
        missing doc
        """

        NormReadings = copy.deepcopy( Readings )

        for Reading in NormReadings:
            Reading = np.divide(Reading, OD)

        return NormReadings



    def AbsorbLineReading(self, Times, Data, lines = np.arange(8)):
        """
        missing doc
        """

        ntimes = Times.shape[0]
        m = np.zeros( (lines.shape[0], ntimes) )

        for time in range(ntimes):
            for line in lines:
                m[line,time] = Data[time][line].mean()

        return m



    ##
    ## Importing and parsing data
    ##

    def findFiles(self, pattern, path = "", extension=".txt"):
        """
        missing doc
        """

        # Creating a list for each of the file types
        FLlist = []
        ODlist = []
        tidx   = []

        # Looping through all files in the pattern
        for file in glob.glob(path + pattern + "*" + extension):
            tidx.append( float( file.split(' ')[-1].split('.')[0] ) )
            FLlist.append(file)
            ODlist.append('OD'+file.split(' ')[1].split('.')[0]+'.txt')


        # Sorting according to the indices
        Idx = np.argsort(tidx)
        self.FLlist = np.array(FLlist)[Idx]
        self.ODlist = np.array(ODlist)[Idx]

        return



    def readFluorescence(self):
        self.fldata = self.read_timeformat(self.FLlist)
        return

    def readOpticalDensity(self):
        self.oddata = self.read_timeformat(self.ODlist, datarow=1)
        return

    def read_timeformat(self, ListOfFiles, nr_header = 2, sep = '\t',
                                  datarow = 1):
        """Importing a set of data, interpreted as a time series. The time is
        supposed to be encoded in the filename. GNANO fluorimeter has
        several export formats. This function reads data from 'time
        format'.

        Data is assumed to be separated by sep character (standard \t).

        list ListOfFiles :: a list with all filenames
        int nr_header    :: number of lines of the header

        """


        ## Getting the headers
        f = open(ListOfFiles[0], 'r')

        for j in range(nr_header): f.readline()

        line = f.readline()
        f.close()
        HEADERS = line.split(sep)[1:-1]
        # To ensure alphabetical order during sorting...
        for j in range(len(HEADERS)):
            if (HEADERS[j][:4] == 'Temp'): HEADERS[j] = 'T(oC)'
            if ( len(HEADERS[j]) < 3 ): HEADERS[j] = HEADERS[j][0] + "0" + HEADERS[j][1]

        TSeries = {}
        for header in HEADERS:
            TSeries[header] = []

        # Getting the data
        for filename in ListOfFiles[1:]:

            f = open(filename, 'r')
            for j in range(nr_header+1):
                f.readline()

            Readings = []
            Times = []

            row = 0
            while True:
                line = f.readline()

                if not line or line == '\r\n' :
                    if row >= datarow: break
                else:
                    row += 1

                if row == datarow:

                    line_usfl = line.split(sep)[1:-1]

                    cnt = 0
                    for item in line_usfl:
                        if ( item == '' ):
                            TSeries[HEADERS[cnt]].append( '' )
                        else:
                            TSeries[HEADERS[cnt]].append( float(item) )
                        cnt += 1

            # Closing the file
            f.close()


        # Turning it into a Pandas DataFrame object
        dictdata = {}
        for header in HEADERS:
            dictdata[header] = TSeries[header]

        TimeReadings = pd.DataFrame(data = dictdata)

        return TimeReadings




    def readfromSpreadSheet(self, InitialRow, ncols = 12, nrows = 8):
        """MUST WRITE...
        """

        LTS = list(string.ascii_uppercase)[:20]

        auxV = np.zeros( (0,nrows*ncols) )
        for file in self.FLlist:

            spsheet = xl.load_workbook(file)['final']

            read = np.zeros( (nrows, ncols) )

            for row in range(nrows):
                line = row + InitialRow
                for col in range(ncols):
                    read[row,col] = float( spsheet[LTS[col+1]+str(line)].value )


            auxV = np.concatenate(
                        (auxV, np.reshape(read, (1,read.shape[0]*read.shape[1]) ) ),
                        axis = 0 )


            columns = []
            for row in range(nrows):
                for col in range(ncols):
                    columns.append( LTS[row] + str(col).zfill(2) )
            TimeReadings = pd.DataFrame(data=auxV, columns=columns)


        return TimeReadings







    def filterTime(self, time):
        """
        missing doc
        """

        tsplit = time.split(':')
        coef = {3 : 3600., 2 : 60., 1 : 1.}
        nterms = len(tsplit)

        ftime = 0.0
        for item in tsplit:
            ftime += coef[nterms]*float(item)
            nterms -= 1

        return ftime









def biolrepl(wells):
    """
    missing doc
    """
    return np.array( [ wells[0:3].mean(), wells[3:6].mean(), wells[6:9].mean() ] ), \
        np.array( [ np.std(wells[0:3]), np.std(wells[3:6]), np.std(wells[6:9]) ] )
