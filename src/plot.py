import numpy as np
import pylab as pl

def simplePlot(timeseries, lw = 1.5, markersize = 11., fillcolor=(0.3,0.5,1.0),
                markeredgecolor=(0.2,0.2,0.2), format = "o", lformat = "-"):

    pl.plot(timeseries, format + lformat, lw=lw, markersize=markersize,
            color=fillcolor, markeredgecolor=markeredgecolor )

    return



##
## colors
##

pallet = {
    "colony" : [
        (0.4,0.6,1.0),
        (0.5,1.0,0.3),
        (1.0,0.5,0.3),
        (1.0,0.2,1.0),
        (0.0,0.2,0.7),
    ],
    "control" : [
        (0.3,0.3,0.3),
        (0.4,0.1,0.1),
        (0.7,0.7,0.7),
        (1.0,0.4,0.1),
        (0.8,0.8,0.1)
    ]
}
