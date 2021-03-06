import pylab as pl
from math import sqrt
import sys

# importing platemate
sys.path.insert(0, '../src')
import platemate



ColumnNames = {
    'C' : "Dev1",
    'D' : "Dev2",
    'E' : "Dev3"
    }

controlNames = {
    'A' : "LB",
    'B' : "LB+Cam",
    'F' : "+control",
    'G' : "-control1",
    'H' : "-control2"
    }


pm = platemate.PlateMate( colonyNames = ColumnNames, controlNames = controlNames )


pm.findFiles("medida")

print 'reading fluorescence...'
pm.readFluorescence()
print 'reading optical density...'
pm.readOpticalDensity()


pm.ANOVA(["Dev1","Dev2","Dev3","-control1"])
res = pm.TukeyHSD(["Dev1","Dev2","Dev3","-control1"])
print res
