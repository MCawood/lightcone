#AUTHOR: Matthew Cawood
#PURPOSE: Creating an HI lightcone from stacked GADGET snapshots
#DATE: May 2014
#Written and tested using python-2.7.6, numpy-1.8.1 and scipy-0.14.0

import numpy as np
import sys
from scipy.interpolate import UnivariateSpline
import re
import time 
from os import path
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib as mpl
#import pyfits

cmdargs = str(sys.argv)

HIfreq = 1420.40575177

omega_matter = 0.3
omega_lamda = 0.7

start = 0
stop = sys.argv[1]
zBins = 0
pixels = 0
elmsPerPix = 11
offset = 0
binsProcessed=0
doDM= False
diag = False
zMax = 0

print "\n \n \n"
print "Lightcone postprocessor"
print "-----------------------"
print "\n"

points = [line.split(" ") for line in open("zr.txt")]
zrFunc = UnivariateSpline(map(float, zip(*points)[2]), map(float, zip(*points)[1]), s=0)

print "Reading parameters"
with open("params.txt") as f:
    content = f.readlines()
    for i in range(len(content)):
        line = content[i].split("=")

        if(line[0] == "FREQBINS"):
            zBins = int(line[1])
	if(line[0] == "PIXELS"):
            pixels = int(line[1])
        if(line[0] == "ZMAX"):
            zMax = float(line[1])

	if(line[0] == "DM"):
            if(line[1].rstrip() == "YES"):
                doDM = True
		elmsPerPix=13
        if(line[0] == "DIAG"):
            if(line[1].rstrip() == "YES"):
                diag = True

dataCube = 0

def calcRadius(z):
    rad = np.tan(np.radians((z+1)/np.sqrt(np.pi)))*zrFunc(z)
    mask = rad < 0.001
    rad[mask] = 0.
    return rad

def calcArea(z):
    return np.square(calcRadius(z))*np.pi

def calcRedshift(p,z):

    sum = np.sum(z, axis=(1,2))
    r = p > 0
    s = np.sum(r, axis=(1,2))
    mask = s > 0

    zs = np.zeros(s.shape)

    zs[mask] = sum[mask]/s[mask]
    out = open('coneZ.txt', 'w')
    bins = zs
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

def calcRC(cube):

    maxZ = 0.58
    freqMax = HIfreq
    freqMin = freqMax/(1+maxZ)
    freqRange = freqMax - freqMin
    channelWidth = freqRange/zBins

    out = np.empty_like(cube)

    freqs = freqMin + channelWidth*(np.arange(zBins))
    for i in range(zBins):

	doMask = cube[i,:,:] > 0.

    	L = np.zeros((pixels,pixels))
    	L[doMask] = (cube[i,doMask]/5.9e-22)*channelWidth*1.e3

    	nF = freqs[i]/freqs
    	out += L * np.power(nF, -0.7)[:,None,None]

    return out
    
#func to read boxes from file
def addBox(i):

    global dataCube
    global offset 
    global binsProcessed
    global zBins

    dir = path.dirname(__file__)
    filename = path.join(dir, 'data/cubeID_'+str(i)+'_data.npy')

    t = np.load(filename)
    cubeDepth = t.shape[0]
    upperBound = min((offset[i]+cubeDepth), zBins)

    boxLimit = upperBound - offset[i]

    #non cumulative data, redshift etc
    dataCube[offset[i]:upperBound,:,:,[1,2]] = np.maximum(t[0:boxLimit,:,:,[1,2]], dataCube[offset[i]:upperBound,:,:,[1,2]])
    
    #cumulative data
    if not doDM:
        dataCube[offset[i]:upperBound,:,:,[0,3,4,5,6,7,8,9,10]] += t[0:boxLimit,:,:,[0,3,4,5,6,7,8,9,10]]
        print "Snapshot:", i, ", gas particles:", int(np.sum(t[0:boxLimit,:,:,0])), ",star particles:", int(np.sum(t[0:boxLimit,:,:,8]))

    else:
	dataCube[offset[i]:upperBound,:,:,[0,3,4,5,6,7,8,9,10,11,12]] += t[0:boxLimit,:,:,[0,3,4,5,6,7,8,9,10,11,12]]
	print "Snapshot:", i, ", gas particles:", int(np.sum(t[0:boxLimit,:,:,0])), ",star particles:", int(np.sum(t[0:boxLimit,:,:,8])), ", DM particles:", int(np.sum(t[0:boxLimit,:,:,10]))

    binsProcessed += cubeDepth
    sys.stdout.flush()

if (stop == "old"):

    print "Using existing dataCube.npy file"
    #read precombined box
    sys.stdout.flush()
    dataCube = np.load('dataCube.npy')
    print "Loaded."

else:

    print "Creating new data cube... \n"
    stop = int(stop)
    sys.stdout.flush()   
    dataCube = np.zeros((zBins, pixels, pixels, elmsPerPix), dtype=np.float32)

    #get cube offsets
    offset = np.zeros(stop-start+1, dtype=np.int32)

    print "Reading snapshot frequency bin offsets"
    with open("cube_offsets.txt") as f:
        content = f.readlines()

	if (len(content) != (stop)):
	    print "ERROR: invalid number of snapshots selected"
            sys.exit()

        for i in range(len(content)):
            line = content[i].split("=")
	    box= int(re.findall('\d+', line[0])[0])
	    
  	    if(box < stop-start+1):
                offset[box] = int(line[1])

    #read boxes
    for i in range(start, stop):
	addBox(i)

    print "Saving new snapshot to dataCube.npy"
    np.save('dataCube', dataCube)

#take desired number of Z bins from raw cube
dataCube = dataCube[0:zBins,:,:,:]

shape = dataCube.shape

mask = np.zeros((shape[0],shape[1],shape[2]), dtype=bool)

p = dataCube[:,:,:,0]
z = dataCube[:,:,:,1]
zDelta = dataCube[:,:,:,2]
totGas = dataCube[:,:,:,3]
h1mass = dataCube[:,:,:,4] 
h2mass = dataCube[:,:,:,5]
flux = dataCube[:,:,:,6]
vel = dataCube[:,:,:,7]
sfr = dataCube[:,:,:,8]
starPart = dataCube[:,:,:,9]
starMass = dataCube[:,:,:,10]

dmPart =0
dmMass =0

if doDM:
    dmPart = dataCube[:,:,:,11]
    dmMass = dataCube[:,:,:,12]

tally = p > 0
m1 = np.sum(tally, axis=(1,2)) > 0
binZs = np.zeros(zBins)
binZs[m1] = np.nan_to_num(np.divide(np.sum(z, axis=(1,2))[m1],np.sum(tally, axis=(1,2))[m1]))
area = calcArea(binZs)
radii = calcRadius(binZs)

Sv = 0

#flatten redshift array to put through Spline function (no multiD support)
zShape = z.shape
distToBin = zrFunc(z.flatten()).reshape(zShape)

def zToFreq (z):    
	return HIfreq/(1.+z)


mass = 0.
thresh = np.max(h1mass)/10

histArr = np.zeros(7)

test =0
inp=0
outp=0



def threeD():

    print "3d data"

    todo = dmMass[:,:,:]
    coords = np.where(abs(todo) > 0)
    data = np.array([coords[0], coords[1], coords[2], todo[coords]])

    np.save('dmMass', data)

#threeD()

def flythrough():

    pixPerSlice = np.sum(p, axis=(1,2))

    slice = np.argmax(pixPerSlice)
    print "busiest freq bin:", slice

    i = 1000
    thickness = 1

    while i < 9500:#(4095-thickness):#


        slice = i

        print "Frequency bins in this slice:", i-thickness, i+thickness, (thickness*2+1)

        cut1 = np.sum(dmMass[slice-thickness:slice+thickness,:,:], axis=0)
        cut2 = np.sum(h1mass[slice-thickness:slice+thickness,:,:], axis=0)
        cut3 = np.sum(totGas[slice-thickness:slice+thickness,:,:], axis=0)
        cut4 = np.sum(starMass[slice-thickness:slice+thickness,:,:], axis=0)

        coords1 = np.where(cut1 > 0)
        coords2 = np.where(cut2 > 0)
        coords3 = np.where(cut3 > 0)
        coords4 = np.where(cut4 > 0)

        print "cords", len(coords1[1]), len(coords2[1])

        data1 = np.array([coords1[0], coords1[1], cut1[coords1]])
        data2 = np.array([coords2[0], coords2[1], cut2[coords2]])
        data3 = np.array([coords3[0], coords3[1], cut3[coords3]])
        data4 = np.array([coords4[0], coords4[1], cut4[coords4]])

        print len(data1[0])

        np.save('dm_'+str(i), data1)
        np.save('h1_'+str(i), data2)
        np.save('totGas_'+str(i), data3)
        np.save('star_'+str(i), data4)

        i += (1 + thickness*2)

#flythrough()


def crossSection():

    pixPerSlice = np.sum(p, axis=(1,2))

    slice = np.argmax(pixPerSlice)
    print "busiest freq bin:", slice

    thickness = 1

    cut1 = np.sum(dmMass[slice-thickness:slice+thickness,:,:], axis=0)
    cut2 = np.sum(h1mass[slice-thickness:slice+thickness,:,:], axis=0)
    cut3 = np.sum(totGas[slice-thickness:slice+thickness,:,:], axis=0)

    #coords =
    coords1 = np.where(cut1 > 0)
    coords2 = np.where(cut2 > 0)
    coords3 = np.where(cut3 > 0)

    print "cords", len(coords1[1]), len(coords2[1])

    data1 = np.array([coords1[0], coords1[1], cut1[coords1]])
    data2 = np.array([coords2[0], coords2[1], cut2[coords2]])
    data3 = np.array([coords3[0], coords3[1], cut3[coords3]])

    print len(data1[0])

    np.save('dm', data1)
    np.save('h1', data2)
    np.save('totGas', data3)
    np.save('flux', flux)

#crossSection()

if (diag):


    out = open('h1.txt', 'w')
    bins = np.sum(h1mass, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()
    out = open('particles.txt', 'w')
    bins = np.sum(p, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()


    out = open('velocity.txt', 'w')
    linePs = np.sum(p, axis=(1,2))
    mask = linePs > 0
    vel = np.sum(vel, axis=(1,2))
    vel1D = np.zeros(len(mask))

    vel1D[mask] = vel[mask]/linePs[mask]

    for i in range(len(vel1D)):
        out.write(str(vel1D[i]) + "\n")
    out.close()
    out = open('flux.txt', 'w')
    bins = np.sum(flux, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()
    out = open('dm.txt', 'w')
    bins = np.sum(dmMass, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

    out = open('dmCount.txt', 'w')
    bins = np.sum(dmPart, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()


    out = open('gascount.txt', 'w')
    bins = np.sum(p, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

    out = open('starCount.txt', 'w')
    bins = np.sum(starPart, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

    out = open('starMass.txt', 'w')
    bins = np.sum(starMass, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

    out = open('h1.txt', 'w')
    bins = np.sum(h1mass, axis=(1,2))
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

    out = open('area.txt', 'w')
    bins = area
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()
    out = open('radii.txt', 'w')
    bins = radii
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()



print "Particles tested", test
print "H1mass threshhold", thresh
print "Non-detections", outp, "Detections", inp

print "Total H1mass", np.sum(h1mass)
print "Maximum H1mass", np.max(h1mass)
print "Amount H1 found", mass
