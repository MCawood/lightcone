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
import pyfits

cmdargs = str(sys.argv)

HIfreq = 1420.40575177

omega_matter = 0.3
omega_lamda = 0.7

start = 0
stop = sys.argv[1]
zBins = 0
pixels = 0
elmsPerPix = 14
offset = 0
doDM= False
diag = False
zMax = 0
rc = False
labels = ['z', 'deltaZ', 'ra', 'dec', 'gasParts', 'ionisedMass', 'hiMass', 'h2Mass', 'hiFlux', 'sfr', 'radioCont', 'perculiarV', 'starParts', 'starMass', 'dmParts', 'dmMass']


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
		elmsPerPix=16
        if(line[0] == "DIAG"):
            if(line[1].rstrip() == "YES"):
                diag = True
        if(line[0] == "RC"):
            if(line[1].rstrip() == "YES"):
                rc = True



dataCube = 0

def calcRadius():

    inP = np.load('dataCube_z.npz', 'r')['dataSet']
    tally = np.sum((inP > 0), axis=(1,2))
    mask = tally > 0
    z=np.zeros(tally.size)
    z[mask] = np.sum(inP, axis=(1,2))[mask]/tally[mask]

    points = [line.split(" ") for line in open("zr.txt")]
    zrFunc = UnivariateSpline(map(float, zip(*points)[2]), map(float, zip(*points)[1]), s=0)

    rad = np.tan(np.radians((z+1)/np.sqrt(np.pi)))*zrFunc(z)
    mask = rad < 0.001
    rad[mask] = 0.

    out = open('radii.txt', 'w')

    for i in range(len(rad)):
        out.write(str(rad[i]) + "\n")

    out.close()

def calcArea(z):
    return np.square(calcRadius(z))*np.pi

def calcRedshift():

    z = np.load('dataCube_z.npz')['dataSet']

    div = np.sum(z>0.01, axis=(1,2))
    tot = np.sum(z, axis=(1,2))

    sel = div > 0

    mean = np.zeros(z.shape[0])
    mean[sel] = tot[sel]/div[sel]

    out = open('coneZ.txt', 'w')
    bins = mean
    for i in range(len(bins)):
        out.write(str(bins[i]) + "\n")
    out.close()

def calcRC():

    global dataCube

    sfr = np.load('dataCube_z.npz')['dataSet']

    maxZ = 0.58
    freqMax = HIfreq
    freqMin = freqMax/(1+maxZ)
    freqRange = freqMax - freqMin
    channelWidth = freqRange/zBins

    freqs = freqMin + channelWidth*(np.arange(zBins))
    for i in range(zBins):

	doMask = sfr[i,:,:] > 0.

    	L = np.zeros((pixels,pixels))
    	L[doMask] = (sfr[i,doMask]/5.9e-22)*channelWidth*1.e3

    	nF = freqs[i]/freqs
    	dataCube += L * np.power(nF, -0.7)[:,None,None]

#func to read boxes from file
def addBox(prop, cube):

    global dataCube
    global offset 
    global zBins

    dir = path.dirname(__file__)
    filename = path.join(dir, 'data/cubeID_'+str(cube)+'.npz')

    print "reading", labels[prop], "from",  filename, "..." 


    t = np.load(filename)[labels[prop]]

    cubeDepth = t.shape[0]

    
    upperBound = min((offset[cube]+cubeDepth), zBins)

    boxLimit = upperBound - offset[cube]

    #non cumulative data, redshift etc

    if prop in (0, 1, 2, 3, 11):

        find = np.where(t[0:boxLimit,:,:] != 0 )
  
        #print dataCube[toPos,find[1],find[2],find[3]].shape

        dataCube[find[0]+offset[cube],find[1],find[2]] = t[find[0],find[1],find[2]]

    
    else:

        #cumulative data
        dataCube[offset[cube]:upperBound,:,:] += t[0:boxLimit,:,:]

    sys.stdout.flush()

if (stop == "old"):

    print "Using existing dataCube files"
    #read precombined box
    sys.stdout.flush()
    #dataCube = np.load('dataCube.npy')

else:

    print "Creating new data cube... \n"
    stop = int(stop)
    sys.stdout.flush()   

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
    for i in range(elmsPerPix):
 	dataCube = np.zeros((zBins, pixels, pixels), dtype=np.float32)
	print "Stitching ", labels[i], "..."
	for j in range(start, stop):
	    addBox(i, j)

	#if processing RC
	if i == 10:
	    #if RC flag
	    if rc:
	        print "Begining radio continuum caclucation..."
	        calcRC()
        	print "Complete."

	print "Saving..."
	
	print str(labels[i]), np.shape(dataCube), np.sum(dataCube), np.count_nonzero(dataCube)
	np.savez_compressed(('dataCube_'+str(labels[i])), dataSet=dataCube[0:zBins,:,:])
	dataCube = 0
	print "Done. \n"

    print "Saving new snapshot to dataCube.npy"
    np.save('dataCube', dataCube)


def zToFreq (z):    
	return HIfreq/(1.+z)


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
        cut3 = np.sum(ion[slice-thickness:slice+thickness,:,:], axis=0)
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
        np.save('ion_'+str(i), data3)
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
    cut3 = np.sum(h2mass[slice-thickness:slice+thickness,:,:], axis=0)
    cut4 = np.sum(ion[slice-thickness:slice+thickness,:,:], axis=0)
    cut5 = np.sum(starMass[slice-thickness:slice+thickness,:,:], axis=0)



    #coords =
    coords1 = np.where(cut1 > 0)
    coords2 = np.where(cut2 > 0)
    coords3 = np.where(cut3 > 0)
    coords4 = np.where(cut4 > 0)
    coords5 = np.where(cut5 > 0)


    print "cords", len(coords1[1]), len(coords2[1])

    data1 = np.array([coords1[0], coords1[1], cut1[coords1]])
    data2 = np.array([coords2[0], coords2[1], cut2[coords2]])
    data3 = np.array([coords3[0], coords3[1], cut3[coords3]])
    data4 = np.array([coords4[0], coords4[1], cut4[coords4]])
    data5 = np.array([coords5[0], coords5[1], cut5[coords5]])



    print len(data1[0])

    np.save('dm', data1)
    np.save('h1', data2)
    np.save('h2', data3)
    np.save('ion', data4)
    np.save('star', data5)



    np.save('flux', flux)
   
#crossSection()

def crossSection2():

    pixPerSlice = np.sum(p, axis=(1,2))

    slice = np.argmax(pixPerSlice)
    slice = 915
    print "busiest freq bin:", slice

    min = 915
    max = 930

    np.save('dm', dmMass[min:max,:,:])
    np.save('h1', h1mass[min:max,:,:])
    np.save('h2', h2mass[min:max,:,:])
    np.save('ion', ion[min:max,:,:])
    np.save('star', starMass[min:max,:,:])

    np.save('ra', ra[min:max,:,:])
    np.save('dec', dec[min:max,:,:])

    print np.min(ra[slice,:,:])
    print np.min(ra)


#crossSection2()

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


print "Post processor complete."
