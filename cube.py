#AUTHOR: Matthew Cawood
#PURPOSE: Creating an HI lightcone from stacked GADGET snapshots
#DATE: May 2014
#Written and tested using python-2.7.6, numpy-1.8.1 and scipy-0.14.0

import sys
import math
import time
import numpy as np
from scipy.interpolate import UnivariateSpline
from readgadget import *
from os import path

#local variables
inP=0
outP=0
binArray = 0
zMax = 0
zMin = 0
zBins = 0
time1 = 0.
time2 = 0.
time3 = 0.
time4 = 0.
neutral_mass=0
HIfreq = 1420.40575177
omega_matter = 0.3
omega_lamda = 0.7
T=0
rho=0
gasMass=0
dmMass=0
starMass=0
sfr=0
fneut=0
hsml=0
vel=0
scaleFactor=0
offSet=0
zUpperLimit=0
doDM=False
OneOverPi = 1/np.sqrt(np.pi)
coneVec = 0
binned =0
h1=0
h2=0

#HI SHIELDING VARS
MHYDR        = 1.673e-24
FSHIELD      = 0.9
P0BLITZ      = 1.7e4    ## Leroy et al 2008, Fig17 (THINGS) Table 6
ALPHA0BLITZ  = 0.8      ## Leroy et al 2008, Fig17 (THINGS) Table 6
XH           = 0.76
NINTERP      = 10000
NHILIM       = 1.73e18  ## Lyman Limit (adjustable)
M_PI = math.pi

KernIntTable = np.zeros((NINTERP+1,3))

#datacube
binArray = 0

#----------------MATH FUNCTIONS--------------------

#vector functions for NumPy arrays
def vec_length(vector):
    return np.linalg.norm(vector,axis=1)
def unit_vector(vector):
    len = vec_length(vector)
    vector = vector/len[:, None]
    return vector
def angle_between(v1, v2_u):
    v1_u = unit_vector(v1)
    angle = np.arccos(np.dot(v1_u, v2_u))
    angle[np.isnan(angle)] = 0.
    return angle

#vector functions for scalars
def vec_length_scalar(vector):
    return np.linalg.norm(vector)
def unit_vector_scalar(vector):
    len = vec_length_scalar(vector)
    if(len > 0):
        return vector / len
    return vector
def angle_between_scalar(v1, v2_u):
    v1_u = unit_vector_scalar(v1)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle

#function to find H as a function of z
def H(z):
    return 70.0*((omega_matter*(1.0+z)**3.0 + omega_lamda)**0.5)

#comoving distance as function of z
def D(z):
    zShape = z.shape
    rz = zrFunc(z.flatten()).reshape(zShape)
    return rz/(1+z)

#recess velocity
def recessV(z):
    return H(z)*D(z)

#calculate flux
def calcFlux(h1, Dl):

     L = 6.27e-9 * h1
     S = L / (4*np.pi*np.square(Dl))
     return S

#-------------------Particle Testing-------------------

#function to test if particle is in or outside of the cone
def test_particle(particle, starFlag, dmFlag):
    
    global time1, OneOverPi, coneVec, binned
    t = time.clock()
    if starFlag:
	print "Starting star particle testing..."
    elif dmFlag:
	print "Starting DM particle testing..."
    else:
        print "Starting gas particle testing..."

    #calculated comoving distance to each particle in snapshot
    length = np.linalg.norm(particle, axis=1)
    
    #redshift to particle - used to find cone openning angle -> coneAng = (1+z)
    z = zrFunc(length)   
        
    coneAng = np.radians((z+1)*OneOverPi)
    #angle from axis of cone to particle in radians
    partAng = angle_between(particle, coneVec)
    
    time1 = time1 + (time.clock() - t)
    
    #find particles in cone and bin them
    binned = abs(partAng) < coneAng

    numParts = np.sum(binned)

    if (numParts > 0):
	#if tested particles are stars
	if starFlag:
            print numParts, "star particles found in this tile \n"
            bin_Misc(particle[binned], length[binned], z[binned], binned, True)

	#if particles are dark matter
	elif dmFlag:
  	    print numParts, "DM particles found in this tile \n"
	    bin_Misc(particle[binned], length[binned], z[binned], binned, False)
    
	#if particles are gas
	elif ~starFlag and ~dmFlag:
            print numParts, "gas particles found in this tile \n"
            bin_Gas(particle[binned], length[binned], z[binned], binned)

	else:
	    print "No particles found in this tile"

#---------------------------------------------------------------------------------------
def bin_Gas(particle, length, z, binned):
    
    print "Begin gas binning..."

    global time2, binArray, cubeZBins, vel, scaleFactor, offSet, zUpperLimit, zBins, coneVec, h1, h2

    t = time.clock()
    #get Y and Z components of particle angle
    cone = length*coneVec[:, None]
    coneRadius =  np.tan(np.radians((z+1)/np.sqrt(np.pi)))*length
    
    #calculate flattened distance in perpendicular plane to axis
    particle = np.transpose(particle)

    yDist = particle[1]-cone[1]
    zDist = particle[2]-cone[2]

    #center [0,0] @ [pixels/2,pixels/2]
    yBin = ((yDist/coneRadius)*(pixels/2)).astype(int)+(pixels/2)
    zBin = ((zDist/coneRadius)*(pixels/2)).astype(int)+(pixels/2)

    xBin = ((z / zUpperLimit)*(zBins-1) - offSet).astype(int)

    #boundary case
    yBin[yBin > pixels-1] = pixels-1
    zBin[zBin > pixels-1] = pixels-1
    xBin[xBin > cubeZBins-1] = cubeZBins-1

    #1 is for z floor
    floor = zMin + (zMax-zMin)*((xBin.astype(float))/cubeZBins)
    binArray[xBin,yBin,zBin,1] = floor

    #2 is for z delta
    ceiling = zMin + (zMax-zMin)*((xBin+1.)/cubeZBins)
    delta = ceiling - floor    
    binArray[xBin,yBin,zBin,2] = delta

    #0 is for particle count
    #3 is for total gas mass
    #4 is for H1Mass
    #5 is for H2Mass
    #6 is for flux
    #7 is for perculiar V
    #8 is for SFR / radio continuum

    calcHI()

    ionized = gasMass[binned]-(h1+h2)

    h1 = h1*1.0e10
    h2 = h1*1.0e10
    ionized = ionized*1.0e10

    v = np.dot(vel[binned], coneVec)/np.sqrt(scaleFactor)

    #luminosity distance
    Dl = length*(1+z)

    flux = calcFlux(h1, Dl)

    #particle summing per bin
    XYZ = np.vstack((xBin,yBin,zBin)).T

    order = np.lexsort(XYZ.T)
    diff = np.diff(XYZ[order], axis=0)
    uniq_mask = np.append(True, (diff != 0).any(axis=1))

    uniq_inds = order[uniq_mask]
    inv_idx = np.zeros_like(order)
    inv_idx[order] = np.cumsum(uniq_mask) - 1

    #value summing
    
    incr = np.bincount(inv_idx)
    ionized = np.bincount(inv_idx, weights=ionized)
    h1 = np.bincount(inv_idx, weights=h1)
    h2 = np.bincount(inv_idx, weights=h2)
    flux = np.bincount(inv_idx, weights=flux)
    v = np.bincount(inv_idx, weights=v)
    starF = np.bincount(inv_idx, weights=sfr[binned])

    #unique co-ords for summed data
    xBin,yBin,zBin = XYZ[uniq_inds].T

    #assignments
    binArray[xBin,yBin,zBin,0] += incr

    binArray[xBin,yBin,zBin,3] += ionized
    binArray[xBin,yBin,zBin,4] += h1  
    binArray[xBin,yBin,zBin,5] += h2
    binArray[xBin,yBin,zBin,6] += flux

    binArray[xBin,yBin,zBin,7] += v    
    binArray[xBin,yBin,zBin,8] += starF

    time2 = time2 + (time.clock() - t)

#seperate function to add DM mass to lightcone if set in param.txt
def bin_Misc(particle, length, z, binned, star):

    print "Begin dark matter binning..."

    global time3, time4, binArray, cubeZBins, scaleFactor, offSet, zUpperLimit, zBins, starMass, dmMass

    t = time.clock()
    #get Y and Z components of particle angle
    cone = length*coneVec[:, None]
    coneRadius =  np.tan(np.radians((z+1)/np.sqrt(np.pi)))*length

    #calculate flattened distance in perpendicular plane to axis
    particle = np.transpose(particle)

    yDist = particle[1]-cone[1]
    zDist = particle[2]-cone[2]

    #center [0,0] @ [pixels/2,pixels/2]
    yBin = (np.around((yDist/coneRadius)*(pixels/2))).astype(int)+(pixels/2)
    zBin = (np.around((zDist/coneRadius)*(pixels/2))).astype(int)+(pixels/2)

    xBin = ((z / zUpperLimit)*(zBins-1) - offSet).astype(int)
    
    yBin[yBin > pixels-1] = pixels-1
    zBin[zBin > pixels-1] = pixels-1
    xBin[xBin > cubeZBins-1] = cubeZBins-1
    
    XYZ = np.vstack((xBin,yBin,zBin)).T

    order = np.lexsort(XYZ.T)
    diff = np.diff(XYZ[order], axis=0)
    uniq_mask = np.append(True, (diff != 0).any(axis=1))

    uniq_inds = order[uniq_mask]
    inv_idx = np.zeros_like(order)
    inv_idx[order] = np.cumsum(uniq_mask) - 1

    #value summing
    incr = np.bincount(inv_idx)

    #9 is for star particles
    #10 is for star mass
    if star:
	pixelMass = np.bincount(inv_idx, weights=starMass[binned])
        xBin,yBin,zBin = XYZ[uniq_inds].T
        pixelMass = pixelMass*1.0e10
        binArray[xBin,yBin,zBin,9] += incr
        binArray[xBin,yBin,zBin,10] += pixelMass
        time3 = time3 + (time.clock() - t)
    
    #11 is for dm particles
    #12 is for dm mass
    else:
        pixelMass = np.bincount(inv_idx, weights=dmMass[binned])
        xBin,yBin,zBin = XYZ[uniq_inds].T
        pixelMass = pixelMass*1.0e10
        binArray[xBin,yBin,zBin,11] += incr
        binArray[xBin,yBin,zBin,12] += pixelMass
        time4 = time4 + (time.clock() - t)


#function to calculate HI for startforming and nonstarforming gas
#---------------------------------------------------------------------------------------
def calcHI():

    global T, rho, gasMass, sfr, fneut, hsml, h1, h2, ionized

    #get subset of snapshot data for particles inside cone
    T_p		= T[binned]
    rho_p	= rho[binned]
    mass_p	= gasMass[binned]
    sfr_p	= sfr[binned]
    fneut_p	= fneut[binned]
    hsml_p	= hsml[binned]

    h1 = np.zeros(np.sum(binned))
    h2 = np.zeros(np.sum(binned))

    H2_frac = 0.

    #if it is not a star forming particle:
    #-------------------------------------------
    nonStarForming = np.where(sfr_p == 0.)[0]

    ilo = np.zeros(nonStarForming.size)
    ihi = np.zeros(nonStarForming.size)
    ihi[:] = (NINTERP-1)

    frh = fneut_p[nonStarForming] * XH * rho_p[nonStarForming] / (MHYDR*hsml_p[nonStarForming]*1.3737)

    #condition mask 1 = while loop 
    loop = ((ihi-ilo) > 1.)
    while(np.count_nonzero(loop) > 0):
	#condition mask 2 = if statement
	mask = ((np.array(KernIntTable[[(ilo[loop]+ihi[loop])/2],1])*frh[loop] < NHILIM)).flatten()
	ihi[mask] = (ilo[mask]+ihi[mask])/2
        ilo[~mask] = (ilo[~mask]+ihi[~mask])/2
	loop = ((ihi-ilo) > 1.)

    mask = np.asarray(np.where((T_p[nonStarForming] < 3.e4) & (ilo > 0.))).flatten()
    index = ((ilo[mask]+ihi[mask])/2).astype(int)
    fneut_p[mask] = ((fneut_p[mask] * KernIntTable[index,0]) + FSHIELD * (1.0 - KernIntTable[index,0]))

    #if it is star forming 
    #------------------------------------
    starForming = np.where(sfr_p > 0.)[0]
    coldphasemassfrac  	 	= (1.0e8-T_p[starForming])/1.0e8;
    Rmol                	= (rho_p[starForming]*T_p[starForming] / (P0BLITZ*MHYDR))**ALPHA0BLITZ
    fneut_p[starForming]        = FSHIELD * coldphasemassfrac / (1.0+Rmol)
    H2_frac			= FSHIELD - fneut[starForming]

    #H1 for SF and nSF
    h1 			= (fneut_p   *  mass_p * XH )
    
    #H2 for SF
    h2[starForming] 	= (H2_frac * mass_p[starForming] * XH)

def InitKernIntTable():
    global KernIntTable
    NSRCHRADKERN = 1.0
    kern = 0.
    ksum = 0.
    kint = 0.
    dx = NSRCHRADKERN/NINTERP
    xw =0.
    for i in range(NINTERP-1,-1,-1):
        xw = i*dx
        if xw<=0.5:
            kern = 1-6*xw*xw+6*xw*xw*xw
        else:
            kern = 2*(1-xw)*(1-xw)*(1-xw)
        kern *= 8./M_PI
        ksum += kern*4*M_PI*xw*xw*dx
        kint += kern*dx
        KernIntTable[i,0] = ksum
        KernIntTable[i,1] = kint
        KernIntTable[i,2] = kern


#-------------------USER INFO--------------------------

start1 = time.clock()

cmdargs = str(sys.argv)
InitKernIntTable()

print "\n"
print "=============================="
print "------------INFO--------------"
print "=============================="
print "\n"

#cone ID, used to determine deptch, 0 = first cube
cubeID = np.int(sys.argv[1])
print "Cone ID=", cubeID

#retrieve coma delimited params

with open("params.txt") as f:
    
    content = f.readlines()
    for i in range(len(content)):
        line = content[i].split("=")

        #get cube size from file
        if(line[0] == "CUBESIZE"):
	    cubeSize = np.float(line[1])
    	    print "Cube size=", cubeSize, "Mpcs"
    
        #convert string list to float and convert to unit vector
        if(line[0] == "CONEAXISVECTOR"):
            coneVec = unit_vector_scalar(np.array((line[1]).split(","), np.float))
            print "Cone Axis Unit Vector=", coneVec
    
    	    #test if user defined cone axis vector meets recommended structure, np.around to avoid floating point comparison issues
     	    if((coneVec[1]+coneVec[2]) / coneVec[0] > 0.3):    
                print "............................................................................................."
	        print "NB! It is recommended to use a cone axis vector [V1, V2, V3], where (V2+V3)/V1 < 0.3."
                print "Continuing anyway..."
	        print "............................................................................................."

	#redshit limit
	if(line[0] == "ZMAX"):
    	    zUpperLimit = np.float(line[1])
    	    print "Maximum redshift = ", zUpperLimit

	#snapshot file
        if(line[0] == "SNAPNAME"):
	    snap = (line[1]).split("\n")[0]
    
    	#append cone ID to given snapfile to get this cube's input snapShot - change this string manipulation if your snapshots 
    	#have a different naming convention
	if(line[0] == "X"):
	    startSnap = int(line[1])  
            # -cubeID because snapShots are in reserve order, IE oldest have lowest file number
    	    snapFile = snap.replace("X", str(startSnap))

	#pixel values - used for datacube
	if(line[0] == "PIXELS"):
    	    pixels = int(line[1])
    
	#frequency bins - used for datacube
	if(line[0] == "FREQBINS"):
	    zBins = int(line[1])
    	    print "Particle binning using", pixels,"x",pixels,"resolution, with",zBins,"frequency bins."

	#do DM calcs?
        if(line[0] == "DM"):
	    if(line[1].rstrip() == "YES"):
		doDM = True
            print "Dark matter selected?", doDM

f.close()

#create linear interpolation function using zr.txt to get Mpcs/H -> Z
points = [line.split(" ") for line in open("zr.txt")]

#zrFunc = interp1d(map(float, zip(*points)[1]), map(float, zip(*points)[3]))
zrFunc = UnivariateSpline(map(float, zip(*points)[1]), map(float, zip(*points)[2]), s=0)

#------------------FIND PARAMS------------------------

#get absolute height of this cone in simulation space
height = cubeID*(cubeSize)
print "Cube Height=",height,"Mpcs"

#get entry points of cone axis into this cube
scale = height / coneVec[0]
startCoOrds = np.array([0.0, (coneVec[1]*scale)%cubeSize, (coneVec[2]*scale)%cubeSize])
print "Entry co-ordinats into cube=",startCoOrds[0],",",startCoOrds[1], ",", startCoOrds[2], "Mpcs"

#get exit point from box - add scaled cone vector to startCoOrds
endCoOrds = np.array(coneVec*(cubeSize/coneVec[0]))
endCoOrds = np.array([cubeSize, startCoOrds[1]+endCoOrds[1], startCoOrds[2]+endCoOrds[2]])
print "Exit co-ordinats from cube=",endCoOrds[0],",",endCoOrds[1], ",", endCoOrds[2], "Mpcs"

#get length of cone axis to this cube
length = np.sqrt((height**2)+((coneVec[1]*scale)**2)+((coneVec[2]*scale)**2))
print "Cone axis length to cube entry=", length

#get observation point (global 0,0,0) reletive to 0,0,0 of this cube
origin = np.array(startCoOrds - (length*(coneVec)))

#correct floating point wierdness
origin[origin > -1.e-5] = 0.

print "Absolute origin=", origin

#------------------TEST BOUNDRIES------------------------

#test the boundary conditions of the cube and cone

breakUp =False
breakDown = False
breakLeft = False
breakRight = False
diag1 = False
diag2 = False
diag3 = False
diag4 = False

#find the radius for the Cone at the entry point to and exit point from the cube 
radiusIn = np.tan(np.radians((zrFunc(vec_length_scalar(startCoOrds - origin))+1))/(np.sqrt(np.pi)))*vec_length_scalar(startCoOrds - origin)
radiusOut = np.tan(np.radians((zrFunc(vec_length_scalar(endCoOrds - origin))+1))/(np.sqrt(np.pi)))*vec_length_scalar(endCoOrds - origin)

print "Cone radius at entry=", radiusIn
print "Cone radius at exit=", radiusOut

#Short note on boundary condition testing:
#this is necessary to detect if the edge of the cone leaves the bounds of the cube space - IE [0->cubeSize]['']['']
#if this occurs, tiling needs to occur, however it is costly to test every particle for every possibility
#below is a shortcut which elimates testing boundaries which the cone never breaks
#it tests if, at the top and bottom of the cube, the radius of the cone is greater than the cube boundaries
#because the cone does not cut through the cube at a right angle, at factor of 1/cos(theta) is applied, where theta is the angle between
#the vertical and the axis of the cube

radiusScaler = 1/np.cos(angle_between_scalar(coneVec, [1.0,0.0,0.0]))

#if cone break bound at either top or bottom of cube in any of the 4 directions, set flag
if((startCoOrds[1]+radiusIn*radiusScaler > cubeSize) or (endCoOrds[1]+radiusOut*radiusScaler > cubeSize)):
    breakUp = True
if((startCoOrds[1]-radiusIn*radiusScaler < 0.0) or (endCoOrds[1]-radiusOut*radiusScaler < 0.0)):
   breakDown = True
if((startCoOrds[2]-radiusIn*radiusScaler < 0.0) or (endCoOrds[2]-radiusOut*radiusScaler < 0.0)):
    breakLeft  = True
if((startCoOrds[2]+radiusIn*radiusScaler > cubeSize) or (endCoOrds[2]+radiusOut*radiusScaler > cubeSize)):    
    breakRight = True

if (breakUp and breakRight):
    diag1 = True
if (breakUp and breakLeft):
    diag2 = True
if (breakDown and breakRight):
    diag3 = True
if (breakDown and breakLeft):
    diag4 = True

#--------------------Z calcs---------------------

#find limits of redshift for this cube to create bounded binning matrix
#calculated using the cone axis vector and radius of cone at upper and lower limits of the cube.
#tested to be correct

groundPlane = np.array([startCoOrds[1],startCoOrds[2]] - unit_vector_scalar([coneVec[1], coneVec[2]])*radiusIn*radiusScaler)
zMin = zrFunc(vec_length_scalar(np.array([0.0,groundPlane[0],groundPlane[1]] - origin)))

#fix Spline function error at z=0
if (zMin < 0.):
    zMin = 0.

ceilingPlane = np.array([endCoOrds[1],endCoOrds[2]] + unit_vector_scalar([coneVec[1], coneVec[2]])*radiusOut*radiusScaler)
zMax = zrFunc(vec_length_scalar(np.array([cubeSize,ceilingPlane[0],ceilingPlane[1]] - origin)))

print "Redshift range for this cube=", zMin,"to", zMax

cubeZBins = math.ceil(((zMax-zMin)/zUpperLimit)*zBins)+1

offSet = int((zMin/zUpperLimit)*zBins)
print "Frequency bins in this cube=", cubeZBins
print "Datacube offset of this cube=", offSet

with open('cube_offsets.txt', "a") as file:
    file.write("CUBE"+str(cubeID)+"="+str(offSet)+"\n")

#---------------SELECT SNAPSHOTS------------------------

#code to find the snapshot with closest redshift to that of the cone at current depth (coneID)

#mean Z for this box
boxZ = zMin + (zMax - zMin)/2

if boxZ < 0.0:
    boxZ=0.0

snapZ = readhead(snapFile, 'redshift');

if(boxZ < snapZ):
    print "ERROR, ealiest snapshot has z=",snapZ, "but this cone segment needs atleast z=",boxZ
    sys.exit()

i = 0

#while not found keep looking
while True:
    i +=1

    index=str(startSnap-i)
    if((startSnap-i) < 100):
      index="0"+str(startSnap-i)

    nextSnap = snap.replace("X", index)
    nextZ = readhead(nextSnap, 'redshift')

    #if next snapshot's Z is further from needed Z than current snap
    if(abs(boxZ-nextZ) > abs(boxZ-snapZ)):
        break;

    else:
        snapZ = nextZ
        snapFile = nextSnap
print "\n"
print "Cone section mean redshift=", boxZ
print "GADGET snapshot for this cube= ", snapFile
print "Snapshot redshift=",snapZ
print "Redshift error margin", abs(snapZ - boxZ)

#------------------READ SNAPSHOTS------------------------

print "\n"
print "=============================="
print "------READING SNAPSHOT--------"
print "=============================="
print "\n"

gasCount = readhead(snapFile, 'gascount')

redShift = readhead(snapFile, 'redshift')

gasPos=readsnap(snapFile,'pos','gas')

rho=readsnap(snapFile,'rho','gas',units=1)
T=readsnap(snapFile,'u','gas',units=1)

gasMass=readsnap(snapFile,'mass','gas')

sfr=readsnap(snapFile,'sfr','gas')
fneut=readsnap(snapFile,'nh','gas')
hsml=readsnap(snapFile,'hsml','gas',units=1)
vel=readsnap(snapFile,'vel','gas')

scaleFactor=readhead(snapFile,'time')
omega_matter=readhead(snapFile,'O0')
omega_lamda=readhead(snapFile,'Ol')

print "\n"
print "Gas particles in this cube=",gasCount
print "\n"

starCount = readhead(snapFile, 'starcount')
starPos=readsnap(snapFile,'pos','star')
starMass=readsnap(snapFile,'mass','star')

print '\n'
print "Star particles in this cube=",starCount
print "\n"

if doDM:
    dmCount = readhead(snapFile, 'dmcount')
    dmPos=readsnap(snapFile,'pos','dm')
    dmMass=readsnap(snapFile,'mass','dm')

    print "\n"
    print "Dark matter particles in this cube=",dmCount
    print "\n"

print "Done.", "\n", "\n"

#-------------------MEM ALOC-----------------------

print "\n"
print "=============================="
print "------ALLOCATING MEMORY-------"
print "=============================="
print "\n"

regions = 0
#center tile

regions = breakUp + breakDown + breakLeft + breakRight + diag1 + diag2 + diag3 + diag4 +1

#create data cube
#include 9th and 10th element for star, 11th,12th for dm

if doDM:
    binArray = np.zeros((cubeZBins, pixels, pixels, 13), dtype=np.float32)
else:
    binArray = np.zeros((cubeZBins, pixels, pixels, 11), dtype=np.float32)


print "Done!"
print "Tiling snapshot", regions, "time(s) for this section"

#------------------TEST PARTICLES------------------------

print "\n"
print "=============================="
print "-----PARTICLE PROCESSING------"
print "=============================="
print "\n"

gasPos = (gasPos/1000.0) - origin
starPos = (starPos/1000.0) - origin
dmPos = (dmPos/1000.0) - origin

#tile particles in directions determined above
up=False
right=False
down=False
left=False
   
def startTile(label, coords):

    print "Starting ",label,"tile..."
    print "-----------------------"
    t= time.clock()
    tiledPartPos = gasPos + cubeSize*np.array(coords)
    test_particle(tiledPartPos, False, False)
    print "Gas comeplete. \n" 

    tiledPartPos = starPos + cubeSize*np.array(coords)
    test_particle(tiledPartPos, True, False)
    print "Stars comeplete. \n"

    if(doDM):
        tiledPartPos = dmPos + cubeSize*np.array(coords)
        test_particle(tiledPartPos, False, True)
	print "DM complete. \n"

    print "Tile complete, after", (time.clock()-t), "seconds"
    print "-----------------------------------"
    print "\n"

#run first untiled box
startTile("center", [0,0,0])
 
#test up
if(breakUp):
    startTile("top", [0,1,0])
    up = True

#test down
if(breakDown):
    startTile("bottom", [0,-1,0])
    down = True

#test right
if(breakRight):
    startTile("right", [0,0,1])
    right = True

#test left
if(breakLeft):
    startTile("left", [0,0,-1])
    left = True

#if out of bound in 2 adjacent directions, test the diagonal between them 
if((up and right) or diag1):
    startTile("diagonal", [0,1,1])

if((up and left) or diag2):
    startTile("diagonal", [0,1,-1])

if((down and right) or diag3):
    startTile("diagonal", [0,-1,1])

if((down and left) or diag4):
    startTile("diagonal", [0,-1,-1])

#average velocities
mask = binArray[:,:,:,0] > 0
binArray[mask,6] = binArray[mask,6]/binArray[mask,0]

#--------------------------------------------
#------------------I/O-----------------------
#--------------------------------------------

print "\n"
print "=============================="
print "-------WRITING TO FILE--------"
print "=============================="
print "\n"

dir = path.dirname(__file__)
filename = path.join(dir, 'data/cubeID_'+str(cubeID)+'_data')
np.save(filename, binArray)

#------------------------------------------

elapsed = (time.clock() - start1)

print "\n", "\n"
print "Done!"
print "Execution time = ", elapsed

print "Particle testing time:", time1
print "Gas binning time:", time2
print "Star binning time:", time3

if doDM:
    print "DM binning time:", time4

