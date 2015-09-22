
#AUTHOR: Matthew Cawood
#PURPOSE: Creating an HI lightcone from stacked GADGET snapshots
#DATE: May 2014
#Written and tested using python-2.7.6, numpy-1.8.1 and scipy-0.14.0

import sys
import scipy
from scipy.interpolate import UnivariateSpline
import numpy as np
import math
import os
import time
import threading 
import platform
import glob 

cpus=0
z=0
box=0
axis=0
toStart=0
numCubes=0
running=0
zBins=0
doDM= False
pixElems = 11
pixels = 0
zBins = 0 


def intro():

    print "-----------------------------------------------------------"
    print "                 Lightcone Tool v0.1                       "
    print "-----------------------------------------------------------"

    print ""

    print "Checking system..."
    print ""

    print "Python Path:    ", sys.executable
    print "Python Version: ", platform.python_version()

    if platform.python_version() != "2.7.6":
        print "NOTE: this code was deveoped with Python version 2.7.6, continuing anyway..."
        print ""

    print "NumPy Version:  ", np.version.version

    if np.version.version != "1.8.1":
        print "NOTE: this code was deveoped with NumPy version 1.8.1, continuing anyway..."
        print ""

    print "SciPy Version:  ", scipy.version.version

    if scipy.version.version != "0.14.0":
        print "NOTE: this code was deveoped with SciPy version 0.14.0, continuing anyway..."
        print ""

    print ""
    print ""

    print "Checking files..."

    if not os.path.isfile("params.txt"):
        print "ERROR, params.txt file not found! Check README"
        sys.exit()

    if not os.path.isfile("zr.txt"):
        print "ERROR, zr.txt file not found! Check README"
        sys.exit()

    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists("data"):
	os.makedirs("data")


    print "Done."
    print ""
    time.sleep(2)

def params():
    
    global cpus, z, box, axis, pixels, zBins, doDM, pixElems
    print "Checking parameters..."

    #read params
    with open("params.txt") as f:
        content = f.readlines()
        for i in range(len(content)):
            line = content[i].split("=")

            if(line[0] == "NCPUS"):
                cpus = int(line[1])

            elif(line[0] == "ZMAX"):
                z = float(line[1])

            elif(line[0] == "CUBESIZE"):
                box = float(line[1])

            elif(line[0] == "CONEAXISVECTOR"):
                axis = np.array(line[1].split(","), np.float)
   
            elif(line[0] == "PIXELS"):
                pixels = int(line[1])

            elif(line[0] == "FREQBINS"):
                zBins = int(line[1])

            elif(line[0] == "DM"):
                if(line[1].rstrip() == "YES"):
                    doDM = True
                    pixElems=13

    #check params
    if(cpus == 0):
        print "Could not find NCPUS in param.txt"
        sys.exit()
    if(z == 0):
        print "Could not find ZMAX in param.txt"
        sys.exit()
    if(box == 0):
        print "Could not find CUBESIZE in param.txt"
        sys.exit()
    if(len(axis) == 1):
        print "Could not find CONEAXISVECTOR in param.txt"
        sys.exit()
    if(zBins == 0):
        print "Could not find FREQBINS in param.txt"
        sys.exit()
    if(pixels == 0):
        print "Could not find PIXELS in param.txt"
        sys.exit()

    print "Done."
    print ""
    time.sleep(2)

#math functions for calculating cone dimensions
def vec_length(vector):
    return np.linalg.norm(vector)

def unit_vector(vector):
    len = vec_length(vector)
    if(len > 0):
        return vector / len
    return vector

def dims():
    global numCubes
    print "Calculating lightcone dimensions..."

    points = [line.split(" ") for line in open("zr.txt")]
    zrFunc = UnivariateSpline(map(float, zip(*points)[2]), map(float, zip(*points)[1]), s=0)
 
    depth = np.nan_to_num(zrFunc(z))
    lenPerBox = vec_length(unit_vector(axis)*(box/unit_vector(axis)[0]))
    numCubes = int(math.ceil(depth / lenPerBox))

    print "Done."
    print numCubes, box, "Mpc snapshots are required for a lightcone of depth z=", z
    print ""
    print ""
    time.sleep(2)

def clean():
    #empty the offsets file
    open("cube_offsets.txt", 'w').close()

    #delete old data files
    for the_file in os.listdir('data/'):
        file_path = os.path.join('data/', the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

    #delete old log files
    for the_file in os.listdir('logs/'):
        file_path = os.path.join('logs/', the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e


    #delete old dataCube files
    for filename in glob.glob("./dataCube_*"):
	os.remove(filename) 

#thread spawner
def startCube(arg, lock):
    
    global running 

    lock.acquire()
    running += 1
    lock.release()    

    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'logs/cube_'+str(arg)+'.log')
    os.system("python cube.py "+str(arg)+" > " +filename ) 

    lock.acquire()	
    running -= 1
    print "Cube", arg, "complete."
    lock.release()
    sys.stdout.flush()

#check versions and files
intro()

#check params
params()

#calculate number of cone sections
dims()

#wipe old data
clean()

#Start execution loop

threads = []
lock = threading.Lock()
t1 = time.time()
toStart = numCubes-1

while 1:

    if (running < cpus) & (toStart >= 0):
	t = threading.Thread(target=startCube, args=(toStart, lock,))
	threads.append(t)

	print "Starting snapshot", toStart, "..."
	sys.stdout.flush()
	t.start()
	toStart -= 1
	
    elif(toStart == -1):
	for x in threads:
	    x.join()

	print "All snapshots complete!"

	t2 = time.time()
	print "Time:", round((t2 - t1)/60, 2), "mins"
	print "Starting postprocessing..."

	sys.stdout.flush()

	dir = os.path.dirname(__file__)
	filename = os.path.join(dir, 'logs/post.log')

	os.system("python postProcessor.py "+ str(numCubes) + " > " + filename)
	os.system("python calcRadius.py")


	print "Postprocessing complete!"
	print "Time:", round((time.time() - t2)/60, 2), "mins"
	print "Total time:", round((time.time()-t1)/60, 2), "mins"

	sys.exit()

    else:
	time.sleep(0.5)
    
