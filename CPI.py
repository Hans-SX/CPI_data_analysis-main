#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:20:42 2022

@author: gianlorenzo
@edit: Shih Xian
    Two cameras. July 4, 2024

"""

import numpy as np
from datetime import datetime
import os, sys, time
from skimage import io
from os.path import join as joinDir
import matplotlib.pyplot as plt
import shutil
# from scipy.interpolate.interpn import interpn as interp




def RefocusSection(corrFun, transf, refArray, dpA, dpB, maxInt=False, correc=False):
    shape = corrFun.shape
    NA, NB= shape[0], shape[0]
    rangeA= (np.arange(NA)-(NA-1)/2)*dpA
    rangeB= (np.arange(NB)-(NB-1)/2)*dpB
    refocused = np.empty((0,2), dtype=object)
    # gridYA, gridXA, gridYB, gridXB = np.meshgrid(rangeA, rangeA, rangeB, rangeB)
    for z in refArray:
        matrix  = transf(z)
        invMat  = np.linalg.inv(matrix)
        dRef    = np.sqrt((invMat[0,0]*dpA)**2 + (invMat[0,1]*dpB)**2)
        maxRef  = (np.abs(invMat[0,0])*dpA*NA + np.abs(invMat[0,1])*dpB*NB)/2
        rangeRef= np.arange(-maxRef, maxRef, dRef)
        dSum    = np.sqrt((matrix[0,1]/dpA)**2 + (matrix[1,1]/dpB)**2)**-1
        maxSum  = maxInt if maxInt else (np.abs(invMat[0,1])*dpA*NA + np.abs(invMat[0,0])*dpB*NB)/2 #Ottima idea! Controlla!
        rangeSum= np.arange(-maxSum, maxSum, dSum)
        points  = (rangeA, rangeA, rangeB, rangeB)
        newPts  = np.array([[[i,j,k,l] for k in rangeSum for l in rangeSum] for i in rangeRef for j in rangeRef])
        def RefocusSinglePixel(newPoints):
            return np.sum(interp(points, corrFun, newPoints))
        
        
        
        
        

def ComputeCorrelations(A, B, differential=False):
    PrintSectionInit('Computing correlations... {}'.format("(With differential)" if differential else ''))
    start = Time()
    Ntot   = A.shape[1]
    NA     = int(np.sqrt(A.shape[0]))
    NB     = int(np.sqrt(B.shape[0]))
    shape  = (NA,NA,NB,NB)
    corrAB = (np.matmul(A, B.T)/Ntot).reshape(shape)
    avgA   = np.sum(A,axis=1)/Ntot
    avgB   = np.sum(B,axis=1)/Ntot
    if differential:
        buckA, buckB = np.sum(A, axis=0), np.sum(B, axis=0)
        avgBA, avgBB = np.sum(buckA)/Ntot, np.sum(buckB)/Ntot
        corrIaBuck   = np.matmul(A, buckA)/Ntot
        corrIbBuck   = np.matmul(B, buckB)/Ntot
        corrIaBIbB   = np.matmul(buckA, buckB)/Ntot
        ka, kb       = avgA/avgBB, avgB/avgBA
        diffTerm     = np.tensordot(ka, corrIbBuck, axes=0)\
                        + np.tensordot(corrIaBuck, kb, axes=0)\
                        - np.tensordot(ka, kb, axes=0)*corrIaBIbB
        diffTerm     = diffTerm.reshape(shape)
    #! This will cause issue when DIFF_BOOL set to False. There is a PlotCorrFunc on diffTerm, can not be plot with diffTerm = None.
    else: diffTerm = None
    stop = Time()
    PrintSectionClose()
    Print("[Elapsed Time]", '{:.3f} s'.format(stop-start))
    print("[G2] shape: {} | dtype: {} | min: {} | max: {}".format(corrAB.shape, corrAB.dtype, 
                                                                  np.min(corrAB), np.max(corrAB)))
    if differential:
        return corrAB - np.tensordot(avgA, avgB, axes=0).reshape(shape), \
               corrAB - diffTerm
    else:
        return corrAB - np.tensordot(avgA, avgB, axes=0).reshape(shape), \
               None

def PseudoGhosts(A, B, outDir, transp=True):
    PrintSectionInit('Calculating pseudo-ghosts...')
    start = Time()
    NA   = int(np.sqrt(A.shape[0]))
    NB   = int(np.sqrt(B.shape[0]))
    Ntot = A.shape[1]
    avgA, avgB       = np.sum(A, axis=1)/Ntot, np.sum(B, axis=1)/Ntot
    bucketA = np.expand_dims(np.sum(A, axis=0),axis=1)
    bucketB = np.expand_dims(np.sum(B, axis=0),axis=1)
    pseudoA = np.matmul(A, bucketB).T[0]/Ntot - avgA*np.sum(bucketB[:,0])/Ntot
    pseudoB = np.matmul(B, bucketA).T[0]/Ntot - avgB*np.sum(bucketA[:,0])/Ntot
    pseudoA, pseudoB = pseudoA.reshape((NA,NA)), pseudoB.reshape((NB,NB))
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,20))
    im1 = ax1.imshow(avgA.reshape((NA,NA)), cmap="gray")
    im2 = ax2.imshow(avgB.reshape((NB,NB)), cmap="gray")
    im3 = ax3.imshow(pseudoA, cmap="gray")
    im4 = ax4.imshow(pseudoB, cmap="gray")
    ax1.set_title("Intensity on A"); ax2.set_title("Intensity on B")
    ax3.set_title("Pseudo-ghost on A"); ax4.set_title('Pseudo-ghost on B')
    fig.colorbar(im1, ax=ax1); fig.colorbar(im2, ax=ax2); fig.colorbar(im3, ax=ax3);fig.colorbar(im4, ax=ax4)    
    fig.savefig(joinDir(outDir, "Pseudo-ghost.tif"),
                dpi='figure',transparent=transp)
    plt.close("all")
    stop = Time()
    Print("[Elapsed Time]", '{:.3f} s'.format(stop-start))
    PrintSectionClose()

def BinInfo(Na, Nb, binA, binB, dp):
    if Na%binA or Nb%binB:
        print('Invalid binning')
        sys.exit()
    else:
        NA, NB = Na//binA, Nb//binB
        dpA,dpB= dp*binA, dp*binB
        Print("[binA,binB]", '[{},{}]'.format(binA, binB))
        Print("[NA_ini,NB_ini]", '[{},{}]'.format(Na, Nb))
        Print("[NA_bin,NB_bin]", '[{},{}]'.format(NA, NB))
        return NA, NB, dpA, dpB

def BinInfo_onecam(Na, binA, dp, arm):
    if Na%binA:
        print('Invalid binning')
        sys.exit()
    else:
        NA = Na//binA
        dpA = dp*binA
        print("[bin"+arm+"]", '[{}]'.format(binA))
        print("["+arm+"_ini]", '[{}]'.format(Na))
        print("["+arm+"_bin]", '[{}]'.format(NA))
        return NA, dpA
    
def ReadAndBin(filename, outDir,
               Na, ax, ay, binA,
               Nb, bx, by, binB,
               dp, dtype='float32'):
    start = Time()
    PrintSectionInit('Loading Data...')
    NA, NB, dpA, dpB = BinInfo(Na, Nb, binA, binB, dp)
    Nfile     = len(filename)
    numExc    = 0
    Print("[N FILES]", Nfile)
    A = np.zeros((NA*NA,1), dtype=dtype)
    B = np.zeros((NB*NB,1), dtype=dtype)
    Ntot = 0
    for ifile in range(Nfile):
        try:
            data = io.imread(filename[ifile])
            nPages = data.shape[0]
            Ntot = Ntot + nPages        
            if ifile==0:
                pkPlotDataOverview(data,outDir,ax,ay,bx,by,Na,Nb)
            Print("[PROGRESS]", '{}/{}'.format(ifile+1, Nfile))
            print("            shape: {} | dtype: {} | min: {} | max: {}".format(
                data.shape, data.dtype, np.min(data), np.max(data)))
            binnedA = bin_ndarray(
                data[:,ay:ay+Na,ax:ax+Na],new_shape=(nPages,NA,NA), operation='mean')
            binnedB = bin_ndarray(
                data[:,by:by+Nb,bx:bx+Nb],new_shape=(nPages,NB,NB), operation='mean')
            A = np.append(A, (binnedA.reshape(nPages,NA*NA)).T, axis=1)
            B = np.append(B, (binnedB.reshape(nPages,NB*NB)).T, axis=1)
            del(data)
        except ValueError:
            numExc += 1
            print("{} invalid file(s)".format(numExc))
            Print("[N FILES]", Nfile - numExc)
            pass
    A = A[:,1:]
    B = B[:,1:]
    stop = Time()
    Print("[N_TOT]", Ntot)
    Print("[A]", " shape: {} | dtype: {} | min: {} | max: {}".format(A.shape, A.dtype, np.min(A), np.max(A)))
    Print("[B]", " shape: {} | dtype: {} | min: {} | max: {}".format(B.shape, B.dtype, np.min(B), np.max(B)))
    Print("[Elapsed Time]", '{:.3f} s'.format(stop-start))
    PrintSectionClose()
    return A, B, dpA, dpB

def ReadAndBin_onecam(filename, outDir,
               Na, binA,
               dp, dtype='float32', arm='spa'):
    """
    bin_ndarry: only take a square shape input (4000, 310, 310)
    NA = Na/binA, needs to be an integer.
    """
    start = Time()
    PrintSectionInit('Loading Data...')
    NA, dpA = BinInfo_onecam(Na, binA, dp, arm)
    Nfile     = len(filename)
    numExc    = 0
    print("[N FILES]", Nfile)
    A = np.zeros((NA*NA,1), dtype=dtype)
    Ntot = 0
    for ifile in range(Nfile):
        try:
            data = io.imread(filename[ifile])
            if data.shape[1] > data.shape[2]:
                diff_half = int((data.shape[1] - data.shape[2])/2)
                data = data[:, diff_half:Na+diff_half, :Na]
            elif data.shape[1] < data.shape[2]:
                diff_half = int((data.shape[2] - data.shape[1])/2)
                data = data[:, :Na, diff_half:Na+diff_half]
            nPages = data.shape[0]
            Ntot = Ntot + nPages
            if ifile==0:
                pkPlotDataOverview_onecam(data,outDir,dp,arm)
            Print("[PROGRESS]", '{}/{}'.format(ifile+1, Nfile))
            print("            shape: {} | dtype: {} | min: {} | max: {}".format(
                data.shape, data.dtype, np.min(data), np.max(data)))
            
            binnedA = bin_ndarray(
                data, new_shape=(nPages,NA,NA), operation='mean')
            A = np.append(A, (binnedA.reshape(nPages,NA*NA)).T, axis=1)
            del(data)
        except ValueError:
            numExc += 1
            print("{} invalid file(s)".format(numExc))
            Print("[N FILES]", Nfile - numExc)
            pass
    A = A[:,1:]
    stop = Time()
    Print("[N_TOT]", Ntot)
    Print("["+arm+"]", " shape: {} | dtype: {} | min: {} | max: {}".format(A.shape, A.dtype, np.min(A), np.max(A)))
    Print("[Elapsed Time]", '{:.3f} s'.format(stop-start))
    PrintSectionClose()
    return A, dpA

def readConfig():
    PrintSectionInit('Reading parameters from config...')
    file = open("config.py")
    cfg  = file.read()
    file.close()
    print(cfg)
    PrintSectionClose()
    return cfg

def readTiff(fileList):
    return None

# DIRECTORY utils
def FilePath():
    '''
    Returns the directory of the current python script
    '''
    return os.path.dirname(os.path.abspath(__file__))

def DataDir(std=True, path=None):
    '''
    Returns the output directory. If std is True, the default directory is 
    given returned. If std is not True, the variable 'path' is returned.

    Parameters
    ----------
    std : TYPE, Boolean
        DESCRIPTION. The default is True.
    path : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    path : string.

    '''
    if std:
        path = joinDir(FilePath(), os.pardir,'data')    
    return path

def OutDir(timeTag=False, std=True, path=None):
    '''
    Returns the output directory. If std is True, the default directory is 
    returned. If std is not True, the variable 'path' is returned. If timeTag
    is True, current date and time is added to the output folder name

    Parameters
    ----------
    timeTag : TYPE, Boolean
        DESCRIPTION. The default is False.
    std : TYPE, Boolen
        DESCRIPTION. The default is True.
    path : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    path : TYPE
        DESCRIPTION.

    '''
    if std:
        path = joinDir(FilePath(), os.pardir,'output')
    if timeTag:
        path += ''+Now()
    return path

def setDirectories(stdData=True, stdOut=True, timeTag=False, 
                   dataPath=None, outPath=None):
    '''
    Creates the folder of interest and returns the output directory and a list
    of files for data reading.

    Parameters
    ----------
    stdData : TYPE, Boole
        DESCRIPTION. The default is True.
    stdOut : TYPE, Boole
        DESCRIPTION. The default is True.
    timeTag : TYPE, Boole
        DESCRIPTION. The default is False.
    dataPath : TYPE, String
        DESCRIPTION. The default is None.
    outPath : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    outPath : String.
    filename : Array of files to read.

    '''
    PrintSectionInit('Setting up folders...')
    
    dataPath = DataDir(stdData, dataPath)
    outPath  = OutDir(timeTag, stdOut, outPath)
    if not os.path.exists(outPath):   
        os.makedirs(outPath)
    filename  = os.listdir(dataPath)
    filename  = [os.path.join(dataPath, filename[i]) 
                 for i in range(len(filename))]
    Print("[DATA PATH]",dataPath)
    Print("[OUT PATH]",outPath)
    
    PrintSectionClose()
    return outPath, filename

def setDirectories_twocams(stdData=True, stdOut=True, timeTag=False, 
                   dataPath=None, outPath=None, armA='spatial', armB='angular'):
    '''
    Creates the folder of interest and returns the output directory and a list
    of files for data reading.

    Parameters
    ----------
    stdData : TYPE, Boole
        DESCRIPTION. The default is True.
    stdOut : TYPE, Boole
        DESCRIPTION. The default is True.
    timeTag : TYPE, Boole
        DESCRIPTION. The default is False.
    dataPath : TYPE, String
        DESCRIPTION. The default is None.
    outPath : TYPE, String
        DESCRIPTION. The default is None.

    Returns
    -------
    outPath : String.
    filename : Array of files to read.

    '''
    PrintSectionInit('Setting up folders...')
    
    dataPath = DataDir(stdData, dataPath)
    armAPath = joinDir(dataPath, armA)
    armBPath = joinDir(dataPath, armB)
    outPath  = OutDir(timeTag, stdOut, outPath)
    if not os.path.exists(outPath):   
        os.makedirs(outPath)
    shutil.copy("./config.py", joinDir(outPath, "config.py"))
    armAfilename  = os.listdir(armAPath)
    armAfilename  = [os.path.join(armAPath, armAfilename[i]) 
                 for i in range(len(armAfilename))]
    armBfilename  = os.listdir(armBPath)
    armBfilename  = [os.path.join(armBPath, armBfilename[i]) 
                 for i in range(len(armBfilename))]
    Print("[DATA PATH]", dataPath)
    Print("[ARM A PATH]", armAPath)
    Print("[ARM B PATH]", armBPath)
    Print("[OUT PATH]", outPath)
    
    PrintSectionClose()
    return outPath, armAfilename, armBfilename

def RefocDir(outDir):
    if not os.path.exists(joinDir(outDir, 'refocused')):   
        os.makedirs(joinDir(outDir, 'refocused'))

# TIME utils
def Now():
    return datetime.now().strftime("%Y-%m-%d %H.%M")
def Time():
    return time.time()

# PRINT Utils
def Print(label, text):
    '''
    Prints "label": "text"

    Parameters
    ----------
    label : string or number.
    text :  string or number.

    Returns
    -------
    None.

    '''
    print("{:>15}: {}".format(label, text))

def PrintSectionInit(text):
    '''
    Prints section name "text" with an indent

    Parameters
    ----------
    text : string

    Returns
    -------
    None.

    '''
    print("{:>17}>> {}".format("\n_______________", text))

def PrintSectionClose():
    '''
    Prints "DONE" at the end of a section

    Returns
    -------
    None.

    '''
    print("{:>17}{}".format("", 'DONE.'))
    
# PLOT utils
def pkPlotDataOverview(data,outputDir,ax1,ay1,bx1,by1,Na,Nb):
    import matplotlib.patches as patches

    #  G(1)                                         
    Nt1      = data.shape[0]
    (Nx, Ny) = (data.shape[2], data.shape[1])
    M = np.max(data)
    m = np.min(data)
    G1tot    = np.sum(data, axis = 0).astype(float)    # Sommo le immagini
    G1totNorm = (G1tot-m)/(M-m)                         # Rinormalizzo la somma tra [0,1], per allinearsi a Wolfram Mathematica
    G1 = G1totNorm/Nt1                                  #  e divido per il loro numero, in pratica si ottiene una media.

    # Create Figure G1_plot
    fig, ax = plt.subplots(1,figsize=(10,4))
    im = ax.imshow(G1, cmap="gray")
    cbar = fig.colorbar(im, ax = ax)
    rectA = patches.Rectangle((ax1,ay1),Na,Na,linestyle='--',edgecolor='b',facecolor='none')
    rectB = patches.Rectangle((bx1,by1),Nb,Nb,linestyle='--',edgecolor='m',facecolor='none')
    ax.add_patch(rectA)
    ax.add_patch(rectB)
    fig.savefig(os.path.join(outputDir, "G1_plot.png"))

    # setup
    from config import dp
    xi = lambda x : dp*(x-float(Nx+1)/2)
    yi = lambda y : dp*(y-float(Ny+1)/2)

    # Create Figure G1_3D_plot
    fig = plt.figure(figsize=(10,4))
    ax = plt.axes(projection='3d')
    x = np.linspace(xi(0), xi(Nx), Nx)
    y = np.linspace(yi(0), yi(Ny), Ny)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, G1[::-1,:], cmap='Wistia',linewidth=0.5, edgecolors='k')
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    ax.set_zlabel('z')
    fig.savefig(os.path.join(outputDir, "G1_plot3D.png"))
    plt.close("all")
    
def pkPlotDataOverview_onecam(data,outputDir,dp,arm):
    """ Take one tif (from one arm) and plot the 2D and 3D of G(1). """

    #  G(1)
    Nt1      = data.shape[0]
    (Nx, Ny) = (data.shape[2], data.shape[1])
    M = np.max(data)
    m = np.min(data)
    G1tot    = np.sum(data, axis = 0).astype(float)    # Sommo le immagini
    G1totNorm = (G1tot-m)/(M-m)                         # Rinormalizzo la somma tra [0,1], per allinearsi a Wolfram Mathematica
    G1 = G1totNorm/Nt1                                  #  e divido per il loro numero, in pratica si ottiene una media.

    # Create Figure G1_plot
    ''' G1_plot is to plot the rectangles on the image.
    No need to draw rectangle in two tif case. Draw two G1 instead.'''
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(G1, cmap="gray")
    cbar = fig.colorbar(im, ax = ax)
    fig.savefig(os.path.join(outputDir, "G1_plot_"+arm+".png"))

    # setup
    # from cpi_refocusing.config import dp
    xi = lambda x : dp*(x-float(Nx+1)/2)
    yi = lambda y : dp*(y-float(Ny+1)/2)

    # Create Figure G1_3D_plot
    fig = plt.figure(figsize=(10,4))
    ax = plt.axes(projection='3d')
    x = np.linspace(xi(0), xi(Nx), Nx)
    y = np.linspace(yi(0), yi(Ny), Ny)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, G1[::-1,:], cmap='Wistia',linewidth=0.5, edgecolors='k')
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    ax.set_zlabel('z')
    fig.savefig(os.path.join(outputDir, "G1_plot3D_"+arm+".png"))
    plt.close("all")

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    # Early return
    if ndarray.shape==new_shape:
        return ndarray

    # Start computation 
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def PlotCorrFunc2D(corrFunc, outDir, differential=False):
    if not differential:
        PrintSectionInit("Printing 2D corr. function")
    else:
        PrintSectionInit("Printing 2D differential corr. function")
    corrDir = joinDir(outDir,"Correlation function")
    if not os.path.exists(corrDir): 
        os.makedirs(corrDir)
    fig, (ax1,ax2) = plt.subplots(2,figsize=(20,20))
    im1 = ax1.imshow(np.sum(corrFunc, axis=(1,3)), cmap="gray")
    im2 = ax2.imshow(np.sum(corrFunc, axis=(0,2)), cmap="gray")
    ax1.set_title("xA-xB Corr. Func."); ax2.set_title("yA-yB Corr. Func.")
    fig.colorbar(im1, ax=ax1); fig.colorbar(im2, ax=ax2)  
    if differential: 
        fig.savefig(joinDir(corrDir, "Corr_func_diff.tif"),
                dpi='figure',transparent=False)
    else: 
        fig.savefig(joinDir(corrDir, "Corr_func.tif"),
                dpi='figure',transparent=False)
    plt.close("all")
    PrintSectionClose()
    



