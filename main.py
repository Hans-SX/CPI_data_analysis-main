#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:30:02 2022

@author: gianlorenzo

Edited by: Shih Xian, July, 2024
"""
print('''\n
      # ========================================================#
      #   Correlation Plenoptic Imaging data analysis           #
      # ========================================================#\n''')

import CPI as cpi
import numpy as np
from scipy.interpolate import interpn as interp
import multiprocessing as mp
import matplotlib.pyplot as plt
import os, sys
import itertools
from os.path import join as joinDir
import argparse

exec(cpi.readConfig())

parser = argparse.ArgumentParser()
parser.add_argument('--targ_dp', type=str)
args = parser.parse_args()
datapath = joinDir(os.getcwd(), os.pardir, args.targ_dp, 'data')
outpath = joinDir(os.getcwd(), os.pardir, args.targ_dp, 'out')
outDir, armAfiles, armBfiles = cpi.setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)
A, dpA = cpi.ReadAndBin_onecam(
               armAfiles, outDir,
               Na, binA, dp, arm='spa')
B, dpB = cpi.ReadAndBin_onecam(
               armBfiles, outDir,
               Nb, binB, dp, arm='ang')
#%%
cpi.PseudoGhosts(A, B, outDir, transp=False)

G2, diffTerm = cpi.ComputeCorrelations(A, B, differential=DIFF_BOOL)

cpi.PlotCorrFunc2D(G2, outDir, differential=False)
cpi.PlotCorrFunc2D(diffTerm, outDir, differential=True)

#%%
# cpi.RefocusSection(G2, transf, REFOC, dpA, dpB, maxInt=maxInt, correc=CORREC_BOOL)
if __name__=="__main__":
    cpi.PrintSectionInit("Refocusing...")
    start = cpi.Time()
    cpi.RefocDir(outDir)
    shape = G2.shape
    NA, NB= shape[0], shape[2]
    rangeA= (np.arange(NA)-(NA-1)/2)*dpA
    rangeB= (np.arange(NB)-(NB-1)/2)*dpB
    counter = 0
    # gridYA, gridXA, gridYB, gridXB = np.meshgrid(rangeA, rangeA, rangeB, rangeB)
    for z in REFOC:
        cpi.Print("Refocusing","{} of {}".format(counter+1,len(REFOC)))
        startRef = cpi.Time()
        matrix  = transf(z)
        invMat  = np.linalg.inv(matrix)
        dRef    = np.sqrt((invMat[0,0]*dpA)**2 + (invMat[0,1]*dpB)**2)
        maxRef  = (np.abs(invMat[0,0])*dpA*NA + np.abs(invMat[0,1])*dpB*NB)/2
        rangeRef= np.arange(-maxRef, maxRef, dRef)
        NR      = len(rangeRef)
        dSum    = np.sqrt((matrix[0,1]/dpA)**2 + (matrix[1,1]/dpB)**2)**-1
        maxSum  = maxInt if maxInt else (np.abs(invMat[1,0])*dpA*NA + np.abs(invMat[1,1])*dpB*NB)/2
        rangeSum= np.arange(-maxSum, maxSum, dSum)
        NS      = len(rangeSum)
        points  = (rangeA, rangeA, rangeB, rangeB)
        refPts  = np.array([x for x in itertools.product(rangeRef, rangeRef)])
        sumPts  = np.array([x for x in itertools.product(rangeSum, rangeSum)])
        newPixels = [x for x in range(len(rangeRef)**2)]
        matrix4D  = np.array(
            [[matrix[0,0],0,matrix[0,1],0],
             [0,matrix[0,0],0,-matrix[0,1]],
             [matrix[1,0],0,matrix[1,1],0],
             [0,-matrix[1,0],0,matrix[1,1]]]
            )
        method='nearest'
        # matrix4D = np.linalg.inv(matrix4D)
        # newPts  = np.array([[[i,j,k,l] for k in rangeSum for l in rangeSum] for i in rangeRef for j in rangeRef])
        def RefocusSinglePixel(pixelCoords):
            newPoints = np.insert(sumPts,[0,0],refPts[pixelCoords],axis=1)
            newPoints = np.matmul(matrix4D, newPoints.T).T
            outcome   = np.sum(interp(points, G2, newPoints, method=method,
                                 bounds_error=False,fill_value=0))
            return outcome
        # pool   = mp.Pool(8)
        refVec = list(map(RefocusSinglePixel, range(len(rangeRef)**2)))
        refVec=np.array(refVec).reshape((NR,NR))
        # pool.close()
        if DIFF_BOOL:
            def RefocusSinglePixel(pixelCoords):
                newPoints = np.insert(sumPts,[0,0],refPts[pixelCoords],axis=1)
                newPoints = np.matmul(matrix4D, newPoints.T).T
                outcome   = np.sum(interp(points, diffTerm, newPoints, method=method,
                                     bounds_error=False,fill_value=0))
                return outcome
            # pool   = mp.Pool(8)
            diff = list(map(RefocusSinglePixel, newPixels))
            diff=np.array(diff).reshape((NR,NR))
            # pool.close()
            if CORREC_BOOL:
                def Correc1D(xRef):
                    corrPoints = np.insert(np.expand_dims(rangeSum, axis=1),
                                           0, xRef, axis=1)
                    corrPoints = np.matmul(matrix, corrPoints.T).T
                    return np.sum(
                        interp((rangeA,rangeB), np.ones((NA,NB)), corrPoints,
                        method=method, bounds_error=False,fill_value=0))
                correc1D = np.array(list(map(Correc1D, rangeRef)))
                correc   = np.tensordot(correc1D, correc1D, 0)
                correc[correc<1]=1
                refVec   = refVec
                refVec  /= correc
                if DIFF_BOOL: 
                    diff = diff
                    diff/= correc
            
            fig, ((ax3,ax4),(ax5,ax6)) = plt.subplots(2,2,figsize=(20,20))
            im1 = ax3.imshow(refVec, cmap="gray")
            im2 = ax4.imshow(diff, cmap="gray")
            refVec[refVec<0]=0
            diff[diff<0]=0
            im3 = ax5.imshow(refVec, cmap="gray")
            im4 = ax6.imshow(diff, cmap="gray")
            ax3.set_title("Refocused image_oof="+str(z))
            ax4.set_title("Differential image_oof="+str(z))
            ax5.set_title("Refocused image pos_oof="+str(z))
            ax6.set_title("Differential image pos_oof="+str(z))
            fig.colorbar(im1, ax=ax3)
            fig.colorbar(im2, ax=ax4)
            fig.colorbar(im3, ax=ax5)
            fig.colorbar(im4, ax=ax6)
            fig.savefig(os.path.join(outDir+"/refocused", "Refocus_plot_"+str(z)+".tif"),
                        dpi='figure',transparent=False)
            plt.close("all")
        else:
            if CORREC_BOOL:
                def Correc1D(xRef):
                    corrPoints = np.insert(np.expand_dims(rangeSum, axis=1),
                                           0, xRef, axis=1)
                    corrPoints = np.matmul(matrix, corrPoints.T).T
                    return np.sum(
                        interp((rangeA,rangeB), np.ones((NA,NB)), corrPoints,
                        method=method, bounds_error=False,fill_value=0))
                correc1D = np.array(list(map(Correc1D, rangeRef)))
                correc   = np.tensordot(correc1D, correc1D, 0)
                correc[correc<1]=1
                refVec   = refVec
                refVec  /= correc
            fig, (ax3,ax4) = plt.subplots(1,2,figsize=(20,20))
            im1 = ax3.imshow(refVec, cmap="gray")
            refVec[refVec<0]=0
            im2 = ax4.imshow(refVec, cmap="gray")
            ax3.set_title("Refocused image_oof="+str(z))
            ax4.set_title("Refocused image Threshold="+str(z))
            fig.colorbar(im1, ax=ax3)
            fig.colorbar(im2, ax=ax4)
            fig.savefig(os.path.join(outDir+"/refocused", "Refocus_plot_"+str(z)+".tif"),
                        dpi='figure',transparent=False)
            plt.close("all")
        print("Time elapsed: {:3f}".format(cpi.Time()-startRef))
        counter +=1
    cpi.PrintSectionClose()
    print("Time elapsed for refocusing: {:3f}".format(cpi.Time()-start))   

