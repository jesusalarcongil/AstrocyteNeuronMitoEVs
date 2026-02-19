#!/usr/bin/env python
# coding: utf-8

# MITOCHONDRIAL NETWORK ANALYSIS PIPELINE

# 1. Introduce variables

import subprocess
import os
import re
import shutil
import tifffile as tiff
from tifffile import imread, imwrite, imsave
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig
from tqdm.notebook import trange
import tifffile
from skimage import io, filters, morphology, measure
from skimage.filters import threshold_otsu, sobel
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes, gaussian_filter, binary_erosion, generate_binary_structure
import mitotnt
import gc

RawNd2Path = "/home/jesusalarcon/Analysis/RawNd2Data"
NChannels = 1
TimeFrameInterval = 2   
LateralPixelSize = 111
ZDistance = 200
PSFZslices = 21
NumAperture = 1.42
RefractIndex = 1.515
NCores = 15
NIter = 20
MitoChannel = 583


TresholdMG = 500 #NLGF1, NLGF2, WT5, WT7
#TresholdMG = 600 #WT5
#TresholdMG = 800 #NLGF3, #NLGF5, #NLGF6
#TresholdMG = 900 #WT1, #WT3, #WT4
#TresholdMG = 1200 #NLGF7
#TresholdMG = 1300 #WT2
#TresholdMG = 1800 #NLGF4
#TresholdMG = 3200 #WT6


# 2. Convert nd2 to tiff

# 2.1. Create tiff folder

RawNd2PathIndex =RawNd2Path.rfind('/')
RawTiffData = RawNd2Path[:(RawNd2PathIndex+1)] + "RawTiffData"
#os.makedirs(RawTiffData)


# 2.2 Create nd2 and tiff paths

RawNd2Files = os.listdir(RawNd2Path)
RawNd2FilePaths = []

for i in range(len(RawNd2Files)):
    NameFileIndex = RawNd2Files[i].rfind('.')
    NameFile = RawNd2Files[i][:NameFileIndex]
    RawTiffFilePath = RawTiffData + "/" + str(NameFile)
    os.makedirs(RawTiffFilePath)
    RawNd2FilePath = RawNd2Path + "/" + str(RawNd2Files[i])
    RawNd2FilePaths.append(RawNd2FilePath)

RawTiffFiles = []

for file in RawNd2Files:
    TiffFile = file.replace(".nd2", "_T%t.tif")
    RawTiffFiles.append(TiffFile)

RawTiffFilePaths = []
RawTiffFolders = os.listdir(RawTiffData)

for i in range(len(RawTiffFiles)):
    for z in range(len(RawTiffFolders)):
        SameFileFolder = re.search(RawTiffFolders[z], RawTiffFiles[i])
        if SameFileFolder != None:
            RawTiffFilePath = RawTiffData + "/" + str(RawTiffFolders[z]) + "/" + str(RawTiffFiles[i])
            RawTiffFilePaths.append(RawTiffFilePath)


# 2.3 Convert the files

for i in range(len(RawNd2FilePaths)):
    for z in range(len(RawTiffFilePaths)):
        NameFolderIndex2 = RawTiffFilePaths[z].rfind('/')
        NameFolderIndex1 = RawTiffFilePaths[z][:NameFolderIndex2].rfind('/')
        NameFolder = RawTiffFilePaths[z][NameFolderIndex1+1:NameFolderIndex2]
        SameFileFolder = re.search(NameFolder, RawNd2FilePaths[i])
        if SameFileFolder != None:
            nd22tiff = ["bash", "-c", "bfconvert " + str(RawNd2FilePaths[i]) + str(" ") + str(RawTiffFilePaths[z])]
            subprocess.run(nd22tiff)


# 3. Deconvolution with Deconwolf

# 3.1. Create deconvolution folders

RawNd2PathIndex = RawNd2Path.rfind('/')
DeconTiffData = RawNd2Path[:(RawNd2PathIndex+1)] + "DeconTiffData"
os.makedirs(DeconTiffData)

EmWavesDeconTiffPaths = []

for i in range(len(RawNd2Files)):
    NameFileIndex = RawNd2Files[i].rfind('.')
    NameFile = RawNd2Files[i][:NameFileIndex]
    DeconTiffFilePath = DeconTiffData + "/" + str(NameFile)
    os.makedirs(DeconTiffFilePath)
    DeconTiffChannelData = DeconTiffFilePath + "/" + str(MitoChannel)
    os.makedirs(DeconTiffChannelData)
    EmWavesDeconTiffPaths.append(DeconTiffChannelData)


# 3.2 Create PSF

PSFData = RawNd2Path[:(RawNd2PathIndex+1)] + "PSFData"
os.makedirs(PSFData)
RawTiffInputFolder = os.listdir(RawTiffData)

for channel in range(NChannels):
    PSFFile = PSFData + "/" + str(MitoChannel) + "PSF.tiff"
    createPSF = ["dw_bw", "--resxy", str(LateralPixelSize), "--resz", str(ZDistance), "--lambda", str(MitoChannel), "--NA", 
                                                                                                  str(NumAperture), "--ni",
                                                                                                  str(RefractIndex), "--nslice",
                                                                                                  str(PSFZslices), PSFFile]
    subprocess.run(createPSF)


# 3.3 Deconvolute tiff files

RawTiffInputFolder = os.listdir(RawTiffData)

for y in range(len(RawTiffInputFolder)):
    RawTiffInputFiles = os.listdir(RawTiffData + "/" + RawTiffInputFolder[y])
    for i in range(len(RawTiffInputFiles)):
        DeconTiffChannelOutput = DeconTiffData + "/" + RawTiffInputFolder[y] + "/"  + str(MitoChannel) + "/" + RawTiffInputFiles[i]
        DeconTiffChannelInput = RawTiffData + "/" + RawTiffInputFolder[y] + "/" + RawTiffInputFiles[i]
        PSFFileInput = PSFData + "/" + str(MitoChannel) + "PSF.tiff"
        createDeconTiffFiles = ["dw", "--threads", str(NCores), "--iter", str(NIter), "--out", DeconTiffChannelOutput, DeconTiffChannelInput, PSFFileInput, "--gpu"]
        subprocess.run(createDeconTiffFiles)


# 3.4 Move the log files to another folder

LogFile = '.log.txt'
DeconLogFolder = RawNd2Path[:(RawNd2PathIndex+1)] + "/" + "DeconLog"
os.makedirs(DeconLogFolder)

DeconTiffInputFolder = os.listdir(DeconTiffData)

for y in range(len(DeconTiffInputFolder)):
    DeconTiffInputFiles = os.listdir(DeconTiffData + "/" + DeconTiffInputFolder[y])
    DeconTiffFolder = DeconTiffData + "/" + DeconTiffInputFolder[y] + "/" + str(MitoChannel)
    DeconTiffFiles = os.listdir(DeconTiffFolder)
    for x in range(len(DeconTiffFiles)):
        DeconLogFile = re.search(LogFile, DeconTiffFiles[x])
        if DeconLogFile != None:
            DeconLogPath = DeconTiffFolder + "/" + DeconTiffFiles[x]
            DeconLogFolderPath = DeconLogFolder + "/" + DeconTiffFiles[x]
            shutil.move(DeconLogPath, DeconLogFolder)


# 4. MitoTNT

# 4.1 Create MitoTNT input folder

DeconTiffData = RawNd2Path[:(RawNd2PathIndex+1)] + "DeconTiffData"
DeconTiffInputFolder = os.listdir(DeconTiffData)
RawNd2Files = os.listdir(RawNd2Path)
MitoGraphFolder = RawNd2Path[:(RawNd2PathIndex+1)] + "MitoGraphData"
#os.makedirs(MitoGraphFolder)

RawNd2Files = ["WT_astro_4mn_2um_0.5FPS_n7.nd2"]

for i in range(len(RawNd2Files)):
    NameFileIndex = RawNd2Files[i].rfind('.')
    NameFile = RawNd2Files[i][:NameFileIndex]
    MitoGraphFilePath = MitoGraphFolder + "/" + str(NameFile) + "/" + str(NameFile)
    os.makedirs(MitoGraphFilePath)

MitoGraphFolders = os.listdir(MitoGraphFolder)

MitoGraphFolders = ["WT_astro_4mn_2um_0.5FPS_n7"]

for z in range(len(MitoGraphFolders)):
    DeconTiffMitoFolder = DeconTiffData + "/" + str(MitoGraphFolders[z]) + "/" + str(MitoChannel)
    DeconTiffMitoFiles = os.listdir(DeconTiffMitoFolder)
    for i in range(len(DeconTiffMitoFiles)):
        DeconTiffMitoFileParts = DeconTiffMitoFiles[i].split('.tif')
        DeconTiffMitoFileIndex = DeconTiffMitoFileParts[0].rfind('T')
        DeconTiffMitoFileFrame = DeconTiffMitoFileParts[0][DeconTiffMitoFileIndex+1:len(DeconTiffMitoFileParts[0])]
        if int(DeconTiffMitoFileFrame) < 60:
            MitoGraphTimeFolder = MitoGraphFolder + "/" + str(MitoGraphFolders[z]) + "/" + str(MitoGraphFolders[z]) + "/" + "frame_" + str(DeconTiffMitoFileFrame)
            os.makedirs(MitoGraphTimeFolder)

DeconMitoInputFolder = os.listdir(DeconTiffData)

DeconMitoInputFolder = ["WT_astro_4mn_2um_0.5FPS_n7"]
DeconTiffInputFolder = ["WT_astro_4mn_2um_0.5FPS_n7"]

for y in range(len(DeconMitoInputFolder)):
    DeconMitoInputPath = DeconTiffData + "/" + str(DeconTiffInputFolder[y]) + "/" + str(MitoChannel)
    DeconMitoInputData = os.listdir(DeconMitoInputPath)
    for i in range(len(DeconMitoInputData)):
        DeconTiffMitoFilePath = DeconMitoInputPath + "/" + DeconMitoInputData[i]
        DeconTiffMitoStack = imread(DeconTiffMitoFilePath)

        DeconTiffMitoStackNorm2 = DeconTiffMitoStack / TresholdMG
        DeconTiffMitoStackNorm2[DeconTiffMitoStackNorm2>1.0] = 255
        #DeconTiffMitoStackNorm3 = DeconTiffMitoStackNorm2 * 255

        DeconTiffMitoStackNormSelec = DeconTiffMitoStackNorm2[:, 0:1024, 0:1024]

        DeconTiffMitoStackNormSelec = DeconTiffMitoStackNormSelec.astype(np.uint8)

        MitoGraphFilesFolders = MitoGraphFolder + "/" + str(DeconTiffInputFolder[y]) + "/" + str(DeconTiffInputFolder[y])
        MitoGraphFrameFolders = os.listdir(MitoGraphFilesFolders)

        for z in range(len(MitoGraphFrameFolders)):
            DeconTiffMitoFileParts = DeconMitoInputData[i].split('.tif')
            DeconTiffMitoFileIndex = DeconTiffMitoFileParts[0].rfind('T')
            DeconTiffMitoFileFrame = DeconTiffMitoFileParts[0][DeconTiffMitoFileIndex+1:len(DeconTiffMitoFileParts[0])]
            DeconTiffMitoFileCompleteFrame = "frame_" + str(DeconTiffMitoFileFrame)
            if int(DeconTiffMitoFileFrame) < 60:
                if DeconTiffMitoFileCompleteFrame == MitoGraphFrameFolders[z]:
                    SaveMitoGraphFileFolder = MitoGraphFolder + "/" + str(DeconTiffInputFolder[y]) + "/" + str(DeconTiffInputFolder[y]) + "/" + str(MitoGraphFrameFolders[z]) + "/" + DeconTiffMitoFileCompleteFrame + ".tif"
                    imwrite(SaveMitoGraphFileFolder, DeconTiffMitoStackNormSelec)


# 4.2. MitoTNT processing

MitoGraphFolder = RawNd2Path[:(RawNd2PathIndex+1)] + "MitoGraphData"
MitoGraphFolderSamples = os.listdir(MitoGraphFolder)
xysize = LateralPixelSize/1000
zsize = ZDistance/1000
mitopattern = "frame"    

for x in range(len(MitoGraphFolderSamples)):
    MitoGraphFile2AnalyzePath = str(MitoGraphFolder) + "/" + str(MitoGraphFolderSamples[x]) + "/" + str(MitoGraphFolderSamples[x])
    seg = mitotnt.MitoSegmenter(MitoGraphFile2AnalyzePath, mitopattern, xysize, zsize)
    seg.run_mitograph_cpu(max_workers=NCores, extra_params = " -analyze")
    seg.visualize_segmented_data()

MitoGraphFolder = RawNd2Path[:(RawNd2PathIndex+1)] + "MitoGraphData"
MitoGraphFolderSamples = os.listdir(MitoGraphFolder)
xysize = LateralPixelSize/1000
zsize = ZDistance/1000
mitopattern = "frame"    

for x in range(len(MitoGraphFolderSamples)):
    MitoGraphFile2AnalyzePath = str(MitoGraphFolder) + "/" + str(MitoGraphFolderSamples[x]) + "/" + str(MitoGraphFolderSamples[x])
    seg = mitotnt.MitoSegmenter(MitoGraphFile2AnalyzePath, mitopattern, xysize, zsize)
    seg.run_mitograph_cpu(max_workers=NCores, extra_params = " -analyze")
    seg.visualize_segmented_data()
    skel = mitotnt.SkeletonizedMito(seg)
    skel.extract_graphs(node_gap_size = 2)

MitoGraphFolder = RawNd2Path[:(RawNd2PathIndex+1)] + "MitoGraphData"
MitoGraphFolderSamples = os.listdir(MitoGraphFolder)
xysize = LateralPixelSize/1000
zsize = ZDistance/1000
mitopattern = "frame"    

for x in range(len(MitoGraphFolderSamples)):
    MitoGraphFile2AnalyzePath = str(MitoGraphFolder) + "/" + str(MitoGraphFolderSamples[x]) + "/" + str(MitoGraphFolderSamples[x])
    seg = mitotnt.MitoSegmenter(MitoGraphFile2AnalyzePath, mitopattern, xysize, zsize)
    seg.run_mitograph_cpu(max_workers=NCores, extra_params = " -analyze")
    seg.visualize_segmented_data()
    skel = mitotnt.SkeletonizedMito(seg)
    skel.extract_graphs(node_gap_size = 2)
    del seg
    tracker = mitotnt.NetworkTracker(skel, frame_interval=TimeFrameInterval, min_track_size = 2, max_gap_size = 1)
    tracked = tracker.run()
    del skel
    viz = mitotnt.Visualizer(tracked)
    viz.generate_chimerax_skeleton()
    viz.generate_tracking_arrows()
    viz.visualize_tracking()
    tracked.detect_fusion_fission()
    tracked.compute_mitochondria_diffusivity()
    del tracker, tracked, viz
    gc.collect()





