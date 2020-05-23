#!/usr/bin/env bash

# @brief  Music Genre Classification
#         This scripts is to genereate f1-score with two algorithms.
# @author Geunsik Lim <leemgs@gmail.com>
# @date   May-22-2020
# @Note
#  - Experimental environment: Ubuntu 18.04 LTS (x86_64), Anaconda (=conda) 20200210, Python 3.6
#  - Genres: classical, hiphop, jazz, metal, pop, rock
#  - Features used: FFT (Fast Fourier Transform), MFCC(Mel-Frequency Cepstral Coefficients)
#  - Classifier: Logistic Regression Classifier, KNeighbors Classifier
#
INSTALL_PACKS=1

# Install required python packages

if [[ $INSTALL_PACKS -eq 1 ]]; then
    conda install -y -c conda-forge scipy
    conda install -y -c contango python_speech_features

    conda install -y -c anaconda scikit-learn 
    conda install -y -c r r-cvst
    conda install -y matplotlib
fi

echo -e "Initializing the .wav files from genres.backup folder..."
rm -rf ./genres.FFT
rm -rf ./genres.MFCC
cp -arfp genres.backup/ genres.FFT
cp -arfp genres.backup/ genres.MFCC

CURRENT_DIR=`pwd`
echo -e "Creating a genres.FFT folder from genres.backup folder..."
python3 extract-features-FFT.py  ${CURRENT_DIR}/genres.FFT/classical/  ${CURRENT_DIR}/genres.FFT/hiphop/  ${CURRENT_DIR}/genres.FFT/jazz/  ${CURRENT_DIR}/genres.FFT/metal/  ${CURRENT_DIR}/genres.FFT/pop/  ${CURRENT_DIR}/genres.FFT/rock/  
if [[ $? -ne 0 ]]; then
    echo -e "Oooops. The task (FFT) is failed. Please fix this issue."
    exit 1
fi

echo -e "Creating a genres.MFCC folder from genres.backup folder..."
python3 extract-features-MFCC.py  ${CURRENT_DIR}/genres.MFCC/classical/  ${CURRENT_DIR}/genres.MFCC/hiphop/  ${CURRENT_DIR}/genres.MFCC/jazz/  ${CURRENT_DIR}/genres.MFCC/metal/  ${CURRENT_DIR}/genres.MFCC/pop/  ${CURRENT_DIR}/genres.MFCC/rock/  
if [[ $? -ne 0 ]]; then
    echo -e "Oooops. The task (MFCC) is failed. Please fix this issue."
    exit 1
fi


echo -e ""
echo -e ""
echo -e "All preparation is finished. Now, run the below statement. !!!!"
echo -e "Method1: ubuntu18.04$ python train-classify.py  genres.FFT/ genres.MFCC/"
echo -e "Method2: ubuntu18.04$ jupyter-notebook train-classify.ipynb"
