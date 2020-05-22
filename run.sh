#!/usr/bin/env bash

# @brief  Music Genre Classification
#         This scripts is to genereate f1-score with two algorithms.
# @author Geunsik Lim <leemgs@gmail.com>
# @date   May-22-2020
# @Note
#  - Experimental environment: Ubuntu 18.04 LTS (x86_64), Anaconda (=conda) 20200210, Python 2.7.18 
#  - Genres: classical, hiphop, jazz, metal, pop, rock
#  - Features used: FFT (Fast Fourier Transform), MFCC(Mel-Frequency Cepstral Coefficients)
#  - Classifier: Logistic Regression Classifier, KNeighbors Classifier
#
INSTALL_PACKS=0

# Install required python packages

if [[ $INSTALL_PACKS -eq 1 ]]; then
    conda install -c anaconda scikit-learn 
    conda install -c r r-cvst
    conda install matplotlib
fi

echo -e "Initializing the .wav files from genres.backup folder..."
rm -rf ./genres.FFT
rm -rf ./genres.MFCC
cp -arfp genres.backup/ genres.FFT
cp -arfp genres.backup/ genres.MFCC

echo -e "Initializing the .wav files from genres.backup folder..."
python extract-features-FFT.py  /work2/aiclass/final/music-genre-classification/genres.FFT/classical/  /work2/aiclass/final/music-genre-classification/genres.FFT/hiphop/  /work2/aiclass/final/music-genre-classification/genres.FFT/jazz/  /work2/aiclass/final/music-genre-classification/genres.FFT/metal/  /work2/aiclass/final/music-genre-classification/genres.FFT/pop/  /work2/aiclass/final/music-genre-classification/genres.FFT/rock/  
if [[ $? -ne 0 ]]; then
    echo -e "Oooops. The task (FFT) is failed. Please fix this issue."
    exit 1
fi

python extract-features-MFCC.py  /work2/aiclass/final/music-genre-classification/genres.MFCC/classical/  /work2/aiclass/final/music-genre-classification/genres.MFCC/hiphop/  /work2/aiclass/final/music-genre-classification/genres.MFCC/jazz/  /work2/aiclass/final/music-genre-classification/genres.MFCC/metal/  /work2/aiclass/final/music-genre-classification/genres.MFCC/pop/  /work2/aiclass/final/music-genre-classification/genres.MFCC/rock/  
if [[ $? -ne 0 ]]; then
    echo -e "Oooops. The task (MFCC) is failed. Please fix this issue."
    exit 1
fi


echo -e ""
echo -e ""
echo -e "All preparation is finished. Now, run the below statement. !!!!"
echo -e "Method1: ubuntu18.04$ python train-classify.py  genres.FFT/ genres.MFCC/"
echo -e "Method2: ubuntu18.04$ jupyter-notebook train-classify.ipynb"