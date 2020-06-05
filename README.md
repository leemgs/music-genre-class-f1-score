# Welcome to my music genre classification project for final exam on 산업인공지능(SFC5015_41).

I set-up a below system environment to develop a music genres classfication software:
* https://github.com/leemgs/music-genre-class-f1-score (GitHub Repository)
 * Ubuntu 18.04 (LTS x86_64)
 * Anaconda3 (20200210)
   * Jupyter 1.0
   * Python 3.6
   * Tensorflow 2.0.1
   * Keras 2.3.1


This repository consists of development code that classifies music genre according to the following six genres: 
* Dataset: (1)Classical, (2)Hiphop, (3)Jazz, (4)Metal, (5)Pop, and (6)Rock


### Features used: 
* FFT (Fast Fourier Transform):
  * F1-score: ~ 50%.

* MFCC (Mel-Frequency Cepstral Coefficients):
  * F1-score: ~ **78%**.


### Choice of classifier:
* Classical Machine Learning:
  * Logistic Regression Classifier (Logistic)
  * KNeighbors Classifier (KNN)
  * Support Vector Machine (SVM)
* Deep Learning: (**Caution**: Use lots of Nvidia GPUs because a training time is too long.)
  * FFNN
    * F1-score: ~ 71%.
  * CNN
    * F1-score: ~ **85%**.
  * Pre-trained network: Desenet, Efficientnet

## How to use:


#### Set-up Evaluation Environment
* Install (Ana)conda environment on Ubuntu 18.04 LTS (x86-64)
```bash
invain@u1804$ curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
invain@u1804$ bash Anaconda3-2020.02-Linux-x86_64.sh
invain@u1804$ source ~/.bashrc
invain@u1804$ conda create -n python36 python=3.6
invain@u1804$ conda info --env
invain@u1804$ conda activate python36
(python36)$ conda install jupyter notebook
(python36)$ jupyter-notebook --debug

```

#### Clasical Machine Learning (./svm/ folder):
* Download a dataset for a training: 
  * Get the dataset from [https://canvas.skku.edu](https://canvas.skku.edu) - **SFC5015_41** - Week06 (exec-06-music-class.zip)
    * You can also get the GTZAN dataset (*.wav) at http://opihi.cs.uvic.ca/sound/genres.tar.gz
  * Extract into a suitable directory: WORK_DIR (e.g., /work2/final/music-genre-class-f1-score/genres.backup/)

* Execute the "**run.sh**" file to do a feature task with FFT & MFCC:
  * Run the "extract-features-FFT.py" file on each dataset sub-directory of WORK_DIR.
  * Run the "extract-features-MFCC.py" file on each dataset sub-directory of WORK_DIR.

* Jupyter Notebooks
  * Train and Classify with the "**"train-classify.ipynb"**" file
  * Run train-classify.ipynb with an interactive Jupyter-notebook software.
  * Then, calculate a precision, recall, and F1-score for performance evaluation.

* Confusion Matrix

![Confusion Matrix for SVM](./image/confusion-matrix-ml.png)

* F1-score of SVM
```bash
           precision    recall  f1-score   support

   classical       0.92      1.00      0.96        24
      hiphop       0.61      0.79      0.69        24
        jazz       0.78      0.50      0.61        28
         pop       0.90      0.86      0.88        21
        rock       0.71      0.81      0.76        27
       metal       0.83      0.77      0.80        26

   micro avg       0.78      0.78      0.78       150
   macro avg       0.79      0.79      0.78       150
weighted avg       0.79      0.78      0.78       150

```
 
#### Deep Learning (./cnn/ folder):

* Download a dataset for a training:
**IMPORTANT NOTE:** the data was too large to store on GitHub.
After downloading the data (*.wav), I combined all of the files into a single folder called "wavfiles." 
For the code to run, this folder must be placed in the empty "data" folder. 
All of the csv files are also set to be exported to this folder.
 
* Jupyter Notebooks
  * Data Gathering
  * Data Cleaning
  * Exploratory Data Analysis
  * Convolutional Neural Network
 

* Confusion Matrix

![Confusion Matrix for CNN](./image/confusion-matrix-dl.png)

* F1-score of SVM
```bash
              precision    recall  f1-score   support

   classical       0.84      0.80      0.82        20
      hiphop       0.83      0.95      0.88        20
        jazz       0.77      0.85      0.81        20
       metal       1.00      0.95      0.97        20
         pop       0.88      0.70      0.78        20
        rock       0.81      0.85      0.83        20

    accuracy                           0.85       120
   macro avg       0.85      0.85      0.85       120
weighted avg       0.85      0.85      0.85       120
```



## Reference

* [Building Machine Learning Systems with Python](http://totoharyanto.staff.ipb.ac.id/files/2012/10/Building-Machine-Learning-Systems-with-Python-Richert-Coelho.pdf)
  * ISBN 978-1-78216-140-0 (290 pages)
  * Author: Willi Richert and Luis Pedro Coelho
  * Publisher: PACKT Publishing (www.packpub.com)
  * Date: Feb-15-2013

  
* [Music Genre Classification using Masked Conditional Neural Networks](https://paperswithcode.com/paper/music-genre-classification-using-masked)
  * Date: Feb-18-2018 
  * Author: Fady Medhat, David Chesmore, John Robinson
  * Network model: MCNN


* [Music genre classification using machine learning technique](https://www.groundai.com/project/music-genre-classification-using-machine-learning-techniques/1)
  * Date: Apr-03-2018
  * Network model: VGG-16


* [Music genre classification with CNN](https://github.com/Ritesh313/Music-genre-classificartion/tree/master/MusicGenre)
  * Date: Jan-12-2020
  * Network model: Densenet, Efficientnet
    

  
