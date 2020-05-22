# Welcome to my music genre classification project

I established a below system environment for development:
  * Ubuntu 18.04 (LTS x86_64)
  * Anaconda3 20200210
  * Python 2.7 (+ sklearn, numpy, plot, ...)

This repository consists of development code that classifies music genre according to the following six genres: 
* Classical
* Hiphop
* Jazz
* Metal
* Pop
* Rock


### Features used: 
* FFT (Fast Fourier Transform)
  * Classification accuracy: ~50%

* MFCC (Mel-Frequency Cepstral Coefficients)
  * Classification accuracy: ~73%


### Choice of classifier:

* Logistic Regression Classifier

* KNeighbors Classifier


## How to use:

* Install (Ana)conda environment on Ubuntu 18.04 LTS (x86-64)
```bash
$ curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
$ bash Anaconda3-2020.02-Linux-x86_64.sh
$ source ~/.bashrc
$ conda create -n python27 python=2.7
$ conda info --env
$ conda activate python27
(python27)$ conda install jupyter notebook
(python27)$ conda upgrade ipykernel
(python27)$ jupyter-notebook --debug

```

* Download a dataset for a training: 
  * Get the dataset from https://canvas.skku.edu - SFC5015_41 - Week06 (exec-06-music-class.zip)
  * Extract into suitable directory: WORK_DIR (e.g., /work2/final/music-genre-class-f1-score/genres.backup/)

* Execute the "run.sh" file to do a feature task with FFT & MFCC:
  * Run the "extract-features-FFT.py" file on each dataset sub-directory of WORK_DIR.
  * Run the "extract-features-MFCC.py" file on each dataset sub-directory of WORK_DIR.

* Train and Classify!!! Then, calculate a precision, recall, and F1-score for performance evaluation:
  * Run the "train-classify.sh" file (or run train-classify.py with Jupyter-notebook program).

## Screenshot
The experimental result is as follows. 

```bash
******USING FFT******
/var/www/invain/anaconda3/envs/python27/lib/python2.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/var/www/invain/anaconda3/envs/python27/lib/python2.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
logistic accuracy = 0.46
logistic_cm:
[[17  2  4  0  2  0]
 [ 4  9  0  8  1  2]
 [ 8  2 10  6  3  1]
 [ 1  8  3 15  2  0]
 [ 0  4  1  7  7  4]
 [ 1  3  0  0  4 11]]
######## [F1-SCORE] CLASSIFICATION REPORT with Logistic Regression ########
              precision    recall  f1-score   support

   classical       0.55      0.68      0.61        25
      hiphop       0.32      0.38      0.35        24
        jazz       0.56      0.33      0.42        30
       metal       0.42      0.52      0.46        29
         pop       0.37      0.30      0.33        23
        rock       0.61      0.58      0.59        19

   micro avg       0.46      0.46      0.46       150
   macro avg       0.47      0.46      0.46       150
weighted avg       0.47      0.46      0.46       150

knn accuracy = 0.43333333333333335
knn_cm:
[[23  0  1  0  0  1]
 [ 4 11  2  3  0  4]
 [18  0  8  2  1  1]
 [ 4 11  2  8  1  3]
 [ 3  3  2  0  5 10]
 [ 1  5  0  0  3 10]]
######## [F1-SCORE] CLASSIFICATION REPORT with KNeighbors Classifier ########
              precision    recall  f1-score   support

   classical       0.55      0.68      0.61        25
      hiphop       0.32      0.38      0.35        24
        jazz       0.56      0.33      0.42        30
       metal       0.42      0.52      0.46        29
         pop       0.37      0.30      0.33        23
        rock       0.61      0.58      0.59        19

   micro avg       0.46      0.46      0.46       150
   macro avg       0.47      0.46      0.46       150
weighted avg       0.47      0.46      0.46       150

*********************

******USING MFCC******
logistic accuracy = 0.6933333333333334
logistic_cm:
[[25  0  1  0  0  0]
 [ 0 13  0  4  2  3]
 [ 3  4 16  1  2  2]
 [ 0  0  0 25  0  0]
 [ 0  8  7  2  7  4]
 [ 0  1  0  0  2 18]]
######## [F1-SCORE] CLASSIFICATION REPORT with Logistic Regression ########
              precision    recall  f1-score   support

   classical       0.89      0.96      0.93        26
      hiphop       0.50      0.59      0.54        22
        jazz       0.67      0.57      0.62        28
       metal       0.78      1.00      0.88        25
         pop       0.54      0.25      0.34        28
        rock       0.67      0.86      0.75        21

   micro avg       0.69      0.69      0.69       150
   macro avg       0.67      0.71      0.68       150
weighted avg       0.68      0.69      0.67       150

knn accuracy = 0.66
knn_cm:
[[24  0  1  0  1  0]
 [ 0 10  3  5  2  2]
 [ 7  4 13  1  3  0]
 [ 0  3  0 20  2  0]
 [ 1  3  5  0 18  1]
 [ 0  3  0  0  4 14]]
######## [F1-SCORE] CLASSIFICATION REPORT with KNeighbors Classifier ########
              precision    recall  f1-score   support

   classical       0.89      0.96      0.93        26
      hiphop       0.50      0.59      0.54        22
        jazz       0.67      0.57      0.62        28
       metal       0.78      1.00      0.88        25
         pop       0.54      0.25      0.34        28
        rock       0.67      0.86      0.75        21

   micro avg       0.69      0.69      0.69       150
   macro avg       0.67      0.71      0.68       150
weighted avg       0.68      0.69      0.67       150

*********************
(python27) invain@u1804:.../final/music-genre-class-f1-score$
```

## Reference

* [Building Machine Learning Systems with Python](http://totoharyanto.staff.ipb.ac.id/files/2012/10/Building-Machine-Learning-Systems-with-Python-Richert-Coelho.pdf)
  * ISBN 978-1-78216-140-0
  * Author: Willi Richert and Luis Pedro Coelho
  * Publisher: PACKT Publishing (www.packpub.com)
  * Date: 2013
  * Pages: 290

* Music genre classification using machine learning technique
  * Date: 2018.04.03
  * Network model: VGG-16
  * https://www.groundai.com/project/music-genre-classification-using-machine-learning-techniques/1

* Music genre classification with CNN
  * Date: 2020.01.12
  * Network model: Densenet,Efficientnet
  * https://github.com/Ritesh313/Music-genre-classificartion/tree/master/MusicGenre
