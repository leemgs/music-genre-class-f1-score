# Welcome to my music genre classification project

I established a below system environment for development:
  * Ubuntu 18.04 (LTS x86_64)
  * Anaconda3 20200210
  * Python 3.6

This repository consists of development code that classifies music genre according to the following six genres: 
* Classical, Hiphop, Jazz, Metal, Pop, and Rock


### Features used: 
* FFT (Fast Fourier Transform): Classification accuracy can be possible until 50%.
* MFCC (Mel-Frequency Cepstral Coefficients): Classification accuracy can be possible until 75%.


### Choice of classifier:
* Classical Machine Learning:
  * Logistic Regression Classifier (Logistic)
  * KNeighbors Classifier (KNN)
  * Support Vector Machine (SVM)
* Deep Learning:
  * VGG-16
  * Desenet
  * Efficientnet

## How to use:

* Install (Ana)conda environment on Ubuntu 18.04 LTS (x86-64)
```bash
invain@u1804$ curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
invain@u1804$ bash Anaconda3-2020.02-Linux-x86_64.sh
invain@u1804$ source ~/.bashrc
invain@u1804$ conda create -n python27 python=2.7
invain@u1804$ conda info --env
invain@u1804$ conda activate python27
(python27)$ conda install jupyter notebook
(python27)$ conda upgrade ipykernel
(python27)$ jupyter-notebook --debug

```

* Download a dataset for a training: 
  * Get the dataset from https://canvas.skku.edu - SFC5015_41 - Week06 (exec-06-music-class.zip)
    * You can also get the dataset (*.wav) at http://opihi.cs.uvic.ca/sound/genres.tar.gz
  * Extract into suitable directory: WORK_DIR (e.g., /work2/final/music-genre-class-f1-score/genres.backup/)

* Execute the "**run.sh**" file to do a feature task with FFT & MFCC:
  * Run the "extract-features-FFT.py" file on each dataset sub-directory of WORK_DIR.
  * Run the "extract-features-MFCC.py" file on each dataset sub-directory of WORK_DIR.

* Train and Classify with the "**"train-classify.sh"**" file
  * Alternatively, you can run train-classify.ipynb if you prefer a Jupyter-notebook program (.ipynb) to a consolw program (.py).
  * Then, calculate a precision, recall, and F1-score for performance evaluation.


## Evaluation
The experimental result is as follows. 

```bash
******USING FFT******
(python27) invain@u1804:.../final/music-genre-class-f1-score$ ./train-classify.sh
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
logistic accuracy = 0.7333333333333334
logistic_cm:
[[24  0  2  1  0  1]
 [ 0 16  3  3  1  2]
 [ 0  5  9  1  3  0]
 [ 0  1  1 25  0  0]
 [ 0  2  4  2 13  1]
 [ 0  3  1  0  3 23]]
######## [F1-SCORE] CLASSIFICATION REPORT with Logistic Regression ########
              precision    recall  f1-score   support

   classical       1.00      0.86      0.92        28
      hiphop       0.59      0.64      0.62        25
        jazz       0.45      0.50      0.47        18
       metal       0.78      0.93      0.85        27
         pop       0.65      0.59      0.62        22
        rock       0.85      0.77      0.81        30

   micro avg       0.73      0.73      0.73       150
   macro avg       0.72      0.71      0.71       150
weighted avg       0.75      0.73      0.74       150

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

```
 
## Reference

* [Building Machine Learning Systems with Python](http://totoharyanto.staff.ipb.ac.id/files/2012/10/Building-Machine-Learning-Systems-with-Python-Richert-Coelho.pdf)
  * ISBN 978-1-78216-140-0
  * Author: Willi Richert and Luis Pedro Coelho
  * Publisher: PACKT Publishing (www.packpub.com)
  * Date: 2013
  * Pages: 290

* [Music genre classification using machine learning technique](https://www.groundai.com/project/music-genre-classification-using-machine-learning-techniques/1)
  * Date: 2018.04.03
  * Network model: VGG-16


* [Music genre classification with CNN](https://github.com/Ritesh313/Music-genre-classificartion/tree/master/MusicGenre)
  * Date: 2020.01.12
  * Network model: Densenet, Efficientnet
