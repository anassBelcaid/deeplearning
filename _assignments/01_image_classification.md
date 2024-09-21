---
type: assignment
date: 2024-10-08T4:00:00+4:30
title: 'Assignment #1 - Image Classification using ML and Two-Layer Neural nets'
# pdf: /assign1/
attachment: /assignments/assignment1.zip
# solutions: /static_files/assignments/asg_solutions.pdf
due_event: 
    type: due
    date: 2024-10-31T23:59:00+3:30
    description: 'Assignment #1 due'
---

In this assignement you will implement several **Image Classification** models. You will use **cross-validation** to best choose the **hyper-parameters** for each model.

---

# Setup

Once you downloaded the attachement. You could work either **locally** or in a [google colab](https://colab.research.google.com/).

## Wokring with google colab.

If you choose to work with **google colab**, you need to adapt the first cell to point to your **google drive** folder. 

1. First Upload the assignement to your **drive** and **unzip-it**.
2. You need to adapt the first **cell** to point the folder that you created.
3. Let's say that we saved our folder as `/Courses/DeepLearning/assignment1/`, then you need to update the `FOLDERNAME` variable like.


```python
# This mounts your Google Drive to the Colab VM.
from google.colab import drive
drive.mount('/content/drive')

# TODO: Enter the foldername in your Drive where you have saved the unzipped
# assignment folder, e.g. 'cs321/assignments/assignment1/'
FOLDERNAME = 'Courses/DeepLearning/assignment1'
assert FOLDERNAME is not None, "[!] Enter the foldername."

# Now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# This downloads the CIFAR-10 dataset to your Drive
# if it doesn't already exist.
%cd /content/drive/My\ Drive/$FOLDERNAME/cs231/datasets/
!bash get_datasets.sh
%cd /content/drive/My\ Drive/$FOLDERNAME

```

## Working locally.

You could also work locally using your **conda** setup. In this case, you don't need to setup any **FOlDERNAME** since it will be in the **current folder**. So your first cell will be as:

```python
# This downloads the CIFAR-10 dataset to your Drive
# if it doesn't already exist.
%cd cs231/datasets/
!bash get_datasets.sh
%cd ../../
```
This will help download the **FIFAR-10** Data set

---

## Q1: k-Nearest Neighbor classifier

The notebook `knn.ipynb` will walk you through implementing the kNN classifier.

---
## Q2: Training a Support Vector Machine

The notebook `svm.ipynb` will walk you through implementing the SVM classifier.

---
## Q3: Implement a Softmax classifier
The notebook `softmax.ipynb` will walk you through implementing the Softmax classifier.

---
## Q4: Two-Layer Neural Network
The notebook `two_layer_net.ipynb` will walk you through the implementation of a two-layer neural network classifier.

---
## Q5: Higher Level Representations: Image Features

The notebook `features.ipynb` will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.
