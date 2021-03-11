# PC-GAIN: Pseudo-label Conditional Generative Adversarial Imputation Networks for Incomplete Data

PC-GAIN utilizes potential category information to further enhance the imputation power. Specifically, we first propose a pre-training
procedure to learn potential category information contained in a subset of low-missing-
rate data. Then an auxiliary classifier is determined based on the synthetic pseudo-
labels. Further, this classifier is incorporated into the generative adversarial framework
to help the generator to yield higher quality imputation results. 

This Tensorflow code implements the model and reproduces the results
from the paper.

## Setup
This code was tested on Windows,
Python 3.6.4, tensorflow-gpu 2.0.0,
cuda_10.0.130_411.31_win10 and 
cudnn-10.0-windows10-x64-v7.4.2.24.

You can run this code on `credit`, `letter` and `news` in the directory `dataset/`.

## Experiments

## Missing Feature Imputation

To impute missing features with PC-GAIN one can use following commands:
```
python main.py
```
After that you will get 5 results through the 5-cross validations off experiment.