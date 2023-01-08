#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data


""" Build the histogram of the years of the songs from the training set and
export the figure to the image file hist_train.png
"""
# TODO À compléter
# X_labeled,y_labeled,X_unlabeled=load_data('/data/YearPredictionMSD_100')

X_labeled,y_labeled,X_unlabeled=load_data('YearPredictionMSD_100.npz')

#plot de l'histogramme
plt.hist(y_labeled)
plt.xlabel('Years')
plt.ylabel('frequence')
plt.savefig('SortieFigure/hist_year.png')
plt.show()

