# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:28:14 2018

@author: i7
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
np.random.seed(42)

m = 1000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]


#what if we plot any two orginal features? Can they separate the class (number)?
plt.figure(figsize=(13,10))
plt.scatter(X[:, 300], X[:, 301], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()





from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()


#what if we plot any two orginal features? Can they separate the class (number)?
plt.figure(figsize=(13,10))
plt.scatter(X[:, 300], X[:, 301], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()