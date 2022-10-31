
# import libraries and datasets
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



mnist = fetch_openml('mnist_784')
x = mnist['data']
y = mnist['target']

digit = x.to_numpy()[601]

# Reshaping Image
digitImage = digit.reshape(28, 28)  

# Plotting pixels 
plt.imshow(digitImage, cmap=matplotlib.cm.binary,interpolation='nearest')

plt.show()

x_train, x_test = x[0:5000], x[6000:7000]
y_train, y_test = y[0:5000], y[6000:7000]



# Creating a 8-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_new = (y_train == 8)
y_test_new = (y_test == 8)

# Training logistic regression classifier
z = LogisticRegression(tol=0.2)
z.fit(x_train, y_train_new)

z.predict([digit])

# Cross Validation
crossVal = cross_val_score(z, x_train, y_train_new, cv=3, scoring="accuracy")

crossVal.mean()