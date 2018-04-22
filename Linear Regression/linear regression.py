import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

# importing the dataset
from sklearn.datasets import load_boston
boston = load_boston()
print boston.keys()
print boston.data.shape

print boston.feature_names

print boston.DESCR

bos = pd.DataFrame(boston.data)

bos.head()

bos.columns = boston.feature_names

bos.head()

bos['PRICE'] = boston.target

from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE',axis =1)
lm = LinearRegression()

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
plt.show()


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, bos.PRICE, test_size = 0.25, random_state = 0)

lm.fit(X_train,Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

print "Fit a model X_train, calculate MSE with Y_train:"
print np.mean((Y_train - pred_train) ** 2)
print "Fit a model X_train, calculate MSE with X_test, Y_test:"
print np.mean((Y_test - pred_test) ** 2)

plt.scatter (lm.predict(X_train),lm.predict(X_train) - Y_train, c ='b',s =40,alpha = 0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test) - Y_test, c ='b',s =40,alpha = 0.5)
plt.hlines(y = 0,xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')
plt.show()
