from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib import cm

data = np.loadtxt("DB/fv.list")
#data = preprocessing.scale(data)
target = np.loadtxt("DB/ev.score").reshape(-1,1)
#print data[:, 0:1].shape
#print target.shape
X_HD = np.hstack([data[:,1:2], target])
#print X_HD.shape

X_HDn = X_HD
#X_HDn=(X_HD - X_HD.mean(axis=0))/X_HD.std(axis=0)
time_vec=X_HDn[:,0:1]
#time_std=time_vec.reshape(-1,1)
f_vec = data[:,6:7]
ev_vec=X_HDn[:,1]
#ev_std=ev_vec.reshape(-1,1)

#X_train, X_test, y_train, y_test = train_test_split(data, ev_vec)

#clf1 = LinearRegression()
#clf1.fit(X_train, y_train)
#predicted_train = clf1.predict(X_train)
#predicted_test = clf1.predict(X_test)
#trains=X_train.reshape(1,-1).flatten()
#tests=X_test.reshape(1,-1).flatten()

#print clf1.coef_, clf1.intercept_

#plt.scatter(time_vec, ev_vec,c='r')
#plt.plot(trains, predicted_train, c='g', alpha=0.5)
#plt.plot(tests, predicted_test, c='g', alpha=0.2)

#plt.scatter(predicted_test, predicted_test- y_test, c='g', s=40)
#plt.scatter(predicted_train, predicted_train- y_train, c='b', s=40, alpha=0.5)
#plt.plot([0.4,2],[0,0])

#plt.show()

#print clf1.score(X_train, y_train), clf1.score(X_test, y_test)

#data = preprocessing.scale(data)
X = time_vec
Y = f_vec
Z = ev_vec
#print X, Y, Z
#from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure(figsize=(12,8))

# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
#ax = fig.add_subplot(1, 2, 1, projection='3d')

#p = ax.plot_surface(time_vec, f_vec, ev_vec, rstride=4, cstride=4, linewidth=0)

# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.scatter(time_vec, f_vec, ev_vec)

fig.show()

